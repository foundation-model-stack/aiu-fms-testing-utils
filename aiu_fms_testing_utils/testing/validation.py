from pathlib import Path
from typing import List, Tuple, Callable, MutableMapping, Any, Optional

import torch
from aiu_fms_testing_utils.utils.aiu_setup import dprint
from aiu_fms_testing_utils._version import version_tuple
import os
from aiu_fms_testing_utils.testing.utils import format_kwargs_to_string

import hashlib


class LogitsExtractorHook(
    Callable[
        [int, torch.Tensor, torch.Tensor, MutableMapping[str, Any]],
        Tuple[torch.Tensor, MutableMapping[str, Any]],
    ]
):
    def __init__(self):
        super().__init__()
        self.extracted_logits: Optional[torch.Tensor] = None

    def __call__(
        self,
        token_position: torch.Tensor,
        logits: torch.Tensor,
        next_val: torch.Tensor,
        kwargs,
    ):
        if self.extracted_logits is None:
            self.extracted_logits = logits.unsqueeze(1)
        else:
            self.extracted_logits = torch.cat(
                (self.extracted_logits, logits.unsqueeze(1)), dim=1
            )
        return next_val, kwargs


class StaticTokenInjectorHook(
    Callable[
        [int, torch.Tensor, torch.Tensor, MutableMapping[str, Any]],
        Tuple[torch.Tensor, MutableMapping[str, Any]],
    ]
):
    def __init__(self, static_tokens: List[torch.Tensor], device_type: str = "cpu"):
        super().__init__()
        self.static_tokens = torch.tensor(
            static_tokens, device=device_type
        ).t()  # transposing so batch tokens per token_position

    def __call__(
        self, token_position: int, logits: torch.Tensor, next_val: torch.Tensor, kwargs
    ):
        next_val.copy_(self.static_tokens[token_position].unsqueeze(1))
        return next_val, kwargs


class GoldenTokenHook(
    Callable[
        [int, torch.Tensor, torch.Tensor, MutableMapping[str, Any]],
        Tuple[torch.Tensor, MutableMapping[str, Any]],
    ]
):
    def __init__(self, static_tokens: torch.Tensor, device_type: str = "cpu"):
        super().__init__()
        self.logits_extractor = LogitsExtractorHook()
        self.extracted_logits = None
        self.token_injector = StaticTokenInjectorHook(
            static_tokens, device_type=device_type
        )

    def __call__(
        self, token_position: int, logits: torch.Tensor, next_val: torch.Tensor, kwargs
    ):
        next_val, kwargs = self.logits_extractor(
            token_position, logits, next_val, kwargs
        )
        self.extracted_logits = self.logits_extractor.extracted_logits
        return self.token_injector(token_position, logits, next_val, kwargs)


class ValidationInfo:
    def __init__(self, validation_info_list):
        super().__init__()

        self._validation_info_list = validation_info_list

    def __iter__(self):
        for vi in self._validation_info_list:
            yield vi

    def get_info(self, info_name):
        return [
            [t.unsqueeze(0) for t in sentence[info_name]]
            for sentence in self._validation_info_list
        ]

    def save(self, save_dir_path: str):
        """Save the validation information into a directory.

        The files will be saved in the following structure:

        save_dir_path/                     # the path to save the files to
        ├── 0.pt                           # prompt 0
        ├── 1.pt                           # prompt 1
        ├── ...
        ├── <batch_size-1>.pt              # prompt <batch_size-1>

        The files will have the following format:

        if containing only tokens - torch.tensor
        if containing tokens and logits - dict[tokens -> torch.tensor, logits -> torch.tensor]

        :param save_dir_path: the path to save to
        """
        dprint(f"saving validation info to {save_dir_path}")
        os.makedirs(save_dir_path, exist_ok=True)

        for sentence_i, sentence in enumerate(self._validation_info_list):
            file_path = os.path.join(save_dir_path, f"{sentence_i}.pt")
            if len(sentence) > 1:
                torch.save(sentence, file_path)
            else:
                torch.save(list(sentence.values())[0], file_path)

    def __len__(self):
        return len(self._validation_info_list)


def get_default_validation_prefix(
    **kwargs,
):
    """
    Args:
        model_id (str): model name used
        max_new_tokens (int): number of max new tokens to generate
        batch_size (int): batch size used
        seq_length (int):sequence length used
        dtype (str): data type
        attn_type (str): type of attention
        aftu_version (str): introduced in v0.3.0 to track changed in log

    Returns:
        str: A hashed prefix that will be prepended to the file name
    """
    aftu_version = kwargs.pop(
        "aftu_version", ".".join([str(_) for _ in version_tuple[:3]])
    )
    kwargs_str = format_kwargs_to_string(**kwargs)

    filename = f"{kwargs_str}"
    hash_object = hashlib.sha256(filename.encode("utf-8"))
    hex_digest = hash_object.hexdigest()
    return f"{hex_digest}_{aftu_version}"


def load_validation_information(
    validation_path, validation_files_type, batch_size, tokenizer=None
):
    """Load the validation information from a directory

    The files will be assumed to be in the following structure:

    save_dir_path/                     # the path to save the files to
    ├── 0.pt                           # prompt 0
    ├── 1.pt                           # prompt 1
    ├── ...
    ├── <batch_size-1>.pt              # prompt <batch_size-1>

    The files will have the following format:

    if containing only tokens - torch.tensor
    if containing tokens and logits - dict[tokens -> torch.tensor, logits -> torch.tensor]
    if containing text - str

    :param validation_path: path to validation info files
    :param validation_files_type: validation file type to load, one of text, tokens, or logits
    :param batch_size: the number of prompts to load
    :param tokenizer: an optional tokenizer, required when validation_files_type=text
    :return: a new validation info
    """
    if isinstance(validation_path, str):
        validation_files_path, sep, glob_pattern = validation_path.partition("*")
    else:
        sep = ""
        glob_pattern = ""
    glob_pattern = sep + glob_pattern

    validation_files_path = Path(os.path.expanduser(validation_files_path))
    validation_files_paths = []

    if validation_files_path.is_dir():
        if glob_pattern != "":
            glob_pattern_list = [glob_pattern]
        else:
            if validation_files_type == "text":
                glob_pattern_list = ["*.txt"]
            elif validation_files_type == "tokens":
                glob_pattern_list = ["*.pt"]
            elif validation_files_type == "logits":
                glob_pattern_list = ["*.pt"]
        for glob_pattern_possibility in glob_pattern_list:
            file_list = list(validation_files_path.glob(glob_pattern_possibility))
            if len(file_list) > 0:
                validation_files_paths = sorted(file_list)
                break

    if validation_files_path.is_file():
        validation_files_paths = [validation_files_path]

    # Check if we found some files
    assert len(validation_files_paths) > 0, (
        f"Can't find any validation files at {validation_files_path}"
    )

    # Check if we have enough files
    assert len(validation_files_paths) >= batch_size, (
        f"Not enough validation files at {validation_files_path} for a batch size of {batch_size}"
    )

    validation_files_paths.sort(key=lambda p: int(p.name.split(".pt")[0]))
    validation_info = []
    for i, validation_file_path in enumerate(validation_files_paths):
        if i == batch_size:
            break
        if validation_files_type == "text":
            if tokenizer is None:
                raise ValueError(
                    "must provide a tokenizer when validation_files_type=text"
                )
            # Text format will get tokenized
            validation_info.append(
                {
                    "tokens": tokenizer.encode(
                        validation_file_path.read_text(encoding="utf-8"),
                        return_tensors="pt",
                    ).squeeze(0),
                    "logits": None,
                }
            )
        elif validation_files_type == "tokens":
            # Tokens are loaded as is
            # Assumption: the file contains the token tensor as-is
            validation_info.append(
                {
                    "tokens": torch.load(validation_file_path, map_location="cpu"),
                    "logits": None,
                }
            )
        elif validation_files_type == "logits":
            # Logits+tokens are loaded as is
            # Assumption: the file contains the dictionary with both tokens and logits
            validation_info.append(torch.load(validation_file_path, map_location="cpu"))

    return ValidationInfo(validation_info)


def extract_validation_information(
    model,
    input_ids,
    max_new_tokens,
    post_iteration_hook,
    attn_algorithm=None,
    eos_token_id=None,
    last_n_tokens=0,
    timing="",
    prefill_chunk_size=0,
    **extra_kwargs,
):
    attention_specific_kwargs = {}
    if "paged" in extra_kwargs.get("attn_name", "sdpa"):
        from aiu_fms_testing_utils.utils.paged import generate

        attention_specific_kwargs["prefill_chunk_size"] = prefill_chunk_size
    else:
        # TODO: Add a unified generation dependent on attn_type
        from fms.utils.generation import generate

        attention_specific_kwargs["contiguous_cache"] = True
        attention_specific_kwargs["max_seq_len"] = input_ids.shape[1] + max_new_tokens

    # Add last_n_tokens optimization
    extra_generation_kwargs = {**extra_kwargs}
    if last_n_tokens != 0:
        extra_generation_kwargs["last_n_tokens"] = last_n_tokens
    if attn_algorithm is not None:
        extra_generation_kwargs["attn_algorithm"] = attn_algorithm

    result = generate(
        model,
        input_ids,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        do_sample=False,
        post_iteration_hook=post_iteration_hook,
        eos_token_id=eos_token_id,
        timing=timing,
        extra_kwargs=extra_generation_kwargs,
        **attention_specific_kwargs,
    )

    if timing != "":
        dprint(
            "=== This timing information might be inaccurate due to extra work being done in generate() for validation"
        )
        result, timings = result
        if timing == "e2e":
            dprint(f"E2E timing information: {timings[0]:.3f}s")
        elif timing == "per-token":
            timings = [f"{t * 1000:.3f}" for t in timings]
            dprint(f"Per-token timing information: {', '.join(timings)} ms")

    if len(result.shape) == 1:
        result = result.unsqueeze(0)

    if hasattr(post_iteration_hook, "extracted_logits"):
        validation_info = [
            {"tokens": t.to("cpu"), "logits": logits.to("cpu")}
            for t, logits in zip(
                torch.unbind(result), torch.unbind(post_iteration_hook.extracted_logits)
            )
        ]
    else:
        validation_info = [{"tokens": t.to("cpu")} for t in torch.unbind(result)]
    return ValidationInfo(validation_info)


def validate_level_0(aiu_tokens_per_sentence, validation_tokens_per_sentence):
    failed_cases = []

    for sentence_idx, (aiu_sentence, validation_sentence) in enumerate(
        zip(aiu_tokens_per_sentence, validation_tokens_per_sentence)
    ):
        for token_idx, (aiu_token, validation_token) in enumerate(
            zip(aiu_sentence, validation_sentence)
        ):
            if aiu_token != validation_token:
                failed_cases.append((sentence_idx, token_idx))
    return failed_cases


def top_k_loss_calculator(
    top_k: int, loss_f: Callable[[torch.Tensor, torch.Tensor], float]
):
    """
    Function which will take the top_k logits indexes / values from a reference validation info and retrieve the same indexes from the test validation info logits
    and perform a loss function over the 2 tensors

    :param top_k: number of values to take from reference
    :param loss_f: a loss function between the reference and test logits
    """

    def loss_func(reference_logits, test_logits):
        reference_logits_prob = reference_logits.to(dtype=torch.float32)
        test_logits_prob = test_logits.to(dtype=torch.float32)

        reference_values, reference_indices = torch.topk(
            reference_logits_prob, top_k, dim=1
        )
        test_values = test_logits_prob[:, reference_indices.squeeze(0)]

        return loss_f(reference_values, test_values)

    return loss_func


def capture_level_1_metrics(
    reference_logits_per_sentence, test_logits_per_sentence, metrics_calculator=None
):
    loss_metrics = []

    for sentence_idx, (reference_sentence, test_sentence) in enumerate(
        zip(reference_logits_per_sentence, test_logits_per_sentence)
    ):
        for token_idx, (reference_logits, test_logits) in enumerate(
            zip(reference_sentence, test_sentence)
        ):
            # computing cross entropy loss per token
            if metrics_calculator is None:
                loss_fn = torch.nn.CrossEntropyLoss()
                metrics_value = loss_fn(
                    reference_logits.to(dtype=torch.float32),
                    test_logits.softmax(dim=1).to(dtype=torch.float32),
                )
            else:
                metrics_value = metrics_calculator(reference_logits, test_logits)

            loss_metrics.append((sentence_idx, token_idx, metrics_value))

    return loss_metrics


def filter_failed_level_1_cases(level_1_loss_metrics, fail_f, print_failed=False):
    failed_cases = []
    for sentence_idx, token_idx, metrics_value in level_1_loss_metrics:
        if fail_f(metrics_value):
            failed_cases.append((sentence_idx, token_idx, metrics_value))
            if print_failed:
                dprint(
                    f"In sentence {sentence_idx + 1}, the metric for token {token_idx} is {metrics_value}"
                )
    return failed_cases


def print_failed_cases(failed_cases, aiu_tokens, validation_tokens, tokenizer):
    for sentence_index, token_index in failed_cases:
        aiu_token = aiu_tokens[sentence_index][token_index]
        validation_token = validation_tokens[sentence_index][token_index]

        aiu_str = tokenizer.decode(aiu_token)
        validation_str = tokenizer.decode(validation_token)
        print(
            f"In sentence {sentence_index + 1}/{len(aiu_tokens)}, token {token_index}, AIU outputs {aiu_token} instead of {validation_token} -- AIU val={aiu_str} -- CPU val={validation_str}"
        )


def get_validation_info_path(
    validation_info_dir: str,
    model_variant: str,
    batch_size: int,
    seq_length: int,
    max_new_tokens: int,
    seed: int,
    attn_type: str,
    aftu_version: Optional[Tuple[int, int, int]] = None,
    device_type: str = "cpu",
    dtype: str = "fp16",
    **kwargs,
):
    if aftu_version is None:
        aftu_version = version_tuple

    sample_key = kwargs.get("sample_key", None)

    validation_file_name = f"{get_default_validation_prefix(aftu_version='.'.join([str(_) for _ in aftu_version[:3]]), model_id=model_variant, max_new_tokens=max_new_tokens, batch_size=batch_size, seq_length=seq_length, dtype=dtype, attn_type=attn_type, sample_key=sample_key)}.{device_type}_validation_info.{seed}.out"
    full_path = os.path.join(validation_info_dir, validation_file_name)
    return full_path


def __decrement_version(version: Tuple[int, int, int], max_minor=25, max_patch=25):
    """
    Function designed to prevent triple nested for loop while decrementing version
    """
    major, minor, patch = version
    if patch > 0:
        return (major, minor, patch - 1)
    elif minor > 0:
        return (major, minor - 1, max_patch)
    elif major > 0:
        return (major - 1, max_minor, max_patch)
    else:
        return None


def find_validation_info_path(
    validation_info_dir: str,
    model_variant: str,
    batch_size: int,
    seq_length: int,
    max_new_tokens: int,
    seed: int,
    attn_type: str,
    aftu_version: Optional[Tuple[int, int, int]] = None,
    version_allow_decrement: bool = False,
    device_type: str = "cpu",
    dtype: str = "fp16",
    **kwargs,
):
    """
    Find the validation info path if it exists, otherwise return None
    """
    sample_key = kwargs.get("sample_key", None)

    if aftu_version is None:
        loc_version_tuple = version_tuple[:3]
    else:
        loc_version_tuple = aftu_version

    result_path: Optional[str] = None

    while result_path is None and loc_version_tuple is not None:
        full_path = get_validation_info_path(
            validation_info_dir,
            model_variant,
            batch_size,
            seq_length,
            max_new_tokens,
            seed,
            attn_type,
            loc_version_tuple,
            device_type,
            dtype,
            sample_key=sample_key,
        )
        # if the path is found, we are done searching and can return
        if os.path.exists(full_path):
            result_path = full_path
        # if allow version decrements, decrement the version and continue
        elif version_allow_decrement:
            loc_version_tuple = __decrement_version(loc_version_tuple)
        # if path is not found and we are not allowing decrementing of version, finish with no result
        else:
            loc_version_tuple = None
    return result_path
