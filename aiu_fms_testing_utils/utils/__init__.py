# Standard
from typing import Optional, List, Tuple
import json
import os
import random
import requests
import time

# Third Party

from aiu_fms_testing_utils.utils.aiu_setup import dprint, rank, world_size
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from fms.utils.generation import pad_input_ids
import torch
import torch.nn as nn
import math
import contextlib
import warnings


@contextlib.contextmanager
def stagger_region(limit: int):
    """
    Limit the number of concurrent processes into this region of code.
    Processes yield from this function when they are allowed to enter the region of code.
    Processes return from this function when all of the processes have completed the region of code.

    :param limit: Number of concurrent processes allowed in the code region if > 0.
    """
    if limit > 0 and limit != world_size:
        for _set in range(math.ceil(world_size / float(limit))):
            if rank < (_set + 1) * limit:
                break
            torch.distributed.barrier()
        dprint(
            f"Stagger: Enter (Set: {_set + 1} of {math.ceil(world_size / float(limit))})"
        )
    yield
    if limit > 0 and limit != world_size:
        for _set in range(math.ceil(world_size / float(limit))):
            if rank >= (_set + 1) * limit:
                continue
            torch.distributed.barrier()
        dprint("Stagger: All Complete")


def warmup_model(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    compile_dynamic_sendnn: bool = False,
    use_cache: bool = True,
    stagger_update_lazyhandle: int = 0,
    **extra_kwargs,
):
    import torch_sendnn

    attention_specific_kwargs = {}
    attn_name = extra_kwargs.get("attn_name", "sdpa")
    if "paged" in attn_name:
        from aiu_fms_testing_utils.utils.paged import generate, adjust_inputs_to_batch
    else:
        # TODO: Add a unified generation dependent on attn_type
        from fms.utils.generation import generate

        attention_specific_kwargs["contiguous_cache"] = True
        attention_specific_kwargs["max_seq_len"] = input_ids.shape[1] + max_new_tokens

    dprint("AIU warmup")
    pt_compile_model_time = time.time()

    # adjust inputs depending on attn_type and dynamic shapes
    _warmup_input_ids = input_ids
    _extra_kwargs = extra_kwargs
    _max_new_tokens = max_new_tokens
    if compile_dynamic_sendnn:
        _max_new_tokens = 2
        # always warmup with batch size 2 when using attn_type=paged
        if "paged" in attn_name:
            _warmup_input_ids, _extra_kwargs = adjust_inputs_to_batch(
                input_ids,
                **extra_kwargs,
            )

    extra_kwargs = {**_extra_kwargs, "only_last_token": "paged" not in attn_name}

    with stagger_region(stagger_update_lazyhandle):
        with torch_sendnn.warmup_mode():
            generate(
                model,
                _warmup_input_ids,
                max_new_tokens=_max_new_tokens,
                do_sample=False,
                use_cache=use_cache,
                extra_kwargs=extra_kwargs,
                **attention_specific_kwargs,
            )
    pt_compile_model_time = time.time() - pt_compile_model_time
    dprint(f"PT compile complete, took {pt_compile_model_time:.3f}s")


def __download_file(url, filename):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(filename, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Successfully downloaded {filename}")

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")


def get_pad_size(prompt_len: int, pad_multiple: int = 64):
    """
    Method to finding nearest prompt length with accepted padding multiple
    i.e.
        prompt length 65 with pad multiple = 64 returns 128
        prompt length 64 with pad multiple = 64 returns 64
    """
    # handle outliers and case where you cannot divide by 0
    if prompt_len <= 0 or pad_multiple == 0:
        if prompt_len < 0:
            warnings.warn(f"{prompt_len=} which should probably be > 0", stacklevel=2)
        return 0
    else:
        return ((prompt_len + pad_multiple - 1) // pad_multiple) * pad_multiple


def _merge_enforce_keep_heterogeneous(
    enforce_list: List[Tuple[str, int]],
    heterogeneous_list: List[Tuple[str, int]],
    batch_size: int,
):
    """
    Method for returning a list that contains both enforced sizes and is heterogeneous

    Args:
        enforce_list: List[Tuple[str, int]], a list of prompt/prompt_len that contains prompt_lens
            that must be enforced
        heterogeneous_list: List[Tuple[str, int]], a list of prompt/prompt_len where all prompt_lens
            are heterogeneous to the extent possible. i.e. if batch size is 3 but only 2 possible
            prompt length exists, this list will contain both prompt lengths with third item sharing
            the same prompt length as one of the previous items.
        batch_size: int, will define the final size of the list.

    Returns:
        List[Tuple[str,int]] that will have all elements from enforce_list
    """
    final_list = enforce_list.copy()
    unique_sizes = {num for _, num in enforce_list}
    for prompt, size in heterogeneous_list:
        if len(final_list) >= batch_size:
            break
        # if the size hasn't been covered by enforce_list, add to list to keep it heterogeneous
        if size not in unique_sizes:
            final_list.append((prompt, size))
            unique_sizes.add(size)
    if len(final_list) > batch_size:
        warnings.warn(
            f"Requested {batch_size=}, which is smaller than the enforced list, will return list larger than requested size",
            stacklevel=2,
        )
    elif len(final_list) < batch_size:
        warnings.warn(
            f"Requested {batch_size=}, than possible combined list. Will return smaller list than batch size",
            stacklevel=2,
        )
    return final_list


def __sample_requests(
    prompt_list: List[str],
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    prompt_length_min: int = 32,
    prompt_length_max: int = 64,
    seed: Optional[int] = None,
    enforce_heterogeneous: bool = False,
    enforce_sizes: List[int] = [],
    pad_multiple: int = 64,
):
    """
    Shuffles dataset, tokenizes the prompts and then filters

    Args:
        prompt_length_min (int): filters out prompts shorter than this value.
        prompt_length_max (int): filters out prompts larger than this value.
        enforce_sizes (List[int]): sample request will grab a prompt with this length if available.
        enforce_heterogeneous (bool): Pads all prompts within batch size to nearest multiple of 64.
        pad_multiple (int): Used only when enforce_heterogeneous is True or enforce_sizes is not empty, asserts that prompt_length would be padded to this multiple
        List[Tuple[str, int]]: a filtered dataset
    """

    # Based on min/max prompt length, one can back out the number of possible heterogeneous values
    max_heterogeneous_combinations = (prompt_length_max // pad_multiple) - (
        (prompt_length_min - 1) // pad_multiple
    )

    # Filter out sequences that are too long or too short
    filtered_dataset: List[Tuple[str, int]] = []
    enforced_dataset: List[Tuple[str, int]] = []

    # To track sizes seen
    seen_sizes: List[int] = []

    if enforce_sizes:
        for size in enforce_sizes:
            # Check that enforced sizes fall within min/max range
            assert prompt_length_min <= size <= prompt_length_max, (
                f"Size {size} in enforced sizes not within {prompt_length_min=}, {prompt_length_max=}"
            )
        if len(enforce_sizes) > num_requests:
            raise ValueError(
                f"{num_requests=} which is smaller than {len(enforce_sizes)=}"
            )

    # Shuffle the dataset.
    if seed is not None:
        random.Random(seed).shuffle(prompt_list)

    for i in range(len(prompt_list)):
        if len(filtered_dataset) == num_requests and not enforce_sizes:
            break

        # Tokenize the prompts and completions.
        prompt = prompt_list[i]
        prompt_token_ids = tokenizer.encode(prompt, return_tensors="pt").squeeze(0)

        prompt_len = len(prompt_token_ids)
        if prompt_len < prompt_length_min or prompt_len > prompt_length_max:
            # Prune too short or too long sequences.
            continue
        # This section is for enforce heterogeneous
        if (
            enforce_heterogeneous
            and max_heterogeneous_combinations > len(filtered_dataset)
            and len(filtered_dataset) < num_requests
        ):
            # for _, size in filtered_dataset:
            current_padded_size = get_pad_size(prompt_len, pad_multiple)

            # If it's in the list of enforce_sizes it is enforced, can remove from list
            if current_padded_size in enforce_sizes:
                enforce_sizes.remove(current_padded_size)
                enforced_dataset.append((prompt, prompt_len))

            if current_padded_size not in seen_sizes:
                filtered_dataset.append((prompt, prompt_len))
                seen_sizes.append(current_padded_size)
        # Forcing search for enforce_sizes
        elif enforce_sizes:
            current_padded_size = get_pad_size(prompt_len, pad_multiple)
            if current_padded_size in enforce_sizes:
                enforce_sizes.remove(current_padded_size)
                enforced_dataset.append((prompt, prompt_len))
        # when not enforcing heterogeneous or when exhausted all possible prompt_lengths
        else:
            filtered_dataset.append((prompt, prompt_len))
    assert not enforce_sizes, "Enforce size should be empty if all lengths are captured"

    if num_requests > max_heterogeneous_combinations:
        print(
            f"There will be prompt size repeats because {num_requests=} while {max_heterogeneous_combinations=}"
        )
    if enforced_dataset:
        filtered_dataset = _merge_enforce_keep_heterogeneous(
            enforced_dataset, filtered_dataset, num_requests
        )

    return filtered_dataset


def sample_sharegpt_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    prompt_length_min: int = 32,
    prompt_length_max: int = 64,
    seed: Optional[int] = None,
    enforce_heterogeneous: bool = False,
    enforce_sizes: List[int] = [],
    pad_multiple: int = 64,
) -> List[Tuple[str, int]]:
    if not os.path.exists(dataset_path):
        print("downloading share-gpt dataset as it does not exist")
        __download_file(
            "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json",
            dataset_path,
        )

    # Load the dataset.
    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    dataset: List[str] = [data["conversations"][0]["value"] for data in dataset]

    return __sample_requests(
        dataset,
        num_requests,
        tokenizer,
        prompt_length_min,
        prompt_length_max,
        seed,
        enforce_heterogeneous,
        enforce_sizes,
        pad_multiple,
    )


def sample_squad_v2_qa_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    prompt_length_min: int = 32,
    prompt_length_max: int = 64,
    seed: Optional[int] = None,
    enforce_heterogeneous: bool = False,
    enforce_sizes: List[int] = [],
    pad_multiple: int = 64,
) -> List[Tuple[str, int]]:
    from datasets import load_dataset

    if os.path.exists(dataset_path):
        ds = load_dataset(dataset_path)["train"]
    else:
        ds = load_dataset("rajpurkar/squad_v2", cache_dir=dataset_path)["train"]

    ds = [f"{data['context']}\n{data['question']}" for data in ds]

    return __sample_requests(
        ds,
        num_requests,
        tokenizer,
        prompt_length_min,
        prompt_length_max,
        seed,
        enforce_heterogeneous,
        enforce_sizes,
        pad_multiple,
    )


def prepare_inputs(
    batch_size, seq_length, tokenizer, ds_path, seed=0, ds_type="sharegpt"
):
    """
    Prepare input IDs and padding kwargs for a batch of questions.

    Args:
        batch_size (int): The number of questions in the batch.
        seq_length (int): The maximum length of the input sequence.
        tokenizer (Tokenizer): A tokenizer object to tokenize the questions.
        ds_path (str): The path to the dataset file.
        seed (int, optional): The random seed for reproducibility. Defaults to 0.
        ds_type (str, optional): The type of dataset to use. Can be "sharegpt" or any other supported dataset type. Defaults to "sharegpt".

    Returns:
        tuple: A tuple containing the input IDs and padding kwargs.
    """
    if "sharegpt" not in ds_type:
        prompts_and_sizes = sample_squad_v2_qa_requests(
            ds_path,
            batch_size,
            tokenizer,
            int(seq_length / 2),
            seq_length,
            seed,
        )
    else:
        prompts_and_sizes = sample_sharegpt_requests(
            ds_path,
            batch_size,
            tokenizer,
            int(seq_length / 2),
            seq_length,
            seed,
        )
    prompt_list = []
    for prompt, _ in prompts_and_sizes:
        prompt_list.append(tokenizer.encode(prompt, return_tensors="pt").squeeze(0))

    input_ids, padding_kwargs = pad_input_ids(prompt_list, min_pad_length=seq_length)
    return input_ids, padding_kwargs
