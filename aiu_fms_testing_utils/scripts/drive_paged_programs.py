import argparse
from dataclasses import dataclass
import datetime
import itertools
import json
import os
from pathlib import Path
import random
import time
from itertools import dropwhile
import re

import torch
from fms.models import get_model
from fms.utils.generation import pad_input_ids
from torch import distributed as dist
from torch.fx.experimental import _config as fx_config
from transformers import AutoTokenizer

from aiu_fms_testing_utils.testing.validation import (
    GoldenTokenHook,
    LogitsExtractorHook,
    capture_level_1_metrics,
    extract_validation_information,
    filter_failed_level_1_cases,
    find_validation_info_path,
    get_validation_info_path,
    load_validation_information,
    top_k_loss_calculator,
)
from aiu_fms_testing_utils.utils import (
    get_pad_size,
    sample_rag_factoid_requests,
    sample_sharegpt_requests,
    stagger_region,
    warmup_model,
)
from aiu_fms_testing_utils.utils.aiu_setup import aiu_dist_setup, dprint, local_rank
from aiu_fms_testing_utils.utils.paged import (
    ProgramCriteria,
    get_programs_prompts,
)
from aiu_fms_testing_utils.utils.dpp_config import DPPRunnerConfig
from aiu_fms_testing_utils.utils.env_utils import scoped_environ
from aiu_fms_testing_utils.testing.utils import format_kwargs_to_string

parser = argparse.ArgumentParser(
    description="Script which will drive paged programs for debugging"
)
parser.add_argument(
    "--programs",
    metavar="N",
    type=str,
    nargs="*",
    default=[],
    help="""
    The list of programs to run. This would take a list where each element would be one of program_id OR <program_id>:<min_batch>,<min_prompt_length>.
    If program_id is specified any prompt that would result in this program would be selected.
    If <program_id>:<min_batch>,<min_prompt_length> is specified, then with the given program_id, select a prompt that satisfies min_batch and min_prompt_length (if none exists, a message will be printed to warn the user)
    If this list is empty, each program will be run once with any prompt that would result in this program being selected.
    """,
)
parser.add_argument(
    "--max_new_tokens",
    type=int,
    default=8,
    help="set this if you want to change the number of tokens generated per sequence (1 prefill + max_new_tokens-1 decodes). Note: If this value is larger than 64, this may result in switching decode programs mid generation",
)
parser.add_argument(
    "--distributed",
    action="store_true",
    help="This is a distributed job (multiple instances run with RANK+WORLD_SIZE)",
)
parser.add_argument(
    "--model_variant",
    type=str,
    default="ibm-ai-platform/micro-g3.3-8b-instruct-1b",
    help="The model id or path to use for this test. Note: must be a huggingface format",
)
parser.add_argument(
    "--timing",
    type=str,
    choices=["e2e", "per-token"],
    default="",
    help="if set, how to time the generation of tokens, e2e or per-token",
)
parser.add_argument(
    "--program_criteria_json_path",
    type=str,
    help="path to json file containing the program criteria list",
)
parser.add_argument(
    "--dataset_path",
    type=str,
    help="path to dataset",
)
parser.add_argument(
    "--dataset_type",
    type=str,
    choices=["rag_factoid", "sharegpt", "custom"],
    default="sharegpt",
    help="selects the correct dataset type for sampling. Must be one of rag_factoid or sharegpt",
)
parser.add_argument(
    "--test_type",
    type=str,
    choices=["tokens", "metrics"],
    default="metrics",
    help="set the type of the test that you would like to run. If metrics, will inject tokens and get metrics. If tokens, will not inject tokens and get tokens",
)

parser.add_argument(
    "--cross_entropy_threshold",
    type=float,
    default=2.5,
    help="threshold to denote passing/failing a given iteration",
)

parser.add_argument(
    "--failure_rate_threshold",
    type=float,
    default=0.1,
    help="the threshold which denotes whether to pass or fail the test. The failure threshold is defined as the number of failing iterations (cross_entropy) over the total iterations. If this value exceeds the failure_rate_threshold, we will fail the test",
)

parser.add_argument(
    "--attention_type",
    type=str,
    default="paged",
    choices=["paged", "paged_fp8"],
    help="The attention type to use",
)
parser.add_argument(
    "--prefill_chunk_size",
    type=int,
    default=0,
    help="if > 0, activate chunked prefill, with chunk_size=this_argument. Only works with paged attention variants.",
)
parser.add_argument(
    "--stagger_load",
    type=int,
    default=0,
    help="Limit the number of concurrent processes executing the model loading phase. Set to 0 to allow all processes",
)
parser.add_argument(
    "--stagger_update_lazyhandle",
    type=int,
    default=0,
    help="Limit the number of concurrent processes executing the AIU update_lazyhandle phase. Set to 0 to allow all processes",
)
parser.add_argument(
    "--dist_timeout",
    type=int,
    default=0,
    help="Timeout to use for messaging in minutes. Default set by PyTorch dist.init_process_group",
)
parser.add_argument(
    "--skip_validation",
    action="store_true",
    help="set to true to skip cpu validation",
)
parser.add_argument(
    "--validation_info_outputs_dir",
    type=str,
    default="/home/senuser/models/validation_info",
    help="path to directory containing validation info outputs",
)
parser.add_argument(
    "--save_validation_info_outputs",
    action="store_true",
    help="set to true to save cpu validation outputs for later consumption",
)
parser.add_argument(
    "--prioritize_large_batch_sizes",
    action="store_true",
    help="set to true if you would like to prioritize large batch sizes",
)
parser.add_argument(
    "--enforce_homogeneous_prompt_programs",
    action="store_true",
    help="set to true ensure that all prompts hit the same prompt program for a given test",
)

args = parser.parse_args()

# interleave the decodes for programs (not 3 separate generates)
max_new_tokens = args.max_new_tokens
model_variant = args.model_variant
DATASET_PATH = args.dataset_path
save_validation_info_outputs = args.save_validation_info_outputs
tokenizer = AutoTokenizer.from_pretrained(model_variant)
custom_shape = None

if args.dataset_type == "custom":
    if local_rank == 0:
        dprint(
            "Using custom prompts from user, programs parameter will be ignored as it will be determined by user prompt"
        )
    directory = Path(DATASET_PATH)
    if not directory.is_dir():
        dprint("when using a custom dataset, you must provide a directory")
        exit()

    result = []
    for fp in directory.iterdir():
        if fp.is_file():
            try:
                content = fp.read_text()
                result.append((content, get_pad_size(len(tokenizer.encode(content)))))
            except Exception as e:
                print(f"Error while reading {fp} for custom dataset: {e}")
                exit()
    custom_shape = (len(result), max([_[1] for _ in result]))

    def __custom_line_sampler(*args, **kwargs):
        return_key = kwargs.get("return_key", False)
        sample_key = format_kwargs_to_string(**kwargs)
        if return_key:
            return result, sample_key
        return result

    sampler = __custom_line_sampler
    allow_truncation = False
elif args.dataset_type == "rag_factoid":
    sampler = sample_rag_factoid_requests
    allow_truncation = False
elif args.dataset_type == "sharegpt":
    sampler = sample_sharegpt_requests
    allow_truncation = True
else:
    raise ValueError("dataset_type must be one of rag_factoid or sharegpt")

if args.skip_validation and args.test_type == "metrics":
    dprint("When skipping validation, only test_type will be ignored")


USE_DISTRIBUTED = args.distributed
TIMING = args.timing
is_fp8 = "fp8" in args.attention_type

attention_map = {
    "sdpa": "sdpa_causal",
    "paged": "spyre_paged_attn",
    "math_fp8": "math_fp8",
    "paged_fp8": "spyre_paged_attn_fp8",
}
ATTN_NAME = attention_map[args.attention_type]
CPU_DTYPE = "fp8" if is_fp8 else "fp32"

torch.manual_seed(42)
torch.set_grad_enabled(False)
os.environ["COMPILATION_MODE"] = "offline_decoder"
os.environ["DT_PROG_CRITERIA_FILEPATH"] = args.program_criteria_json_path
if (
    "VLLM_DT_MAX_CONTEXT_LEN" not in os.environ
    or "VLLM_DT_MAX_BATCH_SIZE" not in os.environ
):
    if local_rank == 0:
        dprint(
            "Please specify VLLM_DT_MAX_CONTEXT_LEN and VLLM_DT_MAX_BATCH_SIZE environment variables"
        )
    exit()

max_batch_size = int(os.environ["VLLM_DT_MAX_BATCH_SIZE"])
max_tkv = int(os.environ["VLLM_DT_MAX_CONTEXT_LEN"])


def __prepare_inputs(batch_size, seq_length, tokenizer, enforce_sizes=[], seed=0):
    start = time.time()
    prompts_and_sizes, sample_key = sampler(
        DATASET_PATH,
        batch_size,
        tokenizer,
        32,
        seq_length * 2 if allow_truncation else seq_length,
        seed,
        enforce_sizes=enforce_sizes,
        truncation=allow_truncation,
        return_key=True,
    )
    end = time.time()
    if local_rank == 0:
        dprint(f"extracted prompts in {(end - start):.4f} seconds")
    prompt_list = []
    for prompt, size in prompts_and_sizes:
        encoded = tokenizer.encode(prompt, return_tensors="pt").squeeze(0)
        if size > seq_length:
            assert allow_truncation
            encoded = encoded[:seq_length]
        prompt_list.append(encoded)

    if not prompt_list:
        raise ValueError(
            f"No valid prompt sample exists in dataset for input shape (Batch Size={batch_size}, Seq Length={seq_length})"
        )
    if len(prompt_list) < batch_size:
        dprint(
            f"You requested {batch_size} prompts but we were only able to get {len(prompt_list)} valid prompts. We will be repeating the first prompt."
        )
        prompt_list = [prompt_list[0]] * (batch_size - len(prompt_list)) + prompt_list

    input_ids, extra_kwargs = pad_input_ids(prompt_list, min_pad_length=seq_length)
    extra_kwargs["mask"] = extra_kwargs["mask"].to(torch.float16)
    return input_ids, extra_kwargs, sample_key


def __maybe_prepare_fp8_weights(model_in, is_fp8):
    if is_fp8:
        for name, param in model_in.named_parameters():
            if param.dtype == torch.bfloat16:
                if param.max() > torch.finfo(torch.float16).max:
                    dprint(
                        f"[WARNING] You are casting param {name} to fp16, which will cause loss of accuracy. You can ignore this warning if this is intended."
                    )
                param.data = param.data.to(dtype=torch.float16)


def __load_validation_info(
    model_variant,
    batch_size,
    seq_length,
    max_new_tokens,
    tokenizer,
    seed,
    attn_type: str,
    **kwargs,
):
    sample_key = kwargs.get("sample_key", None)
    full_path = find_validation_info_path(
        args.validation_info_outputs_dir,
        model_variant,
        batch_size,
        seq_length,
        max_new_tokens,
        seed,
        attn_type,
        version_allow_decrement=True,
        dtype=CPU_DTYPE,
        sample_key=sample_key,
    )
    if full_path is not None:
        dprint(f"cpu validation info found for seed={seed} -- loading it")
        return load_validation_information(full_path, "logits", batch_size, tokenizer)
    else:
        return None


model_path_kwargs = {}
if os.path.exists(model_variant):
    model_path_kwargs = {"model_path": model_variant}
else:
    model_path_kwargs = {"variant": model_variant}

distributed_kwargs = {}
if USE_DISTRIBUTED:
    if args.dist_timeout > 0:
        # Default timeout:
        # https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group
        dist.init_process_group(timeout=datetime.timedelta(minutes=args.dist_timeout))
        dprint(f"NOTICE: init_process_group timeout set to {args.dist_timeout} minutes")
    else:
        dist.init_process_group()
    aiu_dist_setup(dist.get_rank(), dist.get_world_size())
    distributed_kwargs["distributed_strategy"] = "tp"
    distributed_kwargs["group"] = dist.group.WORLD
    save_validation_info_outputs = save_validation_info_outputs and (
        dist.get_rank() == 0
    )

with stagger_region(args.stagger_load):
    model = get_model(
        architecture="hf_pretrained",
        device_type="cpu",
        data_type=None if is_fp8 else torch.float16,
        fused_weights=False,
        **model_path_kwargs,
        **distributed_kwargs,
    )

model.eval()
fx_config.backed_size_oblivious = True

model_config = DPPRunnerConfig()
world_size = dist.get_world_size() if USE_DISTRIBUTED and dist.is_initialized() else 1
model_config.setup_config(
    model_variant, USE_DISTRIBUTED, world_size, args.prefill_chunk_size
)
with scoped_environ(model_config.env_updates()):
    # Temporarily set environment variables needed for compile
    model.compile(backend="sendnn", options={"sendnn.dynamic": True})

__maybe_prepare_fp8_weights(model, is_fp8)

if not args.skip_validation:
    with stagger_region(args.stagger_load):
        validation_model = get_model(
            architecture="hf_pretrained",
            device_type="cpu",
            data_type=None if is_fp8 else torch.float32,
            fused_weights=False,
            **model_path_kwargs,
            **distributed_kwargs,
        )
    validation_model.eval()

# warmup with any input so compiler produces criteria json
# TODO: Swap this with __prepare_inputs once fix for shape_id is available
# input_ids, extra_kwargs, sample_key = __prepare_inputs(2, max_tkv, tokenizer)
prompt_list = [torch.arange(0, 64, dtype=torch.int64)]
# matching vllm warmup to pad to 2 on fp8, and no pad for fp16
if is_fp8:
    prompt_list = prompt_list * 2
input_ids, extra_kwargs = pad_input_ids(prompt_list, min_pad_length=64)

extra_kwargs["mask"] = extra_kwargs["mask"].to(torch.float16)
extra_kwargs["attn_name"] = ATTN_NAME
extra_kwargs["_kvcache_num_blocks_hint"] = model_config.num_blocks
warmup_model(
    model,
    input_ids,
    max_new_tokens=max_new_tokens,
    compile_dynamic_sendnn=True,
    stagger_update_lazyhandle=args.stagger_update_lazyhandle,
    prefill_chunk_size=args.prefill_chunk_size,
    **extra_kwargs,
)

if USE_DISTRIBUTED:
    # wait for rank0 to be finished as it is the only one generating the criteria json
    # this is needed since otherwise we may run into a race condition
    torch.distributed.barrier()


@dataclass
class ProgramInfo:
    program_id: str
    batch_size_limit: int
    batch_size_limit_type: str
    prompt_length_limit: int
    prompt_length_limit_type: str


def parse_program_limit(limit_str: str) -> tuple[int, str]:
    matcher = re.compile(r"^(<|>|<=|>=|==)(\d+)")

    # Default limit to min to maintain backwards compat
    try:
        limit_type = ">="
        limit_val = int(limit_str)
    except ValueError:
        limit_type = None
        match = matcher.fullmatch(limit_str)
        if match is None:
            raise ValueError("Program not well formatted, wrong limit type")
        limit_type = match.group(1)
        limit_val = int(match.group(2))
    return limit_val, limit_type


with open(args.program_criteria_json_path, "r") as f:
    program_criteria_json_list = json.load(f)["programs"]
    program_criteria_list = []
    for i, d in enumerate(program_criteria_json_list):
        program_criteria_list.append(
            ProgramCriteria(
                i,
                d["max_batch"],
                d["max_tkv"],
                d["batch_granularity"],
                d["tkv_granularity"],
            )
        )

    programs = []

    for program_str in args.programs:
        enforce_prompt_split = program_str.split(":")
        program_id = enforce_prompt_split[0]
        if len(enforce_prompt_split) == 1:
            programs.append(
                ProgramInfo(program_id, 0, ">=", 0, ">=")
            )  # this will always satisfy
        else:
            enforce_batch_size, enforce_prompt_length = (
                _ for _ in enforce_prompt_split[1].split(",")
            )

            # Default limit to min to maintain backwards compat
            enforce_batch_size_val, enforce_batch_size_type = parse_program_limit(
                enforce_batch_size
            )
            enforce_prompt_length_val, enforce_prompt_length_type = parse_program_limit(
                enforce_prompt_length
            )

            programs.append(
                ProgramInfo(
                    program_id,
                    enforce_batch_size_val,
                    enforce_batch_size_type,
                    enforce_prompt_length_val,
                    enforce_prompt_length_type,
                )
            )

    if len(programs) == 0:
        programs = [
            ProgramInfo(str(p.program_id), 0, ">=", 0, ">=")
            for p in program_criteria_list
        ]


# FIXME: filter condition for this on prompt and batch
program_map = get_programs_prompts(
    program_criteria_list,
    multiple=64,
    max_batch_size=max_batch_size,
    max_tkv=max_tkv,
    program_cycles=max_new_tokens,
    tkv_limit=model_config.tkv_limit,
    prioritize_large_batch_sizes=args.prioritize_large_batch_sizes,
)
for v in program_map.values():
    random.Random(42).shuffle(v)


# select prompts that fit the batch size criteria
def get_program_prompt_list():
    if custom_shape:
        prompt_found = 0
        for program_criteria_seq, valid_prompt_shapes in program_map.items():
            for valid_prompt_shape in valid_prompt_shapes:
                if valid_prompt_shape == custom_shape:
                    enforce_sizes = [valid_prompt_shape[1]]
                    input_ids, extra_kwargs, sample_key = __prepare_inputs(
                        valid_prompt_shape[0],
                        valid_prompt_shape[1],
                        tokenizer,
                        enforce_sizes=enforce_sizes,
                    )
                    prompt_found = 1
                    yield (
                        program_criteria_seq[0].program_id,
                        custom_shape,
                        input_ids,
                        extra_kwargs,
                        sample_key,
                    )
                    break
            if prompt_found:
                break
    else:
        for program_info in programs:
            program_id = program_info.program_id
            batch_size_limit = program_info.batch_size_limit
            batch_size_limit_type = program_info.batch_size_limit_type
            prompt_length_limit = program_info.prompt_length_limit
            prompt_length_limit_type = program_info.prompt_length_limit_type

            filtered_program_map = program_map
            if program_id.isnumeric():
                filtered_program_map = {
                    k: v
                    for k, v in program_map.items()
                    if k[0] == program_criteria_list[int(program_id)]
                }
            used_keys = set()
            # for each program, we need to check if we have a shape that satisfies the --programs request
            for program_seq_key, valid_prompt_shapes in filtered_program_map.items():
                # if ? or numeric => we need to check if we have found at least one valid key to stop
                if (program_id == "?" or program_id.isnumeric()) and len(used_keys) > 0:
                    break
                # if * => we need to see if we have found the first key to see if we should skip
                elif program_id == "*" and program_seq_key[0] in used_keys:
                    continue

                for valid_prompt_shape in valid_prompt_shapes:
                    # make sure the criteria for batch limit and prompt limit is satisfied
                    # eval is safe here because we have limited what type and limit can be before

                    batch_check = eval(
                        f"valid_prompt_shape[0] {batch_size_limit_type} {batch_size_limit}"
                    )
                    prompt_check = eval(
                        f"valid_prompt_shape[1] {prompt_length_limit_type} {prompt_length_limit}"
                    )
                    if batch_check and prompt_check:
                        # when we enforce homogeneous prompt programs, we will cycle through all sizes between the min of a program and the valid prompt sequence length
                        # if there does not exist enough sequence sizes between this range, we will cycle back to the beginning
                        # in the event we don't have enough sequences that satisfy the enforce_sizes, we will repeat sequences and warn the user
                        enforce_sizes = [valid_prompt_shape[1]]
                        if args.enforce_homogeneous_prompt_programs:
                            # this will get the number of bits for the sequence length and shift to get the power of 2 that is less than or equal to the sequence length
                            tkv_cutoff = 1 << (valid_prompt_shape[1].bit_length() - 1)
                            possible_seq_lengths = [
                                _ for _ in range(tkv_cutoff, valid_prompt_shape[1], 64)
                            ]
                            # favor sequences that are close to the valid prompt length
                            possible_seq_lengths.reverse()
                            enforce_sizes = enforce_sizes + list(
                                itertools.islice(
                                    itertools.cycle(possible_seq_lengths),
                                    valid_prompt_shape[0] - 1,
                                )
                            )
                        try:
                            input_ids, extra_kwargs, sample_key = __prepare_inputs(
                                valid_prompt_shape[0],
                                valid_prompt_shape[1],
                                tokenizer,
                                enforce_sizes=enforce_sizes,
                            )
                            used_keys.add(program_seq_key[0])
                            yield (
                                program_seq_key[0],
                                valid_prompt_shape,
                                input_ids,
                                extra_kwargs,
                                sample_key,
                            )
                            break
                        except ValueError:
                            dprint(
                                f"No valid sample exists in dataset for this input shape {valid_prompt_shape}"
                            )

            if len(used_keys) == 0 and local_rank == 0:
                dprint(
                    f"no valid prompt shape was found which would result in program {program_id} that satisfied batch{batch_size_limit_type}{batch_size_limit} and prompt_length{prompt_length_limit_type}{prompt_length_limit}"
                )


# metric calculator based on the cross-entropy and mean diff for each decode step
def __metric_calculator(r: torch.Tensor, t: torch.Tensor):
    cross_entropy = torch.nn.CrossEntropyLoss()(
        r, t.softmax(dim=1).to(dtype=torch.float32)
    )
    diff = torch.mean(
        torch.abs(
            r.softmax(dim=1).to(dtype=torch.float32)
            - t.softmax(dim=1).to(dtype=torch.float32)
        )
    )
    return (cross_entropy, diff)


failed_cases = []
# for each program and valid prompt (batch size, sequence length)
for (
    program_id,
    valid_prompt,
    input_ids,
    extra_kwargs,
    sample_key,
) in get_program_prompt_list():
    extra_kwargs["attn_name"] = ATTN_NAME
    extra_kwargs["_kvcache_num_blocks_hint"] = model_config.num_blocks

    if local_rank == 0:
        dprint(f"*** testing program {program_id} ***")
        dprint(
            f"program id: {program_id}, valid prompt: {valid_prompt}, input shape: {input_ids.shape}"
        )

    if not args.skip_validation:
        # attempt to load the cpu validation info if it is already computed
        cpu_validation_info = __load_validation_info(
            model_variant,
            valid_prompt[0],
            valid_prompt[1],
            max_new_tokens,
            tokenizer,
            seed=0,
            attn_type=ATTN_NAME,
            sample_key=sample_key,
        )
        # if the cpu validation info is not yet computed, compute it
        if cpu_validation_info is None:
            cpu_validation_info = extract_validation_information(
                validation_model,
                input_ids,
                max_new_tokens,
                LogitsExtractorHook(),
                attn_algorithm="math",
                **extra_kwargs,
            )
            # save the cpu validation info for later consumption
            if save_validation_info_outputs:
                cpu_validation_info.save(
                    get_validation_info_path(
                        args.validation_info_outputs_dir,
                        model_variant,
                        valid_prompt[0],
                        valid_prompt[1],
                        max_new_tokens,
                        0,
                        ATTN_NAME,
                        dtype=CPU_DTYPE,
                        sample_key=sample_key,
                    )
                )

        if args.test_type == "metrics":
            aiu_validation_info = extract_validation_information(
                model,
                input_ids,
                max_new_tokens,
                GoldenTokenHook(cpu_validation_info.get_info("tokens")),
                last_n_tokens=64,
                timing=TIMING,
                prefill_chunk_size=args.prefill_chunk_size,
                **extra_kwargs,
            )

            # capture all level 1 metrics
            level_1_metrics = capture_level_1_metrics(
                cpu_validation_info.get_info("logits"),
                aiu_validation_info.get_info("logits"),
                top_k_loss_calculator(20, __metric_calculator),
            )

            cpu_tokens = cpu_validation_info.get_info("tokens")

            for sentence_idx, token_idx, metrics_value in level_1_metrics:
                if local_rank == 0:
                    aiu_token = torch.argmax(
                        aiu_validation_info.get_info("logits")[sentence_idx][token_idx],
                        dim=-1,
                    )
                    cpu_token = cpu_tokens[sentence_idx][valid_prompt[1] + token_idx]
                    aiu_str = tokenizer.decode(aiu_token).replace(
                        "\n", "<NEWLINE>"
                    )  # remove newlines for readability
                    cpu_str = tokenizer.decode(cpu_token).replace(
                        "\n", "<NEWLINE>"
                    )  # remove newlines for readability
                    dprint(
                        f'For Program {program_id} in sentence {sentence_idx + 1}: the metric for token {token_idx} is {metrics_value}, AIU ID="{aiu_token.item()}" | STR="{aiu_str}" -- CPU ID="{cpu_token.item()}" | CPU STR="{cpu_str}"'
                    )

            ce_fail_responses = filter_failed_level_1_cases(
                level_1_metrics, lambda m: m[0] >= args.cross_entropy_threshold
            )
            failure_rate = len(ce_fail_responses) / len(level_1_metrics)
            if failure_rate >= args.failure_rate_threshold:
                failed_cases.append((program_id, valid_prompt, failure_rate))

        elif args.test_type == "tokens":
            aiu_validation_info = extract_validation_information(
                model,
                input_ids,
                max_new_tokens,
                None,
                last_n_tokens=64,
                timing=TIMING,
                prefill_chunk_size=args.prefill_chunk_size,
                **extra_kwargs,
            )

            if local_rank == 0:
                for sentence_idx, (reference_sentence, test_sentence) in enumerate(
                    zip(
                        cpu_validation_info.get_info("tokens"),
                        aiu_validation_info.get_info("tokens"),
                    )
                ):
                    tokens_prompt = [
                        t.item() for t in reference_sentence[:-max_new_tokens]
                    ]
                    cpu_tokens_generated = [
                        t.item() for t in reference_sentence[-max_new_tokens:]
                    ]
                    aiu_tokens_generated = [
                        t.item() for t in test_sentence[-max_new_tokens:]
                    ]
                    tokens_prompt_without_pad = list(
                        dropwhile(lambda x: x == tokenizer.pad_token_id, tokens_prompt)
                    )
                    prompt_length = len(
                        [token_id for token_id in tokens_prompt_without_pad]
                    )
                    dprint(f"Prompt Length: {prompt_length}")
                    dprint(f"For Program {program_id} in sentence {sentence_idx + 1}:")
                    dprint(f"Prompt:\n{tokenizer.decode(tokens_prompt_without_pad)}")
                    dprint(f"CPU tokens:\n{cpu_tokens_generated}")
                    dprint(f"AIU tokens:\n{aiu_tokens_generated}")
                    dprint(f"CPU output:\n{tokenizer.decode(cpu_tokens_generated)}")
                    dprint(f"AIU output:\n{tokenizer.decode(aiu_tokens_generated)}")
        else:
            raise ValueError("test type must be one of metrics or tokens")
    else:
        aiu_validation_info = extract_validation_information(
            model,
            input_ids,
            max_new_tokens,
            None,
            last_n_tokens=64,
            timing=TIMING,
            prefill_chunk_size=args.prefill_chunk_size,
            **extra_kwargs,
        )

        if local_rank == 0:
            for sentence_idx, test_sentence in enumerate(
                aiu_validation_info.get_info("tokens")
            ):
                tokens_prompt = [t.item() for t in test_sentence[:-max_new_tokens]]
                aiu_tokens_generated = [
                    t.item() for t in test_sentence[-max_new_tokens:]
                ]
                dprint(f"For Program {program_id} in sentence {sentence_idx + 1}:")
                dprint(f"Prompt:\n{tokenizer.decode(tokens_prompt)}")
                dprint(f"AIU tokens:\n{aiu_tokens_generated}")
                dprint(f"AIU output:\n{tokenizer.decode(aiu_tokens_generated)}")

if not args.skip_validation and local_rank == 0:
    if len(failed_cases) != 0:
        dprint("the test failed with the following cases:")
        for failed_case in failed_cases:
            dprint(
                f"Program ID: {failed_case[0]}, Prompt Shape: {failed_case[1]}, Failure Rate: {failed_case[2]}"
            )
    else:
        dprint("all tests passed")
