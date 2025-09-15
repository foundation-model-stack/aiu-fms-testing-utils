from aiu_fms_testing_utils.utils.paged import ProgramCriteria, get_programs_prompts
import torch
import os
import random
from fms.models import get_model
from fms.utils.generation import pad_input_ids
from aiu_fms_testing_utils.utils.aiu_setup import aiu_dist_setup, local_rank, dprint
from aiu_fms_testing_utils.testing.validation import (
    extract_validation_information,
    LogitsExtractorHook,
    GoldenTokenHook,
    capture_level_1_metrics,
    filter_failed_level_1_cases,
    top_k_loss_calculator,
)
from torch import distributed as dist
from torch.fx.experimental import _config as fx_config

from aiu_fms_testing_utils.utils import sample_sharegpt_requests, warmup_model
from transformers import AutoTokenizer
import json
import argparse

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
    "--share_gpt_path",
    type=str,
    help="path to share gpt file",
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

args = parser.parse_args()

# interleave the decodes for programs (not 3 separate generates)
max_new_tokens = args.max_new_tokens
model_variant = args.model_variant
SHARE_GPT_DATASET_PATH = args.share_gpt_path
USE_DISTRIBUTED = args.distributed
TIMING = args.timing
warmed_up = False
is_fp8 = "fp8" in args.attention_type

attention_map = {
    "sdpa": "sdpa_causal",
    "paged": "spyre_paged_attn",
    "math_fp8": "math_fp8",
    "paged_fp8": "spyre_paged_attn_fp8",
}
ATTN_NAME = attention_map[args.attention_type]

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
        if len(enforce_prompt_split) == 1:
            programs.append(
                (int(enforce_prompt_split), 0, 0)
            )  # this will always satisfy
        else:
            program_id = int(enforce_prompt_split[0])
            enforce_batch_size, enforce_prompt_length = (
                int(_) for _ in enforce_prompt_split[1].split(",")
            )
            programs.append((program_id, enforce_batch_size, enforce_prompt_length))

    if len(programs) == 0:
        programs = [(p.program_id, 0, 0) for p in program_criteria_list]

torch.manual_seed(42)
torch.set_grad_enabled(False)
os.environ["COMPILATION_MODE"] = "offline_decoder"
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

# FIXME: filter condition for this on prompt and batch
program_map = get_programs_prompts(
    program_criteria_list,
    multiple=64,
    max_batch_size=max_batch_size,
    max_tkv=max_tkv,
    program_cycles=max_new_tokens,
)
for v in program_map.values():
    random.Random(42).shuffle(v)

# select prompts that fit the batch size criteria
valid_prompts = []
for program_id, min_batch_size, min_prompt_length in programs:
    found_valid_prompt = False
    for valid_prompt_shape in program_map[(program_criteria_list[program_id],)]:
        # make sure the criteria for min batch and min prompt is satisfied
        if (
            valid_prompt_shape[0] >= min_batch_size
            and valid_prompt_shape[1] >= min_prompt_length
        ):
            valid_prompts.append((program_id, valid_prompt_shape))
            found_valid_prompt = True
            break

    if not found_valid_prompt:
        if local_rank == 0:
            dprint(
                f"no valid prompt shape was found which would result in program {program_id} that satisfied min_batch={min_batch_size} and min_prompt_length={min_prompt_length}"
            )


def __prepare_inputs(batch_size, seq_length, tokenizer, seed=0):
    prompts_and_sizes = sample_sharegpt_requests(
        SHARE_GPT_DATASET_PATH,
        batch_size,
        tokenizer,
        32,
        seq_length,
        seed,
        enforce_sizes=[seq_length],
    )
    prompt_list = []
    for prompt, _ in prompts_and_sizes:
        prompt_list.append(tokenizer.encode(prompt, return_tensors="pt").squeeze(0))

    input_ids, extra_kwargs = pad_input_ids(prompt_list, min_pad_length=seq_length)
    return input_ids, extra_kwargs


def __maybe_prepare_fp8_weights(model_in, is_fp8):
    if is_fp8:
        for name, param in model_in.named_parameters():
            if param.dtype == torch.bfloat16:
                if param.max() > torch.finfo(torch.float16).max:
                    dprint(
                        f"[WARNING] You are casting param {name} to fp16, which will cause loss of accuracy. You can ignore this warning if this is intended."
                    )
                param.data = param.data.to(dtype=torch.float16)


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


model_path_kwargs = {}
if os.path.exists(model_variant):
    model_path_kwargs = {"model_path": model_variant}
else:
    model_path_kwargs = {"variant": model_variant}

distributed_kwargs = {}
if USE_DISTRIBUTED:
    dist.init_process_group()
    aiu_dist_setup(dist.get_rank(), dist.get_world_size())
    distributed_kwargs["distributed_strategy"] = "tp"
    distributed_kwargs["group"] = dist.group.WORLD

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
model.compile(backend="sendnn", options={"sendnn.dynamic": True})

__maybe_prepare_fp8_weights(model, is_fp8)

validation_model = get_model(
    architecture="hf_pretrained",
    device_type="cpu",
    data_type=None if is_fp8 else torch.float32,
    fused_weights=False,
    **model_path_kwargs,
    **distributed_kwargs,
)
validation_model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_variant)
failed_cases = []
for program_id, valid_prompt in valid_prompts:  # for each program
    input_ids, extra_kwargs = __prepare_inputs(
        valid_prompt[0], valid_prompt[1], tokenizer
    )
    extra_kwargs["attn_name"] = ATTN_NAME
    # warmup aiu model
    if not warmed_up:
        warmup_model(
            model,
            input_ids,
            max_new_tokens=max_new_tokens,
            compile_dynamic_sendnn=True,
            **extra_kwargs,
        )
        warmed_up = True

    if local_rank == 0:
        dprint(f"*** testing program {program_id} ***")
        dprint(
            f"program id: {program_id}, valid prompt: {valid_prompt}, input shape: {input_ids.shape}"
        )

    cpu_validation_info = extract_validation_information(
        validation_model,
        input_ids,
        max_new_tokens,
        LogitsExtractorHook(),
        attn_algorithm="math",
        **extra_kwargs,
    )

    if args.test_type == "metrics":
        aiu_validation_info = extract_validation_information(
            model,
            input_ids,
            max_new_tokens,
            GoldenTokenHook(cpu_validation_info.get_info("tokens")),
            only_last_token=False,
            timing=TIMING,
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
            only_last_token=False,
            timing=TIMING,
            **extra_kwargs,
        )

        if local_rank == 0:
            for sentence_idx, (reference_sentence, test_sentence) in enumerate(
                zip(
                    cpu_validation_info.get_info("tokens"),
                    aiu_validation_info.get_info("tokens"),
                )
            ):
                tokens_prompt = [t.item() for t in reference_sentence[:-max_new_tokens]]
                cpu_tokens_generated = [
                    t.item() for t in reference_sentence[-max_new_tokens:]
                ]
                aiu_tokens_generated = [
                    t.item() for t in test_sentence[-max_new_tokens:]
                ]
                dprint(f"For Program {program_id} in sentence {sentence_idx + 1}:")
                dprint(f"Prompt:\n{tokenizer.decode(tokens_prompt)}")
                dprint(f"CPU tokens:\n{cpu_tokens_generated}")
                dprint(f"AIU tokens:\n{aiu_tokens_generated}")
                dprint(f"CPU output:\n{tokenizer.decode(cpu_tokens_generated)}")
                dprint(f"AIU output:\n{tokenizer.decode(aiu_tokens_generated)}")
    else:
        raise ValueError("test type must be one of metrics or tokens")

if local_rank == 0:
    if len(failed_cases) != 0:
        dprint("the test failed with the following cases:")
        for failed_case in failed_cases:
            dprint(
                f"Program ID: {failed_case[0]}, Prompt Shape: {failed_case[1]}, Failure Rate: {failed_case[2]}"
            )
    else:
        dprint("all tests passed")
