import argparse
import ast
import os

import torch
from torch import distributed as dist
from aiu_fms_testing_utils.testing.validation import (
    capture_level_1_metrics,
    extract_validation_information,
    LogitsExtractorHook,
    get_default_validation_prefix,
    load_validation_information,
    print_failed_cases,
    validate_level_0,
    GoldenTokenHook,
    top_k_loss_calculator,
)
from aiu_fms_testing_utils.utils import sample_sharegpt_requests
from fms.models import get_model
from fms.utils.generation import pad_input_ids
from transformers import AutoTokenizer

parser = argparse.ArgumentParser(
    description="Script to determine a reasonable logits loss threshold when testing with aiu"
)
parser.add_argument(
    "--architecture",
    type=str,
    help="The model architecture to benchmark",
)
parser.add_argument(
    "--variant",
    type=str,
    default=None,
    help="The model variant (configuration) to benchmark. E.g. 7b, 13b, 70b.",
)
parser.add_argument(
    "--model_path",
    type=str,
    help="Path to the directory containing LLaMa weights (.pth files sharded by tensor parallel rank, not HF weights)",
)
parser.add_argument(
    "--model_source",
    type=str,
    help="Source of the checkpoint. E.g. 'meta', 'hf', None",
)
parser.add_argument(
    "--tokenizer",
    type=str,
    required=True,
    help="Path to the tokenizer (e.g. ~/tokenizer.model)",
)
parser.add_argument(
    "--default_dtype",
    type=str,
    default=None,
    choices=["bf16", "fp16", "fp32"],
    help="If set to one of the choices, overrides the model checkpoint weight format by setting the default pytorch format",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=1,
    help="size of input batch",
)
parser.add_argument(
    "--min_pad_length",
    type=int,
    help="Pad inputs to a minimum specified length. If any prompt is larger than the specified length, padding will be determined by the largest prompt",
    default=0,
)
parser.add_argument(
    "--max_new_tokens",
    type=int,
    help="max number of generated tokens",
    default=100,
)
parser.add_argument(
    "--sharegpt_path",
    type=str,
    help="path to sharegpt data json",
)
parser.add_argument(
    "--output_dir",
    type=str,
    help="output directory",
)
parser.add_argument(
    "--topk_per_token",
    type=int,
    help="top k values per token to generate loss on",
    default=20,
)
parser.add_argument(
    "--num_test_tokens_per_sequence",
    type=int,
    help="number of tokens in test. For instance, if max_new_tokens=128 and num_test_tokens_per_sequence=256, this means we will generate data over 2 sample prompts. If not set, will be set to max_new_tokens",
    default=None,
)
parser.add_argument(
    "--extra_get_model_kwargs",
    nargs="*",
    default={},
    help="Use this to override model configuration values to get model. Example: --extra_get_model_kwargs nlayers=2,...",
)
parser.add_argument(
    "--distributed",
    action="store_true",
    help="This is a distributed job (multiple instances run with RANK+WORLD_SIZE)",
)
parser.add_argument(
    "--skip_computation",
    action="store_true",
    help="Set this if the output is already assumed to be computed and would like to regenerate metrics without model loading or computation",
)
local_rank = int(os.getenv("LOCAL_RANK", 0))
world_size = int(os.getenv("WORLD_SIZE", 1))
args = parser.parse_args()

if args.distributed:
    dist.init_process_group()
    distr_param = "tp"
    torch.cuda.set_device(local_rank)
else:
    distr_param = None

extra_get_model_kwargs = {}
for a in args.extra_get_model_kwargs:
    a_split = a.split("=")
    try:
        extra_get_model_kwargs[a_split[0]] = ast.literal_eval(a_split[1])
    except ValueError:
        extra_get_model_kwargs[a_split[0]] = a_split[1]

# this follows the same pattern of naming in test_shapes. This way we can save and re-use for quicker shape testing.
prefix = get_default_validation_prefix(
    model_id=args.variant,
    max_new_tokens=args.max_new_tokens,
    batch_size=args.batch_size,
    seq_len=args.min_pad_length,
    dtype=args.default_dtype,
)
if os.path.exists(os.path.join(args.output_dir, f"{prefix}.prob_mean.csv")):
    print("skipping metric generation as it has already been done")
    exit(0)

default_dtype = None
dtypes_map = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}
if args.default_dtype is not None:
    default_dtype = dtypes_map[args.default_dtype]

if default_dtype is not None:
    torch.set_default_dtype(default_dtype)

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

torch.set_grad_enabled(False)


def find_eos_index(reference_tokens, eos_token_id):
    result = []
    for sentence in reference_tokens:
        found_eos = False
        for token_idx, token in enumerate(sentence[args.min_pad_length :]):
            if token.item() == eos_token_id:
                found_eos = True
                result.append(token_idx)
                break
        if not found_eos:
            result.append(args.max_new_tokens)
    return result


def filter_before_eos(metrics, filter_indexes):
    from itertools import groupby

    filtered_results = [
        list(g)[: filter_indexes[k]] for k, g in groupby(metrics, key=lambda x: x[0])
    ]
    return [item for sublist in filtered_results for item in sublist]


def __prepare_inputs(batch_size, seq_length, tokenizer, seed=0):
    prompts_and_sizes = sample_sharegpt_requests(
        args.sharegpt_path, batch_size, tokenizer, seq_length // 2, seq_length, seed
    )
    prompt_list = []
    for prompt, _ in prompts_and_sizes:
        prompt_list.append(tokenizer.encode(prompt, return_tensors="pt").squeeze(0))

    input_ids, padding_kwargs = pad_input_ids(prompt_list, min_pad_length=seq_length)
    return input_ids, padding_kwargs


def write_csv(metrics, path, metric_name):
    with open(path, "w") as f:
        f.write(f"{metric_name}\n")
        for t in metrics:
            f.write(f"{t[2].item()}\n")
        f.close()


# prepare the cuda model
if not args.skip_computation:
    cuda_model = get_model(
        architecture=args.architecture,
        variant=args.variant,
        model_path=args.model_path,
        device_type="cuda",
        data_type=default_dtype,
        distributed_strategy=distr_param,
        group=dist.group.WORLD,
        **extra_get_model_kwargs,
    )

    cuda_model.eval()
    print("loaded cuda model")

    # prepare the cpu model (this is the reference)
    cpu_model = get_model(
        architecture=args.architecture,
        variant=args.variant,
        model_path=args.model_path,
        device_type="cpu",
        data_type=torch.float32,
        distributed_strategy=distr_param,
        group=dist.group.WORLD,
        **extra_get_model_kwargs,
    )
    cpu_model.eval()
    print("loaded cpu model")

    ids, padding_kwargs = __prepare_inputs(
        args.batch_size, args.min_pad_length, tokenizer
    )

    # first test validation level 0
    cpu_validation_info = extract_validation_information(
        cpu_model,
        ids,
        args.max_new_tokens,
        LogitsExtractorHook(),
        attn_algorithm="math",
        **padding_kwargs,
    )
    cpu_static_tokens = cpu_validation_info.get_info("tokens")
    print("extracted cpu validation information")

    eos_indexes = find_eos_index(cpu_static_tokens, tokenizer.eos_token_id)
    print(f"valid testing tokens per sequence: {eos_indexes}")

    # generate cpu validation info
    cuda_validation_info = extract_validation_information(
        cuda_model,
        ids.to("cuda"),
        args.max_new_tokens,
        None,
        last_n_tokens=1,
        **{k: v.to("cuda") for k, v in padding_kwargs.items()},
    )
    cuda_static_tokens = cuda_validation_info.get_info("tokens")
    failed_responses = validate_level_0(cpu_static_tokens, cuda_static_tokens)

    print("extracted cuda validation information level 0")
    if local_rank == 0:
        if len(failed_responses) != 0:
            print_failed_cases(
                failed_responses, cpu_static_tokens, cuda_static_tokens, tokenizer
            )

num_test_tokens_per_sequence = args.num_test_tokens_per_sequence
if num_test_tokens_per_sequence is None:
    num_test_tokens_per_sequence = args.max_new_tokens

cross_entropy = lambda r, t: torch.nn.CrossEntropyLoss()(  # noqa: E731
    r, t.softmax(dim=1).to(dtype=torch.float32)
)
prob_mean = lambda r, t: torch.mean(  # noqa: E731
    (
        r.softmax(dim=1).to(dtype=torch.float32)
        / t.softmax(dim=1).to(dtype=torch.float32)
    )
    - 1.0
)
prob_std = lambda r, t: torch.std(  # noqa: E731
    r.softmax(dim=1).to(dtype=torch.float32) / t.softmax(dim=1).to(dtype=torch.float32)
)
diff_mean = lambda r, t: torch.mean(  # noqa: E731
    torch.abs(
        r.softmax(dim=1).to(dtype=torch.float32)
        - t.softmax(dim=1).to(dtype=torch.float32)
    )
)

prob_mean_metrics = []
prob_std_metrics = []
prob_diff_metrics = []
prob_ce_loss_metrics = []

for i in range(num_test_tokens_per_sequence // args.max_new_tokens):
    cpu_path = os.path.join(args.output_dir, f"{prefix}.cpu_validation_info.{i}.out")
    cuda_path = os.path.join(args.output_dir, f"{prefix}.cuda_validation_info.{i}.out")
    if os.path.exists(cpu_path) and os.path.exists(cuda_path):
        print(f"found the logits at {cpu_path}, reusing")
        cpu_validation_info = load_validation_information(
            cpu_path, "logits", args.batch_size, tokenizer
        )
        cuda_validation_info = load_validation_information(
            cuda_path, "logits", args.batch_size, tokenizer
        )
    elif not args.skip_computation:
        ids, padding_kwargs = __prepare_inputs(
            args.batch_size, args.min_pad_length, tokenizer, i
        )

        # only need to compute this once if we aren't generating more test data
        if num_test_tokens_per_sequence > args.max_new_tokens:
            cpu_validation_info = extract_validation_information(
                cpu_model,
                ids,
                args.max_new_tokens,
                LogitsExtractorHook(),
                attn_algorithm="math",
                **padding_kwargs,
            )

        # generate aiu validation info
        cuda_validation_info = extract_validation_information(
            cuda_model,
            ids.to("cuda"),
            args.max_new_tokens,
            GoldenTokenHook(cpu_validation_info.get_info("tokens"), "cuda"),
            last_n_tokens=1,
            **{k: v.to("cuda") for k, v in padding_kwargs.items()},
        )

        print("extracted cuda validation information level 1")

        if local_rank == 0:
            cpu_validation_info.save(cpu_path)
            cuda_validation_info.save(cuda_path)

    eos_indexes = find_eos_index(
        cpu_validation_info.get_info("tokens"), tokenizer.eos_token_id
    )
    level_1_metrics = capture_level_1_metrics(
        cpu_validation_info.get_info("logits"),
        cuda_validation_info.get_info("logits"),
        top_k_loss_calculator(args.topk_per_token, prob_mean),
    )
    prob_mean_metrics.extend(filter_before_eos(level_1_metrics, eos_indexes))

    level_1_metrics = capture_level_1_metrics(
        cpu_validation_info.get_info("logits"),
        cuda_validation_info.get_info("logits"),
        top_k_loss_calculator(args.topk_per_token, prob_std),
    )
    prob_std_metrics.extend(filter_before_eos(level_1_metrics, eos_indexes))

    level_1_metrics = capture_level_1_metrics(
        cpu_validation_info.get_info("logits"),
        cuda_validation_info.get_info("logits"),
        top_k_loss_calculator(args.topk_per_token, cross_entropy),
    )
    prob_ce_loss_metrics.extend(filter_before_eos(level_1_metrics, eos_indexes))

    level_1_metrics = capture_level_1_metrics(
        cpu_validation_info.get_info("logits"),
        cuda_validation_info.get_info("logits"),
        top_k_loss_calculator(args.topk_per_token, diff_mean),
    )
    prob_diff_metrics.extend(filter_before_eos(level_1_metrics, eos_indexes))

if local_rank == 0:
    write_csv(
        prob_mean_metrics,
        os.path.join(args.output_dir, f"{prefix}.prob_mean.csv"),
        "prob_mean",
    )
    write_csv(
        prob_std_metrics,
        os.path.join(args.output_dir, f"{prefix}.prob_std.csv"),
        "prob_std",
    )
    write_csv(
        prob_ce_loss_metrics, os.path.join(args.output_dir, f"{prefix}.ce.csv"), "ce"
    )
    write_csv(
        prob_diff_metrics,
        os.path.join(args.output_dir, f"{prefix}.diff_mean.csv"),
        "diff_mean",
    )
