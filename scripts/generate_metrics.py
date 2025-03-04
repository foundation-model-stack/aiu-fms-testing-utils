import argparse
import json
import os
import random
from typing import List, Optional, Tuple

import torch

from aiu_fms_testing_utils.testing.validation import capture_level_1_metrics, extract_validation_information, LogitsExtractorHook, print_failed_cases, \
    validate_level_0, GoldenTokenHook, top_k_loss_calculator
from aiu_fms_testing_utils.utils import ids_for_prompt
from fms.models import get_model
from fms.utils import tokenizers
from fms.utils.generation import pad_input_ids

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
    default=20
)
args = parser.parse_args()


prefix = f"{args.variant.replace('/', '--')}_max-new-tokens-{args.max_new_tokens}_batch-size-{args.batch_size}_seq-length{args.min_pad_length}_dtype-{args.default_dtype}"
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

tokenizer = tokenizers.get_tokenizer(args.tokenizer)

torch.set_grad_enabled(False)

# prepare the cuda model
cuda_model = get_model(
    architecture=args.architecture,
    variant=args.variant,
    model_path=args.model_path,
    device_type="cuda",
    data_type=default_dtype,
)

print("loaded cuda model")

cuda_model.eval()

# prepare the cpu model (this is the reference)
cpu_model = get_model(
    architecture=args.architecture,
    variant=args.variant,
    model_path=args.model_path,
    device_type="cpu",
    data_type=torch.float32,
)
cpu_model.eval()
print("loaded cpu model")

def sample_sharegpt_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer,
) -> List[Tuple[str, int, int, None]]:
    # Load the dataset.
    with open(dataset_path, encoding='utf-8') as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [(data["conversations"][0]["value"],
                data["conversations"][1]["value"]) for data in dataset]

    # Shuffle the dataset.
    random.Random(42).shuffle(dataset)

    # Filter out sequences that are too long or too short
    filtered_dataset: List[Tuple[str, int, int]] = []
    for i in range(len(dataset)):
        if len(filtered_dataset) == num_requests:
            break

        # Tokenize the prompts and completions.
        prompt = dataset[i][0]
        prompt_token_ids = ids_for_prompt(prompt, tokenizer)
        
        prompt_len = len(prompt_token_ids)
        if prompt_len < 32 or prompt_len > args.min_pad_length:
            # Prune too short sequences.
            continue
        filtered_dataset.append((prompt, prompt_len))

    return filtered_dataset

def find_eos_index(reference_tokens, eos_token_id):
    result = []
    for sentence in reference_tokens:
        found_eos = False
        for token_idx, token in enumerate(sentence[args.min_pad_length:]):
            if token.item() == eos_token_id:
                found_eos = True
                result.append(token_idx)
                break
        if not found_eos:
            result.append(args.max_new_tokens)
    return result

def filter_before_eos(l, filter_indexes):
    from itertools import groupby
    filtered_results = [list(g)[:filter_indexes[k]] for k, g in groupby(l, key=lambda x: x[0])]
    return [item for sublist in filtered_results for item in sublist]
            
prompts_and_lens = sample_sharegpt_requests(args.sharegpt_path, args.batch_size, tokenizer)
print(f"prompt_lengths: {[pl[1] for pl in prompts_and_lens]}")
prompts = [ids_for_prompt(pl[0], tokenizer) for pl in prompts_and_lens]

padding_length = args.min_pad_length

has_padding = args.batch_size > 1 or padding_length != 0
max_len = max([len(prompt) for prompt in prompts])

if has_padding:
    ids, padding_kwargs = pad_input_ids(prompts, min_pad_length=padding_length)
else:
    ids = prompts
    padding_kwargs = {}

# first test validation level 0
cpu_validation_info = extract_validation_information(
    cpu_model,
    ids,
    args.max_new_tokens,
    LogitsExtractorHook(),
    attn_algorithm="math",
    **padding_kwargs
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
    only_last_token=True,
    **{k: v.to("cuda") for k,v in padding_kwargs.items()}
)
cuda_static_tokens = cuda_validation_info.get_info("tokens")
failed_responses = validate_level_0(cpu_static_tokens, cuda_static_tokens)

print("extracted cuda validation information level 0")
if len(failed_responses) != 0:    
    print_failed_cases(failed_responses, cpu_static_tokens, cuda_static_tokens, tokenizer)

# generate aiu validation info
cuda_validation_info = extract_validation_information(
    cuda_model,
    ids.to("cuda"),
    args.max_new_tokens,
    GoldenTokenHook(cpu_static_tokens, "cuda"),
    only_last_token=True,
    **{k: v.to("cuda") for k,v in padding_kwargs.items()}
)

print("extracted cuda validation information level 1")

cross_entropy = lambda r, t: torch.nn.CrossEntropyLoss()(r, t.softmax(dim=1).to(dtype=torch.float32))
prob_mean = lambda r, t: torch.mean((r.softmax(dim=1).to(dtype=torch.float32) / t.softmax(dim=1).to(dtype=torch.float32)) - 1.0)
prob_std = lambda r, t: torch.std(r.softmax(dim=1).to(dtype=torch.float32) / t.softmax(dim=1).to(dtype=torch.float32))
diff_mean = lambda r, t: torch.mean(r.softmax(dim=1).to(dtype=torch.float32) - t.softmax(dim=1).to(dtype=torch.float32))

def write_csv(l, path, metric):
    with open(path, 'w') as f:
        f.write(f'{metric}\n')
        for t in l:
            f.write(f"{t[2].item()}\n") 
        f.close()

prefix = f"{args.variant.replace('/', '--')}_max-new-tokens-{args.max_new_tokens}_batch-size-{args.batch_size}_seq-length{args.min_pad_length}_dtype-{args.default_dtype}"

cpu_validation_info.save(os.path.join(args.output_dir, f"{prefix}.cpu_output_logits.out"))
cuda_validation_info.save(os.path.join(args.output_dir, f"{prefix}.cuda_output_logits.out"))

level_1_metrics = capture_level_1_metrics(
    cpu_validation_info.get_info("logits"),
    cuda_validation_info.get_info("logits"),
    top_k_loss_calculator(args.topk_per_token, prob_mean),
)
loss_metrics = filter_before_eos(level_1_metrics, eos_indexes)
write_csv(loss_metrics, os.path.join(args.output_dir, f"{prefix}.prob_mean.csv"), "prob_mean")

level_1_metrics = capture_level_1_metrics(
    cpu_validation_info.get_info("logits"),
    cuda_validation_info.get_info("logits"),
    top_k_loss_calculator(args.topk_per_token, prob_std),
)
loss_metrics = filter_before_eos(level_1_metrics, eos_indexes)
write_csv(loss_metrics, os.path.join(args.output_dir, f"{prefix}.prob_std.csv"), "prob_std")

level_1_metrics = capture_level_1_metrics(
    cpu_validation_info.get_info("logits"),
    cuda_validation_info.get_info("logits"),
    top_k_loss_calculator(args.topk_per_token, cross_entropy),
)
loss_metrics = filter_before_eos(level_1_metrics, eos_indexes)
write_csv(loss_metrics, os.path.join(args.output_dir, f"{prefix}.ce.csv"), "ce")

level_1_metrics = capture_level_1_metrics(
    cpu_validation_info.get_info("logits"),
    cuda_validation_info.get_info("logits"),
    top_k_loss_calculator(args.topk_per_token, diff_mean),
)
loss_metrics = filter_before_eos(level_1_metrics, eos_indexes)
write_csv(loss_metrics, os.path.join(args.output_dir, f"{prefix}.diff_mean.csv"), "diff_mean")
