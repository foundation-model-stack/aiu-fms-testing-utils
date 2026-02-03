import json
from aiu_fms_testing_utils.testing.validation import (
    LogitsExtractorHook,
    extract_validation_information,
)
from fms.models import get_model
from transformers import AutoTokenizer
# from concurrent.futures import ThreadPoolExecutor
# Ideally we want this script to fetch data in parallel
# But it's proving harder than initially thought
# Making it work for now, making it fast is second step

import argparse
import torch


def load_jsonl(path):
    """
    Loads a JSONL file.
    - If field is None: returns a list of dicts (one per line).
    - If field is a string: returns a list of obj[field] (only non-None values).
    """
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except (ValueError, json.JSONDecodeError) as e:
                print(f"Failed to parse line {idx} in {path}: {e}")
            data.append(obj)
    return data


parser = argparse.ArgumentParser(
    description="Script which will save CPU validation data"
)
parser.add_argument(
    "--attention_type",
    type=str,
    default="paged",
    choices=["paged", "paged_fp8"],
    help="The attention type to use",
)
parser.add_argument(
    "--model_variant",
    type=str,
    default="ibm-ai-platform/micro-g3.3-8b-instruct-1b",
    help="The model id or path to use for this test. Note: must be a huggingface format",
)
parser.add_argument(
    "--max_new_tokens",
    type=int,
    default=8,
    help="set this if you want to change the number of tokens generated per sequence (1 prefill + max_new_tokens-1 decodes). Note: If this value is larger than 64, this may result in switching decode programs mid generation",
)
parser.add_argument(
    "--max_workers",
    type=int,
    default=8,
    help="max workers to run in parallel",
)
parser.add_argument(
    "--dataset_path",
    type=str,
    help="path to dataset",
)
args = parser.parse_args()
max_new_tokens = args.max_new_tokens
is_fp8 = "fp8" in args.attention_type
model_variant = args.model_variant
tokenizer = AutoTokenizer.from_pretrained(model_variant)
model_path_kwargs = {"variant": model_variant}
validation_model = get_model(
    architecture="hf_pretrained",
    device_type="cpu",
    data_type=None if is_fp8 else torch.bfloat16,
    fused_weights=False,
    **model_path_kwargs,
)

# get the input ids for the validation
dataset = load_jsonl(args.dataset_path)


def process_row(row):
    id = row["id"]
    prompt_text = row["prompt"]
    input_ids = tokenizer.encode(prompt_text)
    print("fetching cpu validation info for id: ", id)
    with torch.no_grad():
        cpu_validation_info = extract_validation_information(
            validation_model,
            torch.tensor(input_ids).unsqueeze(0),
            max_new_tokens,
            LogitsExtractorHook(),
            attn_algorithm="math",
        )
    return {"id": id, "input_ids": input_ids, "validation": cpu_validation_info}


# See comment above
# with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
#     results = list(executor.map(process_row, dataset))

# save the results
validation_info = {}
for row in dataset:
    result = process_row(row)
    # for result in results:
    tokens = result["validation"].get_info("tokens")
    generated_tokens_tensor = tokens[0][-max_new_tokens:]
    generated_tokens = [token.item() for token in generated_tokens_tensor]
    logits = result["validation"].get_info("logits")
    top_logprobs = []
    for step_num, logits_for_step in enumerate(logits[0]):
        logprob_for_step = torch.nn.functional.log_softmax(logits_for_step, dim=-1)
        values, indices = torch.topk(logprob_for_step, k=100)
        # in case we want to save a new tensor?
        # but this will also take memory
        # top_logprobs = torch.full_like(logprobs, float('-inf'))
        # top_logprobs.scatter_(1, indices, values)
        top_logprob_dict = {
            int(idx): float(val) for idx, val in zip(indices[0], values[0])
        }
        top_logprobs.append(top_logprob_dict)
    validation_info[result["id"]] = {
        "logprobs": top_logprobs,
        "tokens": generated_tokens,
        "text": tokenizer.decode(generated_tokens),
    }
    with open(f"{result['id']}_cpu_validation_info.json", "w") as f:
        json.dump(validation_info, f, indent=4)
    print(f"Done for {result['id']}")


# save the final result
with open("cpu_validation_info.json", "w") as f:
    json.dump(validation_info, f, indent=4)
print("all done!")
