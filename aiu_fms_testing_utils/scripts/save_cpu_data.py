import json
import os
from aiu_fms_testing_utils.testing.validation import LogitsExtractorHook, extract_validation_information
from fms.models import get_model
from transformers import AutoTokenizer
from concurrent.futures import ThreadPoolExecutor

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
            data_type=None if is_fp8 else torch.float32,
            fused_weights=False,
            **model_path_kwargs,
        )

# get the input ids for the validation
dataset = load_jsonl(args.dataset_path)
def process_row(row):
    id = row["id"]
    prompt_text = row["prompt"]
    input_ids = torch.tensor(tokenizer.encode(prompt_text)).unsqueeze(0)
    print("fetching cpu validation info for id: ", id)
    cpu_validation_info = extract_validation_information(
            validation_model,
            input_ids,
            max_new_tokens,
            LogitsExtractorHook(),
            attn_algorithm="math",
        )
    return {
        "id": id,
        "input_ids": input_ids,
        "validation": cpu_validation_info
    }

for row in dataset:
    result = process_row(row)
# with ThreadPoolExecutor(max_workers=1) as executor:
#     results = list(executor.map(process_row, dataset))

    # save the results
    validation_info = {}
    # for result in results:
    tokens = result["validation"].get_info("tokens")
    generated_tokens = tokens[0][-max_new_tokens:]
    logits = result["validation"].get_info("logits")
    logprobs = []
    for step_num, logits_for_step in enumerate(logits[0]):
        logprobs.append(torch.nn.functional.log_softmax(logits_for_step, dim=-1))
    validation_info[result["id"]] = {
        "logprobs": logprobs,
        "tokens": generated_tokens,
        "text": "".join([tokenizer.decode(tensor.tolist(), \
                                            skip_special_tokens=True) for \
                        tensor in generated_tokens])
    }


    torch.save(validation_info, f"{result["id"]}-cpu_validation_info.pt")
    print("finished saving cpu validation info for id: ", result["id"])

