import itertools
import math
import os
from pathlib import Path
from subprocess import PIPE, Popen

import pytest

FMS_DIR = Path(__file__).parent
AIU_FMS_DIR = os.path.join(FMS_DIR, "../../../aiu-fms-testing-utils/")
INFERENCE_FILE_PATH = os.path.join(AIU_FMS_DIR, "scripts", "inference.py")

common_model_paths = os.environ.get("FMS_TESTING_COMMON_MODEL_PATHS", "")

# pass custom model path list for eg: EXPORT FMS_TESTING_COMMON_MODEL_PATHS="/tmp/models/granite-3-8b-base,/tmp/models/granite-7b-base"
if common_model_paths == "":
    common_model_paths = ["ibm-ai-platform/micro-g3.3-8b-instruct-1b"]
else:
    common_model_paths = common_model_paths.split(",")

common_batch_sizes = [1, 8]
common_seq_lengths = [64]
common_max_new_tokens = [12]
common_attn_types = ["sdpa", "paged"]

common_params = list(
    itertools.product(
        common_model_paths,
        common_batch_sizes,
        common_seq_lengths,
        common_max_new_tokens,
        common_attn_types,
    )
)

current_env = os.environ.copy()


def execute_script(execute_cmd):
    current_env["MAX_SHAREDPROG_ITERS"] = f"{common_max_new_tokens[0]}"

    with Popen(
        execute_cmd,
        stdin=PIPE,
        stdout=PIPE,
        stderr=PIPE,
        universal_newlines=True,
        env=current_env,
    ) as p:
        output, error = p.communicate()
        if p.returncode == 0:
            return output
        else:
            raise Exception(error)


def execute_inference(model_path, batch_size, seq_length, max_new_tokens, attn_type):
    extra_args = []
    if attn_type == "paged":
        extra_args.append("--compile_dynamic_sendnn")
        extra_args.append("--attention_type=paged")

    execute_cmd = [
        "python3",
        INFERENCE_FILE_PATH,
        "--architecture=hf_pretrained",
        f"--variant={model_path}",
        f"--tokenizer={model_path}",
        f"--max_new_tokens={max_new_tokens}",
        f"--min_pad_length={seq_length}",
        f"--batch_size={batch_size}",
        "--unfuse_weights",
        "--no_early_termination",
        "--compile_dynamic",
        "--compile",
        "--device_type=aiu",
        "--default_dtype=fp16",
    ]
    return execute_script(execute_cmd + extra_args)


common_asserts = [
    "### Response:\nProvide a list of instructions for preparing chicken soup",
    "### Response:\nExplain some popular greetings in Spanish.",
    "### Response:\nExplain to me why ignorance is bliss.",
    "### Response:\nI have just come into a very large sum of money",
]


def __repeat_batch_asserts(bs: int) -> list[str]:
    n_repeats = int(math.ceil(bs / len(common_asserts)))
    return (common_asserts * n_repeats)[:bs]


# add the asserts based on batch size
# for batches greater than common_asserts, repeat common_asserts since this follows inference behavior
common_inference_params = [
    common_param + (__repeat_batch_asserts(common_param[1]),)
    for common_param in common_params
]


@pytest.mark.parametrize(
    "model_path,batch_size,seq_length,max_new_tokens,attn_type,asserts",
    common_inference_params,
)
def test_inference_script(
    model_path, batch_size, seq_length, max_new_tokens, attn_type, asserts
):
    result_text = execute_inference(
        model_path, batch_size, seq_length, max_new_tokens, attn_type
    )

    for common_assert in asserts:
        assert common_assert in result_text
