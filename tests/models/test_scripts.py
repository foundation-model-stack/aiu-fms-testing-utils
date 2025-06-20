import pytest, os
from subprocess import Popen, PIPE
from pathlib import Path
import itertools
import math

FMS_DIR = Path(__file__).parent
AIU_FMS_DIR = os.path.join(FMS_DIR, "../../../aiu-fms-testing-utils/")
VALIDATION_FILE_PATH = os.path.join(AIU_FMS_DIR, "scripts", "validation.py")
INFERENCE_FILE_PATH = os.path.join(AIU_FMS_DIR, "scripts", "inference.py")


model_dir = os.environ.get("FMS_TESTING_MODEL_DIR", "/tmp/models")

LLAMA_194M = f"{model_dir}/llama-194m"
GRANITE_7B_BASE = f"{model_dir}/granite-7b-base"
GRANITE_8B_CODE_BASE = f"{model_dir}/granite-8b-code-base"
GRANITE_3_8B_CODE_BASE = f"{model_dir}/granite-3-8b-base"

# pass custom model path list for eg: EXPORT FMS_TESTING_COMMON_MODEL_PATHS="/tmp/models/granite-3-8b-base,/tmp/models/granite-7b-base"
if (
    os.environ.get("FMS_TESTING_COMMON_MODEL_PATHS") == None
    or os.environ.get("FMS_TESTING_COMMON_MODEL_PATHS") == ""
):
    common_model_paths = [LLAMA_194M]
else:
    common_model_paths = os.environ.get("FMS_TESTING_COMMON_MODEL_PATHS").split(",")

common_batch_sizes = [1, 8]
common_seq_lengths = [64]
common_max_new_tokens = [8]

common_params = list(
    itertools.product(
        common_model_paths,
        common_batch_sizes,
        common_seq_lengths,
        common_max_new_tokens,
    )
)
common_asserts = [
    "### Response: Chicken soup is a popular soup that is",
    "### Response: I am sorry, but I am not",
    "### Response: I am ignorant of the fact that I",
    "### Response: I have just come into a very large",
]

current_env = os.environ.copy()
current_env["DT_OPT"] = (
    "varsub=1,lxopt=1,opfusion=1,arithfold=1,dataopt=1,patchinit=1,patchprog=1,autopilot=1,weipreload=0,kvcacheopt=1,progshareopt=1"
)


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


# we are forcing the number of layers to be 2 to reduce the size of the model as we do not care about output, but just consistency between cpu and aiu
def execute_validation(
    validation_level,
    model_path,
    max_new_tokens,
    batch_size,
    seq_length,
    logits_loss_threshold=0.0,
):
    execute_cmd = [
        "python3",
        VALIDATION_FILE_PATH,
        "--architecture=hf_pretrained",
        f"--model_path={model_path}",
        f"--tokenizer={model_path}",
        f"--max_new_tokens={max_new_tokens}",
        f"--min_pad_length={seq_length}",
        f"--batch_size={batch_size}",
        "--unfuse_weights",
        "--no_early_termination",
        f"--validation_level={validation_level}",
        f"--logits_loss_threshold={logits_loss_threshold}",
        "--compile_dynamic",
    ]
    return execute_script(execute_cmd)


def execute_inference(model_path, max_new_tokens, batch_size, seq_length):
    execute_cmd = [
        "python3",
        INFERENCE_FILE_PATH,
        "--architecture=hf_pretrained",
        f"--model_path={model_path}",
        f"--tokenizer={model_path}",
        f"--max_new_tokens={max_new_tokens}",
        f"--min_pad_length={seq_length}",
        f"--batch_size={batch_size}",
        "--unfuse_weights",
        "--no_early_termination",
        "--compile_dynamic",
        "--compile",
        "--device_type=aiu",
    ]
    return execute_script(execute_cmd)


@pytest.mark.parametrize(
    "model_path,batch_size,seq_length,max_new_tokens", common_params
)
def test_level_1_validation_script(model_path, batch_size, seq_length, max_new_tokens):
    result_text = execute_validation(
        1, model_path, max_new_tokens, batch_size, seq_length, 64.0
    )
    assert "The validation has passed!" in result_text


@pytest.mark.parametrize(
    "model_path,batch_size,seq_length,max_new_tokens", common_params
)
def test_level_0_validation_script(model_path, batch_size, seq_length, max_new_tokens):
    result_text = execute_validation(
        0,
        model_path,
        max_new_tokens,
        batch_size,
        seq_length,
    )
    assert "The validation has passed!" in result_text


common_asserts = [
    "### Response: Chicken soup is a popular soup that is",
    "### Response: I am sorry, but I am not",
    "### Response: I am ignorant of the fact that I",
    "### Response: I have just come into a very large",
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
    "model_path,batch_size,seq_length,max_new_tokens,asserts", common_inference_params
)
def test_inference_script(model_path, max_new_tokens, seq_length, batch_size, asserts):
    result_text = execute_inference(model_path, max_new_tokens, batch_size, seq_length)

    for common_assert in asserts:
        assert common_assert in result_text
