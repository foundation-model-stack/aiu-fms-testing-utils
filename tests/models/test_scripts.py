import pytest
import os
from subprocess import Popen, PIPE
from pathlib import Path
import itertools
import math

FMS_DIR = Path(__file__).parent
AIU_FMS_DIR = os.path.join(FMS_DIR, "../../../aiu-fms-testing-utils/")
INFERENCE_FILE_PATH = os.path.join(AIU_FMS_DIR, "scripts", "inference.py")
DPP_FILE_PATH = os.path.join(AIU_FMS_DIR, "scripts", "drive_paged_programs.py")
SHARED_DIR = os.environ.get("FMS_TESTING_SHARED_MODEL_DIRECTORY", "/mnt/home/models")
common_model_paths = os.environ.get("FMS_TESTING_COMMON_MODEL_PATHS", "")

# pass custom model path list for eg: EXPORT FMS_TESTING_COMMON_MODEL_PATHS="/tmp/models/granite-3-8b-base,/tmp/models/granite-7b-base"
if common_model_paths == "":
    common_model_paths = ["ibm-ai-platform/micro-g3.3-8b-instruct-1b"]
else:
    common_model_paths = common_model_paths.split(",")

common_batch_sizes = [1, 4]
common_seq_lengths = [64]
common_max_new_tokens = [8]
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


def execute_inference(
    model_path, batch_size, seq_length, max_new_tokens, attn_type, allow_symbolic_shapes
):
    extra_args = []
    if attn_type == "paged":
        # paged needs symbolic shapes
        extra_args.append("--attention_type=paged")
        # using these options temporarily
        current_env.setdefault("VLLM_DT_MAX_BATCH_TKV_LIMIT", "16384")
        current_env.setdefault("VLLM_DT_MAX_BATCH_SIZE", "4")
        current_env.setdefault("VLLM_DT_MAX_CONTEXT_LEN", "4096")
    else:
        # added in case symbolic shapes used with sdpa
        current_env.setdefault("_PROMPT_LEN", "64")
        current_env.setdefault("_MAX_DECODE_TOKENS", "8")
        current_env.setdefault("_MAX_CONTEXT_LEN", "71")

    if allow_symbolic_shapes is not None and allow_symbolic_shapes:
        extra_args.append("--compile_dynamic_sendnn")

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
    "### Response:\n\n1.\n\nThe following",
    "### Response:\n\n1.\n\nI am",
    "### Response:\n\nI am not sure what you",
    "### Response:\n\nI have just come into a",
]


def __repeat_batch_asserts(bs: int) -> list[str]:
    n_repeats = int(math.ceil(bs / len(common_asserts)))
    return (common_asserts * n_repeats)[:bs]


# add the asserts based on batch size
# for batches greater than common_asserts, repeat common_asserts since this follows inference behavior
common_inference_params = [
    common_param + (__repeat_batch_asserts(common_param[1]), None)
    for common_param in common_params
]
# adding special case where we allow symbolic shapes for batch size 1 using sdpa
common_inference_params.append(
    (common_model_paths[0], 1, 64, 8, "sdpa", [common_asserts[0]], True)
)


@pytest.mark.parametrize(
    "model_path,batch_size,seq_length,max_new_tokens,attn_type,asserts,allow_symbolic_shapes",
    common_inference_params,
)
def test_inference_script(
    model_path,
    batch_size,
    seq_length,
    max_new_tokens,
    attn_type,
    asserts,
    allow_symbolic_shapes,
):
    # force symbolic shapes if paged
    if "paged" in attn_type:
        allow_symbolic_shapes = True
    result_text = execute_inference(
        model_path,
        batch_size,
        seq_length,
        max_new_tokens,
        attn_type,
        allow_symbolic_shapes,
    )

    for common_assert in asserts:
        assert common_assert in result_text

program_possibilities = [None, "*:0,==256", "?:2,>=256", "?:==2,<=256"]
max_new_tokens = [8, 128]
dataset_type = ["sharegpt", "custom"]
test_type = ["metrics", "tokens"]
skip_validation = [True, False]
prioritize_large_batch_sizes = [True, False]
enforce_homogeneous_prompt_programs = [True, False]
attn_types = ["paged", "paged_fp8"]

@pytest.fixture(scope="session")
def shared_tmp_path(tmp_path_factory):
    return tmp_path_factory.mktemp("shared_data")

def execute_dpp(
    attn_type, programs, max_new_tokens, dataset_type, test_type, skip_validation, enforce_homogeneous_prompt_programs, shared_tmp_path
):
    current_env["VLLM_DT_MAX_BATCH_TKV_LIMIT"] = "1024"
    current_env["VLLM_DT_MAX_CONTEXT_LEN"] = "512"
    current_env["VLLM_DT_MAX_BATCH_SIZE"] = "2"
    Path(os.path.join(shared_tmp_path, "sendnn_cache")).mkdir(exist_ok=True)
    current_env["TORCH_SENDNN_CACHE_DIR"] = os.path.join(shared_tmp_path, "sendnn_cache")
    current_env["TORCH_SENDNN_CACHE_ENABLE"] = "1"
    
    command_list = [
        "python3",
        f"{DPP_FILE_PATH}",
    ]

    # add attn_type
    command_list += [f"--attention_type={attn_type}"]

    # add model variant
    if attn_type == "paged":
        command_list += [f"--model_variant=ibm-granite/granite-3.3-8b-instruct"]
    else:
        # FIXME: added fp8 paged
        pass

    # add programs
    if programs is not None:
        command_list += [f"--programs", programs]

    # add max_new_tokens
    command_list += [f"--max_new_tokens={max_new_tokens}"]
    
    # add dataset_path and dataset_type
    if dataset_type == "sharegpt":
        dataset_path = os.path.join(SHARED_DIR, "ShareGPT_V3_unfiltered_cleaned_split.json")
    elif dataset_type == "custom":
        dataset_path = os.path.join(shared_tmp_path, "custom_text.txt")
        with open(dataset_path, 'w') as file:
            file.write("This is the first line:")
            file.write("This is the second line:")
            file.write("This is the third line, it should have more tokens than the first 2:")
    else:
        pytest.fail("please provide a valid dataset_type")
    command_list += [f"--dataset_type={dataset_type}", f"--dataset_path={dataset_path}"]

    # add test_type
    if test_type is not None:
        command_list += [f"--test_type={test_type}"]
    
    if skip_validation:
        command_list += ["--skip_validation"]

    if enforce_homogeneous_prompt_programs:
        command_list += ["--enforce_homogeneous_prompt_programs"]

    # add program criteria path
    program_criteria_path = os.path.join(shared_tmp_path, "program_critera.json")
    command_list += [f"--program_criteria_json_path={program_criteria_path}"]
    
    return execute_script(command_list)

dpp_possibilities = []
dpp_possibilities.append(("paged", None, 8, "sharegpt", "metrics", False, False)) # metrics and run all programs
dpp_possibilities.append(("paged", "*:0,==256", 65, "sharegpt", "tokens", False, False)) # tokens and run all programs that satisfy 256 sequence length
dpp_possibilities.append(("paged", "*:>=2,0", 65, "sharegpt", None, True, True)) # metrics and run all programs that have >=2 batch size
dpp_possibilities.append(("paged", None, 8, "custom", "tokens", False, False)) # tokens running with specific custom dataset

"""
{
  "programs" : [
    {
      "max_batch" : 2,
      "max_tkv" : 512,
      "batch_granularity" : 2,
      "tkv_granularity" : 512
    },
    {
      "max_batch" : 1,
      "max_tkv" : 512,
      "batch_granularity" : 1,
      "tkv_granularity" : 512
    },
    {
      "max_batch" : 2,
      "max_tkv" : 256,
      "batch_granularity" : 2,
      "tkv_granularity" : 256
    },
    {
      "max_batch" : 1,
      "max_tkv" : 256,
      "batch_granularity" : 1,
      "tkv_granularity" : 256
    }
  ]
}
"""
asserts = [
    [
        # program assertions
        [0, 1, 2, 3],
        # program shape assertions
        (">=0", "==256")
        # max_new_tokens assertions
    ]
]
#r"program id: ProgramCriteria\(program_id=\d+\), valid prompt: \(2, 384\), input shape: torch.Size([2, 384])"
# test type asserion
# r"For Program ProgramCriteria\(program_id=\d+\) in sentence 1: the metric for token 3 is",
@pytest.mark.parametrize("attn_type,programs,max_new_tokens,dataset_type,test_type,skip_validation,enforce_homogeneous_prompt_programs", dpp_possibilities)
def test_dpp_script(attn_type, programs, max_new_tokens, dataset_type, test_type, skip_validation, enforce_homogeneous_prompt_programs, shared_tmp_path):
    result_text = execute_dpp(
        attn_type,
        programs,
        max_new_tokens,
        dataset_type,
        test_type,
        skip_validation,
        enforce_homogeneous_prompt_programs,
        shared_tmp_path
    )
    # assert that we find all programs
    if programs is None:
        program_assertions = [0, 1, 2, 3]
        shape_assertions = [">=0", ">=0"]
    else:
        program_assertions = 
    
    if 

    print(result_text)

