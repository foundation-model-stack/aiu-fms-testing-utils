import json
import re
import pytest
import os
from subprocess import Popen, PIPE
from pathlib import Path
import itertools
import math
from aiu_fms_testing_utils.utils.paged import get_programs_prompts, ProgramCriteria

FMS_DIR = Path(__file__).parent
AIU_FMS_DIR = os.path.join(FMS_DIR, "../../../aiu-fms-testing-utils/")
INFERENCE_FILE_PATH = os.path.join(AIU_FMS_DIR, "scripts", "inference.py")
DPP_FILE_PATH = os.path.join(AIU_FMS_DIR, "scripts", "drive_paged_programs.py")
SHARED_DIR = os.environ.get(
    "FMS_TESTING_SHARED_MODEL_DIRECTORY", "/home/senuser/models"
)
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


@pytest.fixture
def isolated_env(monkeypatch):
    monkeypatch.setattr(os, "environ", os.environ.copy())
    yield os.environ


def execute_script(execute_cmd, isolated_env):
    with Popen(
        execute_cmd,
        stdin=PIPE,
        stdout=PIPE,
        stderr=PIPE,
        universal_newlines=True,
        env=isolated_env,
    ) as p:
        output, error = p.communicate()
        if p.returncode == 0:
            return output
        else:
            raise Exception(error)


def execute_inference(
    model_path,
    batch_size,
    seq_length,
    max_new_tokens,
    attn_type,
    allow_symbolic_shapes,
    isolated_env,
):
    extra_args = []
    if attn_type == "paged":
        # paged needs symbolic shapes
        extra_args.append("--attention_type=paged")
        # using these options temporarily
        isolated_env.setdefault("VLLM_DT_MAX_BATCH_TKV_LIMIT", "16384")
        isolated_env.setdefault("VLLM_DT_MAX_BATCH_SIZE", "4")
        isolated_env.setdefault("VLLM_DT_MAX_CONTEXT_LEN", "4096")
    else:
        # added in case symbolic shapes used with sdpa
        isolated_env.setdefault("_PROMPT_LEN", "64")
        isolated_env.setdefault("_MAX_DECODE_TOKENS", "8")
        isolated_env.setdefault("_MAX_CONTEXT_LEN", "71")

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
    return execute_script(execute_cmd + extra_args, isolated_env)


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
    isolated_env,
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
        isolated_env,
    )

    for common_assert in asserts:
        assert common_assert in result_text


@pytest.fixture(scope="session")
def shared_tmp_path(tmp_path_factory):
    return tmp_path_factory.mktemp("shared_data")


def execute_dpp(
    attn_type,
    programs,
    max_new_tokens,
    dataset_type,
    test_type,
    skip_validation,
    enforce_homogeneous_prompt_programs,
    shared_tmp_path,
    isolated_env,
):
    isolated_env["VLLM_DT_MAX_BATCH_TKV_LIMIT"] = "1024"
    isolated_env["VLLM_DT_MAX_CONTEXT_LEN"] = "512"
    isolated_env["VLLM_DT_MAX_BATCH_SIZE"] = "2"
    Path(os.path.join(shared_tmp_path, "sendnn_cache")).mkdir(exist_ok=True)
    os.environ.setdefault(
        "TORCH_SENDNN_CACHE_DIR", os.path.join(shared_tmp_path, "sendnn_cache")
    )
    isolated_env["TORCH_SENDNN_CACHE_ENABLE"] = "1"

    command_list = [
        "python3",
        f"{DPP_FILE_PATH}",
    ]

    # add attn_type
    command_list += [f"--attention_type={attn_type}"]

    # add model variant
    if attn_type == "paged":
        command_list += ["--model_variant=ibm-granite/granite-3.3-8b-instruct"]
    else:
        # FIXME: added fp8 paged
        pass

    # add programs
    if programs is not None:
        command_list += ["--programs", programs]

    # add max_new_tokens
    command_list += [f"--max_new_tokens={max_new_tokens}"]

    # add dataset_path and dataset_type
    if dataset_type == "sharegpt":
        dataset_path = os.path.join(
            SHARED_DIR, "ShareGPT_V3_unfiltered_cleaned_split.json"
        )
    elif dataset_type == "custom":
        dataset_path = os.path.join(shared_tmp_path, "custom_dataset")
        dataset_path.mkdir(exist_ok=True)
        prompts = [
            "This is the first line:",
            "This is the second line:",
            "This is the third line, it should have more tokens than the first 2:",
        ]

        for i, prompt in enumerate(prompts):
            fp = dataset_path / f"{i}.txt"
            fp.write_text(prompt)
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
    command_list += [
        f"--program_criteria_json_path={os.environ['DT_PROG_CRITERIA_FILEPATH']}"
    ]

    return execute_script(command_list, isolated_env)


dpp_possibilities = []
dpp_possibilities.append(
    ("paged", None, 8, "sharegpt", "metrics", False, False)
)  # metrics and run all programs
dpp_possibilities.append(
    ("paged", "*:0,==256", 65, "sharegpt", "tokens", False, False)
)  # tokens and run all programs that satisfy 256 sequence length
dpp_possibilities.append(
    ("paged", "*:>=2,0", 65, "sharegpt", None, True, True)
)  # metrics and run all programs that have >=2 batch size
dpp_possibilities.append(
    ("paged", None, 8, "custom", "tokens", False, False)
)  # tokens running with specific custom dataset


@pytest.mark.parametrize(
    "attn_type,programs,max_new_tokens,dataset_type,test_type,skip_validation,enforce_homogeneous_prompt_programs",
    dpp_possibilities,
)
def test_dpp_script(
    attn_type,
    programs,
    max_new_tokens,
    dataset_type,
    test_type,
    skip_validation,
    enforce_homogeneous_prompt_programs,
    shared_tmp_path,
    isolated_env,
):
    os.environ.setdefault(
        "DT_PROG_CRITERIA_FILEPATH",
        os.path.join(shared_tmp_path, "program_critera.json"),
    )

    result_text = execute_dpp(
        attn_type,
        programs,
        max_new_tokens,
        dataset_type,
        test_type,
        skip_validation,
        enforce_homogeneous_prompt_programs,
        shared_tmp_path,
        isolated_env,
    )
    print(result_text)
    with open(os.environ["DT_PROG_CRITERIA_FILEPATH"], "r") as f:
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

    if programs is None:
        program_assertions = [i for i in range(len(program_criteria_list))]
        shape_assertions = [">=0", ">=0"]
    else:
        program_map = get_programs_prompts(
            program_criteria_list,
            multiple=64,
            max_batch_size=2,
            max_tkv=512,
            program_cycles=max_new_tokens,
        )
        programs_split = programs.split(":")
        program_ids_str = programs_split[0]
        shape_assertions = [
            f">={_}" if _.isnumeric() else _ for _ in programs_split[1].split(",")
        ]

        program_assertions = []
        for program_id_seq, shapes in program_map.items():
            if any(
                (
                    eval(
                        f"shape[0]{shape_assertions[0]} and shape[1]{shape_assertions[1]}"
                    )
                    for shape in shapes
                )
            ):
                program_assertions.append(program_id_seq[0].program_id)

        if program_ids_str == "?":
            program_assertions = program_assertions[:1]
        elif program_ids_str.isnumeric():
            program_assertions = [program_assertions[int(program_ids_str)]]
        elif program_ids_str == "*":
            pass
        else:
            raise ValueError("program_id must be one of numeric, ?, or *")

    # only test that we find the correct programs if not doing custom prompt
    if not dataset_type == "custom":
        # assert that we find all programs
        for program_asserion_id in program_assertions:
            assert (
                f"*** testing program ProgramCriteria(program_id={program_asserion_id})"
                in result_text
            )

        # assert that we don't find any extra programs
        for nf_program_assertion_id in [
            i for i in range(len(program_criteria_list)) if i not in program_assertions
        ]:
            assert (
                f"*** testing program ProgramCriteria(program_id={nf_program_assertion_id})"
                not in result_text
            )

    # assert on the shapes
    shape_pattern = r"program id: ProgramCriteria\(program_id=\d+\), valid prompt: \(\d+, \d+\), input shape: torch.Size\(\[\d+, \d+\]\)"
    all_program_ids = re.findall(shape_pattern, result_text)
    for x in all_program_ids:
        numbers = [int(x) for x in re.findall(r"\d+", x)]  # noqa: F841
        # assert batch
        assert eval(f"numbers[1]{shape_assertions[0]}")
        assert eval(f"numbers[3]{shape_assertions[0]}")
        # assert seq
        assert eval(f"numbers[2]{shape_assertions[1]}")
        assert eval(f"numbers[4]{shape_assertions[1]}")

    if skip_validation:
        assert "CPU tokens:" not in result_text
        assert "the metric for token 0 is (tensor(" not in result_text
    elif test_type == "tokens":
        assert "CPU tokens:" in result_text
    else:
        assert "the metric for token 0 is (tensor(" in result_text
