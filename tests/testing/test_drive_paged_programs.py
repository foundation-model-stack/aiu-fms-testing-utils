import os
from pathlib import Path

import pytest

from aiu_fms_testing_utils.testing.dpp.program_models import (
    DatasetType,
)
from aiu_fms_testing_utils.testing.dpp.run_drive_paged_programs import run_dpp
from aiu_fms_testing_utils.utils.aiu_setup import r0dprint


@pytest.fixture(scope="module")
def dpp_criterion_json_path():
    test_path = Path(__file__).parent.parent / "fixtures" / "dpp-all-criterion.json"
    return str(test_path)


def setup_environment():
    """Sets up the testing environment for driving paged programs."""

    r0dprint("Setting up environment for driving paged programs...")
    os.environ["VLLM_DT_MAX_BATCH_TKV_LIMIT"] = os.environ.get(
        "VLLM_DT_MAX_BATCH_TKV_LIMIT", "131072"
    )
    os.environ["VLLM_DT_MAX_BATCH_SIZE"] = os.environ.get(
        "VLLM_DT_MAX_BATCH_SIZE", "32"
    )
    os.environ["VLLM_DT_MAX_CONTEXT_LEN"] = os.environ.get(
        "VLLM_DT_MAX_CONTEXT_LEN", "32768"
    )
    os.environ["VLLM_DT_CHUNK_LEN"] = os.environ.get("VLLM_DT_CHUNK_LEN", "1024")

    r0dprint("Batch TKV Limit:", os.environ["VLLM_DT_MAX_BATCH_TKV_LIMIT"])
    r0dprint("Max Batch Size:", os.environ["VLLM_DT_MAX_BATCH_SIZE"])
    r0dprint("Max Context Length:", os.environ["VLLM_DT_MAX_CONTEXT_LEN"])
    r0dprint("Chunk Length:", os.environ["VLLM_DT_CHUNK_LEN"])


@pytest.mark.dpp
def test_drive_paged_programs(dpp_criterion_json_path: str):
    """Test driving paged programs with specified configurations."""

    setup_environment()

    programs = ["2:0,<8192"]
    max_new_tokens = 32
    model_variant = "ibm-granite/granite-3.3-8b-instruct"
    validation_info_outputs_dir = os.getenv(
        "VALIDATION_INFO_OUTPUTS_DIR", "/home/senuser/models/validation_info"
    )
    dataset_type = DatasetType.SHAREGPT
    cross_entropy_threshold = 2.6
    failure_rate_threshold = 0.1

    r0dprint(f"Loading criteria from path: {dpp_criterion_json_path}")

    run_dpp(
        program_criteria_json_path=dpp_criterion_json_path,
        dataset_type=dataset_type,
        max_new_tokens=max_new_tokens,
        model_variant=model_variant,
        programs=programs,
        cross_entropy_threshold=cross_entropy_threshold,
        failure_rate_threshold=failure_rate_threshold,
        prefill_chunk_size=1024,
        run_cpu_validation=True,
        prioritize_large_batch_sizes=True,
        enforce_homogeneous_prompt_programs=True,
        validation_info_outputs_dir=validation_info_outputs_dir,
    )
