import os
from pathlib import Path

import pytest

from aiu_fms_testing_utils.testing.dpp.program_models import AttnType, TestType
from aiu_fms_testing_utils.testing.dpp.run_drive_paged_programs import run_dpp
from aiu_fms_testing_utils.utils.aiu_setup import r0dprint
from aiu_fms_testing_utils.utils.model_setup import Timing


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


def test_drive_paged_programs(dpp_criterion_json_path: str):
    """Test driving paged programs with specified configurations."""

    setup_environment()

    programs = ["2:0,<8192"]
    max_new_tokens = 32
    model_variant = "ibm-granite/granite-3.3-8b-instruct"
    dataset_path = "sharegpt"
    cross_entropy_threshold = 2.6
    failure_rate_threshold = 0.1

    r0dprint(f"Loading criteria from path: {dpp_criterion_json_path}")

    run_dpp(
        programs=programs,
        dataset_path=dataset_path,
        max_new_tokens=max_new_tokens,
        model_variant=model_variant,
        timing=Timing.NONE,
        program_criteria_json_path=dpp_criterion_json_path,
        test_type=TestType.TOKENS,
        cross_entropy_threshold=cross_entropy_threshold,
        failure_rate_threshold=failure_rate_threshold,
        attention_type=AttnType.PAGED,
        prefill_chunk_size=1024,
        stagger_load=0,
        stagger_update_lazyhandle=0,
        dist_timeout=0,
        run_cpu_validation=False,
        prioritize_large_batch_sizes=True,
        enforce_homogeneous_prompt_programs=True,
    )
