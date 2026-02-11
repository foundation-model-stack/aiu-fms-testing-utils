import os
from aiu_fms_testing_utils.testing.dpp.run_drive_paged_programs import run_dpp
from aiu_fms_testing_utils.testing.dpp.program_models import AttnType
from aiu_fms_testing_utils.testing.dpp.program_models import TestType
from aiu_fms_testing_utils.utils.model_setup import Timing


def pytest_addoption(parser):
    parser.addoption(
        "--program_criteria_json_path",
        action="store",
        required=True,
        help="path to json file containing the program criteria list",
    )


def pytest_generate_tests(metafunc):
    # This is called for every test. Only get/set command line arguments
    # if the argument is specified in the list of test "fixturenames".
    option_value = metafunc.config.option.program_criteria_json_path
    if (
        "program_criteria_json_path" in metafunc.fixturenames
        and option_value is not None
    ):
        metafunc.parametrize("program_criteria_json_path", [option_value])


def test_drive_paged_programs(program_criteria_json_path):
    """Test driving paged programs with specified configurations."""

    programs = "2:0,<8192"
    max_new_tokens = 32
    model_variant = "ibm-granite/granite-3.3-8b-instruct"
    dataset_path = "sharegpt"
    cross_entropy_threshold = 2.6
    failure_rate_threshold = 0.1

    run_dpp(
        programs=programs,
        dataset_path=dataset_path,
        max_new_tokens=max_new_tokens,
        distributed="PYTEST_XDIST_WORKER" in os.environ,
        model_variant=model_variant,
        timing=Timing.NONE,
        program_criteria_json_path=program_criteria_json_path,
        test_type=TestType.TOKENS,
        cross_entropy_threshold=cross_entropy_threshold,
        failure_rate_threshold=failure_rate_threshold,
        attention_type=AttnType.PAGED,
        prefill_chunk_size=1024,
        stagger_load=0,
        stagger_update_lazyhandle=0,
        dist_timeout=0,
        run_cpu_validation=True,
        prioritize_large_batch_sizes=True,
        enforce_homogeneous_prompt_programs=True,
    )
