# content of conftest.py

from aiu_fms_testing_utils.utils.aiu_setup import aiu_setup, rank, world_size
import os


def pytest_sessionstart(session):
    """
    Called after the Session object has been created and
    before performing collection and entering the run test loop.
    """
    aiu_setup(rank, world_size)
    os.environ.setdefault("SENCORES", "32")
    os.environ.setdefault("SENCORELETS", "2")
    os.environ.setdefault("DATA_PREC", "fp16")
    os.environ.setdefault("FLEX_OVERWRITE_NMB_FRAME", "1")
    os.environ.setdefault("DTCOMPILER_KEEP_EXPORT", "true")

    os.environ.setdefault("COMPILATION_MODE", "offline_decoder")
    os.environ["FLEX_COMPUTE"] = "SENTIENT"
    os.environ.setdefault("FLEX_DEVICE", "PF")

    os.environ.setdefault("DTLOG_LEVEL", "error")
    os.environ.setdefault("DT_DEEPRT_VERBOSE", "-1")

    # NOTE: we should configure the cachedir before importing torchsendnn's
    # graph cache to prevent it from being initialized in the wrong place.
    os.environ["TORCH_SENDNN_CACHE_DIR"] = os.path.join(os.getcwd(), ".cache")


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--capture_expectation",
        action="store_true",
        default=False,
        help="capture the output expectation for a given test",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "capture expectation: expectation was captured")


def pytest_generate_tests(metafunc):
    option_value = metafunc.config.option.capture_expectation
    if "capture_expectation" in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("capture_expectation", [option_value])
