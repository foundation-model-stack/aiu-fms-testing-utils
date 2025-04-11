# content of conftest.py

from aiu_fms_testing_utils.utils.aiu_setup import aiu_setup, rank, world_size
import os
import pytest

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
    os.environ["FLEX_DEVICE"] = "VFIO"
    os.environ.setdefault("DT_OPT", "dtversion=2,varsub=1,lxopt=1,opfusion=1,arithfold=1,dataopt=1,patchinit=1,patchprog=1,autopilot=1,weipreload=0,kvcacheopt=1,progshareopt=1")

    os.environ.setdefault("DTLOG_LEVEL", "error")
    os.environ.setdefault("DT_DEEPRT_VERBOSE", "-1")

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

