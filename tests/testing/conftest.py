import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--program_criteria_json_path",
        action="store",
        required=True,
        help="path to json file containing the program criteria list",
    )


@pytest.fixture(scope="session")
def program_criteria_json_path(request):
    return request.config.getoption("--program_criteria_json_path")
