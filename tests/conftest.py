import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--run-cuda",
        action="store_true",
        default=False,
        help="Run cuda tests",
    )

def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-cuda"):
        skipper = pytest.mark.skip(reason="Only run when --run-cuda is given")
        for item in items:
            if "cuda" in item.keywords:
                item.add_marker(skipper)
