import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-cuda",
        action="store_true",
        default=False,
        help="Run cuda tests",
    )

    parser.addoption(
        "--run-onnx",
        action="store_true",
        default=False,
        help="Run onnx tests",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-cuda"):
        skipper = pytest.mark.skip(reason="Only run when --run-cuda is given")
        for item in items:
            if "cuda" in item.keywords:
                item.add_marker(skipper)

    if not config.getoption("--run-onnx"):
        skipper = pytest.mark.skip(reason="Only run when --run-onnx is given")
        for item in items:
            if "onnx" in item.keywords:
                item.add_marker(skipper)