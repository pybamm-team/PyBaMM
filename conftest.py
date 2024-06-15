import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--examples", action="store_true", default=False, help="run examples tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "examples: mark test as an example")


def pytest_collection_modifyitems(config, items):
    options = {
        "examples": "examples",
    }
    selected_markers = [
        marker for option, marker in options.items() if config.getoption(option)
    ]
    if "examples" not in selected_markers:
        skip_example = pytest.mark.skip(reason="Skipping example tests since --examples option is not provided")
        for item in items:
            if 'examples' in item.keywords:
                item.add_marker(skip_example)
