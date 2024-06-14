
import pytest



def pytest_addoption(parser):
    # parser.addoption(
    #     "--unit", action="store_true", default=False, help="run unit tests"
    # )
    # parser.addoption(
    #     "--integration",
    #     action="store_true",
    #     default=False,
    #     help="run integration tests",
    # )
    parser.addoption(
        "--examples", action="store_true", default=False, help="run examples tests"
    )
    # parser.addoption(
    #     "--plots", action="store_true", default=False, help="run plotting tests"
    # )
    # parser.addoption(
    #     "--notebooks", action="store_true", default=False, help="run notebook tests"
    # )
    # parser.addoption("--docs", action="store_true", default=False, help="run doc tests")


# def pytest_terminal_summary(terminalreporter, exitstatus, config):
#     """Add additional section to terminal summary reporting."""
#     total_time = sum([x.duration for x in terminalreporter.stats.get("passed", [])])
#     num_tests = len(terminalreporter.stats.get("passed", []))
#     print(f"\nTotal number of tests completed: {num_tests}")
#     print(f"Total time taken: {total_time:.2f} seconds")


def pytest_configure(config):
    # config.addinivalue_line("markers", "unit: mark test as a unit test")
    # config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "examples: mark test as an example")
    # config.addinivalue_line("markers", "plots: mark test as a plot test")
    # config.addinivalue_line("markers", "notebook: mark test as a notebook test")
    # config.addinivalue_line("markers", "docs: mark test as a doc test")


def pytest_collection_modifyitems(config, items):
    options = {
        # "unit": "unit",
        "examples": "examples",
        # "integration": "integration",
        # "plots": "plots",
        # "notebooks": "notebooks",
        # "docs": "docs",
    }
    selected_markers = [
        marker for option, marker in options.items() if config.getoption(option)
    ]

    # if (
    #     "notebooks" in selected_markers
    # ):  # Notebooks are meant to be run as an individual session
    #     return

    # If no options were passed, skip all tests
    # if not selected_markers:
    #     skip_all = pytest.mark.skip(
    #         reason="Need at least one of --unit, --examples, --integration, --docs, or --plots option to run"
    #     )
    #     for item in items:
    #         item.add_marker(skip_all)
    #     return

    # Skip tests that don't match any of the selected markers
    for item in items:
        item_markers = {
            mark.name for mark in item.iter_markers()
        }  # Gather markers of the test item
        if not item_markers.intersection(
            selected_markers
        ):  # Skip if there's no intersection with selected markers
            skip_this = pytest.mark.skip(
                reason=f"Test does not match the selected options: {', '.join(selected_markers)}"
            )
            item.add_marker(skip_this)

