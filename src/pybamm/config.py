import uuid
import pathlib
import os


def _get_config_file():
    # Get the home directory
    home_dir = pathlib.Path.home()

    # Create the file path for pybamm config
    config_file = home_dir / ".config" / "pybamm" / "config.yml"

    return config_file


def is_running_tests():
    """
    Detect if the code is being run as part of a test suite.

    Returns:
        bool: True if running tests, False otherwise.
    """
    import sys

    # Check if pytest or unittest is running
    if any(
        test_module in sys.modules for test_module in ["pytest", "unittest", "nose"]
    ):
        return True

    # Check for GitHub Actions environment variable
    if "GITHUB_ACTIONS" in os.environ:
        return True

    # Check for other common CI environment variables
    ci_env_vars = ["CI", "TRAVIS", "CIRCLECI", "JENKINS_URL", "GITLAB_CI"]
    if any(var in os.environ for var in ci_env_vars):
        return True

    # Check for common test runner names in command-line arguments
    test_runners = ["pytest", "unittest", "nose", "trial", "nox", "tox"]
    return any(runner in sys.argv[0].lower() for runner in test_runners)


def generate():
    if is_running_tests():
        return

    config_file = _get_config_file()

    # Create the directory if it doesn't exist
    config_file.parent.mkdir(parents=True, exist_ok=True)

    # Check if the config file already exists
    if read() is not None:
        return

    # Generate a UUID
    unique_id = uuid.uuid4()

    # Write the UUID to the config file in YAML format
    with open(config_file, "w") as f:
        f.write("pybamm:\n")
        f.write(f"  uuid: {unique_id}\n")


def read():
    config_file = _get_config_file()

    # Check if the config file exists
    if not config_file.exists():
        return None

    # Read the UUID from the config file
    with open(config_file) as f:
        content = f.read().strip()

    # Extract the UUID using YAML parsing
    try:
        import yaml

        config = yaml.safe_load(content)
        return config["pybamm"]
    except (yaml.YAMLError, ValueError):
        return None
