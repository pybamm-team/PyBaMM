import uuid
import os
import platformdirs
from pathlib import Path
import pybamm
import select
import sys


def is_running_tests():  # pragma: no cover
    """
    Detect if the code is being run as part of a test suite or building docs with Sphinx.

    Returns:
        bool: True if running tests or building docs, False otherwise.
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
    if any(runner in sys.argv[0].lower() for runner in test_runners):
        return True

    # Check if building docs with Sphinx
    if "sphinx" in sys.modules:
        return True

    return False


def ask_user_opt_in(timeout=10):
    """
    Ask the user if they want to opt in to telemetry.

    Parameters
    ----------
    timeout : float, optional
        The timeout for the user to respond to the prompt. Default is 10 seconds.

    Returns
    -------
    bool
        True if the user opts in, False otherwise.
    """
    print(
        "PyBaMM can collect usage data and send it to the PyBaMM team to "
        "help us improve the software.\n"
        "We do not collect any sensitive information such as models, parameters, "
        "or simulation results - only information on which parts of the code are "
        "being used and how frequently.\n"
        "This is entirely optional and does not impact the functionality of PyBaMM.\n"
        "For more information, see https://docs.pybamm.org/en/latest/source/user_guide/index.html#telemetry"
    )

    while True:
        print("Do you want to enable telemetry? (Y/n): ", end="", flush=True)

        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
        if rlist:
            user_input = sys.stdin.readline().strip().lower()
            if user_input in ["yes", "y", ""]:
                return True
            elif user_input in ["no", "n"]:
                return False
            else:
                print("Invalid input. Please enter 'yes/y' for yes or 'no/n' for no.")
        else:
            print("\nTimeout reached. Defaulting to not enabling telemetry.")
            return False


def generate():  # pragma: no cover
    if is_running_tests():
        return

    # Check if the config file already exists
    if read() is not None:
        return

    # Ask the user if they want to opt in to telemetry
    opt_in = ask_user_opt_in()
    config_file = Path(platformdirs.user_config_dir("pybamm")) / "config.yml"
    write_uuid_to_file(config_file, opt_in)

    if opt_in:
        pybamm.telemetry.capture("user-opted-in")


def read():  # pragma: no cover
    config_file = Path(platformdirs.user_config_dir("pybamm")) / "config.yml"
    return read_uuid_from_file(config_file)


def write_uuid_to_file(config_file, opt_in):
    # Create the directory if it doesn't exist
    config_file.parent.mkdir(parents=True, exist_ok=True)

    # Write the UUID to the config file in YAML format
    with open(config_file, "w") as f:
        f.write("pybamm:\n")
        f.write(f"  enable_telemetry: {opt_in}\n")
        if opt_in:
            unique_id = uuid.uuid4()
            f.write(f"  uuid: {unique_id}\n")


def read_uuid_from_file(config_file):
    # Check if the config file exists
    if not config_file.exists():  # pragma: no cover
        return None

    # Read the UUID from the config file
    with open(config_file) as f:
        content = f.read().strip()

    # Extract the UUID using YAML parsing
    try:
        import yaml

        config = yaml.safe_load(content)
        return config["pybamm"]
    except (yaml.YAMLError, ValueError):  # pragma: no cover
        return None
