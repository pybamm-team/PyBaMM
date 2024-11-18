import uuid
import os
import platformdirs
from pathlib import Path
import pybamm
import sys
import threading
import time


def check_env_opt_out():
    return os.getenv("PYBAMM_DISABLE_TELEMETRY", "false").lower() != "false"


def check_opt_out():
    opt_out = check_env_opt_out()
    config = pybamm.config.read()
    if config:
        opt_out = opt_out or not config["enable_telemetry"]
    return opt_out


def is_running_tests():  # pragma: no cover
    """
    Detect if the code is being run as part of a test suite or building docs with Sphinx.

    Returns:
        bool: True if running tests or building docs, False otherwise.
    """
    # Check if pytest or unittest is running
    if any(
        test_module in sys.modules for test_module in ["pytest", "unittest", "nose"]
    ):
        return True

    # Check for other common CI environment variables
    ci_env_vars = [
        "GITHUB_ACTIONS",
        "CI",
        "TRAVIS",
        "CIRCLECI",
        "JENKINS_URL",
        "GITLAB_CI",
    ]
    if any(var in os.environ for var in ci_env_vars):
        return True

    # Check if building docs with Sphinx
    if any(mod == "sphinx" or mod.startswith("sphinx.") for mod in sys.modules):
        print(
            f"Found Sphinx module: {[mod for mod in sys.modules if mod.startswith('sphinx')]}"
        )
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

    def get_input():  # pragma: no cover
        try:
            user_input = (
                input("Do you want to enable telemetry? (Y/n): ").strip().lower()
            )
            answer.append(user_input)
        except Exception:
            # Handle any input errors
            pass

    time_start = time.time()

    while True:
        if time.time() - time_start > timeout:
            print("\nTimeout reached. Defaulting to not enabling telemetry.")
            return False

        answer = []
        # Create and start input thread
        input_thread = threading.Thread(target=get_input)
        input_thread.daemon = True
        input_thread.start()

        # Wait for either timeout or input
        input_thread.join(timeout)

        if answer:
            if answer[0] in ["yes", "y", ""]:
                print("\nTelemetry enabled.\n")
                return True
            elif answer[0] in ["no", "n"]:
                print("\nTelemetry disabled.\n")
                return False
            else:
                print("\nInvalid input. Please enter 'yes/y' for yes or 'no/n' for no.")
        else:
            print("\nTimeout reached. Defaulting to not enabling telemetry.")
            return False


def generate():
    if is_running_tests() or check_opt_out():
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


def read():
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
