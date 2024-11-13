import os
import sys
import time
import uuid

from pathlib import Path

import pybamm
import platformdirs

from pybamm.util import is_notebook


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

    # Check for GitHub Actions environment variable
    if "GITHUB_ACTIONS" in os.environ:
        return True

    # Check for other common CI environment variables
    ci_env_vars = ["CI", "TRAVIS", "CIRCLECI", "JENKINS_URL", "GITLAB_CI"]
    if any(var in os.environ for var in ci_env_vars):
        return True

    # Check for common test runner names in command-line arguments
    test_runners = ["pytest", "unittest", "nose", "trial", "nox", "tox"]
    if any(runner in arg.lower() for arg in sys.argv for runner in test_runners):
        return True

    # Check if building docs with Sphinx
    if any(mod == "sphinx" or mod.startswith("sphinx.") for mod in sys.modules):
        print(
            f"Found Sphinx module: {[mod for mod in sys.modules if mod.startswith('sphinx')]}"
        )
        return True

    return False


def get_input_or_timeout(timeout):  # pragma: no cover
    """
    Cross-platform input with timeout, using various methods depending on the
    environment. Works in Jupyter notebooks, Windows, and Unix-like systems.

    Args:
        timeout (float): Timeout in seconds

    Returns:
        str | None: The user input if received before the timeout, or None if:
            - The timeout was reached
            - Telemetry is disabled via environment variable
            - Running in a non-interactive environment
            - An error occurred

        The caller can distinguish between an empty response (user just pressed Enter)
        which returns '', and a timeout/error which returns None.
    """
    # Check for telemetry disable flag
    if os.getenv("PYBAMM_DISABLE_TELEMETRY", "false").lower() != "false":
        return None

    if not (sys.stdin.isatty() or is_notebook()):
        return None

    # 1. Handling for Jupyter notebooks. This is a simplified
    # implementation in comparison to widgets because notebooks
    # are usually run interactively, and we don't need to worry
    # about the event loop. The input is disabled when running
    # tests due to the environment variable set.
    if is_notebook():  # pragma: no cover
        try:
            from IPython.display import clear_output

            user_input = input("Do you want to enable telemetry? (Y/n): ")
            clear_output()

            return user_input

        except Exception:  # pragma: no cover
            return None

    # 2. Windows-specific handling
    if sys.platform == "win32":
        try:
            import msvcrt

            start_time = time.time()
            input_chars = []
            sys.stdout.write("Do you want to enable telemetry? (Y/n): ")
            sys.stdout.flush()

            while time.time() - start_time < timeout:
                if msvcrt.kbhit():
                    char = msvcrt.getwche()
                    if char in ("\r", "\n"):
                        sys.stdout.write("\n")
                        return "".join(input_chars)
                    input_chars.append(char)
                time.sleep(0.1)
            return None
        except Exception:
            return None

    # 3. POSIX-like systems will need to use termios
    else:  # pragma: no cover
        try:
            import termios
            import tty
            import select

            # Save terminal settings for later
            old_settings = termios.tcgetattr(sys.stdin)
            try:
                # Set terminal to raw mode
                tty.setraw(sys.stdin.fileno())

                sys.stdout.write("Do you want to enable telemetry? (Y/n): ")
                sys.stdout.flush()

                input_chars = []
                start_time = time.time()

                while time.time() - start_time < timeout:
                    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
                    if rlist:
                        char = sys.stdin.read(1)
                        if char in ("\r", "\n"):
                            sys.stdout.write("\n")
                            return "".join(input_chars)
                        input_chars.append(char)
                        sys.stdout.write(char)
                        sys.stdout.flush()
                return None

            finally:
                # Restore saved terminal settings
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                sys.stdout.write("\n")
                sys.stdout.flush()

        except Exception:  # pragma: no cover
            return None

    return None


def ask_user_opt_in(timeout=10):  # pragma: no cover
    """
    Ask the user if they want to opt in to telemetry.
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

    user_input = get_input_or_timeout(timeout)

    if user_input is None:
        print("\nTimeout reached. Defaulting to not enabling telemetry.")
        return False

    while True:
        if user_input.lower() in ["y", "yes", ""]:
            print("Telemetry enabled.\n")
            return True
        elif user_input.lower() in ["n", "no"]:
            print("Telemetry disabled.")
            return False
        else:
            print("Invalid input. Please enter 'Y/y' for yes or 'n/N' for no:")
            user_input = get_input_or_timeout(timeout)
            if user_input is None:
                print("\nTimeout reached. Defaulting to not enabling telemetry.")
                return False


def generate():
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
