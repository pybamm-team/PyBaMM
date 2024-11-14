from posthog import Posthog
import os
import pybamm
import sys

_posthog = Posthog(
    # this is the public, write only API key, so it's ok to include it here
    project_api_key="phc_bLZKBW03XjgiRhbWnPsnKPr0iw0z03fA6ZZYjxgW7ej",
    host="https://us.i.posthog.com",
)

_posthog.log.setLevel("CRITICAL")


def disable():
    _posthog.disabled = True


def check_opt_out():
    return os.getenv("PYBAMM_DISABLE_TELEMETRY", "false").lower() != "false"


if check_opt_out():  # pragma: no cover
    disable()


def capture(event):  # pragma: no cover
    # don't capture events in automated testing
    if pybamm.config.is_running_tests() or _posthog.disabled:
        return

    if check_opt_out():
        disable()
        return

    properties = {
        "python_version": sys.version,
        "pybamm_version": pybamm.__version__,
    }

    config = pybamm.config.read()
    if config:
        if config["enable_telemetry"]:
            user_id = config["uuid"]
            _posthog.capture(user_id, event, properties=properties)
