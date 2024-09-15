from posthog import Posthog
import os
import pybamm
import sys

_posthog = Posthog(
    # this is the public, write only API key, so it's ok to include it here
    project_api_key="phc_bLZKBW03XjgiRhbWnPsnKPr0iw0z03fA6ZZYjxgW7ej",
    host="https://us.i.posthog.com",
)


def disable():
    _posthog.disabled = True


_opt_out = os.getenv("PYBAMM_OPTOUT_TELEMETRY", "false").lower()
if _opt_out != "false":
    disable()


def capture(event):
    # don't capture events in automated testing
    if pybamm.config.is_running_tests():
        return

    properties = {
        "python_version": sys.version,
        "pybamm_version": pybamm.__version__,
    }

    config = pybamm.config.read()
    if config:
        user_id = config["uuid"]
    else:
        user_id = "anonymous-user-id"
        properties["$process_person_profile"] = False

    # setting $process_person_profile to False mean that we only track what events are
    # being run and don't capture anything about the user
    _posthog.capture(user_id, event, properties=properties)
