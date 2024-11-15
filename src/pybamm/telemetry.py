from posthog import Posthog
import pybamm
import sys


class MockTelemetry:
    def __init__(self):
        class MockLog:
            @staticmethod
            def setLevel(_: str):
                pass

        self.disabled = True
        self.log = MockLog()

    @staticmethod
    def capture(**kwargs):
        pass


if pybamm.config.check_opt_out():
    _posthog = MockTelemetry()
else:
    _posthog = Posthog(
        # this is the public, write only API key, so it's ok to include it here
        project_api_key="phc_bLZKBW03XjgiRhbWnPsnKPr0iw0z03fA6ZZYjxgW7ej",
        host="https://us.i.posthog.com",
    )

_posthog.log.setLevel("CRITICAL")


def disable():
    _posthog.disabled = True


if pybamm.config.check_opt_out():  # pragma: no cover
    disable()


def capture(event):  # pragma: no cover
    # don't capture events in automated testing
    if pybamm.config.is_running_tests() or _posthog.disabled:
        return

    if pybamm.config.check_opt_out():
        disable()
        return

    config = pybamm.config.read()
    if config:
        properties = {
            "python_version": sys.version,
            "pybamm_version": pybamm.__version__,
        }
        user_id = config["uuid"]
        _posthog.capture(user_id, event, properties=properties)
