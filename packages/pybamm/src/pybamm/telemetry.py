import sys

import pybamm


class MockTelemetry:
    def __init__(self):
        self.disabled = True

    @staticmethod
    def capture(**kwargs):  # pragma: no cover
        pass


# Created on first use: importing PostHog would slow every `import pybamm`
_posthog = None


def _client():
    """Return the telemetry client, creating it on first use."""
    global _posthog
    if _posthog is None:
        if pybamm.config.check_opt_out():
            _posthog = MockTelemetry()
        else:  # pragma: no cover
            from posthog import Posthog

            _posthog = Posthog(
                # this is the public, write only API key, so it's ok to include it here
                project_api_key="phc_acTt7KxmvBsAxaE0NyRd5WfJyNxGvBq1U9HnlQSztmb",
                host="https://us.i.posthog.com",
            )
            _posthog.log.setLevel("CRITICAL")
    return _posthog


def disable():
    global _posthog
    if _posthog is None:
        _posthog = MockTelemetry()
    else:
        _posthog.disabled = True


def capture(event):  # pragma: no cover
    if pybamm.config.is_running_tests() or (_posthog is not None and _posthog.disabled):
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
        _client().capture(distinct_id=user_id, event=event, properties=properties)
