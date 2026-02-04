import sys

import pybamm

# Lazily initialized posthog client
_posthog = None
_disabled = False


def _get_posthog():
    """Lazily initialize the posthog client on first use."""
    global _posthog, _disabled

    if _posthog is not None:
        return _posthog

    if pybamm.config.check_opt_out():
        _disabled = True
        return None

    # Import posthog only when needed (this pulls in requests, urllib3, etc.)
    from posthog import Posthog

    _posthog = Posthog(
        # this is the public, write only API key, so it's ok to include it here
        project_api_key="phc_acTt7KxmvBsAxaE0NyRd5WfJyNxGvBq1U9HnlQSztmb",
        host="https://us.i.posthog.com",
    )
    _posthog.log.setLevel("CRITICAL")
    return _posthog


def disable():
    global _disabled
    _disabled = True
    if _posthog is not None:
        _posthog.disabled = True


def capture(event):  # pragma: no cover
    global _disabled

    if pybamm.config.is_running_tests() or _disabled:
        return

    if pybamm.config.check_opt_out():
        disable()
        return

    posthog = _get_posthog()
    if posthog is None:
        return

    config = pybamm.config.read()
    if config:
        properties = {
            "python_version": sys.version,
            "pybamm_version": pybamm.__version__,
        }
        user_id = config["uuid"]
        posthog.capture(distinct_id=user_id, event=event, properties=properties)
