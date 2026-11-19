import os
import subprocess
import sys
from types import SimpleNamespace

import pybamm


class TestTelemetry:
    def test_import_does_not_load_posthog_with_telemetry_on(self, tmp_path):
        # An empty config directory and no opt-out variable leave telemetry on
        code = (
            "import platformdirs; "
            f"platformdirs.user_config_dir = lambda *args, **kwargs: {str(tmp_path)!r}; "
            "import sys, pybamm; "
            "assert not pybamm.config.check_opt_out(); "
            "print('posthog' in sys.modules)"
        )
        env = {k: v for k, v in os.environ.items() if k != "PYBAMM_DISABLE_TELEMETRY"}
        # Skips the consent prompt that `import pybamm` shows outside CI
        env["CI"] = "true"
        result = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        # An editable install's rebuild check can print before the result
        assert result.stdout.splitlines()[-1] == "False"

    def test_client_is_created_on_first_use(self, monkeypatch):
        monkeypatch.setattr(pybamm.telemetry, "_posthog", None)
        monkeypatch.setenv("PYBAMM_DISABLE_TELEMETRY", "true")
        client = pybamm.telemetry._client()
        assert isinstance(client, pybamm.telemetry.MockTelemetry)
        assert pybamm.telemetry._client() is client

    def test_disable(self, monkeypatch):
        # Before any client exists, disabling creates none
        monkeypatch.setattr(pybamm.telemetry, "_posthog", None)
        pybamm.telemetry.disable()
        assert isinstance(pybamm.telemetry._posthog, pybamm.telemetry.MockTelemetry)
        assert pybamm.telemetry._posthog.disabled

        client = SimpleNamespace(disabled=False)
        monkeypatch.setattr(pybamm.telemetry, "_posthog", client)
        pybamm.telemetry.disable()
        assert client.disabled
