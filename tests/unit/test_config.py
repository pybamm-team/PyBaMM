import pytest
import select
import sys

import pybamm
import uuid
from pathlib import Path
import platformdirs


class TestConfig:
    @pytest.mark.parametrize("write_opt_in", [True, False])
    def test_write_read_uuid(self, tmp_path, write_opt_in):
        # Create a temporary file path
        config_file = tmp_path / "config.yml"

        # Call the function to write UUID to file
        pybamm.config.write_uuid_to_file(config_file, write_opt_in)

        # Check that the file was created
        assert config_file.exists()

        # Read the UUID using the read_uuid_from_file function
        config_dict = pybamm.config.read_uuid_from_file(config_file)
        # Check that the UUID was read successfully
        if write_opt_in:
            assert config_dict["enable_telemetry"] is True
            assert "uuid" in config_dict

            # Verify that the UUID is valid
            try:
                uuid.UUID(config_dict["uuid"])
            except ValueError:
                pytest.fail("Invalid UUID format")
        else:
            assert config_dict["enable_telemetry"] is False

    @pytest.mark.parametrize("user_opted_in, user_input", [(True, "y"), (False, "n")])
    def test_ask_user_opt_in(self, monkeypatch, capsys, user_opted_in, user_input):
        # Mock select.select to simulate user input
        def mock_select(*args, **kwargs):
            return [sys.stdin], [], []

        monkeypatch.setattr(select, "select", mock_select)

        # Mock sys.stdin.readline to return the desired input
        monkeypatch.setattr(sys.stdin, "readline", lambda: user_input + "\n")

        # Call the function to ask the user if they want to opt in
        opt_in = pybamm.config.ask_user_opt_in()

        # Check the result
        assert opt_in is user_opted_in

        # Check that the prompt was printed
        captured = capsys.readouterr()
        assert "Do you want to enable telemetry? (Y/n):" in captured.out

    def test_ask_user_opt_in_invalid_input(self, monkeypatch, capsys):
        # Mock select.select to simulate user input and then timeout
        def mock_select(*args, **kwargs):
            nonlocal call_count
            if call_count == 0:
                call_count += 1
                return [sys.stdin], [], []
            else:
                return [], [], []

        monkeypatch.setattr(select, "select", mock_select)

        # Mock sys.stdin.readline to return invalid input
        monkeypatch.setattr(sys.stdin, "readline", lambda: "invalid\n")

        # Initialize call count
        call_count = 0

        # Call the function to ask the user if they want to opt in
        opt_in = pybamm.config.ask_user_opt_in(timeout=1)

        # Check the result (should be False for timeout after invalid input)
        assert opt_in is False

        # Check that the prompt, invalid input message, and timeout message were printed
        captured = capsys.readouterr()
        assert "Do you want to enable telemetry? (Y/n):" in captured.out
        assert (
            "Invalid input. Please enter 'yes/y' for yes or 'no/n' for no."
            in captured.out
        )
        assert "Timeout reached. Defaulting to not enabling telemetry." in captured.out

    def test_ask_user_opt_in_timeout(self, monkeypatch, capsys):
        # Mock select.select to simulate a timeout
        def mock_select(*args, **kwargs):
            return [], [], []

        monkeypatch.setattr(select, "select", mock_select)

        # Call the function to ask the user if they want to opt in
        opt_in = pybamm.config.ask_user_opt_in(timeout=1)

        # Check the result (should be False for timeout)
        assert opt_in is False

        # Check that the prompt and timeout message were printed
        captured = capsys.readouterr()
        assert "Do you want to enable telemetry? (Y/n):" in captured.out
        assert "Timeout reached. Defaulting to not enabling telemetry." in captured.out

    def test_generate_and_read(self, monkeypatch, tmp_path):
        monkeypatch.setattr(pybamm.config, "is_running_tests", lambda: False)
        monkeypatch.setattr(pybamm.config, "check_opt_out", lambda: False)
        monkeypatch.setattr(pybamm.config, "ask_user_opt_in", lambda: True)

        # Mock telemetry capture
        capture_called = False

        def mock_capture(event):
            nonlocal capture_called
            assert event == "user-opted-in"
            capture_called = True

        monkeypatch.setattr(pybamm.telemetry, "capture", mock_capture)

        # Mock config directory
        monkeypatch.setattr(platformdirs, "user_config_dir", lambda x: str(tmp_path))

        # Test generate() creates new config
        pybamm.config.generate()

        # Verify config was created
        config = pybamm.config.read()
        assert config is not None
        assert config["enable_telemetry"] is True
        assert "uuid" in config
        assert capture_called is True

        # Test generate() does nothing if config exists
        capture_called = False
        pybamm.config.generate()
        assert capture_called is False

    def test_read_uuid_from_file_no_file(self):
        config_dict = pybamm.config.read_uuid_from_file(Path("nonexistent_file.yml"))
        assert config_dict is None

    def test_read_uuid_from_file_invalid_yaml(self, tmp_path):
        # Create a temporary directory and file with invalid YAML content
        invalid_yaml = tmp_path / "invalid_yaml.yml"
        with open(invalid_yaml, "w") as f:
            f.write("invalid: yaml: content:")

        config_dict = pybamm.config.read_uuid_from_file(invalid_yaml)

        assert config_dict is None
