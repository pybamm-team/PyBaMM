import pytest
import sys
import os

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


    @pytest.mark.parametrize(
        "user_input,expected_output,expected_message",
        [
            ("y", True, "Telemetry enabled"),
            ("n", False, "Telemetry disabled"),
            ("x", False, "Invalid input"),
            (None, False, "Timeout reached"),
        ],
    )
    def test_ask_user_opt_in_scenarios(
        self, monkeypatch, capsys, user_input, expected_output, expected_message
    ):
        # mock is_running_tests to return False. This is done
        # temporarily here in order to prevent an early return.
        monkeypatch.setattr(pybamm.config, "is_running_tests", lambda: False)
        monkeypatch.setattr(os, "getenv", lambda x, y: "false")
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
        monkeypatch.setattr(pybamm.util, "is_notebook", lambda: False)

        # Mock get_input_or_timeout based on scenario
        def mock_get_input(timeout):
            print("Do you want to enable telemetry? (Y/n): ", end="")
            return user_input, user_input is None

        monkeypatch.setattr(pybamm.config, "get_input_or_timeout", mock_get_input)

        opt_in = pybamm.config.ask_user_opt_in(timeout=1)
        captured = capsys.readouterr()

        assert "Do you want to enable telemetry?" in captured.out
        assert "PyBaMM can collect usage data" in captured.out
        assert expected_message in captured.out
        assert opt_in is expected_output


    @pytest.mark.parametrize(
        "test_scenario",
        [
            "first_generation",  # Test first-time config generation
            "config_exists",  # Test when config already exists
        ],
    )
    def test_generate_and_read(self, monkeypatch, tmp_path, test_scenario, timeout=2):
        # Mock is_running_tests to return False
        monkeypatch.setattr(pybamm.config, "is_running_tests", lambda: False)

        # Mock ask_user_opt_in to return True
        def mock_ask_user_opt_in(timeout=10):
            return True

        monkeypatch.setattr(pybamm.config, "ask_user_opt_in", mock_ask_user_opt_in)

        # Track if capture was called
        capture_called = False

        def mock_capture(event):
            nonlocal capture_called
            assert event == "user-opted-in"
            capture_called = True

        monkeypatch.setattr(pybamm.telemetry, "capture", mock_capture)

        # Mock config directory
        monkeypatch.setattr(platformdirs, "user_config_dir", lambda x: str(tmp_path))

        if test_scenario == "first_generation":
            # Test first-time generation
            pybamm.config.generate()

            # Verify config was created
            config = pybamm.config.read()
            assert config is not None
            assert config["enable_telemetry"] is True
            assert "uuid" in config
            assert (
                capture_called is True
            )  # Should not ask for capturing telemetry when config exists

        else:  # config_exists case
            # First create a config
            pybamm.config.generate()
            capture_called = False  # Reset the flag

            # Now test that generating again does nothing
            pybamm.config.generate()
            assert (
                capture_called is False
            )  # Should not ask for capturing telemetry when config exists

    @pytest.mark.parametrize(
        "file_scenario,expected_output",
        [
            ("nonexistent", None),
            ("invalid_yaml", None),
        ],
    )
    def test_read_uuid_from_file_scenarios(
        self, tmp_path, file_scenario, expected_output
    ):
        if file_scenario == "nonexistent":
            config_dict = pybamm.config.read_uuid_from_file(
                Path("nonexistent_file.yml")
            )
        else:  # invalid_yaml
            # Create a temporary directory and file with invalid YAML content
            invalid_yaml = tmp_path / "invalid_yaml.yml"
            with open(invalid_yaml, "w") as f:
                f.write("invalid: yaml: content:")
            config_dict = pybamm.config.read_uuid_from_file(invalid_yaml)

        assert config_dict is expected_output
