import pytest
from inputimeout import TimeoutOccurred

import pybamm
import uuid


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
    def test_ask_user_opt_in(self, monkeypatch, user_opted_in, user_input):
        # Mock the inputimeout function to return invalid input first, then valid input
        inputs = iter(["invalid", user_input])
        monkeypatch.setattr(
            "pybamm.config.inputimeout", lambda prompt, timeout: next(inputs)
        )

        # Call the function to ask the user if they want to opt in
        opt_in = pybamm.config.ask_user_opt_in()
        assert opt_in is user_opted_in

    def test_ask_user_opt_in_timeout(self, monkeypatch):
        # Mock the inputimeout function to raise a TimeoutOccurred exception
        def mock_inputimeout(*args, **kwargs):
            raise TimeoutOccurred

        monkeypatch.setattr("pybamm.config.inputimeout", mock_inputimeout)

        # Call the function to ask the user if they want to opt in
        opt_in = pybamm.config.ask_user_opt_in()
        assert opt_in is False
