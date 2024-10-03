import pytest

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

    def test_ask_user_opt_in(self, monkeypatch):
        # Mock the input function to always return "y"
        monkeypatch.setattr("builtins.input", lambda _: "y")

        # Call the function to ask the user if they want to opt in
        opt_in = pybamm.config.ask_user_opt_in()

        # Check that the function returns True
        assert opt_in is True
