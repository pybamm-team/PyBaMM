import pytest

import pybamm
import uuid


class TestConfig:
    def test_write_read_uuid(self, tmp_path):
        # Create a temporary file path
        config_file = tmp_path / "config.yml"

        # Call the function to write UUID to file
        pybamm.config.write_uuid_to_file(config_file)

        # Check that the file was created
        assert config_file.exists()

        # Read the UUID using the read_uuid_from_file function
        uuid_dict = pybamm.config.read_uuid_from_file(config_file)

        # Check that the UUID was read successfully
        assert uuid_dict is not None
        assert "uuid" in uuid_dict

        # Verify that the UUID is valid

        try:
            uuid.UUID(uuid_dict["uuid"])
        except ValueError:
            pytest.fail("Invalid UUID format")
