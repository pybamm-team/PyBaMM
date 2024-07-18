#
# Tests the logger class.
#
import pytest
import pybamm


class TestLogger:
    def test_logger(self):
        logger = pybamm.logger
        assert logger.level == 30
        pybamm.set_logging_level("INFO")
        assert logger.level == 20
        pybamm.set_logging_level("ERROR")
        assert logger.level == 40
        pybamm.set_logging_level("VERBOSE")
        assert logger.level == 15
        pybamm.set_logging_level("NOTICE")
        assert logger.level == 25
        pybamm.set_logging_level("SUCCESS")
        assert logger.level == 35

        pybamm.set_logging_level("SPAM")
        assert logger.level == 5
        pybamm.logger.spam("Test spam level")
        pybamm.logger.verbose("Test verbose level")
        pybamm.logger.notice("Test notice level")
        pybamm.logger.success("Test success level")

        # reset
        pybamm.set_logging_level("WARNING")

    def test_exceptions(self):
        with pytest.raises(ValueError):
            pybamm.get_new_logger("test", None)
