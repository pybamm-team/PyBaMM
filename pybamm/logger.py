#
# Logging class for PyBaMM
#
import logging

format = (
    "%(asctime)s - [%(levelname)s] %(module)s.%(funcName)s(%(lineno)d): "
    + "%(message)s"
)
logging.basicConfig(format=format, datefmt="%Y-%m-%d %H:%M:%S")

# Create a custom logger
logger = logging.getLogger(__name__)

def set_logging_level(level):
    logger.setLevel(level)
