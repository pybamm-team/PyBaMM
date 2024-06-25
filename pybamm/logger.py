import logging


def get_log_level_func(value):
    def func(self, message, *args, **kws):
        if self.isEnabledFor(value):
            self._log(value, message, args, **kws)

    return func


# Additional levels inspired by verboselogs
new_levels = {"SPAM": 5, "VERBOSE": 15, "NOTICE": 25, "SUCCESS": 35}
for level, value in new_levels.items():
    logging.addLevelName(value, level)
    setattr(logging.Logger, level.lower(), get_log_level_func(value))


FORMAT = (
    "%(asctime)s.%(msecs)03d - [%(levelname)s] %(module)s.%(funcName)s(%(lineno)d): "
    "%(message)s"
)
LOG_FORMATTER = logging.Formatter(datefmt="%Y-%m-%d %H:%M:%S", fmt=FORMAT)


def set_logging_level(level):
    logger.setLevel(level)


def _get_new_logger(name, filename=None):
    new_logger = logging.getLogger(name)
    if filename is None:
        handler = logging.StreamHandler()
    else:
        handler = logging.FileHandler(filename)
    handler.setFormatter(LOG_FORMATTER)
    new_logger.addHandler(handler)
    return new_logger


# Only the function for getting a new logger with filename not None is exposed
def get_new_logger(name, filename):
    if filename is None:
        raise ValueError("filename must be specified")
    return _get_new_logger(name, filename)


# Create a custom logger
logger = _get_new_logger(__name__)
set_logging_level("WARNING")
