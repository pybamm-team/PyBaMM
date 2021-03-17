#
# Logging class for PyBaMM
# Includes additional logging levels inspired by verboselogs
# https://pypi.org/project/verboselogs/#overview-of-logging-levels
#
# Implementation from stackoverflow
# https://stackoverflow.com/questions/2183233/how-to-add-a-custom-loglevel-to-pythons-logging-facility
#
import logging


def set_logging_level(level):
    logger.setLevel(level)


format = (
    "%(asctime)s - [%(levelname)s] %(module)s.%(funcName)s(%(lineno)d): "
    + "%(message)s"
)
logging.basicConfig(format=format)
logging.Formatter(datefmt="%Y-%m-%d %H:%M:%S", fmt="%(asctime)s.%(msecs)03d")

# Additional levels inspired by verboselogs
SPAM_LEVEL_NUM = 5
logging.addLevelName(SPAM_LEVEL_NUM, "SPAM")

VERBOSE_LEVEL_NUM = 15
logging.addLevelName(VERBOSE_LEVEL_NUM, "VERBOSE")

NOTICE_LEVEL_NUM = 25
logging.addLevelName(NOTICE_LEVEL_NUM, "NOTICE")

SUCCESS_LEVEL_NUM = 35
logging.addLevelName(SUCCESS_LEVEL_NUM, "SUCCESS")


def spam(self, message, *args, **kws):
    if self.isEnabledFor(SPAM_LEVEL_NUM):
        self._log(SPAM_LEVEL_NUM, message, args, **kws)


def verbose(self, message, *args, **kws):
    if self.isEnabledFor(VERBOSE_LEVEL_NUM):
        self._log(VERBOSE_LEVEL_NUM, message, args, **kws)


def notice(self, message, *args, **kws):
    if self.isEnabledFor(NOTICE_LEVEL_NUM):
        self._log(NOTICE_LEVEL_NUM, message, args, **kws)


def success(self, message, *args, **kws):
    if self.isEnabledFor(SUCCESS_LEVEL_NUM):
        self._log(SUCCESS_LEVEL_NUM, message, args, **kws)


logging.Logger.spam = spam
logging.Logger.verbose = verbose
logging.Logger.notice = notice
logging.Logger.success = success

# Create a custom logger
logger = logging.getLogger(__name__)
set_logging_level("WARNING")
