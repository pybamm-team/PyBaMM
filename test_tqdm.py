import logging
import time

# import colorlog
from tqdm import tqdm


class TqdmHandler(logging.StreamHandler):
    def __init__(self):
        logging.StreamHandler.__init__(self)

    def emit(self, record):
        msg = self.format(record)
        tqdm.write(msg)


if __name__ == "__main__":
    for x in tqdm(range(100)):
        logger = logging.getLogger("MYAPP")
        logger.setLevel(logging.DEBUG)
        handler = TqdmHandler()
        # handler.setFormatter(
        #     colorlog.ColoredFormatter(
        #         "%(log_color)s%(name)s | %(asctime)s | %(levelname)s | %(message)s",
        #         datefmt="%Y-%d-%d %H:%M:%S",
        #         log_colors={
        #             "DEBUG": "cyan",
        #             "INFO": "white",
        #             "SUCCESS:": "green",
        #             "WARNING": "yellow",
        #             "ERROR": "red",
        #             "CRITICAL": "red,bg_white",
        #         },
        #     )
        # )

        logger.addHandler(handler)
        logger.debug("Inside subtask: " + str(x))
        time.sleep(0.5)
