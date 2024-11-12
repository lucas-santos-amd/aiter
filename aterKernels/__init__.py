from aterKernels_ import *
import sys
import logging
import multiprocessing
logger = logging.getLogger("aterKernels")


def getLogger():
    global logger
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            fmt="[%(levelname)s] %(asctime)s.%(msecs)03d - %(process)d:%(processName)s - %(pathname)s:%(lineno)d - %(funcName)s\n%(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        console_handler = logging.StreamHandler()
        # console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)

    return logger


if __name__ != "__main__" and multiprocessing.current_process().name == 'MainProcess' and '--multiprocessing-fork'not in sys.argv:
    logger = getLogger()
