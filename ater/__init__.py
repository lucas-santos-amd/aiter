import importlib.util  
if importlib.util.find_spec('ater_') is not None: 
    from ater_ import *
if importlib.util.find_spec('hipbsolidxgemm_') is not None: 
    from hipbsolidxgemm_ import *
if importlib.util.find_spec('rocsolidxgemm_') is not None: 
    from rocsolidxgemm_ import *
from .ops.norm import *
from .ops.smoothquant import *
from .ops.gemm_op_a8w8 import *
import os
import sys
import logging
import multiprocessing
logger = logging.getLogger("ater")
from . import transpose_operator

def getLogger():
    global logger
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        console_handler = logging.StreamHandler()
        if int(os.environ.get('ATER_LOG_MORE', 0)):
            formatter = logging.Formatter(
                fmt="[%(name)s %(levelname)s] %(asctime)s.%(msecs)03d - %(process)d:%(processName)s - %(pathname)s:%(lineno)d - %(funcName)s\n%(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)

    return logger


if __name__ != "__main__" and multiprocessing.current_process().name == 'MainProcess' and '--multiprocessing-fork'not in sys.argv:
    logger = getLogger()
