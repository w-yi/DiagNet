import torch
import datetime
import logging
import sys
import os

def cuda_wrapper(var):
    if torch.cuda.is_available():
        return var.cuda()
    return var


def get_time(format):
    return datetime.datetime.now().strftime(format)


def check_mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def get_logger(log_file):
    logFormatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    fileHandler = logging.FileHandler(log_file)
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)

    return logger
