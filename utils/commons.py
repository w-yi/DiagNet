import torch
import datetime
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
