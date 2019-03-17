import torch

def cuda_wrapper(var):
    if torch.cuda.is_available():
        return var.cuda()
    return var
