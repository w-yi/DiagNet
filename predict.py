import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np
import os
import sys
import config
from models.mfb_baseline import mfb_baseline
from models.mfh_baseline import mfh_baseline
from models.mfb_coatt_glove import mfb_coatt_glove
from models.mfh_coatt_glove import mfh_coatt_glove
from utils import data_provider
from utils.data_provider import VQADataProvider
from utils.eval_utils import exec_validation, drawgraph
from utils.cuda import cuda_wrapper
import json
import datetime
from tensorboardX import SummaryWriter
sys.path.append(config.VQA_TOOLS_PATH)
sys.path.append(config.VQA_EVAL_TOOLS_PATH)
from vqaTools.vqa import VQA
from vqaEvaluation.vqaEval import VQAEval


def pred(opt, model, folder):
    exec_validation(model, opt, mode='test', folder=folder, it=0)


def main():
    opt = config.parse_opt()

    folder = os.path.join(config.OUTPUT_DIR, opt.ID + '_' + opt.TRAIN_DATA_SPLITS)
    if not os.path.exists(folder):
        os.makedirs(folder)

    model = None
    if opt.MODEL == 'mfb':
        if opt.EXP_TYPE == 'glove':
            model = mfb_coatt_glove(opt)
        else:
            model = mfb_baseline(opt)
    elif opt.MODEL == 'mfh':
        if opt.EXP_TYPE == 'glove':
            model = mfh_coatt_glove(opt)
        else:
            model = mfh_baseline(opt)

    if opt.RESUME:
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(opt.RESUME_PATH)
        model.load_state_dict(checkpoint)
    else:
        '''init model parameter'''
        for name, param in model.named_parameters():
            if 'bias' in name:  # bias can't init by xavier
                init.constant_(param, 0.0)
            elif 'weight' in name:
                init.kaiming_uniform_(param)
                # init.xavier_uniform(param)  # for mfb_coatt_glove
    model = cuda_wrapper(model)
    model.eval()

    pred(opt, model, folder)


if __name__ == '__main__':
    main()
