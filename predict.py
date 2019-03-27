import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np
import os
import config
from models.mfb_baseline import mfb_baseline
from models.mfh_baseline import mfh_baseline
from models.mfb_coatt_glove import mfb_coatt_glove
from models.mfh_coatt_glove import mfh_coatt_glove
from utils import data_provider
from utils.data_provider import VQADataProvider
from utils.eval_utils import exec_validation, drawgraph, visualize_pred
from utils.cuda import cuda_wrapper
import json
import datetime
from tensorboardX import SummaryWriter



def pred(opt, folder):

    dp = VQADataProvider(opt, batchsize=opt.VAL_BATCH_SIZE, mode='val')
    opt.quest_vob_size, opt.ans_vob_size = dp.get_vocab_size()

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

    if opt.RESUME_PATH:
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
    exec_validation(model, opt, mode='val', folder=folder, it=0, visualize=True, dp=dp)


def main():
    opt = config.parse_opt()

    folder = os.path.join(config.OUTPUT_DIR, opt.ID + '_pred')

    if not os.path.exists(folder):
        os.makedirs(folder)

    # pred(opt, folder)

    with open('debug.json') as f:
        stat_list = json.load(f)

    visualize_pred(opt, stat_list, folder, 'val')


if __name__ == '__main__':
    main()
