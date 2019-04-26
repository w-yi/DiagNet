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
from models.mfb_coatt_embed_ocr import mfb_coatt_embed_ocr
from models.mfh_coatt_embed_ocr import mfh_coatt_embed_ocr
from utils import data_provider
from utils.data_provider import VQADataProvider
from utils.eval_utils import exec_validation, drawgraph, visualize_pred
from utils.commons import cuda_wrapper, check_mkdir, get_logger
import json
from tensorboardX import SummaryWriter


def get_model(opt):
    """
    args priority:
    OCR > EMBED > not specified (baseline)
    """
    model = None
    if opt.MODEL == 'mfb':
        if opt.OCR:
            assert opt.EXP_TYPE in ['textvqa','textvqa_butd'], 'dataset not supported'
            model = mfb_coatt_embed_ocr(opt)
        elif opt.EMBED:
            model = mfb_coatt_glove(opt)
        else:
            model = mfb_baseline(opt)

    elif opt.MODEL == 'mfh':
        if opt.OCR:
            assert opt.EXP_TYPE in ['textvqa','textvqa_butd'], 'dataset not supported'
            model = mfh_coatt_embed_ocr(opt)
        elif opt.EMBED:
            model = mfh_coatt_glove(opt)
        else:
            model = mfh_baseline(opt)

    return model


def pred(opt, folder, logger):

    assert opt.RESUME_PATH, 'please specify the model file'

    dp = VQADataProvider(opt, batchsize=opt.VAL_BATCH_SIZE, mode='val', logger=logger)
    opt.quest_vob_size, opt.ans_vob_size = dp.get_vocab_size()


    logger.info('==> Resuming from checkpoint..')
    checkpoint = torch.load(opt.RESUME_PATH, map_location='cpu')
    model = get_model(opt)
    model.load_state_dict(checkpoint['model'])
    model = cuda_wrapper(model)

    test_loss, acc_overall, acc_per_ques, acc_per_ans = exec_validation(model, opt, mode='val', folder=folder, it=0, visualize=True, dp=dp, logger=logger)
    logger.info('Test loss: {}'.format(test_loss))
    logger.info('Accuracy: {}'.format(acc_overall))
    logger.info('Test per ans: {}'.format(acc_per_ans))


def main():
    opt = config.parse_opt()

    folder = os.path.join(config.OUTPUT_DIR, opt.ID + '_pred')
    log_file = os.path.join(config.LOG_DIR, opt.ID)

    logger = get_logger(log_file)

    check_mkdir(folder)

    pred(opt, folder, logger)

    # visualize_pred(opt, folder, 'val', logger)


if __name__ == '__main__':
    main()
