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
from models.mfb_coatt_embed_ocr import mfb_coatt_embed_ocr
from models.mfh_coatt_embed_ocr import mfh_coatt_embed_ocr
from utils import data_provider
from utils.data_provider import VQADataProvider
from utils.eval_utils import exec_validation, drawgraph
from utils.commons import cuda_wrapper, get_time, check_mkdir, get_logger
import json
from tensorboardX import SummaryWriter


def train(opt, model, train_Loader, optimizer, lr_scheduler, writer, folder, logger):
    criterion = nn.KLDivLoss(reduction='batchmean')
    train_loss = np.zeros(opt.MAX_ITERATIONS)
    results = []
    for iter_idx, (data, word_length, img_feature, label, embed_matrix, ocr_length, ocr_embedding, _, ocr_answer_flags, epoch) in enumerate(train_Loader):
        if iter_idx >= opt.MAX_ITERATIONS:
            break
        model.train()
        epoch = epoch.numpy()
        # TODO: get rid of these weird redundant first dims
        data = torch.squeeze(data, 0)
        word_length = torch.squeeze(word_length, 0)
        img_feature = torch.squeeze(img_feature, 0)
        label = torch.squeeze(label, 0)
        embed_matrix = torch.squeeze(embed_matrix, 0)
        ocr_length = torch.squeeze(ocr_length, 0)
        ocr_embedding = torch.squeeze(ocr_embedding, 0)
        ocr_answer_flags = torch.squeeze(ocr_answer_flags, 0)

        data = cuda_wrapper(Variable(data)).long()
        word_length = cuda_wrapper(word_length)
        img_feature = cuda_wrapper(Variable(img_feature)).float()
        label = cuda_wrapper(Variable(label)).float()
        optimizer.zero_grad()

        if opt.OCR:
            embed_matrix = cuda_wrapper(Variable(embed_matrix)).float()
            ocr_length = cuda_wrapper(ocr_length)
            ocr_embedding = cuda_wrapper(Variable(ocr_embedding)).float()
            if opt.BINARY:
                ocr_answer_flags = cuda_wrapper(ocr_answer_flags)
                pred = model(data, img_feature, embed_matrix, ocr_length, ocr_embedding, ocr_answer_flags, 'train')
            else:
                pred = model(data, img_feature, embed_matrix, ocr_length, ocr_embedding, 'train')
        elif opt.EMBED:
            embed_matrix = cuda_wrapper(Variable(embed_matrix)).float()

            pred = model(data, img_feature, embed_matrix, 'train')
        else:
            pred = model(data, word_length, img_feature, 'train')

        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()
        train_loss[iter_idx] = loss.data.float()
        lr_scheduler.step()
        if iter_idx % opt.PRINT_INTERVAL == 0 and iter_idx != 0:
            # now = get_time('%Y-%m-%d %H:%M:%S')
            c_mean_loss = train_loss[iter_idx - opt.PRINT_INTERVAL+1:iter_idx+1].mean()
            writer.add_scalar(opt.ID + '/train_loss', c_mean_loss, iter_idx)
            writer.add_scalar(opt.ID + '/lr', optimizer.param_groups[0]['lr'], iter_idx)
            logger.info('Train Epoch: {}\tIter: {}\tLoss: {:.4f}'.format(
                        epoch, iter_idx, c_mean_loss))
        if iter_idx % opt.CHECKPOINT_INTERVAL == 0 and iter_idx != 0:
            save_path = os.path.join(config.CACHE_DIR, opt.ID + '_iter_' + str(iter_idx) + '.pth')
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict()
            }, save_path)
        if iter_idx % opt.VAL_INTERVAL == 0 and iter_idx != 0:
            test_loss, acc_overall, acc_per_ques, acc_per_ans = exec_validation(model, opt, mode='val', folder=folder, it=iter_idx, logger=logger)
            writer.add_scalar(opt.ID + '/val_loss', test_loss, iter_idx)
            writer.add_scalar(opt.ID + 'accuracy', acc_overall, iter_idx)
            logger.info('Test loss: {}'.format(test_loss))
            logger.info('Accuracy: {}'.format(acc_overall))
            logger.info('Test per ans: {}'.format(acc_per_ans))
            results.append([iter_idx, c_mean_loss, test_loss, acc_overall, acc_per_ques, acc_per_ans])
            best_result_idx = np.array([x[3] for x in results]).argmax()
            logger.info('Best accuracy of {} was at iteration {}'.format(
                results[best_result_idx][3],
                results[best_result_idx][0]
            ))
            drawgraph(results, folder, opt.MFB_FACTOR_NUM, opt.MFB_OUT_DIM, prefix=opt.ID)
        if iter_idx % opt.TESTDEV_INTERVAL == 0 and iter_idx != 0:
            exec_validation(model, opt, mode='test-dev', folder=folder, it=iter_idx, logger=logger)


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


def main():
    opt = config.parse_opt()
    # notice that unique id with timestamp is determined here

    # torch.cuda.set_device(opt.TRAIN_GPU_ID)
    # torch.cuda.manual_seed(opt.SEED)
    # print('Using gpu card: ' + torch.cuda.get_device_name(opt.TRAIN_GPU_ID))
    writer = SummaryWriter()

    folder = os.path.join(config.OUTPUT_DIR, opt.ID)
    log_file = os.path.join(config.LOG_DIR, opt.ID)

    logger = get_logger(log_file)

    train_Data = data_provider.VQADataset(opt, config.VOCABCACHE_DIR, logger)
    train_Loader = torch.utils.data.DataLoader(dataset=train_Data, shuffle=True, pin_memory=True, num_workers=2)

    opt.quest_vob_size, opt.ans_vob_size = train_Data.get_vocab_size()

    #model = get_model(opt)
    #optimizer = optim.Adam(model.parameters(), lr=opt.INIT_LERARNING_RATE)
    #lr_scheduler = optim.lr_scheduler.StepLR(optimizer, opt.DECAY_STEPS, opt.DECAY_RATE)
    try:
        if opt.RESUME_PATH:
            logger.info('==> Resuming from checkpoint..')
            checkpoint = torch.load(opt.RESUME_PATH)
            model = get_model(opt)
            model.load_state_dict(checkpoint['model'])
            model = cuda_wrapper(model)
            optimizer = optim.Adam(model.parameters(), lr=opt.INIT_LERARNING_RATE)
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, opt.DECAY_STEPS, opt.DECAY_RATE)
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        else:
            model = get_model(opt)
            '''init model parameter'''
            for name, param in model.named_parameters():
                if 'bias' in name:  # bias can't init by xavier
                    init.constant_(param, 0.0)
                elif 'weight' in name:
                    init.kaiming_uniform_(param)
                    # init.xavier_uniform(param)  # for mfb_coatt_glove
            model = cuda_wrapper(model)
            optimizer = optim.Adam(model.parameters(), lr=opt.INIT_LERARNING_RATE)
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, opt.DECAY_STEPS, opt.DECAY_RATE)

        train(opt, model, train_Loader, optimizer, lr_scheduler, writer, folder, logger)

    except Exception as e:
        logger.exception(str(e))

    writer.close()


if __name__ == '__main__':
    main()
