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


def adjust_learning_rate(optimizer, decay_rate):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate

def train(opt, model, train_Loader, optimizer, writer, folder, use_embed):
    criterion = nn.KLDivLoss(reduction='batchmean')
    train_loss = np.zeros(opt.MAX_ITERATIONS + 1)
    results = []
    for iter_idx, (data, word_length, feature, answer, embed_matrix, cvec_token, token_embedding, original_list_tokens, epoch) in enumerate(train_Loader):
        model.train()
        data = torch.squeeze(data, 0)
        word_length = torch.squeeze(word_length, 0)
        feature = torch.squeeze(feature, 0)
        answer = torch.squeeze(answer, 0)
        epoch = epoch.numpy()

        data = cuda_wrapper(Variable(data)).long()
        word_length = cuda_wrapper(word_length)
        img_feature = cuda_wrapper(Variable(feature)).float()
        label = cuda_wrapper(Variable(answer)).float()
        optimizer.zero_grad()

        if use_embed:
            embed_matrix = torch.squeeze(embed_matrix, 0)
            embed_matrix = cuda_wrapper(Variable(embed_matrix)).float()
            pred = model(data, word_length, img_feature, embed_matrix, cvec_token, token_embedding, 'train')
        else:
            pred = model(data, word_length, img_feature, 'train')

        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()
        train_loss[iter_idx] = loss.data.float()
        if iter_idx % opt.DECAY_STEPS == 0 and iter_idx != 0:
            adjust_learning_rate(optimizer, opt.DECAY_RATE)
        if iter_idx % opt.PRINT_INTERVAL == 0 and iter_idx != 0:
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            c_mean_loss = train_loss[iter_idx - opt.PRINT_INTERVAL:iter_idx].mean()
            writer.add_scalar(opt.ID + '/train_loss', c_mean_loss, iter_idx)
            writer.add_scalar(opt.ID + '/lr', optimizer.param_groups[0]['lr'], iter_idx)
            print('{}\tTrain Epoch: {}\tIter: {}\tLoss: {:.4f}'.format(
                        now, epoch, iter_idx, c_mean_loss), flush=True)
        if iter_idx % opt.CHECKPOINT_INTERVAL == 0 and iter_idx != 0:
            save_path = os.path.join(config.CACHE_DIR, opt.ID + '_iter_' + str(iter_idx) + '.pth')
            torch.save(model.state_dict(), save_path)
        if iter_idx % opt.VAL_INTERVAL == 0 and iter_idx != 0:
            test_loss, acc_overall, acc_per_ques, acc_per_ans = exec_validation(model, opt, mode='val', folder=folder, it=iter_idx)
            writer.add_scalar(opt.ID + '/val_loss', test_loss, iter_idx)
            writer.add_scalar(opt.ID + 'accuracy', acc_overall, iter_idx)
            print('Test loss:', test_loss)
            print('Accuracy:', acc_overall)
            print('Test per ans', acc_per_ans)
            results.append([iter_idx, c_mean_loss, test_loss, acc_overall, acc_per_ques, acc_per_ans])
            best_result_idx = np.array([x[3] for x in results]).argmax()
            print('Best accuracy of', results[best_result_idx][3], 'was at iteration', results[best_result_idx][0], flush=True)
            drawgraph(results, folder, opt.MFB_FACTOR_NUM, opt.MFB_OUT_DIM, prefix=opt.ID)
        if iter_idx % opt.TESTDEV_INTERVAL == 0 and iter_idx != 0:
            exec_validation(model, opt, mode='test-dev', folder=folder, it=iter_idx)


def main():
    opt = config.parse_opt()
    # notice that unique id with timestamp is determined here

    # torch.cuda.set_device(opt.TRAIN_GPU_ID)
    # torch.cuda.manual_seed(opt.SEED)
    # print('Using gpu card: ' + torch.cuda.get_device_name(opt.TRAIN_GPU_ID))
    writer = SummaryWriter()

    folder = os.path.join(config.OUTPUT_DIR, opt.ID + '_' + opt.TRAIN_DATA_SPLITS)


    train_Data = data_provider.VQADataset(opt, config.VOCABCACHE_DIR)
    train_Loader = torch.utils.data.DataLoader(dataset=train_Data, shuffle=True, pin_memory=True, num_workers=2)

    opt.quest_vob_size, opt.ans_vob_size = train_Data.get_vocab_size()

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
    optimizer = optim.Adam(model.parameters(), lr=opt.INIT_LERARNING_RATE)

    train(opt, model, train_Loader, optimizer, writer, folder, train_Data.use_embed())
    writer.close()

if __name__ == '__main__':
    main()
