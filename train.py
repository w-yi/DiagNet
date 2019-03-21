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

def make_answer_vocab(adic, vocab_size):
    """
    Returns a dictionary that maps words to indices.
    """
    adict = {'':0}
    nadict = {'':1000000}
    vid = 1
    for qid in adic.keys():
        answer_obj = adic[qid]
        answer_list = [ans['answer'] for ans in answer_obj]

        for q_ans in answer_list:
            # create dict
            if q_ans in adict:
                nadict[q_ans] += 1
            else:
                nadict[q_ans] = 1
                adict[q_ans] = vid
                vid +=1

    # debug
    nalist = []
    for k,v in sorted(nadict.items(), key=lambda x:x[1]):
        nalist.append((k,v))

    # remove words that appear less than once
    n_del_ans = 0
    n_valid_ans = 0
    adict_nid = {}
    for i, w in enumerate(nalist[:-vocab_size]):
        del adict[w[0]]
        n_del_ans += w[1]
    for i, w in enumerate(nalist[-vocab_size:]):
        n_valid_ans += w[1]
        adict_nid[w[0]] = i

    return adict_nid

def make_question_vocab(qdic):
    """
    Returns a dictionary that maps words to indices.
    """
    vdict = {'':0}
    vid = 1
    for qid in qdic.keys():
        # sequence to list
        q_str = qdic[qid]['qstr']
        q_list = VQADataProvider.seq_to_list(q_str)

        # create dict
        for w in q_list:
            if w not in vdict:
                vdict[w] = vid
                vid +=1

    return vdict

def make_vocab_files(opt):
    """
    Produce the question and answer vocabulary files.
    """
    print('making question vocab...', opt.QUESTION_VOCAB_SPACE)
    qdic, _ = VQADataProvider.load_data(opt.QUESTION_VOCAB_SPACE)
    question_vocab = make_question_vocab(qdic)
    print('making answer vocab...', opt.ANSWER_VOCAB_SPACE)
    _, adic = VQADataProvider.load_data(opt.ANSWER_VOCAB_SPACE)
    answer_vocab = make_answer_vocab(adic, opt.NUM_OUTPUT_UNITS)
    return question_vocab, answer_vocab

def adjust_learning_rate(optimizer, decay_rate):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate

def train(opt, model, train_Loader, optimizer, writer, folder, use_glove):
    criterion = nn.KLDivLoss(reduction='batchmean')
    train_loss = np.zeros(opt.MAX_ITERATIONS + 1)
    results = []
    for iter_idx, (data, word_length, feature, answer, glove, epoch) in enumerate(train_Loader):
        model.train()
        data = np.squeeze(data, axis=0)
        word_length = np.squeeze(word_length, axis=0)
        feature = np.squeeze(feature, axis=0)
        answer = np.squeeze(answer, axis=0)
        epoch = epoch.numpy()

        data = cuda_wrapper(Variable(data)).long()
        word_length = cuda_wrapper(word_length)
        img_feature = cuda_wrapper(Variable(feature)).float()
        label = cuda_wrapper(Variable(answer)).float()
        optimizer.zero_grad()

        if use_glove:
            glove = np.squeeze(glove, axis=0)
            glove = cuda_wrapper(Variable(glove)).float()
            pred = model(data, word_length, img_feature, glove, 'train')
        else:
            pred = model(data, word_length, img_feature, 'train')

        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()
        train_loss[iter_idx] = loss.data.float()
        if iter_idx % opt.DECAY_STEPS == 0 and iter_idx != 0:
            adjust_learning_rate(optimizer, opt.DECAY_RATE)
        if iter_idx % opt.PRINT_INTERVAL == 0 and iter_idx != 0:
            now = str(datetime.datetime.now())
            c_mean_loss = train_loss[iter_idx - opt.PRINT_INTERVAL:iter_idx].mean()
            writer.add_scalar(opt.MODEL + '/train_loss', c_mean_loss, iter_idx)
            writer.add_scalar(opt.MODEL + '/lr', optimizer.param_groups[0]['lr'], iter_idx)
            print('{}\tTrain Epoch: {}\tIter: {}\tLoss: {:.4f}'.format(
                        now, epoch, iter_idx, c_mean_loss))
            sys.stdout.flush()
        if iter_idx % opt.CHECKPOINT_INTERVAL == 0 and iter_idx != 0:
            if not os.path.exists(config.CACHE_DIR):
                os.makedirs(config.CACHE_DIR)
            save_path = os.path.join(config.CACHE_DIR, opt.MODEL + '_iter_' + str(iter_idx) + '.pth')
            torch.save(model.state_dict(), save_path)
        if iter_idx % opt.VAL_INTERVAL == 0 and iter_idx != 0:
            test_loss, acc_overall, acc_per_ques, acc_per_ans = exec_validation(model, opt, mode='val', folder=folder, it=iter_idx)
            writer.add_scalar(opt.MODEL + '/val_loss', test_loss, iter_idx)
            writer.add_scalar(opt.MODEL + 'accuracy', acc_overall, iter_idx)
            print('Test loss:', test_loss)
            print('Accuracy:', acc_overall)
            print('Test per ans', acc_per_ans)
            results.append([iter_idx, c_mean_loss, test_loss, acc_overall, acc_per_ques, acc_per_ans])
            best_result_idx = np.array([x[3] for x in results]).argmax()
            print('Best accuracy of', results[best_result_idx][3], 'was at iteration', results[best_result_idx][0])
            sys.stdout.flush()
            drawgraph(results, folder, opt.MFB_FACTOR_NUM, opt.MFB_OUT_DIM, prefix=opt.MODEL)
        if iter_idx % opt.TESTDEV_INTERVAL == 0 and iter_idx != 0:
            exec_validation(model, opt, mode='test-dev', folder=folder, it=iter_idx)


def main():
    opt = config.parse_opt()
    glove = False
    if 'glove' in opt.MODEL:
        glove = True
    # torch.cuda.set_device(opt.TRAIN_GPU_ID)
    # torch.cuda.manual_seed(opt.SEED)
    # print('Using gpu card: ' + torch.cuda.get_device_name(opt.TRAIN_GPU_ID))
    writer = SummaryWriter()
    folder = os.path.join(config.TRAIN_DIR, opt.MODEL + '_' + opt.TRAIN_DATA_SPLITS)
    if not os.path.exists(folder):
        os.makedirs(folder)
    question_vocab, answer_vocab = {}, {}
    vdict_path = os.path.join(folder, 'vdict.json')
    adict_path = os.path.join(folder, 'adict.json')
    if os.path.exists(vdict_path) and os.path.exists(adict_path):
        print('restoring vocab')
        with open(vdict_path,'r') as f:
            question_vocab = json.load(f)
        with open(adict_path,'r') as f:
            answer_vocab = json.load(f)
    else:
        question_vocab, answer_vocab = make_vocab_files(opt)
        with open(vdict_path,'w') as f:
            json.dump(question_vocab, f)
        with open(adict_path,'w') as f:
            json.dump(answer_vocab, f)
    print('question vocab size:', len(question_vocab))
    print('answer vocab size:', len(answer_vocab))
    opt.quest_vob_size = len(question_vocab)
    opt.ans_vob_size = len(answer_vocab)

    train_Data = data_provider.VQADataset(opt.TRAIN_DATA_SPLITS, opt.BATCH_SIZE, folder, opt, glove)
    train_Loader = torch.utils.data.DataLoader(dataset=train_Data, shuffle=True, pin_memory=True, num_workers=1)

    model = None
    if opt.MODEL == 'mfb_bs':
        model = mfb_baseline(opt)
    elif opt.MODEL == 'mfh_bs':
        model = mfh_baseline(opt)
    elif opt.MODEL == 'mfb_glove':
        model = mfb_coatt_glove(opt)
    elif opt.MODEL == 'mfh_glove':
        model = mfh_coatt_glove(opt)
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
    optimizer = optim.Adam(model.parameters(), lr=opt.INIT_LERARNING_RATE)

    train(opt, model, train_Loader, optimizer, writer, folder, glove)
    writer.close()

if __name__ == '__main__':
    main()
