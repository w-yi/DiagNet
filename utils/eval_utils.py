import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os
import sys
import json
import re
import shutil
from PIL import Image
from PIL import ImageFont, ImageDraw
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils.data_provider import VQADataProvider
from utils.commons import cuda_wrapper, check_mkdir
import config
sys.path.append(config.VQA_TOOLS_PATH)
sys.path.append(config.VQA_EVAL_TOOLS_PATH)
from vqaTools.vqa import VQA
from vqaEvaluation.vqaEval import VQAEval

VISUALIZE_LIMIT = 200

class QTypeGetter:
    def __init__(self, qtype_dict, limit, folder):
        self.qtype_dict = qtype_dict
        self.counter = {}
        self.limit = limit
        self.savepath = {}
        for cand in (list(qtype_dict.keys()) + ['other']):
            self.savepath[cand] = os.path.join(folder, cand)
            check_mkdir(self.savepath[cand])

    def get(self, q_list):
        pattern = '^'.join(q_list[:2])
        qtype = 'other'
        for cand in self.qtype_dict.keys():
            if pattern in self.qtype_dict[cand]:
                qtype = cand
                break
        self.counter[qtype] = self.counter.get(qtype, 0) + 1
        if self.counter[qtype] > self.limit:
            return False
        else:
            return self.savepath[qtype]


def visualize_pred(opt, folder, mode, logger):

    img_prefix = config.DATA_PATHS[opt.EXP_TYPE][mode]['image_prefix']
    qtype_getter = QTypeGetter(config.QTYPES, VISUALIZE_LIMIT, folder)

    with open(os.path.join(folder, 'visualize.json')) as f:
        stat_list = json.load(f)

    logger.info('generating prediction images...')
    for t_question in stat_list:
        q_list = t_question['q_list']
        savepath = qtype_getter.get(q_list)
        if savepath:
            iid = t_question['iid']
            question = ' '.join(q_list) + ' ?'
            ans_list = str(t_question['ans_list'])
            ans = t_question['answer']
            pred = t_question['pred']

            if ans == '':
                prefix = 'NA'
            elif ans == pred:
                prefix = 'correct'
            else:
                prefix = 'wrong'
            img_title = prefix + str(iid) + '.png'

            t_img = Image.open(img_prefix + config.IMAGE_FILENAME[opt.EXP_TYPE](iid))
            fig = plt.figure()
            a = fig.add_subplot(1, 2, 1)
            imgplot = plt.imshow(t_img)
            a.axis('off')
            b = fig.add_subplot(1, 2, 2)
            b.text(1, 5, 'Q: ' + question)
            b.text(1, 4, 'A: ' + ans_list)
            b.text(1, 3, 'ground truth: ' + ans)
            b.text(1, 2, 'prediction: ' + pred)

            b.axis([1, 7, 0, 7])
            b.axis('off')
            plt.savefig(os.path.join(savepath, img_title), bbox_inches='tight')
            plt.close()


def exec_validation(model, opt, mode, folder, it, logger, visualize=False, dp=None):
    """
    execute validation and save predictions as json file for visualization
    avg_loss:       average loss on given validation dataset split
    acc_overall:    overall accuracy
    """
    check_mkdir(folder)
    model.eval()
    criterion = nn.NLLLoss()
    # criterion = nn.KLDivLoss(reduction='batchmean')
    if opt.BINARY:
        criterion2 = nn.BCELoss()
        acc_counter = 0
        all_counter = 0
    if not dp:
        dp = VQADataProvider(opt, batchsize=opt.VAL_BATCH_SIZE, mode=mode, logger=logger)
    epoch = 0
    pred_list = []
    loss_list = []
    stat_list = []
    total_questions = len(dp.getQuesIds())

    percent_counter = 0

    logger.info('Validating...')
    while epoch == 0:
        data, word_length, img_feature, answer, embed_matrix, ocr_length, ocr_embedding, ocr_tokens, ocr_answer_flags, qid_list, iid_list, epoch = dp.get_batch_vec()
        data = cuda_wrapper(Variable(torch.from_numpy(data))).long()
        word_length = cuda_wrapper(torch.from_numpy(word_length))
        img_feature = cuda_wrapper(Variable(torch.from_numpy(img_feature))).float()
        label = cuda_wrapper(Variable(torch.from_numpy(answer)))
        ocr_answer_flags = cuda_wrapper(torch.from_numpy(ocr_answer_flags))

        if opt.OCR:
            embed_matrix = cuda_wrapper(Variable(torch.from_numpy(embed_matrix))).float()
            ocr_length = cuda_wrapper(torch.from_numpy(ocr_length))
            ocr_embedding= cuda_wrapper(Variable(torch.from_numpy(ocr_embedding))).float()
            if opt.BINARY:
                ocr_answer_flags = cuda_wrapper(ocr_answer_flags)
                binary, pred1, pred2 = model(data, img_feature, embed_matrix, ocr_length, ocr_embedding, mode)
            else:
                pred = model(data, img_feature, embed_matrix, ocr_length, ocr_embedding, mode)
        elif opt.EMBED:
            embed_matrix = cuda_wrapper(Variable(torch.from_numpy(embed_matrix))).float()
            pred = model(data, img_feature, embed_matrix, mode)
        else:
            pred = model(data, word_length, img_feature, mode)

        if mode == 'test-dev' or mode == 'test':
            pass
        else:
            if opt.BINARY:
                loss = criterion2(binary, ocr_answer_flags.float()) * opt.BIN_LOSS_RATE
                loss += criterion(pred1[label < opt.MAX_ANSWER_VOCAB_SIZE], label[label < opt.MAX_ANSWER_VOCAB_SIZE].long())
                loss += criterion(pred2[label >= opt.MAX_ANSWER_VOCAB_SIZE], label[label >= opt.MAX_ANSWER_VOCAB_SIZE].long() - opt.MAX_ANSWER_VOCAB_SIZE)
                all_counter += binary.size()[0]
                acc_counter += torch.sum((binary <= 0.5) * (ocr_answer_flags == 0) + (binary > 0.5) * (ocr_answer_flags == 1))
                #print(all_counter, acc_counter)
            else:
                loss = criterion(pred, label.long())
            loss = (loss.data).cpu().numpy()
            loss_list.append(loss)

        if opt.BINARY:
            binary = (binary.data).cpu().numpy()
            pred1 = (pred1.data).cpu().numpy()
            pred2 = (pred2.data).cpu().numpy()
            pred = np.hstack([pred1, pred2])
        else:
            pred = (pred.data).cpu().numpy()
        if opt.OCR:
            # select the largest index within the ocr length boundary
            ocr_mask = np.fromfunction(lambda i, j: j >= (ocr_length[i].cpu().numpy() + opt.MAX_ANSWER_VOCAB_SIZE), pred.shape, dtype=int)
            if opt.BINARY:
                ocr_mask += np.fromfunction(lambda i, j: np.logical_or(np.logical_and(binary[i] <= 0.5, j >= opt.MAX_ANSWER_VOCAB_SIZE), np.logical_and(binary[i] > 0.5, j < opt.MAX_ANSWER_VOCAB_SIZE)), pred.shape, dtype=int)
            masked_pred = np.ma.array(pred, mask=ocr_mask)
            pred_max = np.ma.argmax(masked_pred, axis=1)
            pred_str = [dp.vec_to_answer_ocr(pred_symbol, ocr) for pred_symbol, ocr in zip(pred_max, ocr_tokens)]
        else:
            pred_max = np.argmax(pred, axis=1)
            pred_str = [dp.vec_to_answer(pred_symbol) for pred_symbol in pred_max]

        for qid, iid, ans, pred, ocr in zip(qid_list, iid_list, answer.tolist(), pred_str, ocr_tokens):
            pred_list.append((pred, int(dp.getStrippedQuesId(qid))))
            # prepare pred json file
            if visualize:
                q_list = dp.seq_to_list(dp.getQuesStr(qid), opt.MAX_QUESTION_LENGTH)
                if mode == 'test-dev' or mode == 'test':
                    ans_str = ''
                    ans_list = ['']*10
                else:
                    if opt.OCR:
                        ans_str = dp.vec_to_answer_ocr(int(ans), ocr)
                    else:
                        ans_str = dp.vec_to_answer(int(ans))
                    ans_list = [ dp.getAnsObj(qid)[i]['answer'] for i in range(10)]
                stat_list.append({
                    'qid': qid,
                    'q_list': q_list,
                    'iid': iid,
                    'answer': ans_str,
                    'ans_list': ans_list,
                    'pred': pred,
                    'ocr_tokens': ocr
                })
        percent = 100 * float(len(pred_list)) / total_questions
        if percent <= 100 and percent - percent_counter >= 5:
            percent_counter = percent
            sys.stdout.write('\r' + ('%.2f' % percent) + '%')
            sys.stdout.flush()

    if visualize:
        with open(os.path.join(folder, 'visualize.json'), 'w') as f:
            json.dump(stat_list, f, indent=4, sort_keys=True)

    if opt.BINARY:
        logger.info('Binary Acc: {},({}/{})'.format(acc_counter.item()/all_counter, acc_counter, all_counter))

    logger.info('Deduping arr of len {}'.format(len(pred_list)))
    deduped = []
    seen = set()
    for ans, qid in pred_list:
        if qid not in seen:
            seen.add(qid)
            deduped.append((ans, qid))
    logger.info('New len {}'.format(len(deduped)))
    final_list=[]
    for ans,qid in deduped:
        final_list.append({u'answer': ans, u'question_id': qid})

    if mode == 'val':
        avg_loss = np.array(loss_list).mean()
        valFile = os.path.join(folder, 'val2015_resfile')
        with open(valFile, 'w') as f:
            json.dump(final_list, f)
        # if visualize:
        #     visualize_pred(stat_list,mode)

        exp_type = opt.EXP_TYPE

        annFile = config.DATA_PATHS[exp_type]['val']['ans_file']
        quesFile = config.DATA_PATHS[exp_type]['val']['ques_file']
        vqa = VQA(annFile, quesFile)
        vqaRes = vqa.loadRes(valFile, quesFile)
        vqaEval = VQAEval(vqa, vqaRes, n=2)
        vqaEval.evaluate()
        acc_overall = vqaEval.accuracy['overall']
        acc_perQuestionType = vqaEval.accuracy['perQuestionType']
        acc_perAnswerType = vqaEval.accuracy['perAnswerType']
    elif mode == 'test-dev':
        filename = os.path.join(folder, 'test-dev_results_' + str(it).zfill(8))
        with open(filename+'.json', 'w') as f:
            json.dump(final_list, f)
        # if visualize:
        #     visualize_pred(stat_list,mode)
    elif mode == 'test':
        filename = os.path.join(folder, 'test_results_' + str(it).zfill(8))
        with open(filename+'.json', 'w') as f:
            json.dump(final_list, f)
        # if visualize:
        #     visualize_pred(stat_list,mode)
    return avg_loss, acc_overall, acc_perQuestionType, acc_perAnswerType


def drawgraph(results, folder, k, d, prefix='std', save_question_type_graphs=False):
    # 0:it
    # 1:trainloss
    # 2:testloss
    # 3:oa_acc
    # 4:qt_acc
    # 5:at_acc

    # training curve
    it = np.array([l[0] for l in results])
    loss = np.array([l[1] for l in results])
    valloss = np.array([l[2] for l in results])
    valacc = np.array([l[3] for l in results])

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()

    ax1.plot(it,loss, color='blue', label='train loss')
    ax1.plot(it,valloss, '--', color='blue', label='test loss')
    ax2.plot(it,valacc, color='red', label='acc on val')
    plt.legend(loc='lower left')

    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss Value')
    ax2.set_ylabel('Accuracy on Val [%]')

    plt.savefig(os.path.join(folder, 'result_it_%d_acc_%2.2f_k_%d_d_%d_%s.png'%(it[-1], valacc[-1], k, d, prefix)))
    plt.clf()
    plt.close("all")

    # question type
    it = np.array([l[0] for l in results])
    oa_acc = np.array([l[3] for l in results])
    qt_dic_list = [l[4] for l in results]

    def draw_qt_acc(target_key_list, figname):
        fig = plt.figure()
        for k in target_key_list:
            print(k,type(k))
            t_val = np.array([ qt_dic[k] for qt_dic in qt_dic_list])
            plt.plot(it,t_val,label=str(k))
        plt.legend(fontsize='small')
        plt.ylim(0,100.)
        #plt.legend(prop={'size':6})

        plt.xlabel('Iterations')
        plt.ylabel('Accuracy on Val [%]')

        plt.savefig(figname,dpi=200)
        plt.clf()
        plt.close("all")

    if save_question_type_graphs:
        s_keys = sorted(qt_dic_list[0].keys())
        draw_qt_acc(s_keys[ 0:13]+[s_keys[31],],  './ind_qt_are.png')
        draw_qt_acc(s_keys[13:17]+s_keys[49:], './ind_qt_how_where_who_why.png')
        draw_qt_acc(s_keys[17:31]+[s_keys[32],],  './ind_qt_is.png')
        draw_qt_acc(s_keys[33:49],             './ind_qt_what.png')
        draw_qt_acc(['what color is the','what color are the','what color is',\
            'what color','what is the color of the'],'./qt_color.png')
        draw_qt_acc(['how many','how','how many people are',\
            'how many people are in'],'./qt_number.png')
        draw_qt_acc(['who is','why','why is the','where is the','where are the',\
            'which'],'./qt_who_why_where_which.png')
        draw_qt_acc(['what is the man','is the man','are they','is he',\
            'is the woman','is this person','what is the woman','is the person',\
            'what is the person'],'./qt_human.png')
