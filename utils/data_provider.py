import string
import json
import random
import numpy as np
import torch.utils.data as data
import spacy
import os
from collections import Counter

import config
#from torchnlp.word_to_vector import FastText

QID_KEY_SEPARATOR = '/'
ZERO_PAD = '_PAD'
EMBEDDING_SIZE = 300

class VQADataProvider:

    def __init__(self, opt, batchsize, mode, cache_dir=config.VOCABCACHE_DIR):
        self.opt = opt
        self.mode = mode
        self.batchsize = batchsize
        self.d_vocabulary = None
        self.batch_index = None
        self.batch_len = None
        self.rev_adict = None
        self.max_length = opt.MAX_QUESTION_LENGTH
        self.cache_dir = cache_dir
        self.exp_type = opt.EXP_TYPE
        self.use_embed = opt.EMBED
        self.use_ocr = opt.OCR
        self.max_token_size = opt.MAX_TOKEN_SIZE

        self.qdic, self.adic = VQADataProvider.load_data(self.mode, self.exp_type, self.use_ocr)

        self._get_vocab_files()

        if self.use_embed:
            self.n_ans_vocabulary = len(self.adict)
            self.nlp = spacy.load('en_vectors_web_lg')
            # self.nlp = spacy.load('en_core_web_sm')
            self.embed_dict = {} # word -> embed vector


    def _get_vocab_files(self):
        """
        get vocab files
        load cached files if exists
        """
        question_vocab, answer_vocab = {}, {}
        qdict_path = os.path.join(self.cache_dir, self.exp_type + '_qdict.json')
        adict_prefix = '_adict.json'
        if self.use_ocr:
            adict_prefix = '_ocr' + adict_prefix
        adict_path = os.path.join(self.cache_dir, self.exp_type + adict_prefix)
        if os.path.exists(qdict_path) and os.path.exists(adict_path):
            print('restoring vocab')
            with open(qdict_path,'r') as f:
                q_dict = json.load(f)
            with open(adict_path,'r') as f:
                a_dict = json.load(f)
        else:
            q_dict, a_dict = self._make_vocab_files()
            with open(qdict_path,'w') as f:
                json.dump(q_dict, f)
            with open(adict_path,'w') as f:
                json.dump(a_dict, f)
        print('question vocab size:', len(q_dict))
        print('answer vocab size:', len(a_dict), flush=True)
        self.qdict = q_dict
        self.adict = a_dict

    def _make_vocab_files(self):
        """
        Produce the question and answer vocabulary files.
        """
        print('making question vocab...', self.opt.QUESTION_VOCAB_SPACE)
        qdic, _ = VQADataProvider.load_data(self.opt.QUESTION_VOCAB_SPACE, self.exp_type, self.use_ocr)
        question_vocab = VQADataProvider.make_question_vocab(qdic)
        print('making answer vocab...', self.opt.ANSWER_VOCAB_SPACE)
        qdic, adic = VQADataProvider.load_data(self.opt.ANSWER_VOCAB_SPACE, self.exp_type, self.use_ocr)
        answer_vocab = VQADataProvider.make_answer_vocab(adic, qdic, self.opt.MAX_ANSWER_VOCAB_SIZE, self.use_ocr)
        return question_vocab, answer_vocab

    @staticmethod
    def make_question_vocab(qdic):
        """
        Returns a dictionary that maps words to indices.
        """
        qdict = {'':0}
        vid = 1
        for qid in qdic.keys():
            # sequence to list
            q_str = qdic[qid]['qstr']
            q_list = VQADataProvider.seq_to_list(q_str)

            # create dict
            for w in q_list:
                if w not in qdict:
                    qdict[w] = vid
                    vid +=1

        return qdict

    # @staticmethod
    # def make_answer_vocab(adic, vocab_size):
    #     """
    #     Returns a dictionary that maps words to indices.
    #     """
    #     adict = {'':0}
    #     nadict = {'':1000000}
    #     vid = 1
    #     for qid in adic.keys():
    #         answer_obj = adic[qid]
    #         answer_list = [ans['answer'] for ans in answer_obj]
    #
    #         for q_ans in answer_list:
    #             # create dict
    #             if q_ans in adict:
    #                 nadict[q_ans] += 1
    #             else:
    #                 nadict[q_ans] = 1
    #                 adict[q_ans] = vid
    #                 vid +=1
    #     # debug
    #     nalist = []
    #     for k,v in sorted(nadict.items(), key=lambda x:x[1]):
    #         nalist.append((k,v))
    #
    #     # remove words that appear less than once
    #     n_del_ans = 0
    #     n_valid_ans = 0
    #     adict_nid = {}
    #     for i, w in enumerate(nalist[:-vocab_size]):
    #         del adict[w[0]]
    #         n_del_ans += w[1]
    #     for i, w in enumerate(nalist[-vocab_size:]):
    #         n_valid_ans += w[1]
    #         adict_nid[w[0]] = i
    #
    #     return adict_nid

    @staticmethod
    def make_answer_vocab(adic, qdic, vocab_size, use_ocr):
        """
        Returns a vocab_size dictionary that maps words to indices.
        only keep the words with highest occurrences
        if use_ocr, exclude the answers appearing in token lists
        """

        counter = Counter()
        for qid in adic.keys():
            answer_obj = adic[qid]
            answer_list = [ans['answer'] for ans in answer_obj]
            if use_ocr:
                ocr_tokens = qdic[qid]['ocr_tokens']
                answer_list = [x for x in answer_list if x not in ocr_tokens]

            counter.update(answer_list)

        adict = {}
        # TODO: check whether this is the right implementation
        adict[''] = 0
        idx = 1
        alist = counter.most_common(vocab_size - 1)
        for pair in alist:
            adict[pair[0]] = idx
            idx += 1
        return adict

    @staticmethod
    def load_vqa_json(data_split, exp_type, use_ocr):
        """
        Parses the question and answer json files for the given data split.
        Returns the question dictionary and the answer dictionary.

        question dict format: {
            'question': str,
            'question_id': int,
            'image_id': int,
            'ocr_tokens': list (only required if use_ocr)
        }
        answer dict format: {
            'answers': list of str,
            'question_id': int
        }

        """
        qdic, adic = {}, {}

        with open(config.DATA_PATHS[exp_type][data_split]['ques_file'], 'r') as f:
            qdata = json.load(f)['questions']
            for q in qdata:
                q_key = data_split + QID_KEY_SEPARATOR + str(q['question_id'])
                qdic[q_key] = {
                    'qstr': q['question'],
                    'iid': q['image_id']
                }
                if use_ocr:
                    qdic[q_key]['ocr_tokens'] = q['ocr_tokens']

        if 'test' not in data_split:
            with open(config.DATA_PATHS[exp_type][data_split]['ans_file'], 'r') as f:
                adata = json.load(f)['annotations']
                for a in adata:
                    # TODO: we only use key 'answer' in this a['answers'] list
                    adic[data_split + QID_KEY_SEPARATOR + str(a['question_id'])] = \
                        a['answers']

        print('parsed', len(qdic), 'questions for', data_split)
        return qdic, adic

    @staticmethod
    def load_genome_json(exp_type='baseline'):
        """
        Parses the genome json file. Returns the question dictionary and the
        answer dictionary.
        """
        qdic, adic = {}, {}

        with open(config.DATA_PATHS[exp_type]['genome']['genome_file'], 'r') as f:
            qdata = json.load(f)
            for q in qdata:
                key = 'genome' + QID_KEY_SEPARATOR + str(q['id'])
                qdic[key] = {'qstr': q['question'], 'iid': q['image']}
                adic[key] = [{'answer': q['answer']}]

        print('parsed', len(qdic), 'questions for genome')
        return qdic, adic

    @staticmethod
    def load_data(data_split_str, exp_type, use_ocr):
        all_qdic, all_adic = {}, {}

        for data_split in data_split_str.split('+'):
            assert data_split in config.DATA_PATHS[exp_type].keys(), 'unknown data split'
            if data_split == 'genome':
                qdic, adic = VQADataProvider.load_genome_json(exp_type)
                all_qdic.update(qdic)
                all_adic.update(adic)
            else:
                qdic, adic = VQADataProvider.load_vqa_json(data_split, exp_type, use_ocr)
                all_qdic.update(qdic)
                all_adic.update(adic)
        return all_qdic, all_adic

    # def use_embed(self):
    #     """
    #     check usage of pretrained word embedding
    #     """
    #     if self.exp_type == 'glove' or self.exp_type == 'textvqa':
    #         return True
    #     else:
    #         return False

    def getQuesIds(self):
        return list(self.qdic.keys())

    def getStrippedQuesId(self, qid):
        return qid.split(QID_KEY_SEPARATOR)[1]

    def getImgId(self,qid):
        return self.qdic[qid]['iid']

    def getQuesStr(self,qid):
        return self.qdic[qid]['qstr']

    def getQuesOcrTokens(self, qid):
        if self.use_ocr:
            return self.qdic[qid]['ocr_tokens']
        else:
            return []

    def getAnsObj(self, qid):
        if self.mode == 'test-dev' or self.mode == 'test':
            return -1
        return self.adic[qid]

    def get_vocab_size(self):
        return len(self.qdict), len(self.adict)

    @staticmethod
    def seq_to_list(s, max_length):
        """
        slice question into token list
        cut the remaining tokens if exceeds max_length
        """
        t_str = s.lower()
        t_str = t_str.replace('-', ' ').replace('/', ' ')
        t_str = t_str.translate(str.maketrans('', '', string.punctuation))
        q_list = t_str.strip().split()
        return q_list[:max_length]

    def extract_answer(self, answer_obj):
        """
        Return the answer with maximum occurrences
        Only consider the first 10 answers in the list
        Only return one answer if there is a draw
        """
        if self.mode == 'test-dev' or self.mode == 'test':
            return -1
        answer_list = [answer_obj[i]['answer'] for i in range(10)]
        dic = {}
        for ans in answer_list:
            if ans in dic:
                dic[ans] +=1
            else:
                dic[ans] = 1
        max_key = max((v,k) for (k,v) in dic.items())[1]
        return max_key

#     # This is a useless function that has never been used
#     def extract_answer_prob(self,answer_obj):
#         """ Return the most popular answer in string."""
#         if self.mode == 'test-dev' or self.mode == 'test':
#             return -1

#         answer_list = [ ans['answer'] for ans in answer_obj]
#         prob_answer_list = []
#         for ans in answer_list:
#             if ans in self.adict:
#                 prob_answer_list.append(ans)

    def extract_answer_list(self, answer_obj, token_obj):
        answer_list = [ans['answer'] for ans in answer_obj]
        prob_answer_vec = np.zeros(self.opt.NUM_OUTPUT_UNITS)
        for ans in answer_list:
            if ans in self.adict:
                index = self.adict[ans]
                prob_answer_vec[index] += 1
            if self.use_ocr and ans in token_obj:
                for idx in range(0, len(token_obj)):
                    if token_obj[idx] == ans:
                        prob_answer_vec[self.opt.MAX_ANSWER_VOCAB_SIZE + idx] += 1

        return prob_answer_vec / np.sum(prob_answer_vec)

#         if len(prob_answer_list) == 0:
#             if self.mode == 'val' or self.mode == 'test-dev' or self.mode == 'test':
#                 return 'hoge'
#             else:
#                 raise Exception("This should not happen.")
#         else:
#             return random.choice(prob_answer_list)


    def tokenlist_to_vec(self, token_list):
        # Input: the token list
        # Output: the embedded feature matrix for the token embeddings
        embed_matrix = np.zeros((self.max_token_size, self.opt.TOKEN_EMBEDDING_SIZE))
        for i in range(len(token_list)):
            w = token_list[i]
            if w not in self.embed_dict:
                self.embed_dict[w] = self.nlp(u'%s' % w).vector
            if self.embed_dict[w].shape[0] == self.opt.TOKEN_EMBEDDING_SIZE:
                embed_matrix[i] = self.embed_dict[w]
        return embed_matrix


    def qlist_to_vec(self, max_length, q_list):
        """
        Converts a list of words into a format suitable for the embedding layer.

        Arguments:
        max_length -- the maximum length of a question sequence
        q_list -- a list of words which are the tokens in the question

        Returns:
        qvec -- A max_length length vector containing one-hot indices for each word
        cvec -- A max_length length sequence continuation indicator vector
        """
        qvec = np.zeros(max_length)
        embed_matrix = None
        if self.use_embed:
            embed_matrix = np.zeros((max_length, EMBEDDING_SIZE))
        """  pad on the left   """
        # for i in range(max_length):
        #     if i < max_length - len(q_list):
        #         cvec[i] = 0
        #     else:
        #         w = q_list[i-(max_length-len(q_list))]
        #         # is the word in the vocabulary?
        #         if self.qdict.has_key(w) is False:
        #             w = ''
        #         qvec[i] = self.qdict[w]
        #         cvec[i] = 0 if i == max_length - len(q_list) else 1
        """  pad on the right   """
        for i in range(len(q_list)):
            w = q_list[i]
            if self.use_embed:
                if w not in self.embed_dict:
                    self.embed_dict[w] = self.nlp(u'%s' % w).vector
                embed_matrix[i] = self.embed_dict[w]

            if w not in self.qdict:
                w = ''
            qvec[i] = self.qdict[w]
        return qvec, embed_matrix

    def answer_to_vec(self, ans_str, token_obj):
        """
        Return answer id if the answer is included in common or ocr vocabulary
        otherwise return answer id for ''
        """
        if self.mode =='test-dev' or self.mode == 'test':
            return -1

        # FIXME: what if the answer appears both in the dictionary and the token list?
        if self.use_ocr and ans_str in token_obj:
            ans = self.opt.MAX_ANSWER_VOCAB_SIZE + token_obj.index(ans_str)
        elif ans_str in self.adict:
            ans = self.adict[ans_str]
        else:
            ans = self.adict['']
        return ans

    def vec_to_answer(self, ans_symbol):
        """ Return answer string according to id """
        if self.rev_adict is None:
            rev_adict = {}
            for k,v in self.adict.items():
                rev_adict[v] = k
            self.rev_adict = rev_adict

        return self.rev_adict[ans_symbol]

    def vec_to_answer_ocr(self, ans_symbol, ocr_tokens):
        """
        Return answer string according to id
        if id exceeds ocr length, return ''
        """
        if self.rev_adict is None:
            rev_adict = {}
            for k,v in self.adict.items():
                rev_adict[v] = k
            self.rev_adict = rev_adict

        if ans_symbol < self.opt.MAX_ANSWER_VOCAB_SIZE:
            return self.rev_adict[ans_symbol]
        elif ans_symbol < self.opt.MAX_ANSWER_VOCAB_SIZE + len(ocr_tokens):
            return ocr_tokens[ans_symbol - self.opt.MAX_ANSWER_VOCAB_SIZE]
        else:
            return ''

    def create_batch(self, qid_list):

        qvec = np.zeros((self.batchsize, self.max_length))
        q_length = np.zeros(self.batchsize)
        # placeholder for embed
        embed_matrix = np.zeros((self.batchsize, self.max_length, EMBEDDING_SIZE))
        # placeholder for ocr
        ocr_length = np.zeros(self.batchsize)
        ocr_embedding = np.zeros((self.batchsize, self.max_token_size, self.opt.TOKEN_EMBEDDING_SIZE))
        ocr_tokens = list()

        if self.use_embed:
            ivec = np.zeros((self.batchsize, 2048, self.opt.IMG_FEAT_SIZE))
        else:
            ivec = np.zeros((self.batchsize, 2048))

        if self.mode == 'val' or self.mode == 'test-dev' or self.mode == 'test':
            avec = np.zeros(self.batchsize)
        else:
            avec = np.zeros((self.batchsize, self.opt.NUM_OUTPUT_UNITS))

        for i,qid in enumerate(qid_list):

            # load raw question information
            q_str = self.getQuesStr(qid)
            q_ans = self.getAnsObj(qid)
            q_iid = self.getImgId(qid)
            # for ocr
            q_tokens = self.getQuesOcrTokens(qid)
            ocr_tokens += [q_tokens]

            # convert question to vec
            q_list = VQADataProvider.seq_to_list(q_str, self.max_length)
            t_qvec, t_embed_matrix = self.qlist_to_vec(self.max_length, q_list)

            try:
                qid_split = qid.split(QID_KEY_SEPARATOR)
                data_split = qid_split[0]
                if data_split == 'genome':
                    t_ivec = np.load(config.DATA_PATHS[self.exp_type]['genome']['features_prefix'] + str(q_iid) + '.jpg.npy')
                else:
                    path = config.DATA_PATHS[self.exp_type][data_split]['features_prefix'] + config.FEATURE_FILENAME[self.exp_type](q_iid)
                    if os.path.isfile(path):
                        t_ivec = np.load(path)
                    else:
                        t_ivec = np.load(config.DATA_PATHS[self.exp_type][data_split]['features_prefix_alternative'] + config.FEATURE_FILENAME[self.exp_type](q_iid))

                # reshape t_ivec to D x FEAT_SIZE
                if self.use_embed:
                    t_ivec = t_ivec.reshape((2048, -1))

                t_ivec = ( t_ivec / np.sqrt((t_ivec**2).sum()) )
            except:
                t_ivec = 0.
                print(
                    'data not found for qid : ', q_iid, self.mode, '[',
                    config.DATA_PATHS[self.exp_type][data_split]['features_prefix'], ',',
                    config.DATA_PATHS[self.exp_type][data_split]['features_prefix_alternative'], ']',
                    config.FEATURE_FILENAME[self.exp_type](q_iid)
                )

            # convert answer to vec
            if self.mode == 'val' or self.mode == 'test-dev' or self.mode == 'test':
                q_ans_str = self.extract_answer(q_ans)
                t_avec = self.answer_to_vec(q_ans_str, q_tokens)
            else:
                t_avec = self.extract_answer_list(q_ans, q_tokens)

            qvec[i,...] = t_qvec
            q_length[i] = len(q_list)
            avec[i,...] = t_avec
            if self.use_embed:
                ivec[i,:,0:t_ivec.shape[1]] = t_ivec
                embed_matrix[i,...] = t_embed_matrix
            else:
                ivec[i,...] = t_ivec

            if self.use_ocr:
                t_ocr_embedding = self.tokenlist_to_vec(q_tokens)
                ocr_length[i] = len(q_tokens)
                ocr_embedding[i, ...] = t_ocr_embedding

        return qvec, q_length, ivec, avec, embed_matrix, ocr_length, ocr_embedding, ocr_tokens


    def get_batch_vec(self):
        if self.batch_len is None:
            self.n_skipped = 0
            qid_list = self.getQuesIds()
            random.shuffle(qid_list)
            self.qid_list = qid_list
            self.batch_len = len(qid_list)
            self.batch_index = 0
            self.epoch_counter = 0

        def has_at_least_one_valid_answer(t_qid):
            """
            make sure at least one answer falls in the common answer vocabulary
            TODO: include the ocr token list for checking as well
            """
            answer_obj = self.getAnsObj(t_qid)
            answer_list = [ans['answer'] for ans in answer_obj]
            for ans in answer_list:
                if ans in self.adict:
                    return True
            return False

        counter = 0
        t_qid_list = []
        t_iid_list = []
        while counter < self.batchsize:
            t_qid = self.qid_list[self.batch_index]
            t_iid = self.getImgId(t_qid)
            if self.mode == 'val' or self.mode == 'test-dev' or self.mode == 'test':
                t_qid_list.append(t_qid)
                t_iid_list.append(t_iid)
                counter += 1
            elif has_at_least_one_valid_answer(t_qid):
                t_qid_list.append(t_qid)
                t_iid_list.append(t_iid)
                counter += 1
            else:
                self.n_skipped += 1

            if self.batch_index < self.batch_len-1:
                self.batch_index += 1
            else:
                self.epoch_counter += 1
                qid_list = self.getQuesIds()
                random.shuffle(qid_list)
                self.qid_list = qid_list
                self.batch_index = 0
                print("%d questions were skipped in a single epoch" % self.n_skipped, flush=True)
                self.n_skipped = 0
        t_batch = self.create_batch(t_qid_list)
        return t_batch + (t_qid_list, t_iid_list, self.epoch_counter)


class VQADataset(data.Dataset):
    """
    in order to use the universal api, always return the embed matrix
    embed matrix is None if self.use_embed evals to False
    """

    def __init__(self, opt, cache_dir):
        self.mode = opt.TRAIN_DATA_SPLITS
        if self.mode == 'val' or self.mode == 'test-dev' or self.mode == 'test':
            pass
        else:
            self.dp = VQADataProvider(opt, opt.BATCH_SIZE, self.mode, cache_dir)

    def __getitem__(self, index):
        if self.mode == 'val' or self.mode == 'test-dev' or self.mode == 'test':
            pass
        else:
            word, word_length, feature, answer, embed_matrix, cvec_token, token_embedding, original_list_tokens, _, _, epoch = self.dp.get_batch_vec()
            return word, word_length, feature, answer, embed_matrix, cvec_token, token_embedding, original_list_tokens, epoch

    def __len__(self):
        if self.mode == 'train':
            return 150000   # this number had better bigger than "maxiterations" which you set in config
            # mfh_baseline: 200000
            # mfb_coatt_glove: 100000

    def get_vocab_size(self):
        return self.dp.get_vocab_size()
