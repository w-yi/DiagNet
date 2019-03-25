import numpy as np
import re, json, random
import config
import torch.utils.data as data
import spacy
import os

QID_KEY_SEPARATOR = '/'
ZERO_PAD = '_PAD'
EMBEDDING_SIZE = 300
class VQADataProvider:

    def __init__(self, opt, batchsize, mode, cache_dir=config.VOCABCACHE_DIR, max_length=15):
        self.opt = opt
        self.mode = mode
        self.batchsize = batchsize
        self.d_vocabulary = None
        self.batch_index = None
        self.batch_len = None
        self.rev_adict = None
        self.max_length = max_length
        self.cache_dir = cache_dir
        self.exp_type = opt.EXP_TYPE

        self.qdic, self.adic = VQADataProvider.load_data(self.mode, self.exp_type)

        self._get_vocab_files()

        if self.use_embed():
            self.n_ans_vocabulary = len(self.adict)
            self.nlp = spacy.load('en_vectors_web_lg')
            self.embed_dict = {} # word -> embed vector

    def _get_vocab_files(self):
        """
        get vocab files
        load cached files if exist
        """
        question_vocab, answer_vocab = {}, {}
        qdict_path = os.path.join(self.cache_dir, self.exp_type + '_qdict.json')
        adict_path = os.path.join(self.cache_dir, self.exp_type + '_adict.json')
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
        qdic, _ = VQADataProvider.load_data(self.opt.QUESTION_VOCAB_SPACE, self.exp_type)
        question_vocab = VQADataProvider.make_question_vocab(qdic)
        print('making answer vocab...', self.opt.ANSWER_VOCAB_SPACE)
        _, adic = VQADataProvider.load_data(self.opt.ANSWER_VOCAB_SPACE, self.exp_type)
        answer_vocab = VQADataProvider.make_answer_vocab(adic, self.opt.NUM_OUTPUT_UNITS)
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

    @staticmethod
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

    @staticmethod
    def load_vqa_json(data_split, exp_type='baseline'):
        """
        Parses the question and answer json files for the given data split.
        Returns the question dictionary and the answer dictionary.
        """
        qdic, adic = {}, {}

        with open(config.DATA_PATHS[exp_type][data_split]['ques_file'], 'r') as f:
            qdata = json.load(f)['questions']
            for q in qdata:
                qdic[data_split + QID_KEY_SEPARATOR + str(q['question_id'])] = \
                    {'qstr': q['question'], 'iid': q['image_id']}

        if 'test' not in data_split:
            with open(config.DATA_PATHS[exp_type][data_split]['ans_file'], 'r') as f:
                adata = json.load(f)['annotations']
                for a in adata:
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
    def load_data(data_split_str, exp_type='baseline'):
        all_qdic, all_adic = {}, {}

        for data_split in data_split_str.split('+'):
            assert data_split in config.DATA_PATHS[exp_type].keys(), 'unknown data split'
            if data_split == 'genome':
                qdic, adic = VQADataProvider.load_genome_json(exp_type)
                all_qdic.update(qdic)
                all_adic.update(adic)
            else:
                qdic, adic = VQADataProvider.load_vqa_json(data_split, exp_type)
                all_qdic.update(qdic)
                all_adic.update(adic)
        return all_qdic, all_adic

    def use_embed(self):
        """
        check usage of pretrained word embedding
        """
        if self.exp_type == 'glove':
            return True
        else:
            return False

    def getQuesIds(self):
        return list(self.qdic.keys())

    def getStrippedQuesId(self, qid):
        return qid.split(QID_KEY_SEPARATOR)[1]

    def getImgId(self,qid):
        return self.qdic[qid]['iid']

    def getQuesStr(self,qid):
        return self.qdic[qid]['qstr']

    def getAnsObj(self,qid):
        if self.mode == 'test-dev' or self.mode == 'test':
            return -1
        return self.adic[qid]

    @staticmethod
    def seq_to_list(s):
        t_str = s.lower()
        for i in [r'\?',r'\!',r'\'',r'\"',r'\$',r'\:',r'\@',r'\(',r'\)',r'\,',r'\.',r'\;']:
            t_str = re.sub( i, '', t_str)
        for i in [r'\-',r'\/']:
            t_str = re.sub( i, ' ', t_str)
        q_list = re.sub(r'\?','',t_str.lower()).split(' ')
        q_list = filter(lambda x: len(x) > 0, q_list)
        return list(q_list)

    def extract_answer(self,answer_obj):
        """ Return the most popular answer in string."""
        if self.mode == 'test-dev' or self.mode == 'test':
            return -1
        answer_list = [ answer_obj[i]['answer'] for i in range(10)]
        dic = {}
        for ans in answer_list:
            if ans in dic:
                dic[ans] +=1
            else:
                dic[ans] = 1
        max_key = max((v,k) for (k,v) in dic.items())[1]
        return max_key

    def extract_answer_prob(self,answer_obj):
        """ Return the most popular answer in string."""
        if self.mode == 'test-dev' or self.mode == 'test':
            return -1

        answer_list = [ ans['answer'] for ans in answer_obj]
        prob_answer_list = []
        for ans in answer_list:
            if ans in self.adict:
                prob_answer_list.append(ans)

    def extract_answer_list(self,answer_obj):
        answer_list = [ ans['answer'] for ans in answer_obj]
        prob_answer_vec = np.zeros(self.opt.NUM_OUTPUT_UNITS)
        for ans in answer_list:
            if ans in self.adict:
                index = self.adict[ans]
                prob_answer_vec[index] += 1
        return prob_answer_vec / np.sum(prob_answer_vec)

        if len(prob_answer_list) == 0:
            if self.mode == 'val' or self.mode == 'test-dev' or self.mode == 'test':
                return 'hoge'
            else:
                raise Exception("This should not happen.")
        else:
            return random.choice(prob_answer_list)

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
        cvec = np.zeros(max_length)
        embed_matrix = None
        if self.use_embed():
            # TODO: change this config
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
        for i in range(max_length):
            if i >= len(q_list):
                pass
            else:
                w = q_list[i]

                if self.use_embed():
                    if w not in self.embed_dict:
                        self.embed_dict[w] = self.nlp(u'%s' % w).vector
                    embed_matrix[i] = self.embed_dict[w]

                if w not in self.qdict:
                    w = ''
                qvec[i] = self.qdict[w]
                cvec[i] = 1
        return qvec, cvec, embed_matrix

    def answer_to_vec(self, ans_str):
        """ Return answer id if the answer is included in vocabulary otherwise '' """
        if self.mode =='test-dev' or self.mode == 'test':
            return -1

        if ans_str in self.adict:
            ans = self.adict[ans_str]
        else:
            ans = self.adict['']
        return ans

    def vec_to_answer(self, ans_symbol):
        """ Return answer id if the answer is included in vocabulary otherwise '' """
        if self.rev_adict is None:
            rev_adict = {}
            for k,v in self.adict.items():
                rev_adict[v] = k
            self.rev_adict = rev_adict

        return self.rev_adict[ans_symbol]

    def create_batch(self,qid_list):

        qvec = (np.zeros(self.batchsize*self.max_length)).reshape(self.batchsize,self.max_length)
        cvec = (np.zeros(self.batchsize*self.max_length)).reshape(self.batchsize,self.max_length)

        if self.use_embed():
            ivec = np.zeros((self.batchsize, 2048, self.opt.IMG_FEAT_SIZE))
        else:
            ivec = (np.zeros(self.batchsize*2048)).reshape(self.batchsize,2048)

        if self.mode == 'val' or self.mode == 'test-dev' or self.mode == 'test':
            avec = np.zeros(self.batchsize)
        else:
            avec = np.zeros((self.batchsize, self.opt.NUM_OUTPUT_UNITS))

        if self.use_embed():
            embed_matrix = np.zeros((self.batchsize, self.max_length, EMBEDDING_SIZE))

        for i,qid in enumerate(qid_list):

            # load raw question information
            q_str = self.getQuesStr(qid)
            q_ans = self.getAnsObj(qid)
            q_iid = self.getImgId(qid)

            # convert question to vec
            q_list = VQADataProvider.seq_to_list(q_str)
            t_qvec, t_cvec, t_embed_matrix = self.qlist_to_vec(self.max_length, q_list)

            try:
                qid_split = qid.split(QID_KEY_SEPARATOR)
                data_split = qid_split[0]
                if data_split == 'genome':
                    t_ivec = np.load(config.DATA_PATHS[self.exp_type]['genome']['features_prefix'] + str(q_iid) + '.jpg.npy')
                else:
                    # print(config.DATA_PATH[self.exp_type][data_split]['features_prefix']+config.FEATURE_FILENAE[self.exp_type](q_iid))
                    t_ivec = np.load(config.DATA_PATHS[self.exp_type][data_split]['features_prefix'] + config.FEATURE_FILENAME[self.exp_type](q_iid))

                # reshape t_ivec to D x FEAT_SIZE
                if self.use_embed() and len(t_ivec.shape) > 2:
                    t_ivec = t_ivec.reshape((2048, -1))

                t_ivec = ( t_ivec / np.sqrt((t_ivec**2).sum()) )
            except:
                t_ivec = 0.
                print('data not found for qid : ', q_iid,  self.mode, config.DATA_PATHS[self.exp_type][data_split]['features_prefix']+config.FEATURE_FILENAME[self.exp_type](q_iid))

            # convert answer to vec
            if self.mode == 'val' or self.mode == 'test-dev' or self.mode == 'test':
                q_ans_str = self.extract_answer(q_ans)
                t_avec = self.answer_to_vec(q_ans_str)
            else:
                t_avec = self.extract_answer_list(q_ans)

            qvec[i,...] = t_qvec
            cvec[i,...] = t_cvec
            avec[i,...] = t_avec
            if self.use_embed():
                ivec[i,:,0:t_ivec.shape[0]] = t_ivec.T
                embed_matrix[i,...] = t_embed_matrix
            else:
                ivec[i,...] = t_ivec

        if self.use_embed():
            return qvec, cvec, ivec, avec, embed_matrix
        else:
            return qvec, cvec, ivec, avec, 0


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
            answer_obj = self.getAnsObj(t_qid)
            answer_list = [ans['answer'] for ans in answer_obj]
            for ans in answer_list:
                if ans in self.adict:
                    return True

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
                print("%d questions were skipped in a single epoch" % self.n_skipped)
                self.n_skipped = 0

        t_batch = self.create_batch(t_qid_list)
        return t_batch + (t_qid_list, t_iid_list, self.epoch_counter)

class VQADataset(data.Dataset):
    """
    in order to use the universal api, always return the embed matrix
    embed matrix is None if self.use_embed() evals to False
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
            word, cont, feature, answer, embed_matrix, _, _, epoch = self.dp.get_batch_vec()
            word_length = np.sum(cont, axis=1)
            return word, word_length, feature, answer, embed_matrix, epoch

    def __len__(self):
        if self.mode == 'train':
            return 150000   # this number had better bigger than "maxiterations" which you set in config
            # mfh_baseline: 200000
            # mfb_coatt_glove: 100000

    def get_vocab_size(self):
        return len(self.dp.qdict), len(self.dp.adict)

    def use_embed(self):
        return self.dp.use_embed()
