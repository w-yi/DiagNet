import argparse
import socket
import datetime
import os

# get the project root dir assuming data is located within the same project folder
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if socket.gethostname() == 'DESKTOP-9UNSKEL':
    DATA_DIR = os.path.join(ROOT_DIR, 'VQA')
else:
    DATA_DIR = os.path.join(ROOT_DIR, 'data', 'VQA')
# vqa tools - get from https://github.com/VT-vision-lab/VQA

VQA_TOOLS_DIR = os.path.join(ROOT_DIR, 'data', 'VQA')
VQA_TOOLS_PATH = os.path.join(VQA_TOOLS_DIR, 'PythonHelperTools')
VQA_EVAL_TOOLS_PATH = os.path.join(VQA_TOOLS_DIR, 'PythonEvaluationTools')

TRAIN_DIR = os.path.join(ROOT_DIR, 'training')
CACHE_DIR = os.path.join(ROOT_DIR, 'checkpoint')
# location of the data
VQA_PREFIX = DATA_DIR

# baseline_dir = '' # current dataset only includes baseline features
# glove_dir = '/faster_rcnn_resnet_pool5' # features used for glove models; should be added
DATA_PATHS = {
    'baseline': {
        'train': {
            'ques_file': VQA_PREFIX + '/Questions/OpenEnded_mscoco_train2014_questions.json',
            'ans_file': VQA_PREFIX + '/Annotations/mscoco_train2014_annotations.json',
            'features_prefix': VQA_PREFIX + '/Features/coco_resnet/train2014/COCO_train2014_'
        },
        'val': {
            'ques_file': VQA_PREFIX + '/Questions/OpenEnded_mscoco_val2014_questions.json',
            'ans_file': VQA_PREFIX + '/Annotations/mscoco_val2014_annotations.json',
            'features_prefix': VQA_PREFIX + '/Features/coco_resnet/val2014/COCO_val2014_'
        },
        'test-dev': {
            'ques_file': VQA_PREFIX + '/Questions/OpenEnded_mscoco_test-dev2015_questions.json',
            'features_prefix': VQA_PREFIX + '/Features/coco_resnet/test2015/COCO_test2015_'
        },
        'test': {
            'ques_file': VQA_PREFIX + '/Questions/OpenEnded_mscoco_test2015_questions.json',
            'features_prefix': VQA_PREFIX + '/Features/coco_resnet/test2015/COCO_test2015_'
        },
        'genome': {
            'genome_file': VQA_PREFIX + '/Questions/OpenEnded_genome_train_questions.json',
            'features_prefix': VQA_PREFIX + '/Features/genome/feat_resnet-152/resnet_bgrms_large/'
        }
    },
    'glove': {
        'train': {
            'ques_file': VQA_PREFIX + '/Questions/OpenEnded_mscoco_train2014_questions.json',
            'ans_file': VQA_PREFIX + '/Annotations/mscoco_train2014_annotations.json',
            'features_prefix': VQA_PREFIX + '/Features/BUTD_features/'
        },
        'val': {
            'ques_file': VQA_PREFIX + '/Questions/OpenEnded_mscoco_val2014_questions.json',
            'ans_file': VQA_PREFIX + '/Annotations/mscoco_val2014_annotations.json',
            'features_prefix': VQA_PREFIX + '/Features/BUTD_features/'
        },
        'test-dev': {
            'ques_file': VQA_PREFIX + '/Questions/OpenEnded_mscoco_test-dev2015_questions.json',
            'features_prefix': VQA_PREFIX + '/Features/BUTD_features/'
        },
        'test': {
            'ques_file': VQA_PREFIX + '/Questions/OpenEnded_mscoco_test2015_questions.json',
            'features_prefix': VQA_PREFIX + '/Features/BUTD_features/'
        },
        'genome': {
            'genome_file': VQA_PREFIX + '/Questions/OpenEnded_genome_train_questions.json',
            'features_prefix': VQA_PREFIX + '/Features/genome/BUTD_features/resnet_bgrms_large/'
        }
    },
    'textvqa': {
        'train': {
            'ques_file': os.path.join(ROOT_DIR, 'data', 'shared_textvqa', 'textvqa_questions_train.json'),
            'ans_file': os.path.join(ROOT_DIR, 'data', 'shared_textvqa', 'textvqa_annotations_train.json'),
            'features_prefix': os.path.join(ROOT_DIR, 'data', 'textvqa_features', 'baseline', 'train')
        },
        'val': {
            'ques_file': os.path.join(ROOT_DIR, 'data', 'shared_textvqa', 'textvqa_questions_val.json'),
            'ans_file': os.path.join(ROOT_DIR, 'data', 'shared_textvqa', 'textvqa_annotations_val.json'),
            'features_prefix': os.path.join(ROOT_DIR, 'data', 'textvqa_features', 'baseline', 'val')
        },
        'test': {
            'ques_file': os.path.join(ROOT_DIR, 'data', 'shared_textvqa', 'textvqa_questions_test.json'),
            'features_prefix': os.path.join(ROOT_DIR, 'data', 'textvqa_features', 'baseline', 'test')
        }
    },
}


def baseline_fn(q_iid):
    return str(q_iid).zfill(12) + '.jpg.npy'


def glove_fn(q_iid):
    return str(q_iid) + '.npy'


def textvqa_fn(q_iid):
    return str(q_iid) + '.jpg.npy'


def get_time():
    return datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S")

FEATURE_FILENAME = {
    'baseline': baseline_fn,
    'glove': glove_fn,
    'textvqa': textvqa_fn,
}


def parse_opt():
    parser = argparse.ArgumentParser()
    # Data input settings
    parser.add_argument('MODEL', type=str, choices=['mfb', 'mfh'])
    parser.add_argument('EXP_TYPE', type=str, choices=['baseline', 'glove', 'textvqa'])

    parser.add_argument('--TRAIN_GPU_ID', type=int, default=0)
    parser.add_argument('--TEST_GPU_ID', type=int, default=0)
    parser.add_argument('--SEED', type=int, default=-1)
    parser.add_argument('--BATCH_SIZE', type=int, default=200) # glove: 64
    parser.add_argument('--VAL_BATCH_SIZE', type=int, default=1000) # glove: 32
    parser.add_argument('--NUM_OUTPUT_UNITS', type=int, default=3000)
    parser.add_argument('--MAX_WORDS_IN_QUESTION', type=int, default=15)
    parser.add_argument('--MAX_ITERATIONS', type=int, default=50000) # glove: 100000
    parser.add_argument('--PRINT_INTERVAL', type=int, default=100)
    parser.add_argument('--CHECKPOINT_INTERVAL', type=int, default=5000)
    parser.add_argument('--TESTDEV_INTERVAL', type=int, default=45000) # mfh_glove: 100000
    parser.add_argument('--RESUME', type=bool, default=False)
    parser.add_argument('--RESUME_PATH', type=str, default='./data/***.pth')
    parser.add_argument('--VAL_INTERVAL', type=int, default=5000)
    parser.add_argument('--IMAGE_CHANNEL', type=int, default=2048)
    parser.add_argument('--INIT_LERARNING_RATE', type=float, default=0.0007)
    parser.add_argument('--DECAY_STEPS', type=int, default=20000) # glove: 40000
    parser.add_argument('--DECAY_RATE', type=float, default=0.5)
    parser.add_argument('--MFB_FACTOR_NUM', type=int, default=5)
    parser.add_argument('--MFB_OUT_DIM', type=int, default=1000)
    parser.add_argument('--LSTM_UNIT_NUM', type=int, default=1024)
    parser.add_argument('--LSTM_DROPOUT_RATIO', type=float, default=0.3)
    parser.add_argument('--MFB_DROPOUT_RATIO', type=float, default=0.1)
    parser.add_argument('--TRAIN_DATA_SPLITS', type=str, default='train')
    parser.add_argument('--QUESTION_VOCAB_SPACE', type=str, default='train')
    parser.add_argument('--ANSWER_VOCAB_SPACE', type=str, default='train')

    # glove options
    parser.add_argument('--NUM_IMG_GLIMPSE', type=int, default=2)
    parser.add_argument('--NUM_QUESTION_GLIMPSE', type=int, default=2)
    parser.add_argument('--IMG_FEAT_SIZE', type=int, default=100)

    args = parser.parse_args()

    args.ID = '_'.join([get_time(), args.MODEL, args.EXP_TYPE])

    return args
