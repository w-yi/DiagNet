import argparse
import socket
import os
from utils.commons import get_time, check_mkdir

# get the project root dir assuming data is located within the same project folder
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# vqa tools - get from https://github.com/VT-vision-lab/VQA

VQA_TOOLS_DIR = os.path.join(ROOT_DIR, 'data', 'VQA')
VQA_TOOLS_PATH = os.path.join(VQA_TOOLS_DIR, 'PythonHelperTools')
VQA_EVAL_TOOLS_PATH = os.path.join(VQA_TOOLS_DIR, 'PythonEvaluationTools')

# root directory for training generated data
TRAIN_DIR = os.path.join(ROOT_DIR, 'training')
# plots and valuation files
OUTPUT_DIR = os.path.join(TRAIN_DIR, 'output')
# vocab cache for different datasets
VOCABCACHE_DIR = os.path.join(TRAIN_DIR, 'vocab_cache')
# model checkpoints
CACHE_DIR = os.path.join(TRAIN_DIR, 'checkpoint')
# logging files
LOG_DIR = os.path.join(ROOT_DIR, 'logs')

for dir in [OUTPUT_DIR, VOCABCACHE_DIR, CACHE_DIR, LOG_DIR]:
    check_mkdir(dir)

# location of the data
VQA_PREFIX = os.path.join(ROOT_DIR, 'data', 'VQA')

TEXTVQA_PREFIX = os.path.join(ROOT_DIR, 'data', 'textvqa')

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
            'features_prefix': VQA_PREFIX + '/Features/coco_resnet/val2014/COCO_val2014_',
            'image_prefix': VQA_PREFIX + '/Images/val2014/COCO_val2014_'
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
            'ques_file': TEXTVQA_PREFIX + '/json_OCR/textvqa_questions_train_ocr_complete.json',
            'ans_file': TEXTVQA_PREFIX + '/json_OCR/textvqa_annotations_train_complete.json',
            'features_prefix': TEXTVQA_PREFIX + '/baseline/train/'
        },
        'val': {
            'ques_file': TEXTVQA_PREFIX + '/json_OCR/textvqa_questions_val_ocr_complete.json',
            'ans_file': TEXTVQA_PREFIX + '/json_OCR/textvqa_annotations_val_complete.json',
            'features_prefix': TEXTVQA_PREFIX + '/baseline/val/'
        },
        'test': {
            'ques_file': TEXTVQA_PREFIX + '/json_OCR/textvqa_questions_test_ocr_complete.json',
            'features_prefix': TEXTVQA_PREFIX + '/baseline/test/'
        }
    },
    'textvqa_butd': {
        'train': {
            'ques_file': TEXTVQA_PREFIX + '/OCR_sorted_lower_case/textvqa_questions_train_ocr_partial_sorted_flag_v1.json',
            'ans_file': TEXTVQA_PREFIX + '/json_OCR/textvqa_annotations_train_partial.json',
            'features_prefix': TEXTVQA_PREFIX + '/features_butd/train/',
            'features_prefix_alternative': TEXTVQA_PREFIX + '/baseline/train/'
        },
        'val': {
            'ques_file': TEXTVQA_PREFIX + '/OCR_sorted_lower_case/textvqa_questions_val_ocr_complete_sorted_flag_v1.json',
            'ans_file': TEXTVQA_PREFIX + '/json_OCR/textvqa_annotations_val_complete.json',
            'features_prefix': TEXTVQA_PREFIX + '/features_butd/val/',
            'features_prefix_alternative': TEXTVQA_PREFIX + '/baseline/val/'
        },
        'test-dev': {
            'ques_file': TEXTVQA_PREFIX + '/OCR_sorted_lower_case/textvqa_questions_test_ocr_complete_sorted_flag.json',
            'features_prefix': TEXTVQA_PREFIX + '/features_butd/test/',
            'features_prefix_alternative': TEXTVQA_PREFIX + '/baseline/test/'
        }
    },
}

FEATURE_FILENAME = {
    'baseline': (lambda q_iid: str(q_iid).zfill(12) + '.jpg.npy'),
    'glove': (lambda q_iid: str(q_iid) + '.npy'),
    'textvqa': (lambda q_iid: str(q_iid) + '.jpg.npy'),
    'textvqa_butd': (lambda q_iid: str(q_iid) + '.jpg.npy')
}

IMAGE_FILENAME = {
    'baseline': (lambda q_iid: str(q_iid).zfill(12) + '.jpg')
}

QTYPES = {
    'what_colors': ['what^color'],
    'what_is': ['what^is', 'what^kind', 'what^are'],
    'is': ['is^the', 'is^this', 'is^there'],
    'how_many': ['how^many']
}


def get_ID(args):
    id = '_'.join([get_time('%Y-%m-%dT%H%M%S'), args.MODEL, args.EXP_TYPE])
    if args.EMBED:
        id += '_embed'
    if args.OCR:
        id += '_ocr'
    return id

def parse_opt():
    parser = argparse.ArgumentParser()
    # Data input settings
    parser.add_argument('MODEL', type=str, choices=['mfb', 'mfh'])
    parser.add_argument('EXP_TYPE', type=str, choices=['baseline', 'glove', 'textvqa', 'textvqa_butd'])
    parser.add_argument('--EMBED', action='store_true')
    # use ocr infomation from textvqa dataset
    parser.add_argument('--OCR', action='store_true')
    # use a binary predictor to determine whether answer falls in the ocr text set
    parser.add_argument('--BINARY', action='store_true')
    parser.add_argument('--BIN_HELP', action='store_true')


    parser.add_argument('--LATE_FUSION', action='store_true')
    parser.add_argument('--PROB_PATH', type=str, default='')

    parser.add_argument('--TRAIN_GPU_ID', type=int, default=0)
    parser.add_argument('--TEST_GPU_ID', type=int, default=0)
    parser.add_argument('--SEED', type=int, default=-1)
    parser.add_argument('--BATCH_SIZE', type=int, default=200) # glove: 64
    parser.add_argument('--VAL_BATCH_SIZE', type=int, default=200) # glove: 32
    parser.add_argument('--MAX_ANSWER_VOCAB_SIZE', type=int, default=3000)
    parser.add_argument('--MAX_TOKEN_SIZE', type=int, default=104)
    parser.add_argument('--MAX_QUESTION_LENGTH', type=int, default=15)
    parser.add_argument('--MAX_ITERATIONS', type=int, default=100000) # non-glove: 50000, glove:100000
    parser.add_argument('--PRINT_INTERVAL', type=int, default=100)
    parser.add_argument('--CHECKPOINT_INTERVAL', type=int, default=1000)
    parser.add_argument('--TESTDEV_INTERVAL', type=int, default=100000) # non-mfh_glove: 45000
    parser.add_argument('--RESUME_PATH', type=str, default='')
    parser.add_argument('--VAL_INTERVAL', type=int, default=2000)
    parser.add_argument('--IMAGE_CHANNEL', type=int, default=2048)
    parser.add_argument('--INIT_LERARNING_RATE', type=float, default=0.0007)
    parser.add_argument('--DECAY_STEPS', type=int, default=40000) # non-glove: 20000
    parser.add_argument('--DECAY_RATE', type=float, default=0.5)
    parser.add_argument('--MFB_FACTOR_NUM', type=int, default=5)
    parser.add_argument('--MFB_OUT_DIM', type=int, default=1000)
    parser.add_argument('--LSTM_UNIT_NUM', type=int, default=1024)
    parser.add_argument('--LSTM_DROPOUT_RATIO', type=float, default=0.3)
    parser.add_argument('--MFB_DROPOUT_RATIO', type=float, default=0.1)
    parser.add_argument('--TRAIN_DATA_SPLITS', type=str, default='train')
    parser.add_argument('--QUESTION_VOCAB_SPACE', type=str, default='train')
    parser.add_argument('--ANSWER_VOCAB_SPACE', type=str, default='train')

    parser.add_argument('--TOKEN_EMBEDDING_SIZE', type=int, default=300)

    # embed options
    parser.add_argument('--NUM_IMG_GLIMPSE', type=int, default=2)
    parser.add_argument('--NUM_QUESTION_GLIMPSE', type=int, default=2)
    parser.add_argument('--IMG_FEAT_SIZE', type=int, default=100)

    # OCR options
    parser.add_argument('--NUM_OCR_GLIMPSE', type=int, default=2)

    # BINARY options
    parser.add_argument('--BIN_LOSS_RATE', type=float, default=1.0)
    parser.add_argument('--BIN_TOKEN_RATE', type=float, default=1.0)

    args = parser.parse_args()

    args.ID = get_ID(args)

    # define the dimention of model output
    args.NUM_OUTPUT_UNITS = args.MAX_ANSWER_VOCAB_SIZE
    if args.OCR:
        assert args.EMBED, 'ocr only supported with embed now'
        args.NUM_OUTPUT_UNITS = args.MAX_ANSWER_VOCAB_SIZE + args.MAX_TOKEN_SIZE
    if args.BINARY:
        assert args.OCR, 'binary predictor only enabled with ocr'

    return args
