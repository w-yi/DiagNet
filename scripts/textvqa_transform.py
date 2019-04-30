import json
from collections import Counter
import argparse

def transform(split, input_dir, output_dir)
    with open(input_dir + 'TextVQA_0.5_{}.json'.format(split), 'r') as f:
        dict_original = json.load(f)

    print("@@@ Generating question file for " + split)

    dict_question = dict()
    dict_question['info'] = {
        'description': 'TextVQA dataset.',
         'url': '_',
         'version': '_',
         'year': '_',
         'contributor': '_',
         'date_created': '_'
    }
    dict_question['task_type'] = 'Open-Ended'
    dict_question['data_type'] = 'OpenImage'
    dict_question['license'] = {'url': '_', 'name': '_'}
    dict_question['data_subtype'] = 'TextVQA'

    list_questions = list()
    for cur_data in dict_original['data']:
        dict_tmp_q_ = dict()
        dict_tmp_q_['question'] = cur_data['question']
        dict_tmp_q_['image_id'] = cur_data['image_id']
        dict_tmp_q_['question_id'] = cur_data['question_id']
        tmp_test = cur_data['ocr_info']
        tmp_test_sorted = sorted(tmp_test, key=
                             lambda item:(item['bounding_box']['top_left_x'] + 0.5 * item['bounding_box']['width'],
                                          item['bounding_box']['top_left_y'] - 0.5 * item['bounding_box']['height']))
        tmp_test_sorted_token = [str.lower(i['word']) for i in tmp_test_sorted]
        dict_tmp_q_['ocr_tokens'] = tmp_test_sorted_token
        dict_tmp_q_['ocr_info'] = tmp_test_sorted

        if split == 'test':
            list_questions += [dict_tmp_q_]
        else:
            list_answers = cur_data['answers']
            most_freq_answer = Counter(list_answers).most_common(1)[0][0]
            flag_found = False
            if most_freq_answer in tmp_test_sorted_token:
                flag_found = True
            if flag_found:
                dict_tmp_q_['ocr_answer_flag'] = 1
            else:
                dict_tmp_q_['ocr_answer_flag'] = 0
            list_questions += [dict_tmp_q_]

    dict_question['questions'] = list_questions

    with open(output_dir + 'textvqa_questions_{}_ocr_complete_sorted_flag_v1.json'.format(split), 'w') as f:
        json.dump(dict_question, f)


    if split == 'test':
        return

    print("@@@ Generating annotations file for " + split)

    dict_annotations = dict()
    dict_annotations['info'] = {
        'description': 'TextVQA dataset.',
         'url': '_',
         'version': '_',
         'year': '_',
         'contributor': '_',
         'date_created': '_'}
    dict_annotations['data_type'] = 'OpenImage'
    dict_annotations['license'] = {'url': '_', 'name': '_'}
    dict_annotations['data_subtype'] = 'TextVQA'

    list_annotations = list()
    for cur_data in dict_original['data']:
        dict_tmp_a_ = dict()
        dict_tmp_a_['image_id'] = cur_data['image_id']
        dict_tmp_a_['question_id'] = cur_data['question_id']
        dict_tmp_a_['question_type'] = 'What'
        dict_tmp_a_['answer_type'] = 'other'
        list_tmp_answers = list()
        for idx, cur_a in enumerate(cur_data['answers']):
            list_tmp_answers += [{
                'answer': cur_a,
                'answer_confidence': 'yes',
                'answer_id': idx + 1
            }]
        dict_tmp_a_['answers'] = list_tmp_answers
        list_annotations += [dict_tmp_a_]
    dict_annotations['annotations'] = list_annotations

    with open(output_dir + 'textvqa_annotations_{}_complete.json'.format(split), 'w') as f:
        json.dump(dict_annotations, f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', '-s', choices=['train', 'val', 'test'], help='split')
    parser.add_argument('--input_dir', '-i', type=str, default='/data/home/wennyi/vqa-mfb.pytorch/data/textvqa/origin/', help='original textvqa dataset dir')
    parser.add_argument('--output_dir', '-f', type=str, default='/data/home/wennyi/vqa-mfb.pytorch/data/textvqa/', help='output dir')
    args = parser.parse_args()
    transform(args.split, args.input_dir, args.output_dir)
