import json
from shutil import copyfile

json_file = "TextVQA_0.5_train.json"
src_path = './'
dst_path = './'

annotations = None
with open(json_file) as f:
    annotations = json.load(f)

print(annotations['dataset_type'])

for img in annotations['data']:
    filename = img['image_id'] + '.jpg'
    print(filename)
    copyfile(src_path + filename, dst_path + filename)
