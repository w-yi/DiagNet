import json
import boto3
from tqdm import tqdm


json_file = "TextVQA_0.5_train.json"
dst_path = '/home/syqian/openimages/train'

s3 = boto3.resource('s3')
bucket = s3.Bucket('open-images-dataset')

annotations = None
with open(json_file) as f:
    annotations = json.load(f)

split = annotations['dataset_type']
print('Dataset split:', split)

for img in tqdm(annotations['data']):
    filename = img['image_id'] + '.jpg'
    bucket.download_file(split + '/' + filename, dst_path + filename)
    #print(filename)
