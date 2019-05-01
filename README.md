# DiagNet: Bridging Text and Image
Code for a class project of [EECS 598/498: Deep Learning](https://docs.google.com/document/u/1/d/e/2PACX-1vSZw2CS74V1BEeruYxASJeeFO51tS7vj9NBjWnCvPkK1m-45xpHaAWr6LMG_0EH6HEqSttWEXRFaHua/pub) at University of Michigan, Winter 2019.

Some code is borrowed from this [pyTorch implementation](https://github.com/asdf0982/vqa-mfb.pytorch) of Multi-modal Factorized Bilinear Pooling (MFB) for VQA. The code of extracting BUTD features is adopted from [the official implementation](https://github.com/peteanderson80/bottom-up-attention).

![Figure 1: The DiagNet Network architecture for TextVQA.](https://github.com/WYchelsy/DiagNet/blob/master/imgs/DiagNet_architecture.png)

## Requirements

The training and inference code of our model require python 3.6 and pytorch 1.0.

```bash
# tensorboardX
pip install tensorboardX

# pytorch
conda install pytorch torchvision cudatoolkit=9.0 -c pytorch

# spacy
conda install -c conda-forge spacy
python -m spacy download en
python -m spacy download en_vectors_web_lg
```

In addition, Preparing BUTD features on TextVQA requires Caffe. Please go to [bottom-up-attention](bottom-up-attention) and check out the README. The environment is exactly the same as the original implementation although we modify some code. AWS GPU instance is recommended to set up the environment.

## Preparing Datasets

We use two datasets for our experiments: [VQA v1.0](https://visualqa.org/vqa_v1_download.html) and [TextVQA v0.5](https://textvqa.org/dataset). Each dataset
has three splits: `train|val|test`; each of them has three components:
* `ques_file`: json file with vqa questions.
* `ans_file`: json file with answers to questions.
* `features_prefix`: path to image feature `.npy` files

Following examples are for TextVQA only.

1. Download dataset and corresponding image files
    ```bash
    mkdir -p data/textvqa/origin
    cd data/textvqa/origin
    wget https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5_train.json
    wget https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5_val.json
    wget https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5_test.json
    wget https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip
    unzip train_val_images.zip
    cd ../../..
    ```

2. Generate ResNet image features:
    ```bash
    python scripts/resnet_feature.py [--split] [--image_dir] [--feature_dir]
    ```

3. Generate BUTD image features:
    ```bash
    # generate tsv file (Caffe is required)
    cd bottom-up-attention
    ./gen_faster_rcnn_textvqa.sh

    # convert tsv file to npy
    python scripts/butd_feature.py [--split] [--image_dir] [--feature_dir]
    ```

4. VQA dataset is already in the desired `ques_file|ans_file` format. Generate json files for TextVQA:
    ```bash
    python scripts/textvqa_transform.py [--split] [--input_dir] [--output_dir]
    ```

5. Modify `DATA_PATHS` in `config.py` to match the dataset and image feature paths accordingly.

## Training

Our implementation supports multiple models and datasets. Use the following command for training (looking into `config.py` for option details):

```bash
python train.py [MODEL] [EXP_TYPE] [--options]
```
Some examples:
1. MFH baseline on VQA v1.0:
    ```bash
    python train.py mfh baseline
    ```

2. DiagNet without OCR on VQA v1.0:
    ```bash
    python train.py mfh glove --EMBED
    ```

3. DiagNet on TextVQA v0.5:
    ```bash
    python train.py mfh textvqa_butd --EMBED --OCR --BIN_HELP
    ```

## Prediction Visualization
1. Download image files and modify `image_prefix` of `DATA_PATHS` in `config.py` accordingly.

2. Run training and get the `.pth` model file in `training/checkpoint`. For example:
    ```bash
    python train.py mfh glove --EMBED
    ```

3. Specify the questions of interest by modifying `QTYPES` in `config.py`

4. Run visualization:
    ```bash
    python predict.py mfh glove --EMBED [--RESUME_PATH]
    ```
![Figure 2: Visualization Example.](https://github.com/WYchelsy/DiagNet/blob/master/imgs/correct224477.png)
