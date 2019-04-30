# DiagNet: Bridging Text and Image
This is the code for a class project of [EECS 598/498: Deep Learning](https://docs.google.com/document/u/1/d/e/2PACX-1vSZw2CS74V1BEeruYxASJeeFO51tS7vj9NBjWnCvPkK1m-45xpHaAWr6LMG_0EH6HEqSttWEXRFaHua/pub) at University of Michigan, Winter 2019.

We borrowed code from this [pyTorch implementation](https://github.com/asdf0982/vqa-mfb.pytorch) of Multi-modal Factorized Bilinear Pooling (MFB) for VQA.

![Figure 1: The DiagNet with OCR Network architecture for TextVQA.](https://github.com/WYchelsy/vqa-mfb.pytorch/blob/docs/imgs/DiagNet.png)

## Related Works
### Papers related to our implementation

- CoAtt: [Hierarchical Question-Image Co-Attention for Visual Question Answering](https://arxiv.org/abs/1606.00061)
- MFB: [Multi-modal Factorized Bilinear Pooling with Co-Attention Learning for Visual Question Answering](http://openaccess.thecvf.com/content_iccv_2017/html/Yu_Multi-Modal_Factorized_Bilinear_ICCV_2017_paper.html)
- MFH: [Beyond Bilinear: Generalized Multi-modal Factorized High-order Pooling for Visual Question Answering](https://arxiv.org/abs/1708.03619)
- BUTD: [Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering](https://arxiv.org/abs/1707.07998)
- Pythia: [Pythia v0.1: the Winning Entry to the VQA Challenge 2018](https://arxiv.org/pdf/1807.09956.pdf)
- TextVQA: [Towards VQA Models That Can Read](https://textvqa.org/assets/paper/TextVQA.pdf)

### Baseline implementations

- PyTorch implementation: <https://github.com/asdf0982/vqa-mfb.pytorch>
- Caffe implementation: <https://github.com/yuzcccc/vqa-mfb>
- pythia: <https://github.com/facebookresearch/pythia>
- BUTD: <https://github.com/peteanderson80/bottom-up-attention>
- Fork of BUTD: <https://github.com/yuzcccc/bottom-up-attention>


## Requirements

python 3.6, pytorch 1.0

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

## Preparing Datasets
We use two datasets for our experiments: [VQA v1.0](https://visualqa.org/vqa_v1_download.html) and [TextVQA v0.5](https://textvqa.org/dataset). Each dataset
has three splits: `train|val|test`; each of them has three components:
* `ques_file`: json file with vqa questions.
* `ans_file`: json file with answers to questions.
* `features_prefix`: path to image feature `.npy` files


1. Download datasets and corresponding images:
    ```bash
    pass
    ```

1. Generate ResNet image features:
    ```bash
    python scripts/feature.py [--split]
    ```

1. Generate BUTD image features:
    ```bash
    pass
    ```

1. VQA dataset is already in the desired `ques_file|ans_file` format. Generate json files for TextVQA:
    ```bash
    pass
    ```

1. Modify `DATA_PATHS` in `config.py` accordingly.

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

1. DiagNet without OCR on VQA v1.0:
    ```bash
    python train.py mfh glove --EMBED
    ```

1. DiagNet on TextVQA v0.5:
    ```bash
    python train.py mfh textvqa_butd --EMBED --OCR --BIN_HELP
    ```

## Prediction Visualization
1. Download image files and modify `image_prefix` of `DATA_PATHS` in `config.py` accordingly.

1. Run training and get the `.pth` model file in `training/checkpoint`. For example:
    ```bash
    python train.py mfh glove --EMBED
    ```

1. Specify the questions of interest by modifying `QTYPES` in `config.py`

1. Run visualization:
    ```bash
    python predict.py mfh glove --EMBED --RESUME_PATH path_to_model_file
    ```
![Figure 2: Visualization Example.](https://github.com/WYchelsy/vqa-mfb.pytorch/blob/docs/imgs/correct224477.png)
