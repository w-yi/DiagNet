# Multi-modal Factorized Bilinear Pooling (MFB) for VQA

Original papers and repos

- [Multi-modal Factorized Bilinear Pooling with Co-Attention Learning for Visual Question Answering](http://openaccess.thecvf.com/content_iccv_2017/html/Yu_Multi-Modal_Factorized_Bilinear_ICCV_2017_paper.html)
- [Beyond Bilinear: Generalized Multi-modal Factorized High-order Pooling for Visual Question Answering](https://arxiv.org/abs/1708.03619)
- <https://github.com/asdf0982/vqa-mfb.pytorch>
- <https://github.com/yuzcccc/vqa-mfb>

## Requirements

python 3.7, pytorch 1.0

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

## Extract features

```bash
python feature.py [--split]
```

Helpful links:

- The caffe implementation uses <https://github.com/yuzcccc/bottom-up-attention>

## Training

```bash
python train.py [--options]
```
