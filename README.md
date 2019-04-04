# Multi-modal Factorized Bilinear Pooling (MFB) for VQA

# Important TODOs
Add MAX_TOKEN_SIZE=104 into OPT


Papers related to our implementation

- CoAtt: [Hierarchical Question-Image Co-Attention for Visual Question Answering](https://arxiv.org/abs/1606.00061)
- MFB: [Multi-modal Factorized Bilinear Pooling with Co-Attention Learning for Visual Question Answering](http://openaccess.thecvf.com/content_iccv_2017/html/Yu_Multi-Modal_Factorized_Bilinear_ICCV_2017_paper.html)
- MFH: [Beyond Bilinear: Generalized Multi-modal Factorized High-order Pooling for Visual Question Answering](https://arxiv.org/abs/1708.03619)
- BUTD: [Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering](https://arxiv.org/abs/1707.07998)
- [Pythia v0.1: the Winning Entry to the VQA Challenge 2018](https://arxiv.org/pdf/1807.09956.pdf)
- TextVQA: [Towards VQA Models That Can Read](https://textvqa.org/assets/paper/TextVQA.pdf)

Baseline implementations

- PyTorch implementation: <https://github.com/asdf0982/vqa-mfb.pytorch>
- Caffe implementation: <https://github.com/yuzcccc/vqa-mfb>
- pythia: <https://github.com/facebookresearch/pythia>
- BUTD: <https://github.com/peteanderson80/bottom-up-attention>
- Fork of BUTD: <https://github.com/yuzcccc/bottom-up-attention>

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

## Training

```bash
python train.py [--options]
```
