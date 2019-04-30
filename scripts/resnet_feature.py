import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import os
import io
from skimage import io
import numpy as np
import argparse


class CocoDataset(Dataset):
    def __init__(self, split, image_dir, feature_dir):
        # assert split in ['train', 'val', 'test']
        self.path = image_dir + split
        self.files = os.listdir(self.path)
        self.transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((448, 448)), transforms.ToTensor()])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        img_name = os.path.join(self.path, f)
        img = io.imread(img_name)
        if len(img.shape) == 1:
            img = img[0]
        if not len(img.shape) == 3:
            img = np.stack([img, img, img], -1)
        image = self.transform(img)
        return [image, f]


def get_features(split, image_dir, feature_dir, batch, gpu=True):
    model = models.resnet152(pretrained=True)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    modules = list(model.children())[:-1]
    model = nn.Sequential(*modules)
    if gpu:
        model = model.cuda()
    for split in ['test', 'train', 'val']:
        dataset = CocoDataset(split, image_dir, feature_dir)
        data_loader = DataLoader(dataset, batch, shuffle=False, num_workers=4, pin_memory=gpu, drop_last=False)
        for inputs, targets in data_loader:
            #pass
            if gpu:
                inputs = inputs.cuda()
            outputs = model(inputs).squeeze(-1).squeeze(-1)
            for x, f in zip(outputs, targets):
                np.save(feature_dir + split + '/' + f, x.cpu().data.numpy())
                del x
                del f
            del inputs
            del outputs
            del targets


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', '-s', choices=['train', 'val', 'test'], help='split')
    parser.add_argument('--image_dir', '-i', type=str, default='/data/home/wennyi/vqa-mfb.pytorch/data/textvqa/', help='input images dir')
    parser.add_argument('--feature_dir', '-f', type=str, default='/data/home/wennyi/vqa-mfb.pytorch/data/textvqa_features/baseline/', help='output features dir')
    args = parser.parse_args()
    get_features(args.split, args.image_dir, args.feature_dir, 8)
