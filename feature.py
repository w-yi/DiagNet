import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import os
import io
from skimage import io


class CocoDataset(Dataset):
    def __init__(self, split):
        assert split in ['train', 'val', 'test']
        self.path = "/data/home/wennyi/vqa-mfb.pytorch/data/VQA/Images/" + split + "2014"
        self.files = os.listdir(self.path)
        self.transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor()])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        img_name = os.path.join(self.path, f)
        image = self.transform(io.imread(img_name))
        # print(self.files[idx])
        # print(f)
        return [image, int(f[-16:-4])]


def get_features(split):
    model = models.resnet152(pretrained=False)
    dataset = CocoDataset(split)
    data_loader = DataLoader(dataset, 4, shuffle=False, num_workers=4, pin_memory=False, drop_last=False)
    for inputs, targets in data_loader:
        print(inputs.shape)
        outputs = model(inputs)
        print(outputs.shape, targets)
        break
