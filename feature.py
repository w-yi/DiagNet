import torch
import torch.nn as nn
from torchvision import datasets, transform


class CocoDataset(Dataset):
    def __init__(self, split):
        assert split in ['train', 'val', 'test']
        path = "/data/home/wennyi/vqa-mfb.pytorch/data/VQA/Images/" + split + "2014"
        self.data = datasets.ImageFolder(
            root=path, transform=transforms.Compose([transforms.Resize(448, 448), transforms.ToTensor()])
        )

    def __len__(self):
        return self.data.__len__()

    def __getitem__(self, idx):
        return self.data[idx]


def get_features(split, m=None):
    model = torchvision.models.resnet152(pretrained=False)
    dataset = CocoDataset(split)
    data_loader = data.DataLoader(dataset, 256, shuffle=False, num_workers=4, pin_memory=False, drop_last=False)
    for i, inputs in enumerate(data_loader):
        print(inputs.shape)
        outputs = model(inputs)
        print(outputs.shape)
        if i == m:
            break
