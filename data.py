import os
import csv
import math
import torch
import torch.nn.functional as F
from torch.utils.data import dataset
from torchvision.datasets.folder import default_loader
from utils import list_pictures
from torchvision import transforms
from torch.utils.data import dataloader


class Data:
    def __init__(self, args):
        self.args = args
        transform_list = [
            transforms.Resize((args.height, args.width), interpolation=3),
            transforms.RandomChoice(
                [transforms.RandomHorizontalFlip(),
                 transforms.RandomGrayscale(),
                 transforms.RandomRotation(20)
                 ]
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        transform = transforms.Compose(transform_list)
        self.train_dataset = Dataset(args, transform)
        self.train_loader = dataloader.DataLoader(self.train_dataset,
                                                  shuffle=True,
                                                  batch_size=args.train_batch_size,
                                                  num_workers=args.nThread
												 )


class Dataset(dataset.Dataset):
    def __init__(self, args, transform):
        self.root = args.train_img
        self.transform = transform
        self.labels = [label[0:-1] for label in csv.reader(open(args.train_label, 'r'))]
        self.loader = default_loader

    def __getitem__(self, index):
        name, age = self.labels[index]
        img = self.loader(os.path.join(self.root, name))
        age = int(age)
        label = [normal_sampling(age, i) for i in range(101)]
        label = torch.Tensor(label)
        label = F.normalize(label, p=1, dim=0)

        if self.transform is not None:
            img = self.transform(img)
        return img, label, age

    def __len__(self):
        return len(self.labels)


def normal_sampling(mean, label_k, std=2):
    return math.exp(-(label_k-mean)**2/(2*std**2))/(math.sqrt(2*math.pi)*std)
