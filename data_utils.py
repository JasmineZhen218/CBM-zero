import os
from torchvision import datasets, transforms
from torch.utils import data

class ActivationDataset(data.Dataset):
    def __init__(self, activations):
        self.activations = activations
    def __len__(self):
        return len(self.activations)
    def __getitem__(self, index):
        return self.activations[index]
    
def get_preprocess(dataset_name):
    if "cifar10" in dataset_name or "cifar100" in dataset_name:
        preprocess = transforms.Compose(
            [
                transforms.Resize(32),
                transforms.CenterCrop(32),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )
    else:
        preprocess = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )
    return preprocess

# note
# Put the data in the right place
# For CIFAR-10 and CIFAR-100, the data will be downloaded automatically.
# For ImageNet, download the data from http://www.image-net.org/ and put it in /data/imagenet and place the train and val folders in it.

def get_data(dataset_name, preprocess=None):
    if dataset_name == "cifar10_train":
        data = datasets.CIFAR10(
            root=os.path.expanduser("~/.cache"),
            download=True,
            train=True,
            transform=preprocess,
        )
    elif dataset_name == "cifar10_val":
        data = datasets.CIFAR10(
            root=os.path.expanduser("~/.cache"),
            download=True,
            train=False,
            transform=preprocess,
        )
    elif dataset_name == "cifar100_train":
        data = datasets.CIFAR100(
            root=os.path.expanduser("~/.cache"),
            download=True,
            train=True,
            transform=preprocess,
        )
    elif dataset_name == "cifar100_val":
        data = datasets.CIFAR100(
            root=os.path.expanduser("~/.cache"),
            download=True,
            train=False,
            transform=preprocess,
        )
    elif dataset_name == "imagenet_train":
        data = datasets.ImageNet(
            root="/data/",
            split="train",
            transform=preprocess,
        )
    elif dataset_name == "imagenet_val":
        data = datasets.ImageNet(
            root="/data/",
            split="val",
            transform=preprocess,
        )
    elif dataset_name == "food101_train":
        data = datasets.Food101(
            root="/data/food101",
            split="train",
            transform=preprocess,
            download=True,
        )
    elif dataset_name == "food101_val":
        data = datasets.Food101(
            root="/data/food101",
            split="test",
            transform=preprocess,
            download=True,
        )
    elif dataset_name == "cub_train":
        data = datasets.ImageFolder("data/CUB/train", preprocess)
    elif dataset_name == "cub_val":
        data = datasets.ImageFolder("data/CUB/test", preprocess)
    elif dataset_name == "awa2_train":
        data = datasets.ImageFolder(
            "data/Animals_with_Attributes2/train", preprocess
        )
    elif dataset_name == "awa2_val":
        data = datasets.ImageFolder(
            "data/Animals_with_Attributes2/val", preprocess
        )
    else:
        raise ValueError("Unknown dataset: {}".format(dataset_name))
    return data

def load_labels(data_name):
    data_train = get_data(
        "{}_train".format(data_name),
    )
    data_val = get_data(
        "{}_val".format(data_name),
    )
    if data_name in ["food101"]:
        labels_train = data_train._labels
        labels_val = data_val._labels
    else:
        labels_train = data_train.targets
        labels_val = data_val.targets
    return labels_train, labels_val