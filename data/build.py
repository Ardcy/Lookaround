import torchvision
import torchvision.transforms as transforms
import torch
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from torchvision.datasets.vision import VisionDataset
import os.path
import pickle
from typing import Any, Callable, Optional, Tuple
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2
from IPython import embed
import torchvision
from torchvision.datasets import ImageFolder

class CIFAR10(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    base_folder = "cifar-10-batches-py"
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = "c58f30108f718f92721af3b95e74349a"
    train_list = [
        ["data_batch_1", "c99cafc152244af753f735de768cd75f"],
        ["data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
        ["data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
        ["data_batch_4", "634d18415352ddfa80567beed471001a"],
        ["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"],
    ]

    test_list = [
        ["test_batch", "40351d587109b95175f43aff81a1287e"],
    ]
    meta = {
        "filename": "batches.meta",
        "key": "label_names",
        "md5": "5ff9c542aee3614f3951f8cda6e48888",
    }

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        img_size=32
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)

        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. You can use download=True to download it")

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                if "labels" in entry:
                    self.targets.extend(entry["labels"])
                else:
                    self.targets.extend(entry["fine_labels"])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        data = []
        for i in tqdm(range(self.data.shape[0])):
            data.append(cv2.resize(self.data[i], [img_size, img_size]))

        data = np.vstack(data).reshape(-1, img_size, img_size, 3)
        del self.data
        self.data = []
        for i in tqdm(range(data.shape[0])):
            self.data.append(Image.fromarray(data[i]))

        self._load_meta()

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta["filename"])
        if not check_integrity(path, self.meta["md5"]):
            raise RuntimeError(
                "Dataset metadata file not found or corrupted. You can use download=True to download it")
        with open(path, "rb") as infile:
            data = pickle.load(infile, encoding="latin1")
            self.classes = data[self.meta["key"]]
        self.class_to_idx = {_class: i for i,
                             _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        root = self.root
        for fentry in self.train_list + self.test_list:
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        download_and_extract_archive(
            self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"


class CIFAR100(CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """

    base_folder = "cifar-100-python"
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
    train_list = [
        ["train", "16019d7e3df5f24257cddd939b257f8d"],
    ]

    test_list = [
        ["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"],
    ]
    meta = {
        "filename": "meta",
        "key": "fine_label_names",
        "md5": "7973b15100ade9c7d40fb424638fde48",
    }


def build_transforms(cfg, method):
    method = method

    if cfg.DATASET.NAME == 'cifar100':
        mean, std = cfg.CIFAR100_TRAIN_MEAN, cfg.CIFAR100_TRAIN_STD
    if cfg.DATASET.NAME == 'cifar10':
        mean, std = cfg.CIFAR10_TRAIN_MEAN, cfg.CIFAR10_TRAIN_STD

    imagenet_normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])

    if cfg.DATASET.NAME == 'imagenet':
        if method == 'test':
            return transforms.Compose([
                transforms.Resize(cfg.DATASET.IMAGESIZE+32),
                transforms.CenterCrop(cfg.DATASET.IMAGESIZE),
                transforms.ToTensor(),
                imagenet_normalize,
            ])
        else:
            return transforms.Compose([
                transforms.RandomResizedCrop(cfg.DATASET.IMAGESIZE),
                eval(f"transforms.{method}()"),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                imagenet_normalize,
            ])

    if cfg.DATASET.NAME == 'cifar100':
        if method == 'RandomHorizontalFlip':
            return transforms.Compose([
                transforms.RandomCrop(cfg.DATASET.IMAGESIZE, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        if method == 'RandomVerticalFlip':
            return transforms.Compose([
                transforms.RandomCrop(cfg.DATASET.IMAGESIZE, padding=4),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

        if method == 'RandAugment':
            return transforms.Compose([
                transforms.RandomCrop(cfg.DATASET.IMAGESIZE, padding=4),
                transforms.RandAugment(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        if method == 'AutoAugment':
            return transforms.Compose([
                transforms.RandomCrop(cfg.DATASET.IMAGESIZE, padding=4),
                transforms.AutoAugment(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

        if method == 'test':
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        return transforms.Compose([
            transforms.RandomCrop(cfg.DATASET.IMAGESIZE, padding=4),
            eval(f"transforms.{method}()"),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    if cfg.DATASET.NAME == 'cifar10':
        if method == 'RandomHorizontalFlip':
            return transforms.Compose([
                transforms.RandomCrop(cfg.DATASET.IMAGESIZE, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        if method == 'RandomVerticalFlip':
            return transforms.Compose([
                transforms.RandomCrop(cfg.DATASET.IMAGESIZE, padding=4),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

        if method == 'RandAugment':
            return transforms.Compose([
                transforms.RandomCrop(cfg.DATASET.IMAGESIZE, padding=4),
                transforms.RandAugment(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

        if method == 'test':
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        return transforms.Compose([
            transforms.RandomCrop(cfg.DATASET.IMAGESIZE, padding=4),
            eval(f"transforms.{method}()"),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


def build_dataloader(cfg, transform, data_dir, is_train, batchsize=100):

    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])

    if cfg.DATASET.NAME == 'cifar10':
        dataset = CIFAR10(
            root=data_dir, train=is_train, download=False, transform=transform, img_size=cfg.DATASET.IMAGESIZE)
    if cfg.DATASET.NAME == 'cifar100':
        dataset = CIFAR100(
            root=data_dir, train=is_train, download=False, transform=transform, img_size=cfg.DATASET.IMAGESIZE)
    if cfg.DATASET.NAME == 'imagenet':
        if is_train:
            dataset = ImageFolder(
                os.path.join(data_dir, 'train'),
                transform
            )
        else:
            dataset = ImageFolder(
                os.path.join(data_dir, 'val'),
                transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]))
    if is_train == True:
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=True, num_workers=32, drop_last=True, pin_memory=True)
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batchsize, shuffle=False, num_workers=32, drop_last=True, pin_memory=True)
    return dataloader