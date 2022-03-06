import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, Dataset, ConcatDataset
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor, Normalize, Pad, RandomCrop, RandomHorizontalFlip, RandomErasing
from RandAugment import RandAugment


np.random.seed(2)

class AddTransform(Dataset):
    def __init__(self, dataset, transform):
        self.data = dataset
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x, y = self.data[index]
        x = self.transform(x)
        return x, y

def cifar10_unsupervised_dataloaders():
    print('Data Preparation')
    train_transform = Compose([
        Pad(4),
        RandomCrop(32, fill=128),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        RandomErasing(scale=(0.1, 0.33)),
    ])

    unsupervised_train_transformation = Compose([
        Pad(4),
        RandomCrop(32, fill=128),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # RANDAUGMENT
    unsupervised_train_transformation.transforms.insert(0, RandAugment(3, 9))

    test_transform = Compose([
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # Train dataset with and without labels
    cifar10_train_ds = datasets.ImageFolder(root='../knee 01-34/train')
    train_labelled_ds = cifar10_train_ds
    train_labelled_ds_t = AddTransform(train_labelled_ds, train_transform)
    cifar10_unlabelled_ds = datasets.ImageFolder(root='../archive/data')
    train_unlabelled_ds = ConcatDataset([cifar10_unlabelled_ds,cifar10_train_ds])

    # apply transformation for both
    train_unlabelled_ds_t = AddTransform(train_unlabelled_ds, train_transform)
    train_unlabelled_aug_ds_t = AddTransform(train_unlabelled_ds, unsupervised_train_transformation)

    print('Labelled dataset -- Num_samples: {0}, classes: {1}, \n Unsupervised dataset -- Num_samples {2}, Augmentation -- Num_samples: {3}'
          .format(len(train_labelled_ds_t), 10, len(train_unlabelled_ds_t), len(train_unlabelled_aug_ds_t)))

    # Data loader for labeled and unlabeled train dataset
    train_labelled = DataLoader(
        train_labelled_ds_t,
        batch_size=64,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    train_unlabelled = DataLoader(
        train_unlabelled_ds_t,
        batch_size=64,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    train_unlabelled_aug = DataLoader(
       train_unlabelled_aug_ds_t,
        batch_size=64,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    # Data loader for test dataset
    cifar10_test_ds = datasets.ImageFolder(root='../knee 01-34/test',transform=test_transform)


    print('Test set -- Num_samples: {0}'.format(len(cifar10_test_ds)))

    test = DataLoader(
        cifar10_test_ds, batch_size=64,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    return train_labelled, train_unlabelled, train_unlabelled_aug, test

# train_labelled <- original severity datset (train)
# test <- original everity dataset (test)
# train_Unlabelled <-  new binary dataset

def cifar10_supervised_dataloaders(limit = 0):

    if(limit > 0):
        picks = np.random.permutation(limit)

    train_ds = datasets.ImageFolder(root='knee 01-34/train',
                     transform=Compose([
                         RandomHorizontalFlip(),
                         RandomCrop(32, 4),
                         ToTensor(),
                         Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225]),
                     ]), download=False)

    if(limit > 0):
        train_ds = Subset(train_ds, picks)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=64,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

    print('Loading dataset {0} for training -- Num_samples: {1}, num_classes: {2}'.format(datasets.CIFAR10.__name__,len(train_loader.dataset),10))


    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(root='knee 01-34/test',  transform=Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]), download=True),
        batch_size=64,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    print('Loading dataset {0} for validating -- Num_samples: {1}, num_classes: {2}'.format(datasets.CIFAR10.__name__,len(val_loader.dataset), 10))

    return train_loader, val_loader
