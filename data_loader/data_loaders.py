import torch
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# from base import BaseDataLoader
# not using BaseDataLoader for more flexibility on data loading

class MnistDataLoader():
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, logger, data_dir, batch_size, seed=0, shuffle=True, validation_split=0.1, test_split=0.2, num_workers=1, training=True):
        self.trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.logger = logger
        self.data_dir = data_dir
        self.validation_split = validation_split
        self.test_split = test_split
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.training = training
        
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=self.trsfm)

    def get_data_loader(self):
        self.logger.info('Splitting dataset into train, validation, and test sets using seed {}'.format(self.seed))
        generator = torch.Generator().manual_seed(self.seed)

        if isinstance(self.validation_split, int) or isinstance(self.test_split, int):
            assert self.validation_split > 0 or self.test_split > 0
            assert self.validation_split < len(self.dataset) or self.test_split < len(self.dataset), \
                "validation set size or test set size is configured to be larger than entire dataset."
            train_split = 1 - self.validation_split - self.test_split
            train_dataset, valid_dataset, test_dataset = random_split(self.dataset, [train_split, self.validation_split, self.test_split], generator=generator)
    
        else:
            len_valid = int(len(self.dataset) * self.validation_split)
            len_test  = int(len(self.dataset) * self.test_split)
            len_train = len(self.dataset) - len_valid - len_test
            train_dataset, valid_dataset, test_dataset = random_split(self.dataset, [len_train, len_valid, len_test], generator=generator)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)
        valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)
        test_loader  = DataLoader(test_dataset,  batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)

        return train_loader, valid_loader, test_loader
    
    def get_dataset(self, data_dir):
        trainset = datasets.MNIST(data_dir, train=True, download=False, transform=self.trsfm)
        testset  = datasets.MNIST(data_dir, train=False,  download=False, transform=self.trsfm)
        return trainset, testset
