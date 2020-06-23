from abc import ABC, abstractmethod
from utils import dataset_splitter
from torch.utils.data import DataLoader
import os


class AbstractTrainer(ABC):

    def __init__(self, log_dir, device='cpu'):

        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)

        self.log_dir = log_dir
        self.device = device

        # setup these from set_loader
        self.batch_size = -1
        self.__train_loader = None
        self.__val_loader = None
        self.__test_loader = None

        # setup these with set_model
        self.optimizer = None
        self.loss = None
        self.__model = None

    @property
    def train_loader(self):
        return self.__train_loader

    @property
    def test_loader(self):
        return self.__test_loader

    @property
    def val_loader(self):
        return self.__val_loader

    def set_loader(self, dataset, batch_size=1, split=None, num_cpus=1):

        sub_ds = dataset_splitter(dataset, split)
        print(f"Total data size = {len(dataset):d}")

        self.batch_size = batch_size

        # create data loader
        self.__train_loader = DataLoader(
            sub_ds[0], batch_size=self.batch_size, shuffle=True, num_workers=num_cpus, pin_memory=True
        )
        print(f"Training samples are {len(sub_ds[0])} ({100.0*len(sub_ds[0])/len(dataset):.2f}%)")
        self.__val_loader = DataLoader(
            sub_ds[1], batch_size=self.batch_size, shuffle=True, num_workers=num_cpus, pin_memory=True
        )
        print(f"Validation samples are {len(sub_ds[1])} ({100.0*len(sub_ds[1])/len(dataset):.2f}%)")
        self.__test_loader = DataLoader(
            sub_ds[2], batch_size=self.batch_size, shuffle=False, num_workers=num_cpus, pin_memory=True
        )
        print(f"Test samples are {len(sub_ds[2])} ({100.0*len(sub_ds[2])/len(dataset):.2f}%)")

    def check_valid(self):
        assert self.batch_size > 0, "Batch size must be positive"
        assert all([i is not None for i in [self.__train_loader, self.__test_loader, self.__val_loader]]), \
            "Call 'set_loader' first."
        assert self.__model is not None, "Call 'set_model' first."

    @property
    def model(self):
        return self.__model

    @abstractmethod
    def set_model(self, *args, **kwargs):
        raise NotImplementedError

    def train(self, num_epoch):
        self.check_valid()
        assert num_epoch>0, f"Invalid number of EPOCH (num_epoch={num_epoch})"
