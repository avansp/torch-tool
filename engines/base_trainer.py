from abc import ABC, abstractmethod
from utils import dataset_splitter
from torch.utils.data import DataLoader
import os


class AbstractTrainer(ABC):

    def __init__(self, log_dir, device='cpu'):

        assert not os.path.isdir(log_dir), f"Directory {log_dir} already exists."
        os.makedirs(log_dir)

        self.log_dir = log_dir
        self.device = device

        # setup these from set_loader
        self.batch_size = -1
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        # setup these with set_model
        self.model = None
        self.optimizer = None
        self.loss = None

    def set_loader(self, dataset, split=None, num_cpus=1, batch_size=1):

        sub_ds = dataset_splitter(dataset, split)
        print(f"Total data size = {len(dataset):d}")

        self.batch_size = batch_size

        # create data loader
        self.train_loader = DataLoader(
            sub_ds[0], batch_size=self.batch_size, shuffle=True, num_workers=num_cpus, pin_memory=True
        )
        print(f"Training samples are {len(sub_ds[0])} ({100.0*len(sub_ds[0])/len(dataset):.2f}%)")
        self.val_loader = DataLoader(
            sub_ds[1], batch_size=self.batch_size, shuffle=True, num_workers=num_cpus, pin_memory=True
        )
        print(f"Validation samples are {len(sub_ds[1])} ({100.0*len(sub_ds[1])/len(dataset):.2f}%)")
        self.test_loader = DataLoader(
            sub_ds[2], batch_size=self.batch_size, shuffle=False, num_workers=num_cpus, pin_memory=True
        )
        print(f"Test samples are {len(sub_ds[2])} ({100.0*len(sub_ds[2])/len(dataset):.2f}%)")

    def check_valid(self):
        assert self.batch_size > 0, "Batch size must be positive"
        assert all([i is not None for i in [self.train_loader, self.test_loader, self.val_loader]]), \
            "Call 'set_loader' first."
        assert self.model is not None, "Call 'set_model' first."

    @abstractmethod
    def set_model(self, model, hyper_params=None):
        pass

    @abstractmethod
    def train(self, num_epoch):
        pass
