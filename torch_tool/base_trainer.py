from torch.utils.data import DataLoader
import os
import torch_tool as tt
import torch
from tqdm import tqdm

__all__ = ["BaseTrainer"]


class BaseTrainer():

    def __init__(self, log_dir, model_name='MODEL', device='cpu'):

        self.root_dir = log_dir
        self.log_dir = os.path.join(self.root_dir, model_name)

        if not os.path.isdir(self.log_dir):
            os.makedirs(self.log_dir)

        self.device = device
        self.model_name = model_name
        self.__exp_name = ''

        # setup these from set_loader
        self.__train_loader = None
        self.__val_loader = None
        self.__test_loader = None

        # must setup this before training
        self.optimizer = None
        self.loss = None
        self.model = None

        # fill in during run
        self.train_losses = []
        self.test_losses = []

    @property
    def train_loader(self):
        return self.__train_loader

    @property
    def test_loader(self):
        return self.__test_loader

    @property
    def val_loader(self):
        return self.__val_loader

    @property
    def batch_size(self):
        return -1 if self.__train_loader is None else self.__train_loader.batch_size

    @property
    def train_size(self):
        return 0 if self.__train_loader is None else len(self.__train_loader)

    @property
    def val_size(self):
        return 0 if self.__val_loader is None else len(self.__val_loader)

    @property
    def test_size(self):
        return 0 if self.__test_loader is None else len(self.__test_loader)

    @property
    def train_data_size(self):
        return 0 if self.__train_loader is None else len(self.__train_loader.dataset)

    @property
    def val_data_size(self):
        return 0 if self.__val_loader is None else len(self.__val_loader.dataset)

    @property
    def test_data_size(self):
        return 0 if self.__test_loader is None else len(self.__test_loader.dataset)

    @property
    def data_size(self):
        return self.train_size + self.test_size + self.val_size

    def set_loader(self, dataset, batch_size=1, split=None, num_cpus=1):

        sub_ds = tt.dataset_splitter(dataset, split)
        print(f"Total data size = {len(dataset):d}")

        # create data loader
        self.__train_loader = DataLoader(
            sub_ds[0], batch_size=batch_size, shuffle=True, num_workers=num_cpus, pin_memory=True
        )
        print(f"Training samples are {self.train_data_size} ({100.0*self.train_data_size/len(dataset):.2f}%)")
        self.__val_loader = DataLoader(
            sub_ds[1], batch_size=batch_size, shuffle=True, num_workers=num_cpus, pin_memory=True
        )
        print(f"Validation samples are {self.val_data_size} ({100.0*self.val_data_size/len(dataset):.2f}%)")
        self.__test_loader = DataLoader(
            sub_ds[2], batch_size=batch_size, shuffle=False, num_workers=num_cpus, pin_memory=True
        )
        print(f"Test samples are {self.test_data_size} ({100.0*self.test_data_size/len(dataset):.2f}%)")

    def save_loaders(self, tag=""):
        assert self.__train_loader is not None, "Loaders must be set first"

        fname = os.path.join(self.root_dir, f"{tag}train_loader.pth")
        torch.save(self.train_loader, fname)
        print(f"Saved train loader in {fname}")

        fname = os.path.join(self.root_dir, f"{tag}val_loader.pth")
        torch.save(self.val_loader, fname)
        print(f"Saved validation loader in {fname}")
        
        fname = os.path.join(self.root_dir, f"{tag}test_loader.pth")
        torch.save(self.test_loader, fname)
        print(f"Saved test loader in {fname}")

    def load_loaders(self, tag=""):
        assert os.path.isdir(self.root_dir), f"Log directory {self.root_dir} does not exist. Cannot load loaders."

        self.__train_loader = torch.load(os.path.join(self.root_dir, f"{tag}train_loader.pth"))
        self.__val_loader = torch.load(os.path.join(self.root_dir, f"{tag}val_loader.pth"))
        self.__test_loader = torch.load(os.path.join(self.root_dir, f"{tag}test_loader.pth"))

    def check_valid(self):
        assert self.batch_size > 0, "Batch size must be positive"
        assert all([i is not None for i in [self.__train_loader, self.__test_loader, self.__val_loader]]), \
            "Call 'set_loader' first."
        assert self.model is not None, "Model is undefined."
        assert self.optimizer is not None, "Optimizer has not been set."
        assert self.loss is not None, "Loss function is undefined."

    def run(self, num_epochs):
        """
        Run the whole test & training loop for num_epochs of EPOCH's
        """

        self.check_valid()
        assert num_epochs > 0, f"Invalid number of EPOCH (num_epoch={num_epochs})"

        # allocate training loss matrix
        self.train_losses = torch.zeros((self.train_size, num_epochs), dtype=torch.float)

        self.test()
        for epoch in range(1, num_epochs + 1):
            self.train(epoch)
            self.test()

    def test(self):
        """
        Perform testing.
        """

        # set eval mode ON
        self.model.eval()

        test_loss = 0.0
        correct = 0

        with torch.no_grad():
            for img_in, label in self.test_loader:
                output = self.model(img_in)
                test_loss += self.loss(output, label).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(label.data.view_as(pred)).sum()

        test_loss /= self.test_size
        self.test_losses.append(test_loss)

        print(f"Current test: avg. loss={test_loss:.4f}, acc={correct}/{self.test_data_size} ({100. * correct / self.test_data_size:.2f}%)", flush=True)

    def train(self, epoch):
        """
        Perform only 1 epoch training. No model / loader validity check.
        """

        # Mode TRAINING is set
        self.model.train()

        # use tqdm as progress bar
        with tqdm(total=self.train_data_size) as pbar:

            for batch_idx, (data_in, target) in enumerate(self.train_loader):

                # The 5-statements of torch' training:
                # ------------------------------------
                self.optimizer.zero_grad()
                data_out = self.model(data_in)
                loss = self.loss(data_out, target)
                loss.backward()
                self.optimizer.step()
                # ------------------------------------

                self.train_losses[batch_idx, epoch-1] = loss.item()

                # update the progress bar
                pbar.set_description(f"Epoch {epoch}")
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}"
                )
                pbar.update(n=data_in.shape[0])

            pbar.close()

        # save the model
        self.save_checkpoint(epoch)

    def weight_summary(self):
        if self.model is None:
            return

        print(f"Learned parameters in {self.model_name}:")
        D = self.model.state_dict()
        for param_tensor in D:
            print(f"{param_tensor}:\t{D[param_tensor].size()}")

    def save_checkpoint(self, epoch, **kwargs):
        torch.save(dict({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optim_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'test_losses': self.test_losses
        }, **kwargs), os.path.join(self.log_dir, f"{self.model_name}_EPOCH_{epoch:03d}.pth"))
