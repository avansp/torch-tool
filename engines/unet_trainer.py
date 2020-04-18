from .base_trainer import AbstractTrainer
from models import VanillaUNet
import torch
from tqdm import tqdm

from torch.utils.tensorboard.writer import SummaryWriter
import datetime as dt
import os


class VanillaUNetTrainer(AbstractTrainer):
    """
    Train image segmentation network.
    Training:
        Inputs are tuples of (image, mask)
        Outputs are masks.
    """

    def __init__(self, in_channel, out_channel,
                 **kwargs):
        super(VanillaUNetTrainer, self).__init__(**kwargs)
        self.set_model(in_channel, out_channel)

        # this is hardcoded in Vanilla U-Net
        self.in_img_shape = (572, 572)
        self.out_img_shape = (388, 388)
        self.pad_size = [(572 - 388) // 2] * 4

    def set_model(self, in_channel, out_channel,
                  hyper_params=None):

        self.model = VanillaUNet(in_channel, out_channel).to(self.device)

        if hyper_params is None:
            hyper_params = {
                'lr': 0.1,
                'weight_decay': 1e-8,
                'momentum': 0.9
            }

        # optimizers & criterion
        self.optimizer = torch.optim.RMSprop(self.model.parameters(),
                                             lr=hyper_params['lr'],
                                             weight_decay=hyper_params['weight_decay'],
                                             momentum=hyper_params['momentum'])
        self.loss = torch.nn.BCEWithLogitsLoss()

    def train(self, num_epoch):

        self.check_valid()

        # create writer as a new folder
        now_str = str(dt.datetime.now())
        self.log_dir = os.path.join(self.log_dir, now_str.replace(':', '').replace(' ', '_'))
        assert not os.path.isdir(self.log_dir), f"Something is wrong. {self.log_dir} exists."
        os.makedirs(self.log_dir)
        print(f"[{now_str}]: Starting the experiments.\nLog directory = {self.log_dir}")
        log_writer = SummaryWriter(self.log_dir)
        global_step = 0

        for epoch in range(num_epoch):

            # set the model in training mode
            self.model.train()

            # use tqdm as progress bar
            with tqdm(total=len(self.train_loader.dataset)) as pbar:

                last_loss = 0.00
                for step, (img, mask_true) in enumerate(self.train_loader):

                    # set the device
                    img = img.to(self.device)
                    mask_true = mask_true.to(self.device)

                    # pad the image to match the input size
                    img = torch.nn.functional.pad(img, self.pad_size)

                    # predict
                    mask_pred = self.model(img)

                    # calculate the loss
                    loss = self.loss(mask_pred, mask_true)
                    loss_diff = loss.item() - last_loss
                    last_loss = loss.item()

                    # The three pytorch key sentences:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    # --------------------------------

                    # update the progress bar
                    pbar.set_description(f"Epoch {epoch}")
                    pbar.set_postfix(
                        loss=f"{loss.item():.2f} ({loss_diff:.1f})"
                    )
                    pbar.update(n=img.shape[0])

                    # log
                    global_step += 1
                    log_writer.add_scalars(
                        'Loss/train',
                        {'loss': loss.item(), 'loss_diff': loss_diff },
                        global_step
                    )

        # End of the experiment
        log_writer.close()
        print("FINISHED")
