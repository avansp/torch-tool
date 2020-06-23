from engines import AbstractTrainer
from loaders import ImageIrregularMaskDataset
from models import PartialUNet


class PartialUNetTrainer(AbstractTrainer):
    def __init__(self, log_dir, device):
        super(PartialUNetTrainer, self).__init__(log_dir, device)

    def set_input(self, image_folder, image_pattern, image_size, batch_size, split=None, num_cpus=1):
        ds = ImageIrregularMaskDataset.from_folder(image_folder, image_pattern, image_size)
        n = len(ds)
        assert n > 0, "Data loader returns empty set."
        print(f"PartialUNetTrainer: data loader is ready with {n} images.")
        self.set_loader(ds, split=split, num_cpus=num_cpus, batch_size=batch_size)

    def set_model(self, in_channel, out_channel):
        self.__model = PartialUNet(in_channel, out_channel).to(self.device)
        print(f"PartialUNetTrainer: PartialUNet model is ready.")

    def train(self, num_epoch):
        super(PartialUNetTrainer, self).train(num_epoch)
