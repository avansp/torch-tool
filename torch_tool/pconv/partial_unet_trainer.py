# from trainers import BaseTrainer
# from src import ImageIrregularMaskDataset
# from . import PartialUNet
#
# __all__ = ["PartialUNetTrainer"]
#
# class PartialUNetTrainer(BaseTrainer):
#     def __init__(self, log_dir, device):
#         super(PartialUNetTrainer, self).__init__(log_dir, device)
#
#     def set_input(self, image_folder, image_pattern, image_size, batch_size, split=None, num_cpus=1):
#         ds = ImageIrregularMaskDataset.from_folder(image_folder, image_pattern, image_size)
#         n = len(ds)
#         assert n > 0, "Data loader returns empty set."
#         print(f"PartialUNetTrainer: data loader is ready with {n} images.")
#         self.set_loader(ds, split=split, num_cpus=num_cpus, batch_size=batch_size)
#
#     def initialize(self, in_channel, out_channel):
#         self.model = PartialUNet(in_channel, out_channel).to(self.device)
#         print(f"PartialUNetTrainer: PartialUNet model is ready.")
#
#     def train(self, num_epoch):
#         super(PartialUNetTrainer, self).train(num_epoch)
