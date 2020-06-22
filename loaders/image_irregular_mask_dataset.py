from torch.utils.data import Dataset
import glob
import os
from torchvision import transforms
from PIL import Image
import numpy as np
from utils import MaskGenerator


class ImageIrregularMaskDataset(Dataset):

    def __init__(self, im_size):
        super(ImageIrregularMaskDataset, self).__init__()
        self.__image_files = []
        self.__image_size = im_size
        self.__mask_gen = MaskGenerator(im_size, im_size)

    @property
    def image_files(self):
        return self.__image_files

    def __len__(self):
        return len(self.__image_files)

    def __getitem__(self, idx):
        # read both image and mask
        img = Image.open(self.__image_files[idx])

        # pad to get square size then resize to size
        max_size = np.array(img.size).max()
        pad_size = tuple((max_size - img.size) // 2)

        T = transforms.Compose([
            transforms.Pad(pad_size),
            transforms.Resize(self.__image_size),
            transforms.ToTensor()
        ])

        # convert to tensor
        img = T(img)

        # learning is faster if image is normalized to [-1.0, 1.0]
        img = (2.0 * transforms.Normalize([0.0] * 3, [1.0] * 3)(img)) - 1.0

        # generate irregular mask
        mask = self.__mask_gen.sample()

        return img, mask

    @classmethod
    def from_folder(cls, image_folder, image_pattern, image_size):
        """
        Object creation given image folder of the dataset.
        """

        # create object
        obj = cls(image_size)

        # read files
        obj.__image_files = glob.glob(os.path.join(image_folder, image_pattern))

        return obj