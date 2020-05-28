from numpy.random import randint, seed
import numpy as np
import torch
import cv2
import os


class MaskGenerator():
    """
    Generate masks either randomly or by using a pre-defined mask files.

    Usages:
    1. Generate a mask image size 512x512
       >> mask_generator = MaskGenerator(512, 512)
       >> mask = mask_generator.sample()

    2. Convert tensor output to numpy:
       >> mask_arr = mask.numpy()
    """

    def __init__(self, height, width, rand_seed=None, filepath=None):
        """Convenience functions for generating masks to be used for inpainting training

        Arguments:
            height {int} -- Mask height
            width {width} -- Mask width

        Keyword Arguments:
            rand_seed {[type]} -- Random seed (default: {None})
            filepath {[type]} -- Load masks from filepath. If None, generate masks with OpenCV (default: {None})
        """

        self.height = height
        self.width = width
        self.filepath = filepath

        # If filepath supplied, load the list of masks within the directory
        self.mask_files = []
        if self.filepath:
            filenames = [f for f in os.listdir(self.filepath)]
            self.mask_files = [f for f in filenames
                               if any(filetype in f.lower() for filetype in ['.jpeg', '.png', '.jpg'])]
            print(">> Found {} masks in {}".format(len(self.mask_files), self.filepath))

            # Seed for reproducibility
        if rand_seed:
            seed(rand_seed)

    def _generate_mask(self, n_brush=20):
        """
        Generates a random irregular mask with lines, circles and elipses

        :param n_brush: Number of brushes
        :return: a torch tensor
        """

        img = np.zeros((self.height, self.width), np.uint8)

        # Set size scale
        size = int((self.width + self.height) * 0.03)
        if self.width < 64 or self.height < 64:
            raise Exception("Width and Height of mask must be at least 64!")

        # Draw random lines
        for _ in range(n_brush):
            x1, x2 = randint(1, self.width), randint(1, self.width)
            y1, y2 = randint(1, self.height), randint(1, self.height)
            thickness = randint(2, size)
            cv2.line(img, (x1, y1), (x2, y2), (1, 1, 1), thickness)

        # Draw random circles
        for _ in range(n_brush):
            x1, y1 = randint(1, self.width), randint(1, self.height)
            radius = randint(2, size)
            cv2.circle(img, (x1, y1), radius, (1, 1, 1), -1)

        # Draw random ellipses
        for _ in range(n_brush):
            x1, y1 = randint(1, self.width), randint(1, self.height)
            s1, s2 = randint(1, self.width), randint(1, self.height)
            a1, a2, a3 = randint(3, 180), randint(3, 180), randint(3, 180)
            thickness = randint(2, size)
            cv2.ellipse(img, (x1, y1), (s1, s2), a1, a2, a3, (1, 1, 1), thickness)

        # create tensor
        img_tensor = torch.from_numpy(np.moveaxis(1 - img, -1, 0))

        return img_tensor

    def _load_mask(self, rotation=True, dilation=True, cropping=True):
        """
        Loads a mask from disk, and optionally augments it

        :param rotation: Perform random rotation.
        :param dilation: Perform random dilation.
        :param cropping: Perform random cropping.
        :return: a Torch tensor
        """

        # Read image
        mask = cv2.imread(os.path.join(
            self.filepath, np.random.choice(self.mask_files, 1, replace=False)[0]))
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]

        # Random rotation
        if rotation:
            rand = np.random.randint(-180, 180)
            M = cv2.getRotationMatrix2D((mask.shape[1] / 2, mask.shape[0] / 2), rand, 1.5)
            mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]))

        # Random dilation
        if dilation:
            rand = np.random.randint(5, 47)
            kernel = np.ones((rand, rand), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)

        # Random cropping
        if cropping:
            x = np.random.randint(0, mask.shape[1] - self.width)
            y = np.random.randint(0, mask.shape[0] - self.height)
            mask = mask[y:y + self.height, x:x + self.width]

        # return as tensor
        img = np.moveaxis((mask > 1).astype(np.uint8), -1, 0)

        return torch.from_numpy(img)

    def sample(self, *args, **kwargs):
        """
        Retrieve a random mask. The argument depends on whether the mask is generated randomly or by
        loading predefined mask image files.

        :param args: see arguments in _generate_mask or _load_mask
        :param kwargs:  see arguments in _generate_mask or _load_mask
        :return: a Torch tensor
        """

        if self.filepath and len(self.mask_files) > 0:
            return self._load_mask(*args, **kwargs)
        else:
            return self._generate_mask(*args, **kwargs)

