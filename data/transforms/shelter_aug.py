import torch
import numpy as np
from numpy.random import uniform, randint

class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
        # 在训练时随机把图片的一部分减掉，一点程度的遮挡模拟
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h, w = img.size(1), img.size(2)

        mask = np.ones((h, w), np.float32)

        for _ in range(self.n_holes):

            y = randint(h)
            x = randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


class RandomErasing(object):
    """
    Args:
        probability: The probability that the operation will be performed.
        sl: min erasing area
        sh: max erasing area
        rl: min aspect ratio
        mean: erasing value
    """
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, rl=0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.rl = rl

    def __call__(self, img):
        if np.random.uniform(0, 1) > self.probability:
            return img

        for _ in range(100):

            area = img.size()[1] * img.size()[2]

            target_area = uniform(self.sl, self.sh) * area
            aspect_ratio = uniform(self.rl, 1/self.rl)

            h = int(round(np.sqrt(target_area * aspect_ratio)))
            w = int(round(np.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:

                x1 = randint(0, img.size()[1] - h)
                y1 = randint(0, img.size()[2] - w)

                if img.size()[0] == 3:

                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img[2, x1:x1+h, y1:y1+w] = self.mean[2]

                else:

                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]

                return img

        return img