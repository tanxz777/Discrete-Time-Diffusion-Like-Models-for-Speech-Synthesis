""" from https://github.com/jaywalnut310/glow-tts """

import torch
import numpy as np

def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(int(max_length), dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def fix_len_compatibility(length, num_downsamplings_in_unet=2):
    while True:
        if length % (2**num_downsamplings_in_unet) == 0:
            return length
        length += 1


def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape


def generate_path(duration, mask):
    device = duration.device

    b, t_x, t_y = mask.shape
    cum_duration = torch.cumsum(duration, 1)
    path = torch.zeros(b, t_x, t_y, dtype=mask.dtype).to(device=device)

    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)
    path = path - torch.nn.functional.pad(path, convert_pad_shape([[0, 0], 
                                          [1, 0], [0, 0]]))[:, :-1]
    path = path * mask
    return path


def duration_loss(logw, logw_, lengths):
    loss = torch.sum((logw - logw_)**2) / torch.sum(lengths)
    return loss


class Cutout(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, X0_shape):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        c = X0_shape[0]
        h = X0_shape[1]
        w = X0_shape[2]
        mask = np.ones((c, h, w), np.float32)

        for i in range(c):
            for n in range(self.n_holes):
                y = np.random.randint(h)
                x = np.random.randint(w)

                y1 = np.clip(y - self.length // 2, 0, h)
                y2 = np.clip(y + self.length // 2, 0, h)
                x1 = np.clip(x - self.length // 2, 0, w)
                x2 = np.clip(x + self.length // 2, 0, w)

                mask[i, y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask).to('cuda')

        return mask


def cutout_along_dimension(tensor, l=1, cutout_percentage=0.5):
    batch_size, dim1, t = tensor.size()

    if l == 1:
        # Cutout along the first dimension
        cutout_size = int(dim1 * cutout_percentage)
        mask = torch.ones_like(tensor)
        for i in range(batch_size):
            cutout_indices = torch.randperm(dim1)[:cutout_size]
            mask[i, cutout_indices, :] = 0
        tensor = tensor * mask

    elif l == 2:
        # Cutout along the second dimension
        cutout_size = int(t * cutout_percentage)
        mask = torch.ones_like(tensor)
        for i in range(batch_size):
            cutout_indices = torch.randperm(t)[:cutout_size]
            mask[i, :, cutout_indices] = 0
        tensor = tensor * mask

    else:
        raise ValueError("Parameter l must be 1 or 2.")

    return tensor
