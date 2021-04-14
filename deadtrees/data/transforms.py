import io
import logging

import numpy as np
import skimage
import torch

logger = logging.getLogger(__name__)

# DISABLED FOR NOW
# class Rescale():
#     """Rescale the image in a sample to a given size.

#     Args:
#         output_size (tuple or int): Desired output size. If tuple, output is
#             matched to output_size. If int, smaller of image edges is matched
#             to output_size keeping aspect ratio the same.
#     """

#     def __init__(self, output_size):
#         assert isinstance(output_size, (int, tuple))
#         self.output_size = output_size

#     def __call__(self, sample):
#         image, mask = sample['image'], sample['mask']

#         h, w = image.shape[:2]
#         if isinstance(self.output_size, int):
#             if h > w:
#                 new_h, new_w = self.output_size * h / w, self.output_size
#             else:
#                 new_h, new_w = self.output_size, self.output_size * w / h
#         else:
#             new_h, new_w = self.output_size

#         new_h, new_w = int(new_h), int(new_w)

#         image_scaled = skimage.transform.resize(image, (new_h, new_w))
#         mask_scaled = skimage.transform.resize(mask, (new_h, new_w))

#         sample['image'] = image_scaled
#         sample['mask'] = mask_scaled

#         return sample


class ToTensor:
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, mask = sample["image"], sample["mask"]
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # image = image.transpose((2, 0, 1))

        image = skimage.io.imread(io.BytesIO(image))
        sample["image"] = torch.from_numpy(image) / 255

        mask = skimage.io.imread(io.BytesIO(mask))
        sample["mask"] = torch.from_numpy(mask.astype("uint8"))
        return sample
