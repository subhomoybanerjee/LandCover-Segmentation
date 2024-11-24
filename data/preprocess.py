import numpy as np
import rasterio
import torch
from .auggy import augment1

def read_image(img, BANDS):
    with rasterio.open(img) as src:
        image = src.read(BANDS)
    image = np.moveaxis(image, 0, -1)
    return image


def read_mask(mask_path):
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
    return mask

def preprocess_data(img_path, mask_path, patch, BANDS, auggy ,num_classes):

    # img = read_image(img_path,BANDS)
    mask = read_mask(mask_path)

    img = img_path.split('/')[-1]
    img = patch[img]

    if auggy==True:
        augmentations = augment1()
        augmented = augmentations(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']

    img = torch.tensor(img, dtype=torch.float32)
    mask = torch.nn.functional.one_hot(torch.tensor(mask, dtype=torch.long), num_classes=num_classes).float()

    img = img.permute(2, 0, 1)
    mask = mask.permute(2, 0, 1)

    return img, mask