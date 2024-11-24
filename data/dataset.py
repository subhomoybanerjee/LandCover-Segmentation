import torch
import torch.utils.data as data
import os

class SegmentationDataset(data.Dataset):
    def __init__(self, img_dir, mask_dir, BANDS, preprocess_fn, num_classes, patches,auggy):
        self.img_paths = sorted([os.path.join(img_dir, fname) for fname in os.listdir(img_dir)])
        self.mask_paths = sorted([os.path.join(mask_dir, fname) for fname in os.listdir(mask_dir)])
        self.BANDS = BANDS
        self.preprocess_fn = preprocess_fn
        self.num_classes = num_classes
        self.patches = patches
        self.auggy = auggy

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img, mask = self.preprocess_fn(self.img_paths[idx], self.mask_paths[idx],
                                       self.patches, self.BANDS, self.auggy, self.num_classes)
        return img, mask