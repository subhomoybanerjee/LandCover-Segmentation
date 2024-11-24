import torch
from torch.utils.data import DataLoader
from .dataset import SegmentationDataset

def create_dataloader(img_dir, mask_dir,
                      batch_size, num_classes,
                      BANDS, patches, preprocess_data,shuffle,auggy):
    dataset = SegmentationDataset(img_dir, mask_dir, BANDS, preprocess_data, num_classes, patches, auggy)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=False)
