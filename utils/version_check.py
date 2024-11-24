import os
import sys
import torch
import rasterio
import numpy as np
from matplotlib import pyplot as plt
import segmentation_models_pytorch
import wandb

def check_library_versions():
    library_versions = {
        "Python": sys.version,
        "OS": os.name,
        "torch": torch.__version__,
        "rasterio": rasterio.__version__,
        "numpy": np.__version__,
        "pickle": "built-in library", 
        "matplotlib": plt.matplotlib.__version__,
        "segmentation_models_pytorch": segmentation_models_pytorch.__version__,
        "wandb": wandb.__version__
    }

    for lib, version in library_versions.items():
        print(f"{lib} version: {version}")