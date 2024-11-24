
# **Landcover Semantic Segmentation with U-Net++**

This project implements a **semantic segmentation** pipeline using **U-Net++** to classify land cover types such as **buildings**, **woodlands**, **water**, and **roads**. The model is trained and evaluated on the [LandCover.ai dataset](https://landcover.ai.linuxpolska.com/), a high-resolution satellite image dataset for land cover classification leveraging CUDA for GPU acceleration to optimize training speed and performance.


## **Dataset**

The dataset used in this project is **LandCover.ai**, available for download at [LandCover.ai Official Website](https://landcover.ai.linuxpolska.com/). 

### **Dataset Details**

- **Source**: High-resolution satellite imagery.
- **Classes**:
  - **Buildings** (label: 1)
  - **Woodlands** (label: 2)
  - **Water** (label: 3)
  - **Roads** (label: 4)
- **Resolution**: 512x512 tiles.
- **Format**: RGB images with corresponding labeled masks.


## **Model Architecture**

The model used for this project is **U-Net++** (Nested U-Net), an advanced variant of U-Net with dense skip connections and improved feature extraction capabilities.


### **Dependencies**

- Python 3.11 or higher
- Required libraries:
  ```
  numpy
  pandas
  PyTorch
  Rasterio
  matplotlib
  scikit-learn
  ```

### **Dataset Preparation**

1. Download the dataset from [LandCover.ai](https://landcover.ai.linuxpolska.com/).
2. Run their patching script to patchify the images into desired height and width.
3. Extract the dataset and organize it as follows:
   ```
   ├── datasets
     ├── allPatches
       ├── images
           ├── image1.jpg
           ├── image2.jpg
           ├── ...
       ├── masks
           ├── mask1.png
           ├── mask2.png
           ├── ...
   ```
4. Go to the normalization folder and choose the normalization method to create normalized patches for training.
   

## **Training the Model**

To train the model:
1. Split the data into training, validation, and test sets by going to the utils folder and running the splitPatches.py script.
3. Run the main script.
   

## **Evaluation**

The trained model is evaluated on the test set using the following metrics:
- **IoU (Intersection over Union)**
- **Macro average F1-score**
- 

### **Results**

This will include the training and validation f1,iou and loss curves alongside with the predicted masks.


## **References**

- Dataset: [LandCover.ai](https://landcover.ai.linuxpolska.com/)
- Model: [U-Net++ Paper](https://arxiv.org/abs/1807.10165)


## **Acknowledgements**

- Thanks to the creators of the **LandCover.ai dataset** for providing a valuable resource for semantic segmentation research.
- Model architecture inspired by the [**U-Net++ implementation**](https://github.com/qubvel-org/segmentation_models.pytorch)
