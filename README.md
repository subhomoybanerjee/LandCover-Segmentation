# **Mangrove Monitoring Using U-Net and Multispectral Satellite Imagery**

## **Overview**

This project focuses on automating mangrove extent mapping using deep learning based semantic segmentation. The work was carried out as part of the **AIMVIE Study**, which aims to improve mangrove monitoring, management, and carbon sequestration assessment.

Using U-Net and other segmentation architectures on multispectral satellite imagery, the project delivers a scalable workflow for detecting and segmenting mangrove regions with high spatial accuracy.

---

## **Objectives**

* Build a robust deep learning pipeline for mangrove segmentation.
* Process and analyse multispectral satellite data for feature extraction.
* Improve model performance through systematic experimentation, EDA, and hyperparameter tuning.
* Leverage cloud and TPU acceleration for efficient model training.

---

## **Key Features**

### **1. Deep Learning for Semantic Segmentation**

* Implemented **U-Net**, **Attention U-Net**, and other segmentation variants.
* Trained models using **TensorFlow** and **PyTorch**.
* Tailored architectures to handle multispectral satellite imagery (including NIR channels).

### **2. Cloud-Accelerated Training**

* Deployed and trained models on **Google Cloud VM**.
* Utilised **TPUs** for faster experimentation and large batch processing.
* Managed scalable pipelines suited for high-resolution image datasets.

### **3. Exploratory Data Analysis**

* Performed EDA on satellite tiles to identify patterns, class imbalance, and vegetation signatures.
* Visualised spectral bands, NDVI distributions, and spatial coverage.
* Informed preprocessing choices, augmentation strategies, and model architecture decisions.

---

## **Workflow**

1. **Data Ingestion**
   Import multispectral satellite tiles (RGB + NIR + additional bands when available).

2. **Preprocessing**

   * Normalisation, resizing, band selection, and mask alignment.
   * Computed NDVI and auxiliary vegetation indices to support segmentation.

3. **Model Training**

   * Trained U-Net and segmentation variants on TPUs.
   * Performed hyperparameter tuning and augmentation trials.

4. **Evaluation**

   * Used IoU, Dice Score, Precision, and Recall to track performance.
   * Compared multiple architectures for accuracy and efficiency.

5. **Results & Insights**

   * Models achieved strong boundary detection and class separation.
   * Improved spatial consistency across varying terrain and lighting.

---

## **Tech Stack**

* **Languages:** Python
* **Frameworks:** TensorFlow, PyTorch
* **Cloud:** Google Cloud Platform (GCP), TPU VMs
* **Visualization:** Matplotlib, Seaborn, Rasterio, GDAL
* **Data:** Multispectral satellite imagery (RGB + NIR)

---

## **Impact**

The project supports:

* Better mangrove monitoring workflows.
* Scalable, repeatable segmentation pipelines.
* Improved environmental decision-making for carbon and ecosystem management.

---

If you want, I can also generate:

* A **diagram or architecture figure** for the README
* A short **GitHub description**
* A version tailored for recruiters instead of technical readers
