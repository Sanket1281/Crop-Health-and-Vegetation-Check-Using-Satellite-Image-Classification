# Vegetation Check Using Satellite Image Classification

This project utilizes a U-Net deep learning model to perform semantic segmentation on Sentinel-2 satellite images. The primary goal is to segment land cover into major categories (Vegetation, Water, Built-up) and then use the Normalized Difference Vegetation Index (NDVI) to further sub-classify vegetation types for environmental and agricultural monitoring.

<!-- You can add a screenshot of your final map here -->
<!-- Example: ![Land Cover Showcase for Melbourne](Land_Cover_Showcase_Melbourne.png) -->

## Project Overview

The aim of this project is to develop a satellite image classification system to monitor vegetation conditions over large areas. By leveraging modern deep learning techniques and multispectral satellite data, this system provides accurate, real-time insights into vegetation health, which is valuable for farmers, environmental scientists, and urban planners.

## Data Sources

1.  **Input Images: Sentinel-2**
    * This satellite mission from the Copernicus Programme provides high-resolution, 13-band multispectral imagery.
    * The 13 spectral bands (ranging from 10m to 60m resolution) are essential for distinguishing between land cover types and calculating vegetation indices.
    * The model was trained on 13-channel input images.

2.  **Ground Truth Masks: ESA WorldCover 10m**
    * The [ESA WorldCover 10m v200](https://esa-worldcover.org/en) dataset was used as the ground truth for supervised learning.
    * These tiles, which are pre-segmented into 11 distinct land cover classes, were re-projected and reclassified for training.
    * Training data was sourced from tiles covering diverse regions, including **India** and **Australia**, to ensure model generalization.

## Methodology

This project employs a two-part workflow:
1.  **Part 1: Segmentation:** A U-Net model is trained to identify the primary land cover types.
2.  **Part 2: Classification:** The "Vegetation" class from Part 1 is further classified using NDVI.

### Part 1: U-Net for Land Cover Segmentation

A U-Net architecture was implemented in PyTorch for its effectiveness in pixel-wise semantic segmentation.

#### Pre-processing for Training
The training data was created using the `preprocess_tile` function in the notebook:
1.  **Reprojection:** The ESA WorldCover mask tiles were re-projected using `rasterio.warp.reproject` to perfectly align their pixel grid and Coordinate Reference System (CRS) with the corresponding Sentinel-2 images.
2.  **Reclassification:** The 11 ESA classes were simplified into 4 primary classes for the model:
    * **Class 1 (Vegetation):** Mapped from ESA classes 10, 20, 30, 40 (Trees, Shrubs, Grassland, Cropland).
    * **Class 2 (Water):** Mapped from ESA class 80 (Open water).
    * **Class 3 (Built-up):** Mapped from ESA class 50 (Built-up areas).
    * **Class 0 (Background):** All other classes.
3.  **Tiling:** The large, aligned images and masks were tiled into 128x128 patches.
4.  **Train/Val Split:** Patches were randomly assigned to a training set (80%) or a validation set (20%).

#### Model Training
* The U-Net was trained with an input of 13 channels (for Sentinel-2) and an output of 4 classes.
* A **Weighted Loss Function** (`nn.CrossEntropyLoss`) was used to combat the severe class imbalance in the dataset. The "Built-up" class (Class 3) was given a triple manual weight to improve its detection accuracy.
* The best model was saved based on the highest **F1-score for the "Built-up" class** on the validation dataset.

### Part 2: NDVI for Vegetation Classification

This inference pipeline uses the trained model to produce a more detailed classification map.

1.  **Run Segmentation:** The trained U-Net is loaded and run on a new, full-sized Sentinel-2 image. It uses a patch-based inference method to process the large image and stitches the results back together, producing a 4-class segmentation map (`seg_mask.tif`).
2.  **Calculate NDVI:** The Normalized Difference Vegetation Index (NDVI) is calculated for the same image using its Red (Band 4) and Near-Infrared (Band 8) bands.
    $NDVI = \frac{NIR - RED}{NIR + RED}$
3.  **Combined Analysis:** The script loads both the segmentation mask and the NDVI map. It iterates through all pixels classified as **Vegetation (Class 1)** by the U-Net and re-classifies them based on their NDVI value:
    * **Forest (Class 4):** High NDVI (e.g., $NDVI \ge 0.4$)
    * **Cropland (Class 5):** Medium NDVI (e.g., $0.1 \le NDVI < 0.4$)
    * **Grassland (Class 6):** Low NDVI (e.g., $NDVI < 0.1$)

*(Note: NDVI thresholds are adjustable in the script to suit different geographic regions, such as the Amazon vs. Melbourne).*

## Final Output

The final output is a 6-class land cover map (`veg_type.tif`) that provides a detailed breakdown of vegetation health and type, in addition to water and urban areas. The script also calculates the total area (in hectares) for each class.

*Example: Final 6-class map for Mumbai (0: Background, 2: Water, 3: Built-up, 4: Forest, 5: Cropland, 6: Grassland)*

## Applications

* **Crop Monitoring:** Allows for the early detection of crop stress or disease.
* **Deforestation Detection:** Helps monitor forest health and identifies areas of deforestation.
* **Urban Planning:** Tracks the expansion of built-up areas and their impact on green spaces and water bodies.
* **Climate Change Studies:** Provides data for analyzing changes in land cover and vegetation patterns over time.

## How to Run

### Dependencies
This project requires Python 3 and the following libraries:
* `torch`
* `rasterio`
* `numpy`
* `matplotlib`
* `glob`
* `logging`

### Running the Inference Pipeline

The `ESA 1.ipynb` notebook contains the full code. To run the analysis pipeline on a new image:

1.  **Place Data:** Add your 13-band Sentinel-2 `.tif` image to an input folder.
2.  **Set Paths:** In **Cell 15** of the notebook:
    * Update `img_dir` to your input folder path.
    * Update `out_dir` to your desired output folder path.
    * Ensure `model_path` points to your trained `unet_india_aus_10.pth` file.
3.  **Run Cells:** Execute **Cells 15, 16, 18, and 22** in sequence.
    * **Cell 15:** Generates the 4-class segmentation mask (`seg_mask.tif`).
    * **Cell 16:** Generates the NDVI map (`ndvi.tif`).
    * **Cell 18:** Combines the two maps to create the final 6-class map (`veg_type.tif`) and calculates the area for each class.
    * **Cell 22:** Saves a PNG visualization of the final map for review.
