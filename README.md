# VegSeg

**RGB Vegetation Segmentation Pipeline**

------------------------------------------------------------------------

**VegSeg** is an advanced RGB vegetation segmentation pipeline designed to extract agricultural canopy cover from UAV-captured orthomosaics across multiple spatial scales. It combines two complementary models: a high-resolution canopy cover model trained on the publicly available EWS dataset (Zenkl et al., 2022), and a field-level canopy cover model adapted for lower-resolution UAV mapping images.

The high-resolution model, based on Sadeghi et al. (2017), employs a Random Forest classifier trained on multiple color spaces derived from annotated EWS patches and UAV imagery to accurately segment vegetation with fine spatial detail. Building on this, the field-level model aligns, normalizes, and resizes the high-resolution segmentation results to match the lower-resolution mapping images. This process uses Scale-Invariant Feature Transform (SIFT) for geometric alignment and histogram matching for radiometric normalization, enabling robust multi-scale vegetation segmentation.

By integrating detailed local segmentation with broader spatial context, VegSeg delivers precise and scalable canopy cover mapping optimized for UAV imagery, especially from the DJI Mavic 3E. The pipeline can be either applied using your own high-resolution snapshot images of specific sites or with a pretrained general model trained on corn, sugar beet, and sunflower imagery captured by the DJI Mavic 3E.

------------------------------------------------------------------------

## Table of Contents

-   [Project Structure](#project-structure)
-   [Installation](#installation)
-   [Usage](#usage)
-   [Data](#data)
-   [Contributing](#contributing)
-   [License](#license)
-   [Contact](#contact)

------------------------------------------------------------------------

## Project Structure {#project-structure}

-   `methods/` — Different segmentation methods implemented  
-   `helpers/` — Helper functions and utilities  
-   `data/` — Contains:  
    -   `general_model/` — Pretrained general segmentation model  
    -   `EWS/` — Dataset for training and evaluation   
-   `main.py` — Main script to run the segmentation pipeline  
-   `config.json` — Configuration file with paths and parameters  
-   `requirements_VegSeg.txt` — Python dependencies  

------------------------------------------------------------------------

## Installation {#installation}

Install the required Python packages:

*pip install -r requirements_VegSeg.txt*

------------------------------------------------------------------------

## Usage {#usage}

The main entry point for running the segmentation pipeline is the `main.py` script. Update the `config.json` file before running.

*python main.py*

### Parameters explained

-   **`input_map_file`** (required)\
    Path to the RGB orthomosaic GeoTIFF file in WGS84 coordinate system (EPSG:4326).
-   **`input_snapshot_path`** (optional)\
    Path to a folder containing high-resolution RGB snapshot images of the area of interest.
    -   Provide this path if you want to train a model specific to your snapshots.
    -   Set to `null` to use the pretrained general model for DJI Mavic 3E.
- **`data_path`** (required)  
  Root folder containing all training datasets:  

- **Optional metadata** (used only if snapshot images are provided):  
    - `start_location`: UAV GPS coordinates (latitude, longitude) at takeoff — used for elevation and georeferencing calculations.  
    - `flight_height`: Flight altitude above ground level in meters.  
    - `zoom`: Camera zoom level specific to DJI Mavic 3E.  

------------------------------------------------------------------------

## Data {#data}

This repository includes references and structure for two main datasets used in the pipeline.\
Each dataset includes **RGB images** and their corresponding **binary masks** (with the suffix `_mask`).

### 1. **EWS Dataset**

-   **Source:** [ETH Zurich Research Collection](https://www.research-collection.ethz.ch/handle/20.500.11850/512332)

-   **Content:**

    -   Handheld camera RGB images (**350×350 pixels**)
    -   Corresponding binary masks (`*_mask.*`) for vegetation segmentation

-   **Purpose:** Used to train the **initial high-resolution model** (`SadeghiEtAl2017_model.pkl`).

-   **Location & Structure:**

    -   **data/EWS/**
        -   **train/**
            -   image_01.jpg
            -   image_01_mask.jpg
            -   ...
        -   **test/**
            -   image_11.jpg
            -   image_11_mask.jpg
            -   ...
        -   **validation/**
            -   image_21.jpg
            -   image_21_mask.jpg
            -   ...

### 2. **General Model Training Dataset**

-   **Source:** UAV imagery captured with **DJI Mavic 3E** (sunny white balance mode; crops: corn, sugar beet, sunflower)\
-   **Content:**
    -   RGB patches (**256×256 pixels**)
    -   Corresponding binary masks (`*_mask.*`) for segmentation\
-   **Purpose:** Used to train the **general segmentation model** for DJI Mavic 3E imagery.\
-   **Location & Structure:**
-   **General Model Dataset (DJI Mavic 3E):**
    -   **data/general_model/**
        -   **train_256/**
            -   image_01.jpg
            -   image_01_mask.jpg
            -   ...
        -   **test_256/**
            -   image_11.jpg
            -   image_11_mask.jpg
            -   ...
        -   **validation_256/**
            -   image_21.jpg
            -   image_21_mask.jpg
            -   ...

------------------------------------------------------------------------

## Contributing {#contributing}

Thank you for your interest in contributing to VegSeg!

To maintain quality and consistency, please contact the project maintainer before submitting any changes, suggestions, or issues:\
\[timon.boos.98\@gmail.com\]

Thank you for your cooperation and support!

------------------------------------------------------------------------

## License {#license}

© 2025 Timon Boos. All rights reserved.

Permission to use, copy, modify, or distribute this code must be requested by contacting:\
\[timon.boos.98\@gmail.com\]

------------------------------------------------------------------------

## Acknowledgements

I sincerely thank all collaborators, contributors, funding organizations, and data providers who supported this project, particularly the EWS dataset for providing data and code access (Zenkl et al., 2022). This codebase is adapted from the work of Sadeghi-Tehran et al. (2017). I also gratefully acknowledge the Earth Observation Team at Agroscope, with special thanks to Dr. Helge Aasen for his invaluable support.

------------------------------------------------------------------------

### References

-   Zenkl, R., Timofte, R., Kirchgessner, N., Roth, L., Hund, A., Van Gool, L., Walter, A., & Aasen, H. (2022). Outdoor Plant Segmentation With Deep Learning for High-Throughput Field Phenotyping on a Diverse Wheat Dataset. *Frontiers in Plant Science, 12*. <https://doi.org/10.3389/fpls.2021.774068>

-   Sadeghi-Tehran, P., Virlet, N., Sabermanesh, K., et al. (2017). Multi-feature machine learning model for automatic segmentation of green fractional vegetation cover for high-throughput field phenotyping. *Plant Methods, 13*, 103. <https://doi.org/10.1186/s13007-017-0253-8>
