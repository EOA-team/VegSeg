#%%
import os
import pandas as pd
import cv2
import numpy as np
from osgeo import gdal

def resize_image(image, target_shape):
    return cv2.resize(image, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_AREA)

def ensure_landscape(image):
    """Rotate image if it's in portrait orientation (height > width)."""
    if image.shape[0] > image.shape[1]:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return image

def ensure_landscape_array(array):
    """Rotate array if it's in portrait orientation (height > width)."""
    if array.shape[0] > array.shape[1]:
        return np.rot90(array, k=-1)  # 90° clockwise
    return array

def process_images(input_folder_rgb):
    base_name = os.path.basename(input_folder_rgb)
    
    metadata_file = os.path.join(input_folder_rgb, f'{base_name}_metadata.csv')
    input_folder_masks = os.path.join(input_folder_rgb, f'{base_name}_masks')
    cropped_tif_folder = os.path.join(input_folder_rgb, f'{base_name}_ROIcropped')
    output_folder = os.path.join(input_folder_rgb, f'{base_name}_resized')
    os.makedirs(output_folder, exist_ok=True)

    metadata = pd.read_csv(metadata_file)

    for index, row in metadata.iterrows():
        image_name_v = row['imagename']
        image_name = image_name_v.replace("_V.JPG", "")
        
        rgb_image_path = os.path.join(input_folder_rgb, f"{image_name}_V.JPG")
        mask_image_path = os.path.join(input_folder_masks, f"{image_name}_V_prediction.png")
        cropped_tif_path = os.path.join(cropped_tif_folder, f"{image_name}_ROIcropped.tif")
        
        if not os.path.exists(rgb_image_path) or not os.path.exists(mask_image_path) or not os.path.exists(cropped_tif_path):
            print(f"Skipping {image_name} because one of the files is missing.")
            continue

        # Load and orient cropped GeoTIFF
        cropped_tif = gdal.Open(cropped_tif_path)
        cropped_tif_array = cropped_tif.ReadAsArray().transpose((1, 2, 0))  # HWC
        cropped_tif_array = ensure_landscape_array(cropped_tif_array)
        target_shape = cropped_tif_array.shape[:2]

        # Load and orient RGB and mask
        rgb_image = ensure_landscape(cv2.imread(rgb_image_path))
        binary_mask = ensure_landscape(cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE))

        # Resize
        resized_rgb_image = resize_image(rgb_image, target_shape)
        resized_binary_mask = resize_image(binary_mask, target_shape)

        # Save
        cv2.imwrite(os.path.join(output_folder, f"{image_name}_resized.png"), resized_rgb_image)
        cv2.imwrite(os.path.join(output_folder, f"{image_name}_mask_resized.png"), resized_binary_mask)

        print(f"Resized and saved images for {image_name}.")

    print("Processing complete.")

# Example usage
#process_images(
#   input_folder_rgb='/home/f60041558@agsad.admin.ch/mnt/eo-nas1/eoa-share/projects/011_experimentEObserver/data/Final/Segmentation_HR/RE_20250602/Snapshot_20250602'
#)

# %%
