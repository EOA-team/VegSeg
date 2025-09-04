#%%
import os
import rasterio
from rasterio.windows import from_bounds
from rasterio.transform import from_bounds as transform_from_bounds, Affine
import pandas as pd
from pyproj import Transformer
import cv2
import numpy as np

def ensure_directory_exists(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except Exception as e:
        print(f"Error creating directory {path}: {e}")

def extract_geotiff_window(metadata_row, inputpath_geotiff, path_geotiff_window, base_filename):
    try:
        image_number = str(metadata_row['imagenumber']).zfill(4)
        easting_LV95 = metadata_row['easting_LV95']
        northing_LV95 = metadata_row['northing_LV95']

        transformer = Transformer.from_crs("epsg:2056", "epsg:4326", always_xy=True)
        left_wgs84, bottom_wgs84 = transformer.transform(easting_LV95 - 2.5, northing_LV95 - 2.5)
        right_wgs84, top_wgs84 = transformer.transform(easting_LV95 + 2.5, northing_LV95 + 2.5)

        with rasterio.open(inputpath_geotiff) as src:
            window = from_bounds(left_wgs84, bottom_wgs84, right_wgs84, top_wgs84, src.transform)
            window_data = src.read((1, 2, 3), window=window)
            window_transform = transform_from_bounds(left_wgs84, bottom_wgs84, right_wgs84, top_wgs84, window_data.shape[2], window_data.shape[1])

            new_meta = src.meta.copy()
            new_meta.update({
                "driver": "GTiff",
                "height": window_data.shape[1],
                "width": window_data.shape[2],
                "transform": window_transform,
                "count": 3
            })

            output_geotiff_window_path = f"{path_geotiff_window}/{base_filename}_ROIwindow.tif"
            
            if os.path.exists(output_geotiff_window_path):
                os.remove(output_geotiff_window_path)
            with rasterio.open(output_geotiff_window_path, "w", **new_meta) as dest:
                dest.write(window_data)
            print(f"GeoTIFF window image saved to {output_geotiff_window_path}")
            return output_geotiff_window_path
    except Exception as e:
        print(f"Error extracting GeoTIFF window for {image_number}: {e}")
        return None

def crop_geotiff_using_sift_or_center(metadata_row, geotiff_window_path, image_number, inputpath_image_dir, path_geotiff_cropped, base_filename, metadata_df, index):
    try:
        jpg_filename = f"{base_filename}_V.JPG"
        jpg_path = os.path.join(inputpath_image_dir, jpg_filename)

        if not os.path.exists(jpg_path):
            print(f"JPG file does not exist: {jpg_path}")
            metadata_df.at[index, 'Cropping'] = 'centercropped'
            return crop_centered(metadata_row, geotiff_window_path, path_geotiff_cropped, base_filename)

        img1 = cv2.imread(jpg_path, cv2.IMREAD_GRAYSCALE)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        img1 = clahe.apply(img1)

        with rasterio.open(geotiff_window_path) as dataset:
            img2 = dataset.read([1, 2, 3])
            img2_rgb = img2.transpose((1, 2, 0))
            img2_gray = cv2.cvtColor(img2_rgb, cv2.COLOR_RGB2GRAY)
            img2_gray = clahe.apply(img2_gray)
            transform = dataset.transform

        if img1 is None or img2_gray is None:
            print(f"Error loading images for {image_number}")
            metadata_df.at[index, 'Cropping'] = 'centercropped'
            return crop_centered(metadata_row, geotiff_window_path, path_geotiff_cropped, base_filename)

        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2_gray, None)

        if des1 is None or des2 is None:
            print(f"No descriptors found for {image_number}, possibly blank image.")
            metadata_df.at[index, 'Cropping'] = 'centercropped'
            return crop_centered(metadata_row, geotiff_window_path, path_geotiff_cropped, base_filename)

        des1, des2 = np.float32(des1), np.float32(des2)
        flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
        matches = flann.knnMatch(des1, des2, k=2)

        RATIO_THRESHOLD = 0.6
        good = [m for m, n in matches if m.distance < RATIO_THRESHOLD * n.distance]

        MIN_MATCH_COUNT = 3
        print(f"Found {len(good)} good matches for image {image_number}")

        if len(good) >= MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10.0)

            if M is None or mask is None or not mask.any():
                print(f"Homography failed for image {image_number}")
                metadata_df.at[index, 'Cropping'] = 'centercropped'
                return crop_centered(metadata_row, geotiff_window_path, path_geotiff_cropped, base_filename)

            h, w = img1.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)

            min_x, min_y = np.int32(dst.min(axis=0).ravel() - 0.5)
            max_x, max_y = np.int32(dst.max(axis=0).ravel() + 0.5)

            min_x = max(min_x, 0)
            min_y = max(min_y, 0)
            max_x = min(max_x, img2.shape[2])  # width
            max_y = min(max_y, img2.shape[1])  # height

            cropped_width = max_x - min_x
            cropped_height = max_y - min_y
            print(f"Cropped box: ({min_x}, {min_y}) to ({max_x}, {max_y}) => size: {cropped_width}x{cropped_height}")

            if cropped_width < 100 or cropped_height < 100:
                print(f"Cropped area too small for {image_number}, skipping.")
                metadata_df.at[index, 'Cropping'] = 'centercropped'
                return crop_centered(metadata_row, geotiff_window_path, path_geotiff_cropped, base_filename)

            # Aspect ratio check
            expected_ratio = w / h
            actual_ratio = cropped_width / cropped_height if cropped_height > 0 else 0
            if abs(actual_ratio - expected_ratio) > 0.75:
                print(f"Aspect ratio mismatch (expected ~{expected_ratio:.2f}, got {actual_ratio:.2f}) for {image_number}, skipping.")
                metadata_df.at[index, 'Cropping'] = 'centercropped'
                return crop_centered(metadata_row, geotiff_window_path, path_geotiff_cropped, base_filename)

            cropped_img = img2[:, min_y:max_y, min_x:max_x]
            new_transform = transform * Affine.translation(min_x, min_y)

            output_cropped_path = f"{path_geotiff_cropped}/{base_filename}_ROIcropped.tif"
            new_meta = dataset.meta.copy()
            new_meta.update({
                "height": cropped_img.shape[1],
                "width": cropped_img.shape[2],
                "transform": new_transform
            })

            if os.path.exists(output_cropped_path):
                os.remove(output_cropped_path)
            with rasterio.open(output_cropped_path, 'w', **new_meta) as dst:
                dst.write(cropped_img)

            print(f"Result saved to {output_cropped_path}")
            metadata_df.at[index, 'Cropping'] = 'siftcropped'
            return output_cropped_path
        else:
            print(f"Not enough matches found for {image_number} - {len(good)}/{MIN_MATCH_COUNT}")
            metadata_df.at[index, 'Cropping'] = 'centercropped'
            return crop_centered(metadata_row, geotiff_window_path, path_geotiff_cropped, base_filename)
    except Exception as e:
        print(f"Error cropping GeoTIFF for {image_number}: {e}")
        metadata_df.at[index, 'Cropping'] = 'centercropped'
        return crop_centered(metadata_row, geotiff_window_path, path_geotiff_cropped, base_filename)


def crop_centered(metadata_row, geotiff_window_path, path_geotiff_cropped, base_filename):
    try:
        with rasterio.open(geotiff_window_path) as src:
            # CRS of geotiff_window is EPSG:4326

            # Convert center point from LV95 to WGS84
            transformer = Transformer.from_crs("epsg:2056", "epsg:4326", always_xy=True)
            center_lon, center_lat = transformer.transform(center_x, center_y)

            # Get pixel size in degrees from the transform (src.transform)
            pixel_width = src.transform.a  # pixel width in CRS units (degrees)
            pixel_height = -src.transform.e  # pixel height in CRS units (degrees), usually negative

            half_width_deg = pixel_width * (metadata_row['ImageWidth'] / 2)
            half_height_deg = pixel_height * (metadata_row['ImageHeight'] / 2)

            left_wgs84 = center_lon - half_width_deg
            right_wgs84 = center_lon + half_width_deg
            bottom_wgs84 = center_lat - half_height_deg
            top_wgs84 = center_lat + half_height_deg

            window = from_bounds(left_wgs84, bottom_wgs84, right_wgs84, top_wgs84, src.transform)
            window_data = src.read((1, 2, 3), window=window)
            window_transform = transform_from_bounds(left_wgs84, bottom_wgs84, right_wgs84, top_wgs84, window_data.shape[2], window_data.shape[1])

            new_meta = src.meta.copy()
            new_meta.update({
                "driver": "GTiff",
                "height": window_data.shape[1],
                "width": window_data.shape[2],
                "transform": window_transform
            })

            output_cropped_path = f"{path_geotiff_cropped}/{base_filename}_ROIcropped.tif"
            if os.path.exists(output_cropped_path):
                os.remove(output_cropped_path)
            with rasterio.open(output_cropped_path, "w", **new_meta) as dest:
                dest.write(window_data)
            print(f"Cropped centered GeoTIFF saved to {output_cropped_path}")
            return output_cropped_path
    except Exception as e:
        print(f"Error cropping centered GeoTIFF: {e}")
        return None


def process_folder(input_folder, inputpath_geotiff):
    folder_name = os.path.basename(input_folder)
    metadata_csv_path = f"{input_folder}/{folder_name}_metadata.csv"
    path_geotiff_window = f"{input_folder}/{folder_name}_ROIwindow"
    path_geotiff_cropped = f"{input_folder}/{folder_name}_ROIcropped"

    ensure_directory_exists(path_geotiff_window)
    ensure_directory_exists(path_geotiff_cropped)

    try:
        metadata_df = pd.read_csv(metadata_csv_path)
        # Check if the 'Cropping' column already exists
        if 'Cropping' not in metadata_df.columns:
            metadata_df['Cropping'] = ''  # Add a new column if it doesn't exist
    except Exception as e:
        print(f"Error reading metadata CSV {metadata_csv_path}: {e}")
        return

    for index, metadata_row in metadata_df.iterrows():
        try:
            # Validate critical fields
            if pd.isna(metadata_row['imagenumber']) or pd.isna(metadata_row['easting_LV95']) or pd.isna(metadata_row['northing_LV95']):
                print(f"Skipping row {index} due to missing critical data.")
                continue

            image_number = str(int(metadata_row['imagenumber'])).zfill(4)
            base_filename = f"DJI_{metadata_row['imagedate']}_{image_number}"
            print(f"Processing image {image_number}")

            geotiff_window_path = extract_geotiff_window(metadata_row, inputpath_geotiff, path_geotiff_window, base_filename)
            if geotiff_window_path:
                # Pass metadata_df and index to the function
                crop_geotiff_using_sift_or_center(metadata_row, geotiff_window_path, image_number, input_folder, path_geotiff_cropped, base_filename, metadata_df, index)
        except Exception as e:
            print(f"Error processing row {index}: {e}")

    # Save updated metadata to CSV
    metadata_df.to_csv(metadata_csv_path, index=False)
    print(f"Updated metadata saved to {metadata_csv_path}")

#input_folder = '/home/f60041558@agsad.admin.ch/mnt/eo-nas1/eoa-share/projects/011_experimentEObserver/data/Final/Segmentation_HR/Test/Snapshot/DJI_202406061353_114_EschikonSteinmueri1'
#inputpath_geotiff = '/home/f60041558@agsad.admin.ch/mnt/eo-nas1/eoa-share/projects/011_experimentEObserver/data/Final/Segmentation_HR/Test/Mapping/202406061419_115_EschikonSteinmueri1MappingLayer.tif'
#process_folder(input_folder, inputpath_geotiff)
# %%
