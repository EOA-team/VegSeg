#%%
import os
import glob
import numpy as np
import rasterio
from rasterio.windows import Window
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from skimage.color import rgb2lab, rgb2luv, rgb2hsv, rgb2ycbcr, rgb2yuv
from methods import base  # Assuming 'methods' contains 'base.BenchmarkMethod'
from rasterio.merge import merge

# Function to dynamically segment the GeoTIFF
def segment_geotiff(inputlayer, segment_size_pixels=1000):
    input_directory = os.path.dirname(inputlayer)
    input_filename = os.path.splitext(os.path.basename(inputlayer))[0]
    new_folder = os.path.join(input_directory, input_filename)

    if not os.path.exists(new_folder):
        os.makedirs(new_folder)

    new_inputlayer = os.path.join(new_folder, os.path.basename(inputlayer))
    outputfolder = os.path.join(new_folder, f"{input_filename}_segmented_{segment_size_pixels}")

    if not os.path.exists(outputfolder):
        os.makedirs(outputfolder)

    with rasterio.open(inputlayer) as src:
        width = src.width
        height = src.height
        bands = [1, 2, 3] if src.count >= 3 else list(range(1, src.count + 1))
        meta = src.meta.copy()
        meta.update({'count': len(bands), 'transform': None})

        num_segments_x = (width + segment_size_pixels - 1) // segment_size_pixels
        num_segments_y = (height + segment_size_pixels - 1) // segment_size_pixels

        for i in range(num_segments_x):
            for j in range(num_segments_y):
                window = Window(
                    i * segment_size_pixels,
                    j * segment_size_pixels,
                    min(segment_size_pixels, width - i * segment_size_pixels),
                    min(segment_size_pixels, height - j * segment_size_pixels)
                )

                if window.width != segment_size_pixels or window.height != segment_size_pixels:
                    continue

                segment = src.read(bands, window=window)
                segment_transform = src.window_transform(window)

                meta.update({
                    'height': window.height,
                    'width': window.width,
                    'transform': segment_transform
                })

                output_file = os.path.join(outputfolder, f"{input_filename}_segment_{i}_{j}.tif")
                with rasterio.open(output_file, 'w', **meta) as dst:
                    dst.write(segment)

                print(f"Saved segment {i}_{j} to {output_file}")

# Function to merge segmented GeoTIFF files
def merge_geotiff_segments(input_file):
    input_directory = os.path.dirname(input_file)
    input_filename = os.path.splitext(os.path.basename(input_file))[0]
    segmented_folder = os.path.join(input_directory,input_filename, f"{input_filename}_segmented_1000_prediction")


    if not os.path.exists(segmented_folder):
        print(f"Segmented folder not found: {segmented_folder}")
        return

    tiff_files = glob.glob(os.path.join(segmented_folder, '*.tif'))
    if not tiff_files:
        print(f"No GeoTIFF files found in the directory: {segmented_folder}")
        return

    datasets = [rasterio.open(tiff_file) for tiff_file in tiff_files]
    mosaic, out_transform = merge(datasets) #mosaic, out_transform = rasterio.merge.merge(datasets)

    out_meta = datasets[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_transform,
        "compress": "lzw"
    })

    output_file = os.path.join(input_directory, f"{input_filename}_prediction.tif")
    with rasterio.open(output_file, "w", **out_meta) as dest:
        dest.write(mosaic)

    for dataset in datasets:
        dataset.close()

    print(f"Merged GeoTIFF saved as {output_file}")


class SadeghiEtAl2017(base.BenchmarkMethod):
    def __init__(self):
        super().__init__()
        self.scaler = StandardScaler()

        self.model = RandomForestClassifier(max_depth=95,
                                            max_features=6,
                                            min_samples_leaf=6,
                                            min_samples_split=4,
                                            n_estimators=55,
                                            bootstrap=False,
                                            random_state=42,
                                            n_jobs=-1)

    def preprocess_image(self, image: np.array):
        # Reshape to RGB vector
        rgb = image.reshape((-1, image.shape[2]))

        # Do color transformations
        lab = rgb2lab(rgb)
        luv = rgb2luv(rgb)
        hsv = rgb2hsv(rgb)
        hsi = rgb2hsi(rgb)
        ycbcr = rgb2ycbcr(rgb)
        yuv = rgb2yuv(rgb)

        # Merge to single array
        sample = np.concatenate((rgb, lab, luv, hsv, hsi, ycbcr, yuv), axis=1)

        # Convert to Channels x Pixels
        sample = sample.reshape((21, -1))

        return sample

    def train(self, train_path, val_path):
        # Load all training data into memory
        x = []
        y = []

        for mask_path in glob.glob(train_path + '/*mask.png'):
            image_path = mask_path.replace('_mask', '')

            mask_i = self.read_image(mask_path, rgb=False)
            y.append(mask_i.reshape((-1, 1)))

            image_i = self.read_image(image_path, rgb=True)
            x.append(self.preprocess_image(image_i))

        x = np.array(x)
        y = np.array(y)

        # Reshape images to individual pixels as samples for 256x256 images
        x = x.reshape((-1, 21))
        y = y.reshape((-1,)).astype(np.uint8)

        x = self.scaler.fit_transform(x)

        # Train model
        self.model.fit(X=x, y=y)

        # Calculate training metrics
        preds = self.model.predict(x)

        # Reshape back to images for appropriate metrics computations
        preds = preds.reshape((-1, 65536)).astype(np.uint8)  # 256 * 256 = 65536 pixels
        y = y.reshape((-1, 65536))

        train_metrics = self.calculate_metrics(preds, y)

        # Load all validation data into memory
        x_val = []
        y_val = []

        for mask_path in glob.glob(val_path + '/*mask.png'):
            image_path = mask_path.replace('_mask', '')

            mask_i = self.read_image(mask_path, rgb=False)
            y_val.append(mask_i.reshape((-1, 1)))

            image_i = self.read_image(image_path, rgb=True)
            x_val.append(self.preprocess_image(image_i))

        x_val = np.array(x_val)
        y_val = np.array(y_val)

        # Reshape images to individual pixels as samples for 256x256 images
        x_val = x_val.reshape((-1, 21))
        y_val = y_val.reshape((-1,)).astype(np.uint8)

        x_val = self.scaler.transform(x_val)

        # Calculate validation metrics
        preds_val = self.model.predict(x_val)

        # Reshape back to images for appropriate metrics computations
        preds_val = preds_val.reshape((-1, 65536))  # 256 * 256 = 65536 pixels
        y_val = y_val.reshape((-1, 65536))

        val_metrics = self.calculate_metrics(preds_val, y_val)

        return train_metrics, val_metrics

    def test(self, test_path='data/test'):
        # Load all testing data into memory
        x_test = []
        y_test = []

        for mask_path in sorted(glob.glob(test_path + '/*mask.png')):
            image_path = mask_path.replace('_mask', '')

            mask_i = self.read_image(mask_path, rgb=False)
            y_test.append(mask_i.reshape((-1, 1)))

            image_i = self.read_image(image_path, rgb=True)
            x_test.append(self.preprocess_image(image_i))

        x_test = np.array(x_test)
        y_test = np.array(y_test)

        # Reshape images to individual pixels as samples for 256x256 images
        x_test = x_test.reshape((-1, 21))
        y_test = y_test.reshape((-1, 1))

        x_test = self.scaler.transform(x_test)

        # Calculate predictions
        preds_test = self.model.predict(x_test)

        # Reshape back to images for appropriate metrics computations
        preds_test = preds_test.reshape((-1, 256, 256)).astype(np.uint8)  # Adjust to 256x256
        y_test = y_test.reshape((-1, 256, 256)).astype(np.uint8)  # Adjust to 256x256

        test_metrics = self.calculate_metrics(preds_test, y_test)

        print("Testing metrics:", test_metrics)

        return test_metrics, preds_test, y_test
        
    def plot_predictions_geotiff(self, tif_folder, segment_size_pixels=1000):
        # Automatically create the output folder by appending "_prediction" to the input folder name
        output_folder = f"{tif_folder}_prediction"

        # Create the output directory if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Get the list of all TIFF files in the folder
        image_paths = sorted(glob.glob(os.path.join(tif_folder, '*.tif')))

        # Loop through each TIFF file
        for image_path in image_paths:
            x_test = []

            # Open the current TIFF file
            with rasterio.open(image_path) as src:
                # Loop through the image in segments
                for i in range(0, src.width, segment_size_pixels):
                    for j in range(0, src.height, segment_size_pixels):
                        window = Window(i, j, segment_size_pixels, segment_size_pixels)
                        segment = src.read(window=window)

                        if segment.shape[0] != 3:  # Ensure it's an RGB image
                            continue

                        segment = np.transpose(segment, (1, 2, 0))  # Transform to (height, width, channels)
                        x_test.append(self.preprocess_image(segment))

            x_test = np.array(x_test)

            # Debugging print to check the state of x_test
            print(f"x_test shape before reshaping: {x_test.shape}")

            if x_test.size == 0:
                print(f"No valid segments found or preprocessing failed for {image_path}.")
                continue

            x_test = x_test.reshape((-1, 21))
            x_test = self.scaler.transform(x_test)

            # Calculate predictions
            preds_test = self.model.predict(x_test)

            # Reshape back to segments for appropriate metrics computations
            preds_test = preds_test.reshape((-1, segment_size_pixels, segment_size_pixels))

            # Apply a strict threshold to ensure binary values (0 or 1)
            binary_preds_test = (preds_test <= 0.5).astype(np.uint8)

            # Make sure pixels with value 0 from the input TIFF are never classified as 1
            with rasterio.open(image_path) as src:
                input_data = src.read(1)  # Read the first band (assuming grayscale mask or first band for simplicity)
                
                # Loop through the image in segments and apply the mask to each segment
                for idx, (i, j) in enumerate([(i, j) for i in range(0, src.width, segment_size_pixels) for j in range(0, src.height, segment_size_pixels)]):
                    window = Window(i, j, segment_size_pixels, segment_size_pixels)
                    segment_mask = input_data[j:j+segment_size_pixels, i:i+segment_size_pixels] == 0

                    # Apply the mask to binary_preds_test for the corresponding segment
                    binary_preds_test[idx][segment_mask] = 0

            # Print the frequency of each value in binary_preds_test
            unique, counts = np.unique(binary_preds_test, return_counts=True)
            print(f"Frequency of each value in binary_preds_test for {image_path}:")
            for value, count in zip(unique, counts):
                print(f"Value: {value}, Count: {count}")

            # Extract the base filename (without .tif extension) to use in the output filename
            base_filename = os.path.splitext(os.path.basename(image_path))[0]
            output_file = os.path.join(output_folder, f"{base_filename}_pred.tif")

            # Save the predicted masks as GeoTIFF images
            with rasterio.open(image_path) as src:
                meta = src.meta.copy()
                # Update necessary fields according to your requirements
                meta.update({
                    'driver': 'GTiff',
                    'dtype': 'uint8',
                    'nodata': 0.0,
                    'count': 1,           # Single band for binary prediction
                    'crs': src.crs,       # Preserve the CRS
                    'transform': src.transform,  # Update this transform dynamically for each window
                    'blockysize': 8,
                    'tiled': False,
                    'compress': 'lzw',
                    'interleave': 'band'
                })

                for idx, (i, j) in enumerate([(i, j) for i in range(0, src.width, segment_size_pixels) for j in range(0, src.height, segment_size_pixels)]):
                    window = Window(i, j, segment_size_pixels, segment_size_pixels)
                    transform = src.window_transform(window)

                    pred_segment = binary_preds_test[idx]

                    # Update the transform for the segment
                    meta.update({
                        'transform': transform,
                        'width': segment_size_pixels,
                        'height': segment_size_pixels
                    })

                    with rasterio.open(output_file, 'w', **meta) as dst:
                        dst.write(pred_segment, 1)

            print(f"Predicted images for {image_path} saved to {output_folder}")



def rgb2hsi(rgb: np.array):
    # Method according to https://www.vocal.com/video/rgb-and-hsvhsihsl-color-space-conversion/

    h = rgb2hsv(rgb)[:, 0]
    c_min = np.min(rgb, axis=1)
    i = np.mean(rgb, axis=1)

    # Handle division by zero and NaN values
    s = np.zeros_like(i)
    valid_mask = i != 0
    s[valid_mask] = 1 - c_min[valid_mask] / i[valid_mask]
    s[np.isnan(s)] = 0

    return np.stack((h, s, i), axis=1)

import json

def run_segmentation_pipeline(map_file, rgb_folder=None):
    # Use relative path based on this script's location
    base_path = os.path.dirname(os.path.realpath(__file__))

    if rgb_folder is None:
        print("Running pipeline using the general model...")

        # Define general model data path
        general_data_path = os.path.join(base_path, '..', 'data', 'general_model')

        # Define training, validation, and test paths
        train_path = os.path.join(general_data_path, 'train_256')
        validation_path = os.path.join(general_data_path, 'validation_256')
        test_path = os.path.join(general_data_path, 'test_256')
    else:
        print("Running pipeline using RGB-folder-specific model...")
        # Extract base name and construct paths
        base_name = os.path.basename(rgb_folder)
        train_path = os.path.join(rgb_folder, f'{base_name}_resized_histogrammatched_training', 'train_256')
        validation_path = os.path.join(rgb_folder, f'{base_name}_resized_histogrammatched_training', 'validation_256')
        test_path = os.path.join(rgb_folder, f'{base_name}_resized_histogrammatched_training', 'test_256')

    # Ensure required paths exist
    for path_label, path in [("Training", train_path), ("Validation", validation_path)]:
        if not os.path.exists(path):
            raise ValueError(f"{path_label} path does not exist: {path}")

    # Segment the GeoTIFF
    segment_geotiff(map_file)

    # Train the model
    model = SadeghiEtAl2017()
    train_metrics, val_metrics = model.train(train_path=train_path, val_path=validation_path)
    print("Training metrics:", train_metrics)
    print("Validation metrics:", val_metrics)

    # Save metrics
    input_directory = os.path.dirname(map_file)
    base_name = os.path.splitext(os.path.basename(map_file))[0]
    tif_folder = os.path.join(input_directory, base_name, base_name + "_segmented_1000")
    os.makedirs(tif_folder, exist_ok=True)

    metrics_data = {
        "training_metrics": train_metrics,
        "validation_metrics": val_metrics
    }
    metadata_file = os.path.join(tif_folder, f"{base_name}_model_metrics.json")
    with open(metadata_file, 'w') as f:
        json.dump(metrics_data, f, indent=4)

    print(f"Model metrics saved as metadata to {metadata_file}")

    # Predict and plot
    model.plot_predictions_geotiff(tif_folder=tif_folder, segment_size_pixels=1000)

    # Merge segmented predictions
    merge_geotiff_segments(map_file)

#map_file = "/home/f60041558@agsad.admin.ch/mnt/eo-nas1/eoa-share/projects/011_experimentEObserver/data/Final/Segmentation_HR/Test_RE/Mapping/re112o_20250527_mapping.tif"
#rgb_folder = "/home/f60041558@agsad.admin.ch/mnt/eo-nas1/eoa-share/projects/011_experimentEObserver/data/Final/Segmentation_HR/Test_RE/Snapshot"
#run_segmentation_pipeline(map_file, rgb_folder)

# %%
