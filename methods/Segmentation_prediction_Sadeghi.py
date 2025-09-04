# %%
from PIL import Image
from methods import base
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from skimage.color import rgb2lab, rgb2luv, rgb2hsv, rgb2ycbcr, rgb2yuv
import glob
import os
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

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

        # merge to single array
        sample = np.concatenate((rgb, lab, luv, hsv, hsi, ycbcr, yuv), axis=1)

        # Convert to Channels x Pixels
        sample = sample.reshape((21, -1))

        return sample

    def train(self, train_path='data/train', val_path='data/validation'):

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

        # Reshape images to individual pixels as samples
        x = x.reshape((-1, 21))
        y = y.reshape((-1,)).astype(np.uint8)

        x = self.scaler.fit_transform(x)

        # Train model
        self.model.fit(X=x, y=y)

        # Calculate training metrics
        preds = self.model.predict(x)

        # Reshape back to images for appropriate metrics computations
        preds = preds.reshape((-1, 122500)).astype(np.uint8)
        y = y.reshape((-1, 122500))

        train_metrics = self.calculate_metrics(preds, y)

        # Load all training data into memory
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

        # Reshape images to individual pixels as samples
        x_val = x_val.reshape((-1, 21))
        y_val = y_val.reshape((-1,)).astype(np.uint8)

        x_val = self.scaler.transform(x_val)

        # Calculate training metrics
        preds_val = self.model.predict(x_val)

        # Reshape back to images for appropriate metrics computations
        preds_val = preds_val.reshape((-1, 122500))
        y_val = y_val.reshape((-1, 122500))

        val_metrics = self.calculate_metrics(preds_val, y_val)

        # Debugging step 1: Check shapes and types
        print("Shape of preds_val:", preds_val.shape)
        print("Shape of y_val:", y_val.shape)
        print("Type of preds_val:", type(preds_val))
        print("Type of y_val:", type(y_val))
        
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

        # Reshape images to individual pixels as samples
        x_test = x_test.reshape((-1, 21))
        y_test = y_test.reshape((-1, 1))

        x_test = self.scaler.transform(x_test)

        # Calculate predictions
        preds_test = self.model.predict(x_test)

        # Reshape back to images for appropriate metrics computations
        preds_test = preds_test.reshape((-1, 500, 245)).astype(np.uint8)
        y_test = y_test.reshape((-1, 500, 245)).astype(np.uint8)

        test_metrics = self.calculate_metrics(preds_test, y_test)

        print("Testing metrics:", test_metrics)

        return test_metrics, preds_test, y_test


    def plot_predictions_jpg(self, jpg_path, batch_size=2):
        image_paths = sorted(glob.glob(os.path.join(jpg_path, '*.JPG')))
        if not image_paths:
            print(f"No JPG images found in the directory: {jpg_path}")
            return

        folder_name = os.path.basename(os.path.normpath(jpg_path))
        output_path = os.path.join(jpg_path, f"{folder_name}_masks")
        os.makedirs(output_path, exist_ok=True)

        for batch_start in range(0, len(image_paths), batch_size):
            batch_end = min(batch_start + batch_size, len(image_paths))
            batch_paths = image_paths[batch_start:batch_end]

            x_test = []
            image_shapes = []  # store original image shapes for reshaping preds
            try:
                print(f"Processing images {batch_start + 1} to {batch_end}...")

                # Load and preprocess images in the batch
                for image_path in batch_paths:
                    image_i = self.read_image(image_path, rgb=True)
                    x_test.append(self.preprocess_image(image_i))
                    image_shapes.append(image_i.shape[:2])  # height, width

                x_test = np.array(x_test)
                if x_test.size == 0:
                    print(f"Batch {batch_start + 1} to {batch_end}: No images found or preprocessing failed.")
                    continue

                # Reshape and scale
                x_test = x_test.reshape((-1, 21))
                x_test = self.scaler.transform(x_test)

                preds_test = self.model.predict(x_test)

                # Now split preds_test for each image dynamically
                # preds_test is 1D with length = sum of (h*w) for all images in batch
                start_idx = 0
                for i, (image_path, shape) in enumerate(zip(batch_paths, image_shapes)):
                    h, w = shape
                    length = h * w
                    pred_img_flat = preds_test[start_idx:start_idx + length]
                    start_idx += length

                    pred_img = pred_img_flat.reshape((h, w)).astype(np.uint8)

                    # Convert to binary mask (0 or 255)
                    binary_pred_img = (pred_img > 0.5).astype(np.uint8) * 255

                    pred_image = Image.fromarray(binary_pred_img, mode='L')
                    output_file = os.path.join(output_path, os.path.basename(image_path).replace('.JPG', '_prediction.png'))
                    pred_image.save(output_file, format='PNG')

                print(f"Processed and saved predictions for images {batch_start + 1} to {batch_end}")

            except Exception as e:
                print(f"Error processing images {batch_start + 1} to {batch_end}: {str(e)}")
                continue

        print(f"All predicted images saved to {output_path}")


def rgb2hsi(rgb: np.array):
        # Method according to https://www.vocal.com/video/rgb-and-hsvhsihsl-color-space-conversion/

        h = rgb2hsv(rgb)[:, 0]
        c_min = np.min(rgb, axis=1)
        i = np.mean(rgb, axis=1)
        s = np.where(np.isnan(i), 0, np.where(i != 0, 1 - c_min / i, np.zeros_like(i)))
        # original: s = np.where(i != 0, 1 - c_min / i, np.zeros_like(i))

        return np.stack((h, s, i), axis=1)

def convert_png_to_jpg(png_path='data/prediction_test_output', jpg_path='data/prediction_test_output_jpg'):
    # Create output directory if it doesn't exist
    if not os.path.exists(jpg_path):
        os.makedirs(jpg_path)

    # Read all PNG files in the specified directory
    image_paths = sorted(glob.glob(png_path + '/*.png'))

    for image_path in image_paths:
        # Read the image
        image = Image.open(image_path)

        # Convert the image to RGB (JPEG doesn't support grayscale)
        image = image.convert('RGB')

        # Save the image as JPEG
        output_file = os.path.join(jpg_path, image_path.split('/')[-1].replace('.png', '.jpg'))
        image.save(output_file, format='JPEG', quality=100)

    print(f"Converted PNG images saved to {jpg_path}")

# %%
#import os
#from skimage import io  # Add this line for io module

# Instantiate the model
#model = SadeghiEtAl2017()

# Train the model
#train_metrics, val_metrics = model.train(train_path='data/train', val_path='data/validation')

# %%
# Call the plot_predictions method
#model.plot_predictions_jpg(jpg_path='/home/f60041558@agsad.admin.ch/mnt/eo-nas1/eoa-share/projects/011_experimentEObserver/data/UAV/Sampling/Reckenholz/Eingang/DJI_202407151341_164_ReckenholzEingang')
# %%
