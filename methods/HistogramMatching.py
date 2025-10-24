import os
import shutil
import numpy as np
from skimage import io, img_as_float
from skimage.util import img_as_ubyte

# Function for histogram matching
def match_histogram(src, ref):
    matched = np.zeros_like(src)
    for channel in range(3):
        src_flat = src[:, :, channel].ravel()
        ref_flat = ref[:, :, channel].ravel()

        src_values, src_counts = np.unique(src_flat, return_counts=True)
        ref_values, ref_counts = np.unique(ref_flat, return_counts=True)

        src_cdf = np.cumsum(src_counts).astype(np.float64) / src_flat.size
        ref_cdf = np.cumsum(ref_counts).astype(np.float64) / ref_flat.size

        interp_values = np.interp(src_cdf, ref_cdf, ref_values)

        matched[:, :, channel] = np.interp(src_flat, src_values, interp_values).reshape(src[:, :, channel].shape)

    return matched

def replace_last_occurrence(text, old, new):
    return new.join(text.rsplit(old, 1))

def process_images(input_folderpath):
    print(f"\n📂 Starting histogram matching in: {input_folderpath}")
    
    # Dynamically detect subfolders
    subdirs = [d for d in os.listdir(input_folderpath) if os.path.isdir(os.path.join(input_folderpath, d))]

    resized_folder = None
    cropped_folder = None

    for subdir in subdirs:
        if 'resized' in subdir.lower():
            resized_folder = os.path.join(input_folderpath, subdir)
        elif 'cropped' in subdir.lower() or 'roicropped' in subdir.lower():
            cropped_folder = os.path.join(input_folderpath, subdir)

    if not resized_folder or not cropped_folder:
        print(f"❌ Error: Could not find resized or cropped directories in {input_folderpath}")
        return

    print(f"✅ Resized folder: {resized_folder}")
    print(f"✅ Cropped folder: {cropped_folder}")

    output_folder = replace_last_occurrence(resized_folder, 'resized', 'histogrammatched')
    os.makedirs(output_folder, exist_ok=True)
    print(f"📁 Output folder created at: {output_folder}")

    # Transfer mask files
    print("\n🔁 Transferring mask files...")
    for file_name in os.listdir(resized_folder):
        if file_name.endswith('_mask_resized.png'):
            source_path = os.path.join(resized_folder, file_name)
            destination_path = os.path.join(output_folder, file_name)
            shutil.copy(source_path, destination_path)
            print(f"  📤 Transferred: {file_name}")

    # Perform histogram matching
    print("\n🎨 Performing histogram matching...")
    for file_name in os.listdir(resized_folder):
        if file_name.endswith('_resized.png') and not file_name.endswith('_mask_resized.png'):
            base_name = file_name.replace('_resized.png', '')
            roi_image_name = base_name + '_ROIcropped.tif'

            resized_image_path = os.path.join(resized_folder, file_name)
            roi_image_path = os.path.join(cropped_folder, roi_image_name)

            if os.path.exists(roi_image_path):
                try:
                    resized_image = img_as_float(io.imread(resized_image_path))
                    roi_image = img_as_float(io.imread(roi_image_path))

                    matched_image = match_histogram(resized_image, roi_image)

                    output_image_path = os.path.join(output_folder, f'{base_name}_matched.png')
                    io.imsave(output_image_path, img_as_ubyte(matched_image))
                    print(f"  ✅ Matched and saved: {output_image_path}")
                except Exception as e:
                    print(f"  ⚠️ Error processing {file_name}: {e}")
            else:
                print(f"  ⚠️ No corresponding ROI image found for: {file_name}")
