import os
import shutil
import random
from PIL import Image

def distribute_images(base_folder, train_ratio=0.7, validation_ratio=0.2, test_ratio=0.1):
    """
    Distributes images and their corresponding masks into train, validation, and test folders based on given ratios.

    Parameters:
    - base_folder: Path to the base folder (e.g., "/path/to/DJI_202405191334_085_ReckenholzEingang/").
    - train_ratio: Proportion of images to be placed in the train folder (default is 0.7).
    - validation_ratio: Proportion of images to be placed in the validation folder (default is 0.2).
    - test_ratio: Proportion of images to be placed in the test folder (default is 0.1).
    """
    
    # Extract the base name of the folder, e.g., "DJI_202405191334_085_ReckenholzEingang"
    folder_name = os.path.basename(os.path.normpath(base_folder))
    
    # Define the path to the folder containing both matched and mask files
    input_folder = os.path.join(base_folder, f"{folder_name}_histogrammatched")

    # Define the output folder
    output_folder = os.path.join(base_folder, f"{folder_name}_resized_histogrammatched_training")

    # Create output directories for train, validation, and test, with simplified names
    output_train_folder = os.path.join(output_folder, "train")
    output_validation_folder = os.path.join(output_folder, "validation")
    output_test_folder = os.path.join(output_folder, "test")

    os.makedirs(output_train_folder, exist_ok=True)
    os.makedirs(output_validation_folder, exist_ok=True)
    os.makedirs(output_test_folder, exist_ok=True)

    # Get list of files in the input folder that are matched images
    input_files = [f for f in os.listdir(input_folder) if f.endswith("_matched.png")]

    # Shuffle the list of files to ensure random assignment
    random.shuffle(input_files)

    # Calculate the number of files for each set based on the ratios
    total_files = len(input_files)
    train_count = int(train_ratio * total_files)
    validation_count = int(validation_ratio * total_files)
    test_count = total_files - train_count - validation_count

    # Split the files into train, validation, and test sets
    train_files = input_files[:train_count]
    validation_files = input_files[train_count:train_count + validation_count]
    test_files = input_files[train_count + validation_count:]

    # Function to copy files to their respective folders
    def copy_files(files, destination_folder):
        for filename in files:
            mask_filename = filename.replace("_matched.png", "_mask_resized.png")
            mask_path = os.path.join(input_folder, mask_filename)
            
            if os.path.exists(mask_path):
                shutil.copy(os.path.join(input_folder, filename), os.path.join(destination_folder, filename))
                shutil.copy(mask_path, os.path.join(destination_folder, mask_filename))
            else:
                print(f"Warning: Mask file {mask_filename} not found.")

    # Copy files to respective folders
    copy_files(train_files, output_train_folder)
    copy_files(validation_files, output_validation_folder)
    copy_files(test_files, output_test_folder)

    print(f"Files have been successfully distributed into '{output_folder}' with train/validation/test splits.")

def random_crop(image, mask, crop_size=(256, 256)):
    """Crop a random 256x256 window from the image and mask."""
    width, height = image.size
    crop_width, crop_height = crop_size

    # Skip images that are smaller than the crop size
    if width < crop_width or height < crop_height:
        print(f"Skipping crop for image. Size: ({width}, {height}) is smaller than the crop size: {crop_size}")
        return None, None  # Return None for both image and mask if they are too small

    x = random.randint(0, width - crop_width)
    y = random.randint(0, height - crop_height)

    image_cropped = image.crop((x, y, x + crop_width, y + crop_height))
    mask_cropped = mask.crop((x, y, x + crop_width, y + crop_height))

    return image_cropped, mask_cropped

def crop_and_save_images(input_folder, output_folder, crop_size=(256, 256)):
    """Crop all _matched and _mask_resized images in a folder and save them with updated filenames."""
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith("_matched.png"):
            image_path = os.path.join(input_folder, filename)
            mask_filename = filename.replace("_matched.png", "_mask_resized.png")
            mask_path = os.path.join(input_folder, mask_filename)

            if os.path.exists(image_path) and os.path.exists(mask_path):
                image = Image.open(image_path)
                mask = Image.open(mask_path)

                cropped_image, cropped_mask = random_crop(image, mask, crop_size)

                # Skip saving if the image or mask was too small and not cropped
                if cropped_image is not None and cropped_mask is not None:
                    # Update filenames
                    new_image_filename = filename.replace("_matched.png", ".png")
                    new_mask_filename = mask_filename.replace("_mask_resized.png", "_mask.png")
                    
                    cropped_image.save(os.path.join(output_folder, new_image_filename))
                    cropped_mask.save(os.path.join(output_folder, new_mask_filename))
                else:
                    print(f"Skipping saving {filename} due to small size.")
            else:
                print(f"Skipping {filename}. Corresponding mask not found.")

# Main function to process the train, validation, and test folders
def process_all_folders(base_folder):
    """Process the train, validation, and test folders to crop images and save them with updated filenames in new folders."""
    
    # Step 1: Distribute the images into train, validation, and test folders
    distribute_images(base_folder)
    
    # Step 2: Crop the images and save them into their respective 256x256 subfolders
    sub_folders = ["train", "validation", "test"]

    for sub_folder in sub_folders:
        input_folder = os.path.join(base_folder, f"{os.path.basename(base_folder)}_resized_histogrammatched_training", sub_folder)
        output_folder = os.path.join(base_folder, f"{os.path.basename(base_folder)}_resized_histogrammatched_training", f"{sub_folder}_256")

        if os.path.exists(input_folder):
            print(f"Processing folder: {input_folder} -> {output_folder}")
            crop_and_save_images(input_folder, output_folder)
        else:
            print(f"Folder {input_folder} does not exist, skipping.")
