import os
from datetime import datetime
import joblib  # To save and load the trained model

# Importing necessary functions from the modules
from .Metadata import update_metadata_file
from .Segmentation_prediction_Sadeghi import SadeghiEtAl2017
from .FeatureMatching import process_folder as feature_matching_process_folder
from .Resize import process_images as resize_process_image
from .HistogramMatching import process_images as histogram_matching_process_images
from .TrainingPreparation import process_all_folders as training_prep_process_all_folders

# Get the directory of the current file (this Python script)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Model saving/loading path inside the same directory as this script
model_file_path = os.path.join(current_dir, 'SadeghiEtAl2017_model.pkl')


def try_parse_date(date_str):
    """
    Attempt to parse the date using various formats.
    """
    formats = ['%Y%m%d', '%Y%m%d%H%M']
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None

def find_closest_map(rgb_folder, map_folder):
    """
    Find the map file in the map_folder that matches the exact date of the RGB folder's date.
    Assumes that the folder or file name contains a date in the format 'YYYYMMDD' or 'YYYYMMDDHHMM'.
    """
    rgb_folder_name = os.path.basename(rgb_folder)
    if "_" not in rgb_folder_name:
        print(f"Folder name does not contain a valid date: {rgb_folder_name}")
        return None

    try:
        rgb_date_str = rgb_folder_name.split('_')[1][:8]  # Extract 'YYYYMMDD' part
    except IndexError:
        print(f"Error parsing RGB folder name: {rgb_folder_name}")
        return None

    rgb_date = try_parse_date(rgb_date_str)
    if rgb_date is None:
        print(f"Could not parse date from RGB folder: {rgb_folder_name}")
        return None

    matching_map = None
    for map_file in os.listdir(map_folder):
        if map_file.endswith('.tif'):
            try:
                map_date_str = map_file.split('_')[0][:8]  # Assume 'YYYYMMDD'
            except IndexError:
                print(f"Error parsing map file name: {map_file}")
                continue

            map_date = try_parse_date(map_date_str)
            if map_date and map_date.date() == rgb_date.date():
                matching_map = map_file
                break

    if matching_map:
        return os.path.join(map_folder, matching_map)
    else:
        print(f"No matching map found for RGB folder: {rgb_folder_name}")
        return None

def load_or_train_model(train_path='data/train', val_path='data/validation'):
    try:
        if os.path.exists(model_file_path):
            print("Loading existing trained model.")
            model = joblib.load(model_file_path)
        else:
            print("Training model for the first time...")
            model = SadeghiEtAl2017()
            train_metrics, val_metrics = model.train(train_path=train_path, val_path=val_path)
            print(f"Model training completed: Train Metrics: {train_metrics}, Val Metrics: {val_metrics}")
            joblib.dump(model, model_file_path)  # Save model in the same folder as this script
            print(f"Model saved to {model_file_path}")
        return model
    except Exception as e:
        print(f"Error during model loading/training: {e}")
        raise



def run_pipeline(input_RGB, input_map_file, startlocation=None, flightheight=None, zoom=None,
                 train_path='data/train', val_path='data/validation'):
    if not input_RGB:
        print("No RGB input path provided. Skipping pipeline1 and applying general model.")
        return

    print(f"Starting pipeline for RGB: {input_RGB} and Map File: {input_map_file}")

    input_map = input_map_file
    if not os.path.isfile(input_map):
        print(f"Map file does not exist: {input_map}")
        return

    try:
        model = load_or_train_model(train_path=train_path, val_path=val_path)
    except Exception as e:
        print("Aborting pipeline due to model issue.")
        return

    #try:
        #print("Running prediction...")
        #model.plot_predictions_jpg(input_RGB)
        #print("Prediction complete.")
    #except Exception as e:
        #print(f"Prediction error: {e}")
        #return

    try:
        update_metadata_file(input_RGB, startlocation, flightheight, zoom)
        print("Metadata updated.")
    except Exception as e:
        print(f"Metadata update error: {e}")
        return

    try:
        feature_matching_process_folder(input_RGB, input_map)
        print("Feature matching done.")
    except Exception as e:
        print(f"Feature matching error: {e}")
        return

    try:
        resize_process_image(input_RGB)
        print("Image resizing done.")
    except Exception as e:
        print(f"Resizing error: {e}")
        return

    try:
        histogram_matching_process_images(input_RGB)
        print("Histogram matching done.")
    except Exception as e:
        print(f"Histogram matching error: {e}")
        return

    try:
        training_prep_process_all_folders(input_RGB)
        print("Training prep done.")
    except Exception as e:
        print(f"Training prep error: {e}")
        return

    print("Pipeline executed successfully.")


