import json
import os
from methods.SegmentationPipeline1 import run_pipeline
from methods.SegmentationPipeline2 import run_segmentation_pipeline

if __name__ == "__main__":
    # Load configuration from JSON file
    with open("config.json", "r") as config_file:
        config = json.load(config_file)

    # Define data paths from the single root directory
    data_root = config["data_path"]

    general_model_path = os.path.join(data_root, "general_model")
    train_path = os.path.join(data_root, "EWS", "train")
    val_path = os.path.join(data_root, "EWS", "validation")
    test_path = os.path.join(data_root, "EWS", "test") 

    # Run pipeline with snapshot images (if available)
    run_pipeline(
        config["input_snapshot_path"],
        config["input_map_file"],
        startlocation=tuple(config["start_location"]),
        flightheight=config["flight_height"],
        zoom=config["zoom"],
        train_path=train_path,
        val_path=val_path
    )

    # Run the second segmentation pipeline
    run_segmentation_pipeline(
        config["input_map_file"],
        config["input_snapshot_path"],
        general_model_path
    )
