import json
from methods.SegmentationPipeline1 import run_pipeline
from methods.SegmentationPipeline2 import run_segmentation_pipeline

if __name__ == "__main__":
    # Load configuration from JSON file
    with open("config.json", "r") as config_file:
        config = json.load(config_file)

    # Run pipeline with snapshot images (if available)
    run_pipeline(
        config["input_snapshot_path"],
        config["input_map_file"],
        startlocation=tuple(config["start_location"]),
        flightheight=config["flight_height"],
        zoom=config["zoom"],
        train_path=config["train_data_path"],
        val_path=config["val_data_path"]
    )

    # Run the second segmentation pipeline, typically for inference or further segmentation
    run_segmentation_pipeline(
        config["input_map_file"],
        config["input_snapshot_path"],
        config["general_model_path"]
    )