"""
main.py

Entry point for the VegSeg RGB vegetation segmentation pipeline.

This script runs two segmentation pipelines on UAV-captured RGB orthomosaic images,
optionally using local high-resolution snapshot images for model training or applying
a pretrained general model.

Usage:
    Edit the input paths and optional metadata below, then run:
        python main.py
"""

from methods.SegmentationPipeline1 import run_pipeline
from methods.SegmentationPipeline2 import run_segmentation_pipeline

if __name__ == "__main__":
    # Define input paths and parameters here

    # Path to the RGB orthomosaic GeoTIFF file.
    # Must be in WGS84 coordinate system (EPSG:4326).
    input_map_file = "/home/f60041558@agsad.admin.ch/mnt/eo-nas1/eoa-share/projects/011_experimentEObserver/data/Final/Segmentation_HR/Test2/mapping/re112o_20250602_mapping.tif"

    # Path to the folder containing high-resolution snapshot images.
    # Optional: if you do not have snapshot images or want to use the pretrained general model,
    # set this to None.
    input_snapshot_path = "/home/f60041558@agsad.admin.ch/mnt/eo-nas1/eoa-share/projects/011_experimentEObserver/data/Final/Segmentation_HR/Test2/snapshot_20250602"

    # Optional metadata used only if snapshot images are provided:
    # - startlocation: UAV takeoff GPS coordinates (latitude, longitude)
    # - flightheight: UAV flight altitude in meters
    # - zoom: Camera zoom level specific to DJI Mavic 3E
    startlocation = (47.430487, 8.522988)
    flightheight = 40
    zoom = 7 

    # Run the first segmentation pipeline, which uses snapshot images for training if provided.
    run_pipeline(input_snapshot_path, input_map_file, startlocation=startlocation, flightheight=flightheight, zoom=zoom)

    # Run the second segmentation pipeline, typically for inference or further segmentation.
    run_segmentation_pipeline(input_map_file, input_snapshot_path)
