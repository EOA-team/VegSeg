import os
import pandas as pd
from PIL import Image, ExifTags
from PIL.ExifTags import TAGS, GPSTAGS
from fractions import Fraction
import pyproj
import requests

def create_metadata_file(folderpath):
    if not os.path.isdir(folderpath):
        raise ValueError(f"The provided path '{folderpath}' is not a valid directory.")
    
    folder_name = os.path.basename(os.path.normpath(folderpath))
    metadata_file_name = f"{folder_name}_metadata.csv"
    metadata_file_path = os.path.join(folderpath, metadata_file_name)
    
    df = pd.DataFrame()
    df.to_csv(metadata_file_path, index=False)
    
    print(f"Metadata file created at: {metadata_file_path}")

def get_exif_data(image_path):
    try:
        with Image.open(image_path) as image:
            exif_data = image._getexif()
            if exif_data is not None:
                return {
                    ExifTags.TAGS.get(tag_id, tag_id): value
                    for tag_id, value in exif_data.items()
                }
            else:
                print(f"No EXIF data found in {image_path}")
                return None
    except Exception as e:
        print(f"Failed to extract EXIF data from {image_path}: {e}")
        return None

def get_gps_info(exif_data):
    if exif_data is None:
        return None, None

    gps_info = exif_data.get('GPSInfo', None)
    if gps_info is None:
        print("No GPS info found")
        return None, None

    gps_data = {
        GPSTAGS.get(tag_id, tag_id): value
        for tag_id, value in gps_info.items()
    }

    def convert_to_degrees(value):
        d = float(value[0].numerator) / float(value[0].denominator)
        m = float(value[1].numerator) / float(value[1].denominator)
        s = float(value[2].numerator) / float(value[2].denominator)
        return d + (m / 60.0) + (s / 3600.0)
    
    lat = gps_data.get('GPSLatitude', None)
    lon = gps_data.get('GPSLongitude', None)
    lat_ref = gps_data.get('GPSLatitudeRef', None)
    lon_ref = gps_data.get('GPSLongitudeRef', None)
    
    if lat and lon and lat_ref and lon_ref:
        latitude = convert_to_degrees(lat)
        longitude = convert_to_degrees(lon)
        
        if lat_ref != 'N':
            latitude = -latitude
        if lon_ref != 'E':
            longitude = -longitude
            
        return latitude, longitude
    else:
        return None, None

def convert_wgs84_to_lv95(latitude, longitude):
    wgs84 = pyproj.CRS('EPSG:4326')
    lv95 = pyproj.CRS('EPSG:2056')
    transformer = pyproj.Transformer.from_crs(wgs84, lv95, always_xy=True)
    easting, northing = transformer.transform(longitude, latitude)
    return easting, northing

def get_elevation(latitude, longitude):
    easting, northing = convert_wgs84_to_lv95(latitude, longitude)
    url = 'https://api3.geo.admin.ch/rest/services/height'
    full_url = f"{url}?easting={easting}&northing={northing}"

    response = requests.get(full_url)
    if response.status_code == 200:
        data = response.json()
        elevation = float(data['height'])
        return elevation
    else:
        print(f"Failed to retrieve elevation data: {response.status_code} - {response.reason}")
        return None

def get_elevation_from_metadata(metadata):
    elevations = []
    for index, row in metadata.iterrows():
        latitude = row['latitude_WGS84']
        longitude = row['longitude_WGS84']
        
        if latitude is not None and longitude is not None:
            easting, northing = convert_wgs84_to_lv95(latitude, longitude)
            url = 'https://api3.geo.admin.ch/rest/services/height'
            full_url = f"{url}?easting={easting}&northing={northing}"

            response = requests.get(full_url)
            if response.status_code == 200:
                data = response.json()
                elevation = float(data['height'])
                elevations.append(elevation)
            else:
                print(f"Failed to retrieve data for index {index}: {response.status_code} - {response.reason}")
                elevations.append(None)
        else:
            elevations.append(None)
    return elevations

def get_startlocation(flightnumber):
    startlocations_path = 'path_to_file'
    try:
        df_startlocations = pd.read_excel(startlocations_path, engine='openpyxl')
    except ImportError as e:
        raise ImportError("Missing optional dependency 'openpyxl'. Install it via pip or conda.") from e
    
    match = df_startlocations[df_startlocations['Flightnumber'] == int(flightnumber)]
    if not match.empty:
        start_location = match.iloc[0]['Startlocation']
        lat, lon = map(float, start_location.split(','))
        return lat, lon
    else:
        return None, None

def calculate_and_add_adjusted_dimensions(flight_height_m, zoom=10):
    sensor_width_mm = 6.4
    sensor_height_mm = 4.8
    effective_focal_length_mm = 31.15
    image_width_pixel = 4000
    image_height_pixel = 3000

    if zoom <= 7:
        digital_zoom = 1
    else:
        digital_zoom = zoom / 7  # dynamic zoom factor

    GSD_width = (sensor_width_mm * flight_height_m) / (effective_focal_length_mm * image_width_pixel)
    GSD_height = (sensor_height_mm * flight_height_m) / (effective_focal_length_mm * image_height_pixel)

    width_m = image_width_pixel * GSD_width
    height_m = image_height_pixel * GSD_height

    adj_width_m = width_m / digital_zoom
    adj_height_m = height_m / digital_zoom

    return adj_width_m, adj_height_m

def update_metadata_file(folderpath, startlocation=None, flightheight=None, zoom=None):
    if not os.path.isdir(folderpath):
        raise ValueError(f"The provided path '{folderpath}' is not a valid directory.")

    folder_name = os.path.basename(os.path.normpath(folderpath))
    metadata_file_name = f"{folder_name}_metadata.csv"
    metadata_file_path = os.path.join(folderpath, metadata_file_name)

    image_files = [f for f in os.listdir(folderpath) if f.endswith('.JPG')]

    # Try to parse folder name, but fall back to None or placeholders if it fails
    parts = folder_name.split('_')
    flightstart = parts[1] if len(parts) > 1 else None
    flightnumber = parts[2] if len(parts) > 2 else None
    site, field = (('_'.join(parts[3:])).split('_', 1) if len(parts) > 3 else (None, None))

    if startlocation:
        start_latitude, start_longitude = startlocation
        print(f"Using manually provided start location: {start_latitude}, {start_longitude}")
    elif flightnumber:
        start_latitude, start_longitude = get_startlocation(flightnumber)
        print(f"Using start location from Excel: {start_latitude}, {start_longitude}")
    else:
        start_latitude, start_longitude = None, None
        print("No valid flightnumber found. Skipping start location lookup.")

    start_elevation = get_elevation(start_latitude, start_longitude) if start_latitude and start_longitude else None

    if start_elevation is not None and flightheight is not None:
        start_flight_height = start_elevation + flightheight
        print(f"Computed StartFlightHeight = {start_elevation} + {flightheight} = {start_flight_height} m")
    else:
        start_flight_height = None
        print("StartFlightHeight not set")

    metadata = []
    for image in image_files:
        parts = image.split('_')
        imagedate = parts[1] if len(parts) > 1 else None
        imagenumber = parts[2] if len(parts) > 2 else None

        image_path = os.path.join(folderpath, image)
        exif_data = get_exif_data(image_path)
        latitude, longitude = get_gps_info(exif_data)

        if latitude is not None and longitude is not None:
            easting, northing = convert_wgs84_to_lv95(latitude, longitude)
            elevation = None  # Will fill later
        else:
            latitude, longitude, easting, northing, elevation = [None] * 5

        metadata.append({
            'imagename': image,
            'Flightstart': flightstart,
            'Flightnumber': flightnumber,
            'Site': site,
            'Field': field,
            'Startlocation': f"{start_latitude},{start_longitude}" if start_latitude and start_longitude else None,
            'StartFlightHeight': start_flight_height,
            'imagedate': imagedate,
            'imagenumber': imagenumber,
            'latitude_WGS84': latitude,
            'longitude_WGS84': longitude,
            'easting_LV95': easting,
            'northing_LV95': northing,
            'elevation': elevation
        })

    df = pd.DataFrame(metadata)
    df['elevation'] = get_elevation_from_metadata(df)

    if start_flight_height is not None:
        df['FlightHeight'] = (start_flight_height - df['elevation']).clip(lower=1)
    else:
        df['FlightHeight'] = None

    adjusted_dimensions = df['FlightHeight'].apply(lambda fh: calculate_and_add_adjusted_dimensions(fh, zoom) if pd.notnull(fh) else (None, None))
    df['ImageWidth'] = adjusted_dimensions.apply(lambda x: x[0])
    df['ImageHeight'] = adjusted_dimensions.apply(lambda x: x[1])

    df.to_csv(metadata_file_path, index=False)
    print(f"Metadata file updated at: {metadata_file_path}")

#update_metadata_file(folderpath, startlocation=47.429470,8.514512, flightheight=10, zoom=10)
