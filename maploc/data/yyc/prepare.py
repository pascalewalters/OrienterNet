# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm
import json
from maploc.osm.tiling import TileManager
from maploc.utils.geo import BoundaryBox, Projection
from maploc.data.yyc.dataset import YYCDataModule
from maploc.osm.viz import GeoPlotter
import logging
import os
import shutil
from collections import defaultdict
import random

# Set up logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_val_split(geojson_path, output_path):
    with open(geojson_path, 'r') as f:
        data = json.load(f)

    images_by_map = defaultdict(list)
    
    # Group images by their map ID
    for feature in data['features']:
        map_id = feature['properties']['map']
        image_name = feature['properties']['imageUrl'].replace('.jpg', '')
        images_by_map[map_id].append(image_name)
    
    split_data = {
        "train": {},
        "val": {}
    }
    
    for map_id, images in images_by_map.items():
        random.shuffle(images)
        
        split_idx = int(len(images) * 0.95) # bad split but maximize train for now, and can manually inspect model performance at first? 
        
        split_data["train"][map_id] = images[:split_idx]
        split_data["val"][map_id] = images[split_idx:]
    
    # Save splits to output directory
    splits_path = output_path / 'image_splits.json'
    with open(splits_path, 'w') as f:
        json.dump(split_data, f, indent=2)

def split_photos_by_map(geojson_path, photos_dir, output_path):
    """Organize photos into floor-specific directories."""
    # Read and parse the geojson file
    with open(geojson_path, 'r') as f:
        data = json.load(f)

    # Dictionary to keep track of photos count per map
    map_counts = {}

    # Process each feature
    for feature in data['features']:
        map_id = feature['properties']['map']
        image_name = feature['properties']['imageUrl']
        floor_id = f"f_{map_id}"
        
        # Create images directory within floor directory
        images_dir = output_path / floor_id / 'images'
        images_dir.mkdir(exist_ok=True, parents=True)

        # Source and destination paths
        source_path = Path(photos_dir) / image_name
        dest_path = images_dir / image_name

        # Copy the file if it exists
        if source_path.exists():
            shutil.copy2(source_path, dest_path)
            map_counts[floor_id] = map_counts.get(floor_id, 0) + 1
        else:
            logger.warning(f"Image not found: {image_name}")
            
    logger.info("\nPhoto organization complete!")
    logger.info("\nStatistics:")
    logger.info("-----------")
    for floor_id, count in map_counts.items():
        logger.info(f"Floor {floor_id}: {count} photos")
    logger.info(f"\nTotal floors: {len(map_counts)}")
    logger.info(f"Total photos organized: {sum(map_counts.values())}")


def prepare_osm(
    osm_path,
    geojson_path,
    data_dir,
    output_path,
    floor_id,
    tile_margin=512,
    ppm=2,
):
    with open(osm_path, 'r') as f:
        osm_data = json.load(f)
        
    # Load image locations from GeoJSON
    with open(geojson_path, 'r') as f:
        geojson_data = json.load(f)
        
    floor_features = [
        feature for feature in geojson_data['features']
        if f"f_{feature['properties']['map']}" == f"{floor_id}"
    ]
    
    if not floor_features:
        logger.warning(f"No features found for floor {floor_id}")
        return None
        
    all_latlon = []
    for element in osm_data['elements']:
        if 'lat' in element and 'lon' in element:
            all_latlon.append([element['lat'], element['lon']])
            
            
    image_locations = []
    outside_locations = []
    for feature in floor_features:
        coords = feature['geometry']['coordinates']
        all_latlon.append([coords[1], coords[0]])
        if coords[1] > osm_data['bounds']['maxlat'] or coords[1] < osm_data['bounds']['minlat']:
            print(f"Image: {feature['properties']['imageUrl']} LATITUDE: {coords[1]} outside of bounds: max: {osm_data['bounds']['maxlat']} min: {osm_data['bounds']['minlat']}")
        if coords[0] > osm_data['bounds']['maxlon'] or coords[0] < osm_data['bounds']['minlon']:
            print(f"Image: {feature['properties']['imageUrl']} LONGITUDE: {coords[0]} outside of bounds: max: {osm_data['bounds']['maxlon']} min: {osm_data['bounds']['minlon']}")
        if coords[1] > osm_data['bounds']['maxlat'] or coords[1] < osm_data['bounds']['minlat'] or coords[0] > osm_data['bounds']['maxlon'] or coords[0] < osm_data['bounds']['minlon']:
            outside_locations.append([coords[1], coords[0]])
            continue
        image_locations.append([coords[1], coords[0]])            
    
            
    if not all_latlon:
        raise ValueError(f"No lat/lon coordinates found in {osm_data}")
    
    all_latlon = np.array(all_latlon)
    
    projection = Projection.from_points(all_latlon)
    all_xy = projection.project(all_latlon)
    
    bbox_map = BoundaryBox(all_xy.min(0), all_xy.max(0)) + 64 # tile_margin


    # Create visualization for this floor
    plotter = GeoPlotter()
    # Plot OSM points in red
    n_osm_points = len(osm_data['elements'])
    if n_osm_points > 0:
        plotter.points(all_latlon[:n_osm_points], "red", name="OSM Points")
    # Plot image locations in green
    if image_locations:
        plotter.points(np.array(image_locations), "green", name="Image Locations")
    if outside_locations:
        plotter.points(np.array(outside_locations), "blue", name="outside")
    plotter.bbox(projection.unproject(bbox_map), "blue", "Map Boundary")
    plotter.fig.write_html(f"yyc_data_visualization_{floor_id}.html")

    tile_manager = TileManager.from_bbox(
        projection,
        bbox_map,
        ppm,
        path=osm_path,
    )
    
    floor_output_path = output_path / floor_id 
    floor_output_path.mkdir(exist_ok=True, parents=True)

    tile_manager.save(floor_output_path / 'tiles.pkl')
    
    
    return tile_manager
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", 
        type=Path, 
        default="./datasets/YYC"
    )
    parser.add_argument("--pixel_per_meter", type=int, default=2)
    parser.add_argument("--osm_dir", 
                        type=Path, 
                        default="/home/kevinmeng/workspace/mappedin/VPS/OrienterNet/yyc_osms")
    parser.add_argument("--geojson_path", 
                       type=Path,
                       default="/home/kevinmeng/workspace/mappedin/VPS/Mappedin_VPS_Data-20250127T163206Z-001/Mappedin_VPS_Data/YYC_VPS/combined-output.geojson"
                    )
    parser.add_argument("--photos_dir",
                        type=Path,
                        default="/home/kevinmeng/workspace/mappedin/VPS/Mappedin_VPS_Data-20250127T163206Z-001/Mappedin_VPS_Data/YYC_VPS/photos"
                    )
    
    args = parser.parse_args()
    
    logger.info("Organizing photos into respective floor ID folder")
    split_photos_by_map(args.geojson_path, args.photos_dir, args.data_dir)
    
    logger.info("Creating train/val splits...")
    train_val_split(args.geojson_path, args.data_dir)

    
    # Process each OSM file in the directory
    for osm_file in args.osm_dir.glob('*_osm.json'):
        # Extract floor ID from filename
        floor_id = osm_file.stem.replace('_osm', '')
        
        logger.info(f"Processing floor: {floor_id}")
        
        # Create floor-specific output directory
        floor_data_dir = args.data_dir / floor_id
        
        if not osm_file.exists():
            raise FileNotFoundError(f"No OSM data file at {osm_file}")
        if not args.geojson_path.exists():
            raise FileNotFoundError(f"No GeoJSON file at {args.geojson_path}")
        
        
        prepare_osm(
            osm_file,
            args.geojson_path,
            floor_data_dir,
            args.data_dir,
            floor_id,
            ppm=args.pixel_per_meter
        )
    
    logger.info("Done! Created tiles.pkl and split files for all floors.")