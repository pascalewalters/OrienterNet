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

# Set up logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
    for feature in floor_features:
        coords = feature['geometry']['coordinates']
        all_latlon.append([coords[1], coords[0]])
        image_locations.append([coords[1], coords[0]])            
    
            
    if not all_latlon:
        raise ValueError(f"No lat/lon coordinates found in {osm_data}")
    
    all_latlon = np.array(all_latlon)
    
    projection = Projection.from_points(all_latlon)
    all_xy = projection.project(all_latlon)
    bbox_map = BoundaryBox(all_xy.min(0), all_xy.max(0)) + tile_margin


    # Create visualization for this floor
    plotter = GeoPlotter()
    # Plot OSM points in red
    n_osm_points = len(osm_data['elements'])
    if n_osm_points > 0:
        plotter.points(all_latlon[:n_osm_points], "red", name="OSM Points")
    # Plot image locations in green
    if image_locations:
        plotter.points(np.array(image_locations), "green", name="Image Locations")
    plotter.bbox(projection.unproject(bbox_map), "blue", "Map Boundary")
    plotter.fig.write_html(f"yyc_data_visualization_{floor_id}.html")

    tile_manager = TileManager.from_bbox(
        projection,
        bbox_map,
        ppm,
        path=osm_path,
    )
    
    # Create floor-specific output directory
    floor_output_path = output_path / floor_id
    floor_output_path.mkdir(parents=True, exist_ok=True)    

    # Ensure output_path is a file path ending in .pkl
    tile_manager.save(floor_output_path / 'tiles.pkl')
    
    # Create train/val/test splits for this floor
    # SPLITS ARE CURRENTLY NOT CREATED FROM HERE
    # create_data_splits(floor_output_path, {'features': floor_features})
    
    return tile_manager

def create_data_splits(data_dir, geojson_data):
    """Create JSON files for train/val/test splits"""
    features = geojson_data['features']
    n_samples = len(features)
    
    # Create random indices for splits
    indices = np.random.permutation(n_samples)
    n_train = int(n_samples * 0.95)  # 70% train
    n_val = int(n_samples * 0.05)   # 15% val
    
    splits = {
        'train': indices[:n_train],
        'val': indices[n_train:n_train+n_val],
    }
    
    # Save split indices
    splits_dir = data_dir / 'splits'
    splits_dir.mkdir(exist_ok=True)
    
    for split_name, split_indices in splits.items():
        with open(splits_dir / f'{split_name}_indices.json', 'w') as f:
            json.dump({
                'indices': split_indices.tolist(),
                'total_samples': n_samples
            }, f)
            
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", 
        type=Path, 
        default=Path(YYCDataModule.default_cfg["data_dir"])
    )
    parser.add_argument("--pixel_per_meter", type=int, default=2)
    parser.add_argument("--osm_dir", 
                        type=Path, 
                        default="/home/kevinmeng/workspace/mappedin/VPS/OrienterNet/yyc_osms")
    parser.add_argument("--geojson_path", 
                       type=Path,
                       default="/home/kevinmeng/workspace/mappedin/VPS/Mappedin_VPS_Data-20250127T163206Z-001/Mappedin_VPS_Data/YYC_VPS/combined-output.geojson"
                    )
    args = parser.parse_args()

    args.data_dir.mkdir(exist_ok=True, parents=True)
    
    # Process each OSM file in the directory
    for osm_file in args.osm_dir.glob('*_osm.json'):
        # Extract floor ID from filename
        floor_id = osm_file.stem.replace('_osm', '')
        
        logger.info(f"Processing floor: {floor_id}")
        
        # Create floor-specific output directory
        floor_data_dir = args.data_dir / floor_id
        floor_data_dir.mkdir(exist_ok=True, parents=True)
        
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