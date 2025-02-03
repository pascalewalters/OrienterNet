import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np

from ...osm.viz import GeoPlotter

class MVFDataset:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.manifest = self._load_manifest()
        self.floorstack = self._load_floorstack()
        self.floors = self._load_floors()
        self.spaces = self._load_spaces()
        self.nodes = self._load_nodes()
        self.connections = self._load_connections()
        self.entrances = self._load_features('entrance')
        self.obstructions = self._load_features('obstruction')
        self.windows = self._load_windows()

    def _load_manifest(self) -> Dict:
        with open(self.data_dir / 'manifest.geojson', 'r') as f:
            return json.load(f)

    def _load_floorstack(self) -> Dict:
        with open(self.data_dir / 'floorstack.json', 'r') as f:
            return json.load(f)

    def _load_floors(self) -> Dict:
        floors = {}
        floor_file = self.data_dir / 'floor.geojson'
        if floor_file.exists():
            with open(floor_file, 'r') as f:
                floors = json.load(f)
        return floors

    def _load_spaces(self) -> Dict:
        spaces = {}
        space_dir = self.data_dir / 'space'
        for space_file in space_dir.glob('m_*.geojson'):
            with open(space_file, 'r') as f:
                spaces[space_file.stem] = json.load(f)
        return spaces

    def _load_nodes(self) -> Dict:
        with open(self.data_dir / 'node.geojson', 'r') as f:
            return json.load(f)

    def _load_connections(self) -> Dict:
        with open(self.data_dir / 'connection.json', 'r') as f:
            return json.load(f)

    def _load_features(self, feature_type: str) -> Dict:
        features = {}
        feature_dir = self.data_dir / feature_type
        if feature_dir.exists():
            for feature_file in feature_dir.glob('m_*.geojson'):
                with open(feature_file, 'r') as f:
                    features[feature_file.stem] = json.load(f)
        return features

    def _load_windows(self) -> Dict:
        windows = {}
        window_dir = self.data_dir / 'window'
        if window_dir.exists():
            for window_file in window_dir.glob('m_*.geojson'):
                with open(window_file, 'r') as f:
                    windows[window_file.stem] = json.load(f)
        return windows

def prepare_mvf_data(
    data_dir: Path,
    output_dir: Path,
    tile_margin: int = 512,
    ppm: int = 2,
):
    """Prepare MVF data for training.
    
    Args:
        data_dir: Directory containing MVF data
        output_dir: Directory to save processed data
        tile_margin: Margin around the building in pixels
        ppm: Pixels per meter for rasterization
    """
    # Load MVF dataset
    dataset = MVFDataset(data_dir)
    
    # Get building center point from manifest
    center_point = dataset.manifest['features'][0]['geometry']['coordinates']
    
    # Create visualization
    plotter = GeoPlotter()
    plotter.points(np.array([center_point]), "red", name="Building Center")
    
    # Process each floor
    for floor in dataset.floors['features']:
        floor_id = floor['properties']['id']
        
        # Collect all geometries for this floor
        geometries = {
            'spaces': [space for space in dataset.spaces.values()
                      if space['properties']['floorId'] == floor_id],
            'nodes': [node for node in dataset.nodes['features']
                     if node['properties']['floorId'] == floor_id],
            'connections': [connection for connection in dataset.connections.values()
                            if connection['properties']['floorId'] == floor_id],
            'windows': [window for window in dataset.windows['features']
                        if window['properties']['floorId'] == floor_id],
            'obstructions': [obstruction for obstruction in dataset.obstructions['features']]
            # Add other relevant geometries...
        }
        
        # Create floor-specific visualizations and processing
        # TODO: Add floor-specific processing logic
        
    plotter.fig.write_html(output_dir / "mvf_visualization.html")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--pixel_per_meter', type=int, default=2)
    args = parser.parse_args()

    args.output_dir.mkdir(exist_ok=True, parents=True)
    prepare_mvf_data(args.data_dir, args.output_dir, ppm=args.pixel_per_meter)

if __name__ == "__main__":
    main()