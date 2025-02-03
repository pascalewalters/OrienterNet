# Copyright (c) Meta Platforms, Inc. and affiliates.

import collections
import collections.abc
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils.data as torchdata
from omegaconf import OmegaConf

from ... import YYC_DATASET_PATH, logger
from ...osm.tiling import TileManager
from ..dataset import MapLocDataset
from ..torch import collate, worker_init_fn
import json

class YYCDataset(torchdata.Dataset):
    def __init__(self, stage, cfg, image_paths, data, tile_managers):
        self.stage = stage
        self.cfg = cfg
        self.image_paths = image_paths
        self.data = data
        self.tile_managers = tile_managers

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Get basic data
        scene_name, map_id, image_name = self.image_paths[idx]
        data = {k: v[idx] for k, v in self.data.items()}
        
        # Get the correct tile manager for this image
        tile_manager = self.tile_managers[map_id]
        
        # Extract position and bearing
        position = data['position_and_bearing'][:2]  # [x, y]
        bearing = data['position_and_bearing'][2]    # rotation angle in degrees

        # Calculate crop boundaries
        crop_size = self.cfg.crop_size_meters
        bbox = BoundaryBox(
            position - crop_size/2,
            position + crop_size/2
        )
        
        # Get map crop using the correct tile manager
        map_crop = tile_manager.query(bbox)

        # Load and process image
        image_path = self.cfg.data_dir / 'photos' / image_name  # data_dir should be YYC_VPS
        image = read_image(image_path)
        image = self.transforms(image)

        # Add noise to position and bearing during training
        if self.stage == 'train':
            position = self.add_position_noise(position)
            bearing = self.add_bearing_noise(bearing)

        return {
            'image': image,
            'map': torch.from_numpy(map_crop.raster).long(),
            'position': torch.from_numpy(position).float(),
            'bearing': torch.tensor(bearing).float(),
            'map_id': map_id,
            'image_name': image_name
        }

    # def add_position_noise(self, position):
    #     """Add random noise to position during training"""
    #     if self.stage != 'train':
    #         return position
    #     noise = np.random.uniform(
    #         -self.cfg.max_init_error,
    #         self.cfg.max_init_error,
    #         size=2
    #     )
    #     return position + noise

    # def add_bearing_noise(self, bearing):
    #     """Add random noise to bearing during training"""
    #     if self.stage != 'train':
    #         return bearing
    #     noise = np.random.uniform(
    #         -self.cfg.max_init_error_rotation,
    #         self.cfg.max_init_error_rotation
    #     )
    #     return bearing + noise




class YYCDataModule(pl.LightningDataModule):
    default_cfg = {
        "name": "yyc",
        "data_dir": YYC_DATASET_PATH,
        "tiles_dir": "tiles",
        "geojson_file": "combined-output.geojson",
        "splits": {
            "train": 0.95,
            "val": 0.05,
        },
        "loading": {
            "train": {"batch_size": 32, "num_workers": 4},
            "val": {"batch_size": 1, "num_workers": 0},
        },
        "crop_size_meters": 64,
        "max_init_error": 20,  # meters
        "max_init_error_rotation": 10,  # degrees
    }

    def __init__(self, cfg, tile_manager: Optional[TileManager] = None):
        super().__init__()
        default_cfg = OmegaConf.create(self.default_cfg)
        OmegaConf.set_struct(default_cfg, True)  # cannot add new keys
        self.cfg = OmegaConf.merge(default_cfg, cfg)
        self.root = Path(self.cfg.data_dir)

    def prepare_data(self):
        if not (self.root.exists()):
            raise FileNotFoundError(f"Cannot find the YYC dataset at {self.root}")

    def setup(self, stage: Optional[str] = None):
        if stage == "fit":
            stages = ["train", "val"]
        elif stage is None:
            stages = ["train", "val"]
        else:
            stages = [stage]
            
        # Load geojson data if not already loaded
        if not hasattr(self, 'geojson_data'):
            with open(self.root / self.cfg.geojson_file) as f:
                self.geojson_data = json.load(f)
                
        # Load tile managers if not already loaded
        if not hasattr(self, 'tile_managers'):
            self.tile_managers = {}
            for feature in self.geojson_data['features']:
                map_id = feature['properties']['map']
                if map_id not in self.tile_managers:
                    tile_path = self.root / self.cfg.tiles_dir / f"{map_id}_tiles.pkl"
                    self.tile_managers[map_id] = TileManager.load(tile_path)

        # Split data if not already split
        if not hasattr(self, 'splits'):
            all_features = self.geojson_data['features']
            n_samples = len(all_features)
            indices = np.random.permutation(n_samples)
            
            n_train = int(n_samples * self.cfg.splits['train'])
            
            self.splits = {
                'train': indices[:n_train],
                'val': indices[n_train:],
            }

        # Pack data for each stage
        self.pack_data(stages)

    def pack_data(self, stages):
        for stage in stages:
            data = defaultdict(list)
            image_paths = []
            
            # Process features for this stage
            for idx in self.splits[stage]:
                feature = self.geojson_data['features'][idx]
                d = self.get_picture_data(feature)
                
                for k, v in d.items():
                    if k != 'image_url' and k != 'map_id':
                        data[k].append(v)
                
                # Store image path and map_id
                map_id = feature['properties']['map']
                image_name = feature['properties']['imageUrl']
                image_paths.append(('yyc', map_id, image_name))

            # Convert lists to numpy arrays/tensors
            for k in data:
                data[k] = torch.from_numpy(np.stack(data[k]))

            self.data[stage] = data
            self.image_paths[stage] = np.array(image_paths)

    def get_picture_data(self, feature):
        """Extract data from a GeoJSON feature"""
        coords = feature['geometry']['coordinates']
        properties = feature['properties']
        
        # Convert GPS coordinates to local coordinates using tile manager
        map_id = properties['map']
        tile_manager = self.tile_managers[map_id]
        local_xy = tile_manager.projection.project(
            np.array([coords[1], coords[0]])  # lat, lon
        )
        
        # Combine position and bearing
        position_and_bearing = np.array([
            local_xy[0],
            local_xy[1],
            float(properties['bearing'])
        ], dtype=np.float32)

        return {
            'position_and_bearing': position_and_bearing,
        }

    def dataset(self, stage: str) -> YYCDataset:
        return YYCDataset(
            stage,
            self.cfg,
            self.image_paths[stage],
            self.data[stage],
            self.tile_managers
        )

    def train_dataloader(self):
        return self._get_dataloader('train', shuffle=True)

    def val_dataloader(self):
        return self._get_dataloader('val', shuffle=False)

    def _get_dataloader(self, stage: str, shuffle: bool):
        dataset = self.dataset(stage)
        cfg = self.cfg.loading[stage]
        
        return torchdata.DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            shuffle=shuffle,
            pin_memory=True,
            persistent_workers=cfg.num_workers > 0,
            worker_init_fn=worker_init_fn,
            collate_fn=collate,
        )