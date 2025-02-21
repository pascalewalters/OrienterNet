# Copyright (c) Meta Platforms, Inc. and affiliates.

import json
import os
import shutil
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from ... import DATASETS_PATH, logger
from ...osm.tiling import TileManager
from ..dataset import MapLocDataset
from ..torch import collate, worker_init_fn
from torch.utils.data import Dataset, DataLoader

class YYCDataset(Dataset):
    # dump_filename = "dump.json"
    images_archive = "images.tar.gz"
    images_dirname = "images/"

    default_cfg = {
        "data":{
            **MapLocDataset.default_cfg,
            "name": "yyc",
            # paths and fetch
            "paths": {
                "data_dir": None,
                "osm_dir": None,
                "combined_geojson_path": None,
                "photos_dir": None,
                "mvf_dir": None,
            },
            "local_dir": None,
            "tiles_filename": "tiles.pkl",
            "scenes": "???",
            "split": None,
            "loading": {
                "train": "???",
                "val": "${.test}",
                "test": {"batch_size": 1, "num_workers": 0},
            },
            "filter_for": None,
            "filter_by_ground_angle": None,
            "min_num_points": "???",
        }
    }    
    
    def __init__(self, cfg: Dict[str, Any], stage: str):
        super().__init__()
        # this cfg is orienternet.yaml 
        default_cfg = OmegaConf.create(self.default_cfg)
        OmegaConf.set_struct(default_cfg, False)  # CANNOT add new keys
        self.cfg = OmegaConf.merge(default_cfg, cfg)
        self.stage = stage
            
        self.local_dir = self.cfg.data.local_dir or os.environ.get("TMPDIR")
        self.data_dir = Path(self.cfg.data.paths.data_dir)
            
        if self.local_dir is not None:
            self.local_dir = Path(self.local_dir, "YYC")
        if self.cfg.data.crop_size_meters < self.cfg.data.max_init_error:
            raise ValueError("The ground truth location can be outside the map.")
        
        self.prepare_data()
        self.setup()
        
        # Create the actual dataset using MapLocDataset
        self.dataset = MapLocDataset(
            self.stage,  # Use stage instead of split
            self.cfg.data,
            self.splits[self.stage],  # Use stage
            self.data[self.stage],    # Use stage
            self.image_dirs,
            self.tile_managers,
            image_ext=".jpg"
        )

        
    def prepare_data(self):
        for scene in self.cfg.data.scenes:
            dump_dir = self.data_dir / scene
            # assert (dump_dir / self.dump_filename).exists(), dump_dir
            assert (dump_dir / self.cfg.data.tiles_filename).exists(), dump_dir
            if self.local_dir is None:
                assert (dump_dir / self.images_dirname).exists(), dump_dir
                continue
            # Cache the folder of images locally to speed up reading
            local_dir = self.local_dir / scene
            if local_dir.exists():
                shutil.rmtree(local_dir)
            local_dir.mkdir(exist_ok=True, parents=True)
            images_archive = dump_dir / self.images_archive
            logger.info("Extracting the image archive %s.", images_archive)
            with tarfile.open(images_archive) as fp:
                fp.extractall(local_dir)        
        
    def setup(self):
        self.dumps = {}
        self.tile_managers = {}
        self.image_dirs = {}
        names = []
        
        
        combined_geojson_path = Path(self.cfg.data.paths.combined_geojson_path)
        with open(combined_geojson_path, 'r') as f:
            geojson_data = json.load(f)
            
        
        for scene in self.cfg.data.scenes:
            logger.info("Loading scene %s.", scene)
            dump_dir = self.data_dir / scene
            
            self.dumps[scene] = {}
            
            for feature in geojson_data['features']:
                if f"f_{feature['properties']['map']}" == scene:
                    image_id = feature['properties']['imageUrl'].replace('.png', '')
                    self.dumps[scene][image_id] = {
                        'latitude': feature['geometry']['coordinates'][1],
                        'longitude': feature['geometry']['coordinates'][0],
                        'bearing': float(feature['properties']['bearing'])
                    }
                    names.append((scene, image_id))
                    
            
            
            logger.info("Loading map tiles %s.", self.cfg.data.tiles_filename)
            self.tile_managers[scene] = TileManager.load(
                dump_dir / self.cfg.data.tiles_filename
            )
            groups = self.tile_managers[scene].groups
            
            # TODO: ensure num_classes set to only wall for current dataset
            if self.cfg.data.num_classes:  # check consistency
                if set(groups.keys()) != set(self.cfg.data.num_classes.keys()):
                    raise ValueError(
                        "Inconsistent groups: "
                        f"{groups.keys()} {self.cfg.data.num_classes.keys()}"
                    )
                for k in groups:
                    if len(groups[k]) != self.cfg.data.num_classes[k]:
                        raise ValueError(
                            f"{k}: {len(groups[k])} vs {self.cfg.data.num_classes[k]}"
                        )
            ppm = self.tile_managers[scene].ppm
            if ppm != self.cfg.data.pixel_per_meter:
                raise ValueError(
                    "The tile manager and the config/model have different ground "
                    f"resolutions: {ppm} vs {self.cfg.data.pixel_per_meter}"
                )
            
            

            self.image_dirs[scene] = (
                (self.local_dir or self.data_dir) / scene / self.images_dirname
            )
            assert self.image_dirs[scene].exists(), self.image_dirs[scene]

        print(f"Total number of images found: {len(names)}")

        self.splits = self.parse_splits(self.cfg.data.split, names)
        # if self.cfg.filter_for is not None:
        #     self.filter_elements()
        self.data = self.pack_data()
        
    def parse_splits(self, split_arg, names):
        """Parse dataset splits based on configuration"""
        if split_arg is None:
            return {
                "train": names,
                "val": names,
            }
        elif isinstance(split_arg, int):
            names = np.random.RandomState(self.cfg.seed).permutation(names).tolist()
            return {
                "train": names[split_arg:],
                "val": names[:split_arg],
            }
        elif isinstance(split_arg, DictConfig):
            scenes_val = set(split_arg.val)
            scenes_train = set(split_arg.train)
            assert len(scenes_val - set(self.cfg.data.scenes)) == 0
            assert len(scenes_train - set(self.cfg.data.scenes)) == 0
            return {
                "train": [n for n in names if n[0] in scenes_train],
                "val": [n for n in names if n[0] in scenes_val],
            }
        elif isinstance(split_arg, str):
            with (self.data_dir / split_arg).open("r") as fp:
                splits = json.load(fp)
            
            splits = {
                k: {f"f_{loc}": set(ids) for loc, ids in split.items()}
                for k, split in splits.items()
            }
            
            result = {}
            for k, split in splits.items():
                matching_names = []
                for n in names:
                    scene_id, image_id = n
                    image_id = image_id.replace('.jpg', '')
                    if scene_id in split and image_id in split[scene_id]:
                        matching_names.append(n)
                result[k] = matching_names
            return result
        else:
            raise ValueError(f"Invalid split argument: {split_arg}")


    def pack_data(self):
        """Pack data into tensors for efficient loading"""
        stage_data = {}
        
        for stage, names in self.splits.items():
            stage_data[stage] = {
                'latitude': [],
                'longitude': [],
                'bearing': []
            }
            
            for scene, image_id in names:
                image_data = self.dumps[scene][image_id]
                stage_data[stage]['latitude'].append(image_data['latitude'])
                stage_data[stage]['longitude'].append(image_data['longitude'])
                stage_data[stage]['bearing'].append(image_data['bearing'])
                
            # Convert to tensors
            for k in stage_data[stage]:
                stage_data[stage][k] = torch.tensor(stage_data[stage][k])
        
        return stage_data        

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
    
def create_dataloader(dataset, cfg, split):
    """Helper function to create dataloaders"""
    return DataLoader(
        dataset,
        batch_size=cfg.data.loading[split].batch_size,
        num_workers=cfg.data.loading[split].num_workers,
        shuffle=(split == 'train'),
        pin_memory=True,
        persistent_workers=cfg.data.loading[split].num_workers > 0,
        worker_init_fn=worker_init_fn,
        collate_fn=collate,
    )
