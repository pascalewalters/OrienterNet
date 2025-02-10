# Copyright (c) Meta Platforms, Inc. and affiliates.

import json
import os
import shutil
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils.data as torchdata
from omegaconf import DictConfig, OmegaConf

from ... import DATASETS_PATH, logger
from ...osm.tiling import TileManager
from ..dataset import MapLocDataset
from ..torch import collate, worker_init_fn


class MapillaryDataModule(pl.LightningDataModule):
    dump_filename = "dump.json"
    images_archive = "images.tar.gz"
    images_dirname = "images/"

    default_cfg = {
        **MapLocDataset.default_cfg,
        "name": "mapillary",
        # paths and fetch
        "data_dir": DATASETS_PATH / "YYC",
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

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        default_cfg = OmegaConf.create(self.default_cfg)
        OmegaConf.set_struct(default_cfg, True)  # cannot add new keys
        self.cfg = OmegaConf.merge(default_cfg, cfg)
        self.root = Path(self.cfg.data_dir)
        self.local_dir = self.cfg.local_dir or os.environ.get("TMPDIR")
        if self.local_dir is not None:
            self.local_dir = Path(self.local_dir, "MGL")
        if self.cfg.crop_size_meters < self.cfg.max_init_error:
            raise ValueError("The ground truth location can be outside the map.")

    def prepare_data(self):
        for scene in self.cfg.scenes:
            dump_dir = self.root / scene
            assert (dump_dir / self.dump_filename).exists(), dump_dir
            assert (dump_dir / self.cfg.tiles_filename).exists(), dump_dir
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

    def setup(self, stage: Optional[str] = None):
        self.dumps = {}
        self.tile_managers = {}
        self.image_dirs = {}
        names = []
        
        
        geojson_path = "/home/kevinmeng/workspace/mappedin/VPS/Mappedin_VPS_Data-20250127T163206Z-001/Mappedin_VPS_Data/YYC_VPS/combined-output.geojson"
        with open(geojson_path, 'r') as f:
            geojson_data = json.load(f)
            
        
        for scene in self.cfg.scenes:
            logger.info("Loading scene %s.", scene)
            dump_dir = self.root / scene
            
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
                    
            
            
            logger.info("Loading map tiles %s.", self.cfg.tiles_filename)
            self.tile_managers[scene] = TileManager.load(
                dump_dir / self.cfg.tiles_filename
            )
            groups = self.tile_managers[scene].groups
            
            # TODO: ensure num_classes set to only wall for current dataset
            if self.cfg.num_classes:  # check consistency
                if set(groups.keys()) != set(self.cfg.num_classes.keys()):
                    raise ValueError(
                        "Inconsistent groups: "
                        f"{groups.keys()} {self.cfg.num_classes.keys()}"
                    )
                for k in groups:
                    if len(groups[k]) != self.cfg.num_classes[k]:
                        raise ValueError(
                            f"{k}: {len(groups[k])} vs {self.cfg.num_classes[k]}"
                        )
            ppm = self.tile_managers[scene].ppm
            if ppm != self.cfg.pixel_per_meter:
                raise ValueError(
                    "The tile manager and the config/model have different ground "
                    f"resolutions: {ppm} vs {self.cfg.pixel_per_meter}"
                )
            
            

            self.image_dirs[scene] = (
                (self.local_dir or self.root) / scene / self.images_dirname
            )
            assert self.image_dirs[scene].exists(), self.image_dirs[scene]

        print(f"Total number of images found: {len(names)}")

        self.parse_splits(self.cfg.split, names)
        # if self.cfg.filter_for is not None:
        #     self.filter_elements()
        self.pack_data()

    def pack_data(self):
        # We pack the data into compact tensors
        # that can be shared across processes without copy.
        stage_data = {}  # Create a temporary dictionary
        
        # print(f"SPLITS ITEMS: {self.splits.items()}")
        
        for stage, names in self.splits.items():
            print(f"Processing stage: {stage} with {len(names)} samples")
            
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

            print(f"Packed {stage} data with {len(stage_data[stage]['latitude'])} samples")


        save_path = "stage_data_structure.txt"
        with open(save_path, 'w') as f:
            f.write("Stage Data Structure:\n\n")
            for stage in stage_data:
                f.write(f"\nStage: {stage}\n")
                f.write("-" * 50 + "\n")
                for key, value in stage_data[stage].items():
                    f.write(f"{key}:\n")
                    f.write(f"  Type: {type(value)}\n")
                    f.write(f"  Shape: {value.shape}\n")
                    f.write(f"  First few values: {value[:5]}\n\n")

        self.data = stage_data
        
        print(f"Final data structure keys: {self.data.keys()}")
        for stage in self.data:
            print(f"Data for {stage}: {self.data[stage].keys()}")


    def parse_splits(self, split_arg, names):
        if split_arg is None:
            self.splits = {
                "train": names,
                "val": names,
            }
        elif isinstance(split_arg, int):
            names = np.random.RandomState(self.cfg.seed).permutation(names).tolist()
            self.splits = {
                "train": names[split_arg:],
                "val": names[:split_arg],
            }
        elif isinstance(split_arg, DictConfig):
            scenes_val = set(split_arg.val)
            scenes_train = set(split_arg.train)
            assert len(scenes_val - set(self.cfg.scenes)) == 0
            assert len(scenes_train - set(self.cfg.scenes)) == 0
            self.splits = {
                "train": [n for n in names if n[0] in scenes_train],
                "val": [n for n in names if n[0] in scenes_val],
            }
        elif isinstance(split_arg, str):
            with (self.root / split_arg).open("r") as fp:
                splits = json.load(fp)
                
            print(f"Loaded splits from {split_arg}: {list(splits.keys())}")
                
            splits = {
                k: {f"f_{loc}": set(ids) for loc, ids in split.items()}
                for k, split in splits.items()
            }
            
            # print(f"Transformed splits structure: {splits}")
            
            self.splits = {}
            for k, split in splits.items():
                matching_names = []
                for n in names:
                    scene_id, image_id = n
                    
                    image_id = image_id.replace('.jpg', '')
                    
                    if scene_id in split and image_id in split[scene_id]:
                        matching_names.append(n)
                self.splits[k] = matching_names
        else:
            raise ValueError(split_arg)

    def dataset(self, stage: str):
        return MapLocDataset(
            stage,
            self.cfg,
            self.splits[stage],
            self.data[stage],
            self.image_dirs,
            self.tile_managers,
            image_ext=".jpg",
        )

    def dataloader(
        self,
        stage: str,
        shuffle: bool = False,
        num_workers: int = None,
        sampler: Optional[torchdata.Sampler] = None,
    ):
        dataset = self.dataset(stage)
        cfg = self.cfg["loading"][stage]
        num_workers = cfg["num_workers"] if num_workers is None else num_workers
        loader = torchdata.DataLoader(
            dataset,
            batch_size=cfg["batch_size"],
            num_workers=num_workers,
            shuffle=shuffle or (stage == "train"),
            pin_memory=True,
            persistent_workers=num_workers > 0,
            worker_init_fn=worker_init_fn,
            collate_fn=collate,
            sampler=sampler,
        )
        return loader

    def train_dataloader(self, **kwargs):
        return self.dataloader("train", **kwargs)

    def val_dataloader(self, **kwargs):
        return self.dataloader("val", **kwargs)

    def test_dataloader(self, **kwargs):
        return self.dataloader("test", **kwargs)