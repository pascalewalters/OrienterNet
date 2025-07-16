# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModel
from .feature_extractor import FeatureExtractor


class MapEncoder(BaseModel):
    default_conf = {
        "embedding_dim": "???",
        "output_dim": None,
        "num_classes": "???",
        "backbone": "???",
        "unary_prior": False,
    }

    def _init(self, conf):
        # self.embeddings = torch.nn.ModuleDict(
        #     {
        #         k: torch.nn.Embedding(n + 1, conf.embedding_dim)
        #         for k, n in conf.num_classes.items()
        #     }
        # )

        self.num_classes = conf.num_classes
        num_classes = sum(conf.num_classes.values())
        self.embeddings = torch.nn.Embedding(num_classes + 1, conf.embedding_dim)

        # input_dim = len(conf.num_classes) * conf.embedding_dim
        input_dim = conf.embedding_dim
        output_dim = conf.output_dim
        if output_dim is None:
            output_dim = conf.backbone.output_dim
        if conf.unary_prior:
            output_dim += 1
        if conf.backbone is None:
            self.encoder = nn.Sequential(
                nn.Conv2d(input_dim, output_dim, 1),
                nn.BatchNorm2d(output_dim),
                nn.ReLU(inplace=True),
            )
        elif conf.backbone == "simple":
            self.encoder = nn.Sequential(
                nn.Conv2d(input_dim, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, output_dim, 3, padding=1),
                nn.BatchNorm2d(output_dim),
                nn.ReLU(inplace=True),
            )
        else:
            self.encoder = FeatureExtractor(
                {
                    **conf.backbone,
                    "input_dim": input_dim,
                    "output_dim": output_dim,
                }
            )

    def _forward(self, data):
        # Put the areas down first
        segmentation_map = data["map"][:, 0].detach()
        ways_map = data["map"][:, 1].detach()
        nodes_map = data["map"][:, 2].detach()
        segmentation_map = segmentation_map * (ways_map == 0) + (
            ways_map + self.num_classes["areas"]
        ) * (ways_map > 0)
        segmentation_map = segmentation_map * (nodes_map == 0) + (
            nodes_map + self.num_classes["areas"] + self.num_classes["ways"]
        ) * (nodes_map > 0)

        embeddings = self.embeddings(segmentation_map)  # [B, H, W, E]
        combined = embeddings.permute(0, 3, 1, 2)  # [B, E, H, W]

        # Get encoder features
        if isinstance(self.encoder, BaseModel):
            features = self.encoder({"image": combined})["feature_maps"]
        else:
            features = [self.encoder(combined)]

        return {"map_features": features}
