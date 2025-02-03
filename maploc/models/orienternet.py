# Copyright (c) Meta Platforms, Inc. and affiliates.

import numpy as np
import torch
from torch.nn.functional import normalize
import torch.nn.functional as F
from torch import nn

from . import get_model
from .base import BaseModel
from .bev_net import BEVNet
from .bev_projection import SimpleCartesianProjection
from .map_encoder import MapEncoder
from .metrics import AngleError, AngleRecall, Location2DError, Location2DRecall
from .voting import (
    TemplateSampler,
    argmax_xy,
    conv2d_fft_batchwise,
    expectation_xy,
    log_softmax_spatial,
    mask_yaw_prior,
    nll_loss_xyr,
    nll_loss_xyr_smoothed,
)
import torchmetrics


class OrienterNet(BaseModel):
    default_conf = {
        "image_encoder": "???",
        "map_encoder": "???",
        "bev_net": "???",
        "latent_dim": "???",
        "matching_dim": "???",
        "scale_range": [0, 9],  # useless?
        "num_scale_bins": "???", # useless?
        "z_min": None,
        "z_max": None,
        "x_max": None,
        "pixel_per_meter": "???",
        "bearing_loss_weight": "???",
        "num_rotations": "???",
        "add_temperature": False,
        "normalize_features": False,
        "padding_matching": "replicate",
        "apply_map_prior": True,
        "do_label_smoothing": False,
        "sigma_xy": 1,
        "sigma_r": 2,
        # depcreated
        # "depth_parameterization": "scale",
        "norm_depth_scores": False,
        "normalize_scores_by_dim": False,
        "normalize_scores_by_num_valid": True,
        "prior_renorm": True,
        "retrieval_dim": None,
    }

    def _init(self, conf):
        # assert not self.conf.norm_depth_scores
        # assert self.conf.depth_parameterization == "scale"
        # assert not self.conf.normalize_scores_by_dim
        # assert self.conf.normalize_scores_by_num_valid
        # assert self.conf.prior_renorm

        Encoder = get_model(conf.image_encoder.get("name", "feature_extractor_v2"))
        self.image_encoder = Encoder(conf.image_encoder.backbone)
        self.map_encoder = MapEncoder(conf.map_encoder)
        self.bev_net = None if conf.bev_net is None else BEVNet(conf.bev_net)
        
        image_out_dim = (
            conf.image_encoder.backbone.output_dim  # If using feature_extractor_v2
            if hasattr(conf.image_encoder.backbone, "output_dim")
            else conf.latent_dim  # Fallback to latent_dim
        )

        ppm = conf.pixel_per_meter
        # self.projection_polar = PolarProjectionDepth(
        #     conf.z_max,
        #     ppm,
        #     conf.scale_range,
        #     conf.z_min,
        # )
        self.projection_bev = SimpleCartesianProjection(
            x_max=conf.x_max,
            ppm=ppm
        )
        self.template_sampler = TemplateSampler(
            self.projection_bev.grid_x, 
            ppm, 
            conf.num_rotations
        )

        if conf.bev_net is None:
            self.feature_projection = torch.nn.Linear(
                conf.latent_dim, conf.matching_dim
            )
        if conf.add_temperature:
            temperature = torch.nn.Parameter(torch.tensor(0.0))
            self.register_parameter("temperature", temperature)
            
        self.bearing_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(image_out_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
            
    def exhaustive_voting(self, f_bev, f_map, valid_bev=None, confidence=None):
        """2D feature matching with rotation"""
        B, C, H, W = f_map.shape
        
        # Normalize features if configured
        if self.conf.normalize_features:
            f_bev = F.normalize(f_bev, dim=1)
            f_map = F.normalize(f_map, dim=1)

        # Compute correlation between BEV features and map features
        # This gives us a similarity score for each position
        scores = torch.einsum('bchw,bcHW->bhwHW', f_bev, f_map)
        
        # Apply confidence weighting if available
        if confidence is not None:
            scores = scores * confidence[:, None, None]
            
        # Apply valid mask if available
        if valid_bev is not None:
            scores = scores * valid_bev[:, None]
            
        # Apply temperature scaling if configured
        if hasattr(self, 'temperature'):
            scores = scores * torch.exp(self.temperature)

        return scores

    # def exhaustive_voting(self, f_bev, f_map, valid_bev, confidence_bev=None):
    #     if self.conf.normalize_features:
    #         f_bev = normalize(f_bev, dim=1)
    #         f_map = normalize(f_map, dim=1)

    #     # Build the templates and exhaustively match against the map.
    #     if confidence_bev is not None:
    #         f_bev = f_bev * confidence_bev.unsqueeze(1)
    #     f_bev = f_bev.masked_fill(~valid_bev.unsqueeze(1), 0.0)
    #     templates = self.template_sampler(f_bev)
    #     with torch.autocast("cuda", enabled=False):
    #         scores = conv2d_fft_batchwise(
    #             f_map.float(),
    #             templates.float(),
    #             padding_mode=self.conf.padding_matching,
    #         )
    #     if self.conf.add_temperature:
    #         scores = scores * torch.exp(self.temperature)

    #     # Reweight the different rotations based on the number of valid pixels in each
    #     # template. Axis-aligned rotation have the maximum number of valid pixels.
    #     valid_templates = self.template_sampler(valid_bev.float()[None]) > (1 - 1e-4)
    #     num_valid = valid_templates.float().sum((-3, -2, -1))
    #     scores = scores / num_valid[..., None, None]
    #     return scores

    def _forward(self, data):
        # Get map features
        f_map = self.map_encoder(data['map'])
        
        # Get image features
        f_image = self.image_encoder(data['image'])
        
        # Project to top-down view using ground truth bearing for training
        f_bev = self.projection_bev(f_image, data['bearing'])
        
        # Add a bearing estimation head
        bearing_pred = self.bearing_head(f_image)  # New bearing estimation
        
        # Match features for position estimation
        scores = self.exhaustive_voting(f_bev, f_map)
        log_probs = log_softmax_spatial(scores)
        
        # Get position and bearing estimates
        with torch.no_grad():
            xy_max = argmax_xy(scores).to(scores)
            xy_avg, _ = expectation_xy(log_probs.exp())
        
        return {
            "scores": scores,
            "log_probs": log_probs,
            "xy_max": xy_max,
            "xy_expectation": xy_avg,
            "bearing_pred": bearing_pred,  # Add bearing prediction
            "features_image": f_image,
            "features_bev": f_bev,
        }

    def loss(self, pred, data):
        """Calculate losses for both position and bearing"""
        # Position loss from scores
        pos_loss = -pred['log_probs'].max(dim=1)[0].mean()
        
        # Bearing loss (circular)
        bearing_loss = circular_loss(
            pred['bearing_pred'].squeeze(-1),
            data['bearing'],
            period=360.0
        )
        
        # Combine losses
        total_loss = pos_loss + self.conf.bearing_loss_weight * bearing_loss
        
        return {
            'total': total_loss,
            'position': pos_loss,
            'bearing': bearing_loss,
        }

    def metrics(self, pred=None, data=None):
        """Calculate metrics for both position and bearing"""
        if pred is None or data is None:
            # Return empty metrics during initialization
            return {
                'position_error': torchmetrics.MeanMetric(),
                'bearing_error': torchmetrics.MeanMetric(),
            }
            
        with torch.no_grad():
            # Position error (in meters)
            pos_error = torch.norm(
                pred['xy_expectation'] - data['position'],
                dim=-1
            )
            
            # Bearing error (in degrees)
            bearing_error = circular_distance(
                pred['bearing_pred'].squeeze(-1),
                data['bearing'],
                period=360.0
            )
        
        return {
            'position_error': pos_error,
            'bearing_error': bearing_error,
        }

def circular_loss(pred, target, period=360.0):
    """Calculate circular MSE loss for bearing"""
    diff = (pred - target + period/2) % period - period/2
    return (diff ** 2).mean()

def circular_distance(pred, target, period=360.0):
    """Calculate absolute circular distance for bearing"""
    diff = (pred - target + period/2) % period - period/2
    return torch.abs(diff)