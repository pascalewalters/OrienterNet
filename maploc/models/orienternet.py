# Copyright (c) Meta Platforms, Inc. and affiliates.

import numpy as np
import torch
from torch.nn.functional import normalize, softmax, mse_loss
import torchmetrics

from . import get_model
from .base import BaseModel
from .bev_net import BEVNet
from .bev_projection import CartesianProjection, PolarProjectionDepth
from .map_encoder import MapEncoder
from .metrics import AngleError, AngleRecall, Location2DError, Location2DRecall
from .voting import (
    TemplateSampler,
    argmax_xyr,
    conv2d_fft_batchwise,
    expectation_xyr,
    log_softmax_spatial,
    mask_yaw_prior,
    nll_loss_xyr,
    nll_loss_xyr_smoothed,
)


class OrienterNet(BaseModel):
    default_conf = {
        "image_encoder": "???",
        "map_encoder": "???",
        "bev_net": "???",
        "latent_dim": "???",
        "matching_dim": "???",
        "scale_range": [0, 9],
        "num_scale_bins": "???",
        "z_min": None,
        "z_max": "???",
        "x_max": "???",
        "pixel_per_meter": "???",
        "num_rotations": "???",
        "add_temperature": False,
        "normalize_features": False,
        "padding_matching": "replicate",
        "apply_map_prior": True,
        "do_label_smoothing": False,
        "sigma_xy": 1,
        "sigma_r": 2,
        # depcreated
        "depth_parameterization": "scale",
        "norm_depth_scores": False,
        "normalize_scores_by_dim": False,
        "normalize_scores_by_num_valid": True,
        "prior_renorm": True,
        "retrieval_dim": None,
    }

    def _init(self, conf):
        assert not self.conf.norm_depth_scores
        assert self.conf.depth_parameterization == "scale"
        assert not self.conf.normalize_scores_by_dim
        assert self.conf.normalize_scores_by_num_valid
        assert self.conf.prior_renorm

        Encoder = get_model(conf.image_encoder.get("name", "feature_extractor_v2"))
        self.image_encoder = Encoder(conf.image_encoder.backbone)
        self.map_encoder = MapEncoder(conf.map_encoder)
        self.bev_net = None if conf.bev_net is None else BEVNet(conf.bev_net)

        ppm = conf.pixel_per_meter
        self.projection_polar = PolarProjectionDepth(
            conf.z_max,
            ppm,
            conf.scale_range,
            conf.z_min,
        )
        self.projection_bev = CartesianProjection(
            conf.z_max, conf.x_max, ppm, conf.z_min
        )
        self.template_sampler = TemplateSampler(
            self.projection_bev.grid_xz, ppm, conf.num_rotations
        )

        self.scale_classifier = torch.nn.Linear(conf.latent_dim, conf.num_scale_bins)
        if conf.bev_net is None:
            self.feature_projection = torch.nn.Sequential(
                torch.nn.Dropout(p=conf.regularization.dropout_rate),
                torch.nn.Linear(conf.latent_dim, conf.matching_dim),
                (
                    torch.nn.BatchNorm1d(conf.matching_dim)
                    if conf.regularization.feature_norm
                    else torch.nn.Identity()
                ),
            )
        if conf.add_temperature:
            temperature = torch.nn.Parameter(torch.tensor(0.0))
            self.register_parameter("temperature", temperature)

    def exhaustive_voting(self, f_bev, f_map, valid_bev, confidence_bev=None):
        if self.conf.normalize_features:
            f_bev = normalize(f_bev, dim=1)
            f_map = normalize(f_map, dim=1)

        # Build the templates and exhaustively match against the map.
        if confidence_bev is not None:
            f_bev = f_bev * confidence_bev.unsqueeze(1)
        f_bev = f_bev.masked_fill(~valid_bev.unsqueeze(1), 0.0)
        templates = self.template_sampler(f_bev)
        with torch.autocast("cuda", enabled=False):
            scores = conv2d_fft_batchwise(
                f_map.float(),
                templates.float(),
                padding_mode=self.conf.padding_matching,
            )
        if self.conf.add_temperature:
            scores = scores * torch.exp(self.temperature)

        # Reweight the different rotations based on the number of valid pixels in each
        # template. Axis-aligned rotation have the maximum number of valid pixels.
        valid_templates = self.template_sampler(valid_bev.float()[None]) > (1 - 1e-4)
        # Add small epsilon to prevent division by zero
        num_valid = valid_templates.float().sum((-3, -2, -1)) + 1e-6
        scores = scores / num_valid[..., None, None]
        return scores

    def _forward(self, data):
        pred = {}
        pred_map = pred["map"] = self.map_encoder(data)
        f_map = pred_map["map_features"][0]

        # Extract image features.
        level = 0
        f_image = self.image_encoder(data)["feature_maps"][level]
        camera = data["camera"].scale(1 / self.image_encoder.scales[level])
        camera = camera.to(data["image"].device, non_blocking=True)

        # Estimate the monocular priors.
        pred["pixel_scales"] = scales = self.scale_classifier(f_image.moveaxis(1, -1))
        f_polar = self.projection_polar(f_image, scales, camera)

        # Map to the BEV.
        with torch.autocast("cuda", enabled=False):
            f_bev, valid_bev, _ = self.projection_bev(
                f_polar.float(), None, camera.float()
            )
        pred_bev = {}
        if self.conf.bev_net is None:
            # channel last -> classifier -> channel first
            f_bev = self.feature_projection(f_bev.moveaxis(1, -1)).moveaxis(-1, 1)
        else:
            pred_bev = pred["bev"] = self.bev_net({"input": f_bev})
            f_bev = pred_bev["output"]

        # Normalize features before matching
        if self.conf.regularization.feature_norm:
            f_bev = torch.nn.functional.normalize(f_bev, dim=1)
            f_map = torch.nn.functional.normalize(f_map, dim=1)

        scores = self.exhaustive_voting(
            f_bev, f_map, valid_bev, pred_bev.get("confidence")
        )
        scores = scores.moveaxis(1, -1)  # B,H,W,N
        if "log_prior" in pred_map and self.conf.apply_map_prior:
            scores = scores + pred_map["log_prior"][0].unsqueeze(-1)
        # pred["scores_unmasked"] = scores.clone()
        if "map_mask" in data:
            # scores.masked_fill_(~data["map_mask"][..., None], -np.inf)
            # Ensure at least one valid prediction per batch
            map_mask = data["map_mask"]
            # Broadcast mask to match scores shape
            if map_mask.shape[-2:] != scores.shape[1:3]:  # Compare H,W dimensions
                map_mask = (
                    torch.nn.functional.interpolate(
                        map_mask.float().unsqueeze(1),  # Add channel dim
                        size=scores.shape[1:3],  # Target H,W
                        mode="nearest",
                    )
                    .squeeze(1)
                    .bool()
                )  # Remove channel dim and convert back to bool
            # if map_mask.sum(dim=(1, 2)) == 0:
            #     # If mask is empty, don't apply it
            #     print("Warning: Empty map mask detected")
            # else:
            scores.masked_fill_(
                ~map_mask[..., None], -1e4
            )  # Use finite value instead of -inf

        if "yaw_prior" in data:
            mask_yaw_prior(scores, data["yaw_prior"], self.conf.num_rotations)
        log_probs = log_softmax_spatial(scores)
        with torch.no_grad():
            uvr_max = argmax_xyr(scores).to(scores)
            uvr_avg, _ = expectation_xyr(log_probs.exp())

        return {
            **pred,
            "scores": scores,
            "log_probs": log_probs,
            "uvr_max": uvr_max,
            "uv_max": uvr_max[..., :2],
            "yaw_max": uvr_max[..., 2],
            "uvr_expectation": uvr_avg,
            "uv_expectation": uvr_avg[..., :2],
            "yaw_expectation": uvr_avg[..., 2],
            "features_image": f_image,
            "features_bev": f_bev,
            "valid_bev": valid_bev.squeeze(1),
        }

    def loss(self, pred, data):
        xy_gt = data["uv"]
        yaw_gt = data["roll_pitch_yaw"][..., -1]
        # Check if predictions are valid
        if torch.isnan(pred["log_probs"]).any():
            print("Warning: NaN values in log_probs")
            # Replace NaN with large negative value
            pred["log_probs"] = torch.nan_to_num(pred["log_probs"], nan=-1e4)

        if self.conf.do_label_smoothing:
            # Add small epsilon to mask to prevent complete masking
            mask = data.get("map_mask")
            if mask is not None:
                mask = mask | (mask.sum(dim=(1, 2), keepdim=True) == 0)
            nll = nll_loss_xyr_smoothed(
                pred["log_probs"],
                xy_gt,
                yaw_gt,
                self.conf.sigma_xy / self.conf.pixel_per_meter,
                self.conf.sigma_r,
                mask=data.get("map_mask"),
            )
        else:
            nll = nll_loss_xyr(pred["log_probs"], xy_gt, yaw_gt)

        # Clip loss to prevent explosion
        nll = torch.clamp(nll, max=1e4)

        # l2_loss = 0
        # for param in self.parameters():
        #     l2_loss += torch.norm(param, p=2)

        # Add auxiliary losses for intermediate predictions
        # aux_loss = 0
        # if "features_bev" in pred:
        #     aux_loss += mse_loss(
        #         pred["features_bev"], pred["map"], reduction="mean"
        #     )

        # total_loss = nll + self.conf.regularization.l2_weight * l2_loss

        # # Clip gradients
        # torch.nn.utils.clip_grad_norm_(
        #     self.parameters(), self.conf.regularization.gradient_clip
        # )

        # loss = {"total": total_loss, "nll": nll, "l2": l2_loss}
        loss = {"nll": nll, "total": nll}
        if self.training and self.conf.add_temperature:
            loss["temperature"] = self.temperature.expand(len(nll))
        # loss["aux"] = aux_loss
        return loss

    def metrics(self):
        base_metrics = {
            "xy_max_error": Location2DError("uv_max", self.conf.pixel_per_meter),
            "xy_expectation_error": Location2DError(
                "uv_expectation", self.conf.pixel_per_meter
            ),
            "yaw_max_error": AngleError("yaw_max"),
            "xy_recall_2m": Location2DRecall(2.0, self.conf.pixel_per_meter, "uv_max"),
            "xy_recall_5m": Location2DRecall(5.0, self.conf.pixel_per_meter, "uv_max"),
            "yaw_recall_2°": AngleRecall(2.0, "yaw_max"),
            "yaw_recall_5°": AngleRecall(5.0, "yaw_max"),
        }

        # Add debug metrics
        return {
            **base_metrics,
            # "debug_uv_max": _debug_metric("uv_max"),
            # "debug_uv_gt": _debug_metric("uv"),
            # "debug_yaw_max": _debug_metric("yaw_max"),
            # "debug_yaw_gt": _debug_metric("roll_pitch_yaw", lambda x: x[..., -1]),
        }

    def validation_step(self, data):
        pred = self.forward(data)
        loss = self.loss(pred, data)

        # Debug predictions vs ground truth
        with torch.no_grad():
            print("\nValidation Step Debug:")
            print(f"Predicted UV max: {pred['uv_max'][0]}")
            print(f"Ground truth UV: {data['uv'][0]}")
            print(
                f"UV error (pixels): {torch.norm(pred['uv_max'][0] - data['uv'][0], p=2)}"
            )
            print(
                f"UV error (meters): {torch.norm(pred['uv_max'][0] - data['uv'][0], p=2) / self.conf.pixel_per_meter}"
            )

            # Check if predictions are within mask
            if "map_mask" in data:
                valid_pred = data["map_mask"][
                    0, int(pred["uv_max"][0, 1]), int(pred["uv_max"][0, 0])
                ]
                print(f"Prediction within mask: {valid_pred}")

        return {**loss, **self._compute_metrics(pred, data)}


class _debug_metric(torchmetrics.Metric):
    """Helper to debug metric calculations"""

    full_state_update = True

    def __init__(self, key, transform=None):
        super().__init__()
        self.add_state("value", default=[], dist_reduce_fx="cat")
        self.key = key
        self.transform = transform

    def update(self, pred, data):
        values = data[self.key] if self.key in data else pred[self.key]
        if self.transform is not None:
            values = self.transform(values)
        self.value.append(values)

    def compute(self):
        if not self.value:  # Check if list is empty
            return torch.tensor(0.0, device=self.device)

        # Debug output
        values = torch.cat(self.value)
        print(f"\nDebug {self.key}:")
        print(f"Shape: {values.shape}")
        print(f"First 2 values:\n{values[:2]}")
        print(f"Min: {values.min():.3f}, Max: {values.max():.3f}")
        print(f"Mean: {values.mean():.3f}, Std: {values.std():.3f}")

        return values.mean()
