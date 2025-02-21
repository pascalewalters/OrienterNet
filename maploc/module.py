# Copyright (c) Meta Platforms, Inc. and affiliates.

from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf, open_dict
from torchmetrics import MeanMetric, MetricCollection

from . import logger
from .models import get_model
import yaml

class AverageKeyMeter(MeanMetric):
    def __init__(self, key, *args, **kwargs):
        self.key = key
        super().__init__(*args, **kwargs)

    def update(self, dict):
        value = dict[self.key]
        value = value[torch.isfinite(value)]
        return super().update(value)

class ONGenericModule(nn.Module):
    def __init__(self, config_path):
        super().__init__()
        
        # this config is default orienternet.yaml
        self.config = config_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")
        
        # only get configs if not training on clearml
        self.clearml = self.config.get("clearml", False)
        if not self.clearml:
            with open(config_path, "r") as file:
                self.config = yaml.safe_load(file)
                
        name = self.config["model"]["name"]
        self.model = get_model(name)(self.config["model"])
        
        self.metrics_val = {}
        for name, metric in self.model.metrics().items():
            self.metrics_val[name] = metric.to(self.device)  # Move metric to device
            
            
        self.train_losses = {}
        self.val_losses = None
        self.to(self.device)
        
    def forward(self, batch):
        pred = self.model(batch)
        # Ensure prediction is on the same device as the input batch
        pred = {k: v.to(batch['map'].device) if torch.is_tensor(v) else v 
               for k, v in pred.items()}
        return pred
    
    def reset_train_losses(self):
        self.train_losses = {}
    
    def training_step(self, batch):
        self.train()
        pred = self(batch)
        losses = self.model.loss(pred, batch)
        
        if not self.train_losses:
            self.train_losses = {k: [] for k in losses.keys()}
        for k, v in losses.items():
            self.train_losses[k].append(v.mean().detach())
        
        return losses["total"].mean()
    
    def get_epoch_train_metrics(self):
        epoch_losses = {}
        for k, v in self.train_losses.items():
            epoch_losses[f"loss/{k}/train"] = torch.stack(v).mean()
        self.reset_train_losses()  # Reset for next epoch
        return epoch_losses   
    
    
    def validation_step(self, batch):
        self.eval()
        with torch.no_grad():
            pred = self(batch)
            losses = self.model.loss(pred, batch)
            
            # Initialize loss meters if not exists
            if self.val_losses is None:
                self.val_losses = {
                    k: AverageKeyMeter(k).to(self.device) for k in losses  # Move to device
                }
            
            # Update metrics
            for metric in self.metrics_val.values():
                metric(pred, batch)
            
            # Update losses
            for meter in self.val_losses.values():
                meter.update(losses)
            
            return losses["total"].mean()
        
        
    def get_validation_metrics(self):
        metrics_dict = {}
        
        # Get metric values
        for name, metric in self.metrics_val.items():
            metrics_dict[f"val/{name}"] = metric.compute()
            metric.reset()
        
        # Get loss values
        if self.val_losses is not None:
            for name, meter in self.val_losses.items():
                metrics_dict[f"loss/{name}/val"] = meter.compute()
                meter.reset()
            
        self.val_losses = None
        return metrics_dict

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.training.lr)
        
        cfg_scheduler = self.config.training.get("lr_scheduler")
        if cfg_scheduler is not None:
            scheduler = getattr(torch.optim.lr_scheduler, cfg_scheduler.name)(
                optimizer=optimizer, **cfg_scheduler.get("args", {})
            )
            return optimizer, scheduler
        return optimizer       
            
    

# class GenericModule(pl.LightningModule):
#     def __init__(self, cfg):
#         super().__init__()
#         name = cfg.model.get("name")
#         name = "orienternet" if name in ("localizer_bev_depth", None) else name
#         self.model = get_model(name)(cfg.model)
#         self.cfg = cfg
#         self.save_hyperparameters(cfg)
#         self.metrics_val = MetricCollection(self.model.metrics(), prefix="val/")
#         self.losses_val = None  # we do not know the loss keys in advance

#     def forward(self, batch):
#         return self.model(batch)

#     def training_step(self, batch):
#         pred = self(batch)
#         losses = self.model.loss(pred, batch)
#         self.log_dict(
#             {f"loss/{k}/train": v.mean() for k, v in losses.items()},
#             prog_bar=False,
#             rank_zero_only=True,
#             on_step=False,
#             on_epoch=True
#         )
#         return losses["total"].mean()

#     def validation_step(self, batch, batch_idx):
#         pred = self(batch)
#         losses = self.model.loss(pred, batch)
#         if self.losses_val is None:
#             self.losses_val = MetricCollection(
#                 {k: AverageKeyMeter(k).to(self.device) for k in losses},
#                 prefix="loss/",
#                 postfix="/val",
#             )
#         self.metrics_val(pred, batch)
#         self.log_dict(self.metrics_val, 
#                       sync_dist=True,
#                       on_step=False,
#                       on_epoch=True)
#         self.losses_val.update(losses)
#         self.log_dict(self.losses_val, 
#                       sync_dist=True,
#                       on_step=False,
#                       on_epoch=True)

#     def validation_epoch_start(self, batch):
#         self.losses_val = None

#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.training.lr)
#         ret = {"optimizer": optimizer}
#         cfg_scheduler = self.cfg.training.get("lr_scheduler")
#         if cfg_scheduler is not None:
#             scheduler = getattr(torch.optim.lr_scheduler, cfg_scheduler.name)(
#                 optimizer=optimizer, **cfg_scheduler.get("args", {})
#             )
#             ret["lr_scheduler"] = {
#                 "scheduler": scheduler,
#                 "interval": "epoch",
#                 "frequency": 1,
#                 "monitor": "loss/total/val",
#                 "strict": True,
#                 "name": "learning_rate",
#             }
#         return ret

#     @classmethod
#     def load_from_checkpoint(
#         cls,
#         checkpoint_path,
#         map_location=None,
#         hparams_file=None,
#         strict=True,
#         cfg=None,
#         find_best=False,
#     ):
#         assert hparams_file is None, "hparams are not supported."

#         checkpoint = torch.load(
#             checkpoint_path, map_location=map_location or (lambda storage, loc: storage)
#         )
#         if find_best:
#             best_score, best_name = None, None
#             modes = {"min": torch.lt, "max": torch.gt}
#             for key, state in checkpoint["callbacks"].items():
#                 if not key.startswith("ModelCheckpoint"):
#                     continue
#                 mode = eval(key.replace("ModelCheckpoint", ""))["mode"]
#                 if best_score is None or modes[mode](
#                     state["best_model_score"], best_score
#                 ):
#                     best_score = state["best_model_score"]
#                     best_name = Path(state["best_model_path"]).name
#             logger.info("Loading best checkpoint %s", best_name)
#             if best_name != checkpoint_path:
#                 return cls.load_from_checkpoint(
#                     Path(checkpoint_path).parent / best_name,
#                     map_location,
#                     hparams_file,
#                     strict,
#                     cfg,
#                     find_best=False,
#                 )

#         logger.info(
#             "Using checkpoint %s from epoch %d and step %d.",
#             checkpoint_path.name,
#             checkpoint["epoch"],
#             checkpoint["global_step"],
#         )
#         cfg_ckpt = checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY]
#         if list(cfg_ckpt.keys()) == ["cfg"]:  # backward compatibility
#             cfg_ckpt = cfg_ckpt["cfg"]
#         cfg_ckpt = OmegaConf.create(cfg_ckpt)

#         if cfg is None:
#             cfg = {}
#         if not isinstance(cfg, DictConfig):
#             cfg = OmegaConf.create(cfg)
#         with open_dict(cfg_ckpt):
#             cfg = OmegaConf.merge(cfg_ckpt, cfg)

#         return pl.core.saving._load_state(cls, checkpoint, strict=strict, cfg=cfg)
