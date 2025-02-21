# Copyright (c) Meta Platforms, Inc. and affiliates.

import os.path as osp
from pathlib import Path
from typing import Optional

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only

from . import EXPERIMENTS_PATH, logger, pl_logger
# from .data import modules as data_modules
# from .module import GenericModule
from clearml import Task, Dataset
import random
import numpy as np
from .module import ONGenericModule
from .data.yyc.dataset import YYCDataset, create_dataloader
from torch.utils.tensorboard import SummaryWriter


class ClearMLCallback(pl.callbacks.Callback):
    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        current_epoch = pl_module.current_epoch
        # Add debug print
        print(f"Reporting epoch {current_epoch} as iteration")
        task = Task.current_task()
        if task:
            results = {
                **dict(pl_module.metrics_val.items()),
                **dict(pl_module.losses_val.items()),
            }
            for k, v in results.items():
                task.get_logger().report_scalar(
                    title=k,
                    series="Validation",
                    value=v.compute(),
                    iteration=current_epoch
                )

class CleanProgressBar(pl.callbacks.TQDMProgressBar):
    def get_metrics(self, trainer, model):
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)  # don't show the version number
        items.pop("loss", None)
        return items


class SeedingCallback(pl.callbacks.Callback):
    def on_epoch_start_(self, trainer, module):
        seed = module.cfg.experiment.seed
        is_overfit = module.cfg.training.trainer.get("overfit_batches", 0) > 0
        if trainer.training and not is_overfit:
            seed = seed + trainer.current_epoch

        # Temporarily disable the logging (does not seem to work?)
        pl_logger.disabled = True
        try:
            pl.seed_everything(seed, workers=True)
        finally:
            pl_logger.disabled = False

    def on_train_epoch_start(self, *args, **kwargs):
        self.on_epoch_start_(*args, **kwargs)

    def on_validation_epoch_start(self, *args, **kwargs):
        self.on_epoch_start_(*args, **kwargs)

    def on_test_epoch_start(self, *args, **kwargs):
        self.on_epoch_start_(*args, **kwargs)


class ConsoleLogger(pl.callbacks.Callback):
    @rank_zero_only
    def on_train_epoch_start(self, trainer, module):
        logger.info(
            "New training epoch %d for experiment '%s'.",
            module.current_epoch,
            module.cfg.experiment.name,
        )

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, module):
        results = {
            **dict(module.metrics_val.items()),
            **dict(module.losses_val.items()),
        }
        results = [f"{k} {v.compute():.3E}" for k, v in results.items()]
        logger.info(f'[Validation] {{{", ".join(results)}}}')


def find_last_checkpoint_path(experiment_dir):
    cls = pl.callbacks.ModelCheckpoint
    path = osp.join(experiment_dir, cls.CHECKPOINT_NAME_LAST + cls.FILE_EXTENSION)
    if osp.exists(path):
        return path
    else:
        return None


def prepare_experiment_dir(experiment_dir, cfg, rank):
    config_path = osp.join(experiment_dir, "config.yaml")
    last_checkpoint_path = find_last_checkpoint_path(experiment_dir)
    if last_checkpoint_path is not None:
        if rank == 0:
            logger.info(
                "Resuming the training from checkpoint %s", last_checkpoint_path
            )
        if osp.exists(config_path):
            with open(config_path, "r") as fp:
                cfg_prev = OmegaConf.create(fp.read())
            compare_keys = ["experiment", "data", "model", "training"]
            if OmegaConf.masked_copy(cfg, compare_keys) != OmegaConf.masked_copy(
                cfg_prev, compare_keys
            ):
                raise ValueError(
                    "Attempting to resume training with a different config: "
                    f"{OmegaConf.masked_copy(cfg, compare_keys)} vs "
                    f"{OmegaConf.masked_copy(cfg_prev, compare_keys)}"
                )
    if rank == 0:
        Path(experiment_dir).mkdir(exist_ok=True, parents=True)
        with open(config_path, "w") as fp:
            OmegaConf.save(cfg, fp)
    return last_checkpoint_path

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def save_checkpoint(model, optimizer, epoch, path, is_best=False):
    checkpoint = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(checkpoint, path)
    if is_best:
        best_path = osp.join(osp.dirname(path), 'best.pt')
        torch.save(checkpoint, best_path)

def train(config: DictConfig, job_id: Optional[int] = None):
    
    torch.set_float32_matmul_precision("medium")
    OmegaConf.resolve(config)
    set_seed(config.experiment.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() and config.experiment.gpus > 0 else "cpu")
    if device.type == "cpu":
        logger.warning("Will train on CPU...")
        
    model = ONGenericModule(config).to(device)
    logger.info("Network:\n%s", model.model)
    
    # setup directories
    experiment_dir = osp.join(EXPERIMENTS_PATH, config.experiment.name)
    Path(experiment_dir).mkdir(parents=True, exist_ok=True)
    
    optimizer = model.configure_optimizers()
    scheduler = None
    if isinstance(optimizer, tuple):
        optimizer, schedule = optimizer
        
    # init dataloaders
    if config.clearml.dataset_id:
    # ClearML init
        task = Task.init(project_name="OrienterNet",
                        task_name=config.experiment.name,
                        output_uri=True)
        task.force_requirements_env_freeze(force=True, requirements_file=None)
        task.connect(config)
        
        dataset = Dataset.get(dataset_id=config.clearml.dataset_id)
        local_data_path = dataset.get_local_copy()
        # Update the data_dir path in the nested config
        config.data.paths.data_dir = str(local_data_path)
        config.data.paths.combined_geojson_path = str(local_data_path) + "/combined-output.geojson"
        
    # Create datasets using stages
    train_dataset = YYCDataset(config, stage='train')
    val_dataset = YYCDataset(config, stage='val')
    
    # Create subsets while preserving the cfg attribute
    train_subset = torch.utils.data.Subset(train_dataset, range(10))
    val_subset = torch.utils.data.Subset(val_dataset, range(min(5, len(val_dataset))))
    
    # Manually add the cfg attribute to the subsets
    train_subset.cfg = train_dataset.cfg
    val_subset.cfg = val_dataset.cfg
    
    print(f"Training dataset size: {len(train_subset)}")
    print(f"Validation dataset size: {len(val_subset)}")
    
    train_loader = create_dataloader(train_subset, config, 'train')
    val_loader = create_dataloader(val_subset, config, 'val')
    
    print(experiment_dir)
    
    writer = SummaryWriter(osp.join(experiment_dir, 'tensorboard'))
    
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(config.train.training.trainer.max_epochs):
        model.reset_train_losses()
        # train epoch
        for batch_idx, batch in enumerate(train_loader):
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            
            optimizer.zero_grad()
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            
        train_metrics = model.get_epoch_train_metrics()
            
        # Validation loop
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                model.validation_step(batch)
                
        val_metrics = model.get_validation_metrics()
    
    
        for name, value in {**train_metrics, **val_metrics}.items():
            writer.add_scalar(name, value, epoch)
            if config.clearml.dataset_id:
                task.get_logger().report_scalar(
                    title=name,
                    series="Metrics",
                    value=value,
                    iteration=epoch
                )
            
            save_checkpoint(
                model, optimizer, epoch,
                osp.join(experiment_dir, f'checkpoint-epoch-{epoch:02d}.pt')
            )
            
            val_loss = val_metrics.get('loss/total/val', float('inf'))
            # TODO: implement saving best model? 
        
            # Step scheduler if it exists
            if scheduler is not None:
                scheduler.step()
            
        logger.info(f'Epoch {epoch}: train_loss={train_metrics["loss/total/train"]:.4f}, '
                f'val_loss={val_loss:.4f}')    
    


@hydra.main(
    config_path=osp.join(osp.dirname(__file__), "conf"), config_name="orienternet"
)
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()