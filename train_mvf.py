# python -m maploc.train experiment.name=OrienterNet_YYC  experiment.gpus=1 data.loading.train.batch_size=1

import os.path as osp
from pathlib import Path
import random
import numpy as np
import json
import hydra
import torch
from torch.utils.tensorboard import SummaryWriter
from omegaconf import DictConfig, OmegaConf
from clearml import Task

from maploc import EXPERIMENTS_PATH, logger
from maploc.module import ONGenericModule
from maploc.data.yyc.dataset import create_dataloader
from maploc_mvf.mvf_dataset import YYCDatasetMVF


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(model, optimizer, epoch, path, is_best=False):
    checkpoint = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, path)
    if is_best:
        best_path = osp.join(osp.dirname(path), "best.pt")
        torch.save(checkpoint, best_path)


def load_pretrained_weights(model, checkpoint_path, device):
    """Load pretrained weights handling different number of classes"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["state_dict"]
    model_dict = model.state_dict()

    # Filter out embedding layers and their sizes
    pretrained_dict = {}
    for k, v in state_dict.items():
        # Skip embedding layers
        if "embeddings" not in k:
            if k in model_dict and model_dict[k].shape == v.shape:
                pretrained_dict[k] = v
            else:
                logger.warning(
                    f"Skipping layer {k} due to shape mismatch: "
                    f"pretrained {v.shape} vs model {model_dict[k].shape if k in model_dict else 'N/A'}"
                )

    # Update model weights except embeddings
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)

    logger.info("Successfully loaded pretrained weights (excluding embedding layers)")
    return model


def train(config: DictConfig):
    """
    Run training
    Args:
        config: the configuration object
    """

    torch.set_float32_matmul_precision("medium")
    OmegaConf.resolve(config)
    set_seed(config.experiment.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and config.experiment.gpus > 0 else "cpu"
    )
    if device.type == "cpu":
        logger.warning("Will train on CPU...")

    model = ONGenericModule(config).to(device)
    # Load state dict, handling potential key mismatches
    load_pretrained_weights(model, config.train.experiment.pretrained_path, device)

    # setup directories
    experiment_dir = osp.join(EXPERIMENTS_PATH, config.experiment.name)
    Path(experiment_dir).mkdir(parents=True, exist_ok=True)
    logger.info("Experiment directory: %s", experiment_dir)

    optimizer = model.configure_optimizers()
    scheduler = None
    if isinstance(optimizer, tuple):
        optimizer, schedule = optimizer

    # init dataloaders
    # if config.clearml.dataset_id:
    #     # ClearML init
    if config.train.experiment.clearml:
        task = Task.init(
            project_name="OrienterNet",
            task_name=config.experiment.name,
            output_uri=True,
            # _allow_omegaconf_edit=True,
        )
        task.force_requirements_env_freeze(force=True, requirements_file=None)
        task.connect(config)

        #     dataset = Dataset.get(dataset_id=config.clearml.dataset_id)
        #     local_data_path = dataset.get_local_copy()
        #     # Update the data_dir path in the nested config
        #     config.data.paths.data_dir = str(local_data_path)
        #     config.data.paths.combined_geojson_path = (
        #         str(local_data_path) + "/combined-output.geojson"
        #     )  # TODO: CHANGE FROM e57 when training normal

    train_dataset = YYCDatasetMVF(config, stage="train")
    val_dataset = YYCDatasetMVF(config, stage="val")
    test_dataset = YYCDatasetMVF(config, stage="test")

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    train_loader = create_dataloader(train_dataset, config, "train")
    val_loader = create_dataloader(val_dataset, config, "val")
    test_loader = create_dataloader(test_dataset, config, "test")

    writer = SummaryWriter(osp.join(experiment_dir, "tensorboard"))

    # Training loop
    best_val_loss = float("inf")

    for epoch in range(config.train.training.trainer.max_epochs):
        model.reset_train_losses()
        # train epoch
        for _, batch in enumerate(train_loader):
            batch = {
                k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()
            }

            optimizer.zero_grad()
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()

        train_metrics = model.get_epoch_train_metrics()

        # Validation loop
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                batch = {
                    k: v.to(device) if torch.is_tensor(v) else v
                    for k, v in batch.items()
                }
                model.validation_step(batch)

        val_metrics = model.get_validation_metrics()

        for name, value in {
            **train_metrics,
            **val_metrics,
        }.items():
            writer.add_scalar(name, value, epoch)
            if config.clearml.dataset_id:
                task.get_logger().report_scalar(
                    title=name, series="Metrics", value=value, iteration=epoch
                )

        if epoch % config.train.training.trainer.save_epoch == 0:
            save_checkpoint(
                model,
                optimizer,
                epoch,
                osp.join(experiment_dir, f"checkpoint-epoch-{epoch:02d}.pt"),
            )

        val_loss = val_metrics.get("loss/total/val", float("inf"))

        if val_loss < best_val_loss:
            save_checkpoint(
                model,
                optimizer,
                epoch,
                osp.join(experiment_dir, f"best-model-epoch-{epoch:02d}.pt"),
            )
            best_val_loss = val_loss

        # Step scheduler if it exists
        if scheduler is not None:
            scheduler.step()

        logger.info(
            f'Epoch {epoch}: train_loss={train_metrics["loss/total/train"]:.4f}, '
            f"val_loss={val_loss:.4f}"
        )

    # Final evaluation on test set
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            batch = {
                k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()
            }
            model.test_step(batch)

    final_test_metrics = model.get_test_metrics()
    logger.info(f"Final test metrics: {final_test_metrics}")

    # Save final test metrics
    output_final_test_metrics = {}
    for name, value in final_test_metrics.items():
        output_final_test_metrics[name] = value.item()
    with open(osp.join(experiment_dir, "test_metrics.json"), "w") as f:
        json.dump(output_final_test_metrics, f, indent=2)

    # Report to ClearML if enabled
    if config.train.experiment.clearml and config.clearml.dataset_id:
        for name, value in final_test_metrics.items():
            task.get_logger().report_scalar(
                title=name, series="Final Test", value=value, iteration=0
            )


@hydra.main(
    version_base=None,
    config_path=osp.join(osp.dirname(__file__), "maploc", "conf"),
    config_name="orienternet_mvf",
)
def main(cfg: DictConfig) -> None:
    """Run training with the given configuration."""
    # python train_mvf.py experiment.name=OrienterNet_MVF_YYC experiment.gpus=1
    train(cfg)


if __name__ == "__main__":
    main()
