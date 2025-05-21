# python -m maploc.train experiment.name=OrienterNet_YYC  experiment.gpus=1 data.loading.train.batch_size=1

import os.path as osp
from pathlib import Path
import random
import numpy as np
import json
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from clearml import Task, Dataset
import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from maploc import EXPERIMENTS_PATH, logger
from maploc.module import ONGenericModule
from maploc.data.yyc.dataset import create_dataloader

# from maploc_mvf.mvf_dataset import YYCDatasetMVF
from maploc_mvf.mvf_dataset_selfserve import NaverDatasetMVF
from maploc.evaluation.run import resolve_checkpoint_path


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
    checkpoint_path = resolve_checkpoint_path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["state_dict"]
    model_dict = model.state_dict()

    # Filter out embedding layers and their sizes
    pretrained_dict = {}
    for k, v in state_dict.items():
        # Skip embedding layers
        # if "embeddings" not in k:
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


def visualize_predictions(batch, pred, epoch, output_dir, batch_number=0):
    """
    Visualize model predictions during validation
    Args:
        batch: Dictionary containing input data
        pred: Dictionary containing model predictions
        epoch: Current epoch number
        output_dir: Directory to save visualizations
    """

    # Create visualization directory
    viz_dir = Path(output_dir) / "visualizations" / f"epoch_{epoch:03d}"
    viz_dir.mkdir(parents=True, exist_ok=True)

    for i in range(min(3, len(batch["image"]))):  # Visualize first 3 samples
        fig, axes = plt.subplots(2, 3, figsize=(15, 15))

        # Plot input image
        axes[0, 0].imshow(batch["image"][i].cpu().permute(1, 2, 0))
        axes[0, 0].set_title("Input Image")

        # Plot map with ground truth and predicted position
        # Convert raster from tensor [C,H,W] to numpy [H,W,C]
        raster = batch["map"][i].cpu().permute(1, 2, 0).numpy()

        # Create colored visualization where each channel gets its own color
        colors = [
            [1, 0, 0],  # Red for channel 0
            [0, 1, 0],  # Green for channel 1
            [0, 0, 1],  # Blue for channel 2
        ]

        # Create RGB visualization
        colored_raster = np.zeros((raster.shape[0], raster.shape[1], 3))
        for channel in range(raster.shape[-1]):  # For each channel
            mask = raster[:, :, channel] > 0
            colored_raster[mask] = colors[channel]

        # Clip to [0,1] range
        colored_raster = np.clip(colored_raster, 0, 1)

        axes[0, 1].imshow(colored_raster)
        gt_uv = batch["uv"][i].cpu().numpy()
        pred_uv = pred["uv_max"][i].cpu().numpy()
        print(f"GT UV: {gt_uv}, Pred UV: {pred_uv}")
        axes[0, 1].scatter(gt_uv[0], gt_uv[1], c="w", marker="x", label="Ground Truth")
        axes[0, 1].scatter(pred_uv[0], pred_uv[1], c="w", marker="+", label="Prediction")
        axes[0, 1].legend()
        axes[0, 1].set_title("Map Predictions")

        # Plot map features
        map_features = pred["map"]["map_features"][i][0].cpu().numpy()
        # Reshape to 2D array [H*W, C]
        features_2d = np.transpose(map_features, (1, 2, 0)).reshape(-1, map_features.shape[0])
        # Apply PCA
        pca = PCA(n_components=3)
        features_pca = pca.fit_transform(features_2d)
        # Reshape back to image shape [H, W, 3]
        features_rgb = features_pca.reshape(map_features.shape[1], map_features.shape[2], 3)
        # Normalize to [0, 1]
        features_rgb = (features_rgb - features_rgb.min()) / (
            features_rgb.max() - features_rgb.min()
        )
        axes[0, 2].imshow(features_rgb)
        axes[0, 2].set_title("Map Features (PCA)")

        image_features = pred["features_image"][i].cpu().numpy()
        features_2d = np.transpose(image_features, (1, 2, 0)).reshape(-1, image_features.shape[0])
        features_pca = pca.fit_transform(features_2d)
        features_rgb = features_pca.reshape(image_features.shape[1], image_features.shape[2], 3)
        features_rgb = (features_rgb - features_rgb.min()) / (
            features_rgb.max() - features_rgb.min()
        )
        axes[1, 0].imshow(features_rgb)
        axes[1, 0].set_title("Image Features (PCA)")

        bev_features = pred["features_bev"][i].cpu().numpy()
        features_2d = np.transpose(bev_features, (1, 2, 0)).reshape(-1, bev_features.shape[0])
        features_pca = pca.fit_transform(features_2d)
        features_rgb = features_pca.reshape(bev_features.shape[1], bev_features.shape[2], 3)
        features_rgb = (features_rgb - features_rgb.min()) / (
            features_rgb.max() - features_rgb.min()
        )
        axes[1, 1].imshow(features_rgb)
        axes[1, 1].set_title("BEV Features (PCA)")

        scores = pred["scores"][i].cpu().numpy()
        clipped_scores = np.clip(scores, -3, 10)
        features_2d = clipped_scores.reshape(-1, scores.shape[-1])
        features_pca = pca.fit_transform(features_2d)
        features_rgb = features_pca.reshape(scores.shape[0], scores.shape[1], 3)
        features_rgb = (features_rgb - features_rgb.min()) / (
            features_rgb.max() - features_rgb.min()
        )
        axes[1, 2].imshow(features_rgb)
        axes[1, 2].set_title("Scores (PCA)")

        plt.tight_layout()
        plt.savefig(viz_dir / f"sample_{batch_number}.png")
        plt.close()

def delete_previous_checkpoint(experiment_dir, current_epoch):
    """Delete the previous epoch's checkpoint to save disk space"""
    if current_epoch > 0:
        prev_latest = osp.join(experiment_dir, f"latest-model-epoch-{current_epoch-1:02d}.pt")
        
        # Delete previous checkpoints if they exist
        if osp.exists(prev_latest):
            try:
                Path(prev_latest).unlink()
            except Exception as e:
                logger.warning(f"Failed to delete checkpoint {prev_latest}: {e}")


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
        optimizer, scheduler = optimizer

    if config.train.experiment.clearml:
        Task.add_requirements("networkx", "3.1")
        Task.add_requirements("torch", "2.4.1+cu121")
        task = Task.init(
            project_name="OrienterNet",
            task_name=config.experiment.name,
            output_uri=True,
            # _allow_omegaconf_edit=True,
        )
        # task.force_requirements_env_freeze(force=True, requirements_file=None)
        task.connect(config)

        # dataset = Dataset.get(dataset_id=config.clearml.dataset_id)
        # dataset = Dataset.get(
        #     dataset_name="basemaps_no_symbols",
        #     dataset_project="SymbolDetection",
        #     dataset_version="1.0.0"
        # )

        # local_data_path = dataset.get_local_copy()
        # # Update the data_dir path in the nested config
        # config.data.paths.data_dir = str(local_data_path)
        # config.data.paths.combined_geojson_path = (
        #     str(local_data_path) + "/merged.geojson"
        # )
        # config.data.paths.photos_dir = str(local_data_path) + "/merged_images"
        # config.data.paths.valid_dir = str(local_data_path) + "/valid"
        # config.data.paths.split_file = str(local_data_path) + "/merged_splits.json"
        # config.data.paths.mvf_data_path = str(local_data_path) + "/mvf_data/"
        # config.data.paths.raster_map_path = str(local_data_path) + "/raster_maps/"
        # config.data.paths.area_index_path = (
        #     str(local_data_path) + "/area_index_mapping.json"
        # )
        # config.data.paths.line_index_path = (
        #     str(local_data_path) + "/line_index_mapping.json"
        # )

    train_dataset = NaverDatasetMVF(config, stage="train")
    val_dataset = NaverDatasetMVF(config, stage="val")
    test_dataset = NaverDatasetMVF(config, stage="test")

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    train_loader = create_dataloader(train_dataset, config, "train")
    val_loader = create_dataloader(val_dataset, config, "val")
    test_loader = create_dataloader(test_dataset, config, "test")

    best_val_loss = float("inf")

    for epoch in range(config.train.training.trainer.max_epochs):
        model.reset_train_losses()

        # Train loop
        for _, batch in tqdm.tqdm(
            enumerate(train_loader), desc="Training", total=len(train_loader)
        ):
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
            for val_batch_idx, batch in enumerate(val_loader):
                batch = {
                    k: v.to(device) if torch.is_tensor(v) else v
                    for k, v in batch.items()
                }
                model.validation_step(batch)
                # if val_batch_idx % 100 == 0:
                #     pred = model.forward(batch)
                #     visualize_predictions(batch, pred, epoch, experiment_dir)

        val_metrics = model.get_validation_metrics()

        for name, value in {
            **train_metrics,
            **val_metrics,
        }.items():
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

        # Save the latest checkpoint
        delete_previous_checkpoint(experiment_dir, epoch)
        save_checkpoint(
            model,
            optimizer,
            epoch,
            osp.join(experiment_dir, f"latest-model-epoch-{epoch:02d}.pt"),
        )

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
    config_name="orienternet_mvf_naver",
)
def main(cfg: DictConfig) -> None:
    """Run training with the given configuration."""
    # python train_mvf.py experiment.name=OrienterNet_MVF_YYC experiment.gpus=1
    train(cfg)


if __name__ == "__main__":
    main()
