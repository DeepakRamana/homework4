"""
Usage:
    python3 -m homework.train_planner --model_name mlp_planner --num_epoch 40 --lr 1e-3
    python3 -m homework.train_planner --model_name transformer_planner --num_epoch 60 --lr 5e-4
    python3 -m homework.train_planner --model_name cnn_planner --num_epoch 40 --lr 1e-3
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb

from .datasets.road_dataset import load_data
from .metrics import PlannerMetric
from .models import MODEL_FACTORY, save_model


def train(
    exp_dir: str = "logs",
    model_name: str = "mlp_planner",
    transform_pipeline: str | None = None,
    num_epoch: int = 40,
    lr: float = 1e-3,
    batch_size: int = 64,
    num_workers: int = 2,
    weight_decay: float = 1e-4,
    seed: int = 2024,
    train_data: str = "drive_data/train",
    val_data: str = "drive_data/val",
    **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("CUDA/MPS not available, training on CPU.")
        device = torch.device("cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = tb.SummaryWriter(log_dir)

    if transform_pipeline is None:
        transform_pipeline = "default" if model_name == "cnn_planner" else "state_only"

    train_loader = load_data(
        train_data,
        transform_pipeline=transform_pipeline,
        return_dataloader=True,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = load_data(
        val_data,
        transform_pipeline=transform_pipeline,
        return_dataloader=True,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=False,
    )

    model = MODEL_FACTORY[model_name]().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch)

    train_metric = PlannerMetric()
    val_metric = PlannerMetric()
    global_step = 0
    best_val_l1 = float("inf")

    for epoch in range(num_epoch):
        model.train()
        train_metric.reset()
        train_losses = []

        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            waypoints = batch["waypoints"]
            waypoints_mask = batch["waypoints_mask"]

            if model_name == "cnn_planner":
                pred = model(image=batch["image"])
            else:
                pred = model(track_left=batch["track_left"], track_right=batch["track_right"])

            mask = waypoints_mask[..., None].float()
            diff = (pred - waypoints) * mask
            loss = (diff.abs().sum() + (diff ** 2).sum()) / (mask.sum() * 2 + 1e-8)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_metric.add(pred.detach(), waypoints, waypoints_mask)
            train_losses.append(loss.item())
            logger.add_scalar("train/loss", loss.item(), global_step)
            global_step += 1

        scheduler.step()

        model.eval()
        val_metric.reset()
        with torch.inference_mode():
            for batch in val_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                waypoints = batch["waypoints"]
                waypoints_mask = batch["waypoints_mask"]

                if model_name == "cnn_planner":
                    pred = model(image=batch["image"])
                else:
                    pred = model(track_left=batch["track_left"], track_right=batch["track_right"])

                val_metric.add(pred, waypoints, waypoints_mask)

        train_stats = train_metric.compute()
        val_stats = val_metric.compute()

        logger.add_scalar("train/longitudinal_error", train_stats["longitudinal_error"], epoch)
        logger.add_scalar("train/lateral_error", train_stats["lateral_error"], epoch)
        logger.add_scalar("val/longitudinal_error", val_stats["longitudinal_error"], epoch)
        logger.add_scalar("val/lateral_error", val_stats["lateral_error"], epoch)
        logger.add_scalar("val/l1_error", val_stats["l1_error"], epoch)

        print(
            f"Epoch {epoch + 1:3d}/{num_epoch} "
            f"loss={np.mean(train_losses):.4f} "
            f"train lon={train_stats['longitudinal_error']:.3f} lat={train_stats['lateral_error']:.3f} | "
            f"val lon={val_stats['longitudinal_error']:.3f} lat={val_stats['lateral_error']:.3f} "
            f"l1={val_stats['l1_error']:.3f}"
        )

        if val_stats["l1_error"] < best_val_l1:
            best_val_l1 = val_stats["l1_error"]
            saved_path = save_model(model)
            print(f"  -> saved best model to {saved_path} (val l1={best_val_l1:.3f})")

    print(f"Training complete. Best val l1_error: {best_val_l1:.3f}")
    logger.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, default="mlp_planner", choices=list(MODEL_FACTORY.keys()))
    parser.add_argument("--transform_pipeline", type=str, default=None)
    parser.add_argument("--num_epoch", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--train_data", type=str, default="drive_data/train")
    parser.add_argument("--val_data", type=str, default="drive_data/val")

    args = parser.parse_args()
    train(**vars(args))


if __name__ == "__main__":
    main()
