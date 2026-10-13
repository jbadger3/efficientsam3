"""Load the student and latest checkpoint from a stage1_mod experiment output directory.

Usage:
    PYTHONPATH=. python stage1_mod/analyze_embeddings.py output/<experiment_dir>
"""

import argparse
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from yacs.config import CfgNode as CN

from config import _C
from data import build_loader
from model import build_image_student_model, resolve_head_name

CKPT_PATTERN = re.compile(r"^ckpt_epoch_(\d+)\.pth$")


def load_experiment_config(output_dir):
    """Rebuild the training config from the config.json (YAML dump) saved by the trainer."""
    path = os.path.join(output_dir, "config.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No config.json in {output_dir}")
    with open(path, "r") as f:
        saved = yaml.safe_load(f)
    saved.pop("BASE", None)  # Already merged into the dump.

    config = _C.clone()
    config.defrost()
    config.merge_from_other_cfg(CN(saved))
    config.freeze()
    return config


def find_latest_checkpoint(output_dir):
    """Return (path, epoch) of the highest-numbered ckpt_epoch_N.pth."""
    found = []
    for name in os.listdir(output_dir):
        match = CKPT_PATTERN.match(name)
        if match:
            found.append((int(match.group(1)), os.path.join(output_dir, name)))
    if not found:
        raise FileNotFoundError(f"No ckpt_epoch_*.pth files in {output_dir}")
    epoch, path = max(found)
    return path, epoch


def load_student(config, checkpoint_path, device):
    model = build_image_student_model(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = {k.removeprefix("module."): v for k, v in checkpoint["model"].items()}
    model.load_state_dict(state)
    return model.to(device).eval()


def build_data_loader(config, num_samples):
    """Build the training dataset/dataloader (with saved teacher embeddings) for num_samples images."""
    # build_loader and its sampler require an initialized process group.
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend="gloo", store=torch.distributed.HashStore(), rank=0, world_size=1
        )
    config = config.clone()
    config.defrost()
    config.DATA.NUM_SAMPLES = num_samples
    config.DISTILL.SAVE_TEACHER_EMBED = False
    config.freeze()
    dataset, _, data_loader, _ = build_loader(config, build_val=False)
    dataset.set_epoch(0)
    data_loader.sampler.set_epoch(0)
    return dataset, data_loader


def valid_pixel_mask(img_size, size_before_pad, embed_size):
    """Boolean (H, W) mask of embedding pixels that cover real image content, as in training."""
    h, w = (int(v) for v in size_before_pad[1:])
    valid = torch.zeros(1, 1, img_size, img_size)
    valid[:, :, :h, :w] = 1
    valid = F.interpolate(valid, size=(embed_size, embed_size), mode="bilinear", align_corners=False)
    return (valid[0, 0] > 0.5)


def compute_stats(student, teacher, valid):
    """student/teacher: (C, H, W) float tensors; valid: (H, W) bool. Stats use valid pixels only."""
    cos_map = F.cosine_similarity(student, teacher, dim=0)  # (H, W)
    s = student[:, valid]  # (C, N)
    t = teacher[:, valid]
    cos = cos_map[valid]

    mse = (s - t).square().mean()
    teacher_ms = t.square().mean()

    s_norm = s.norm(dim=0) # Norm of student embeddings per pixel
    t_norm = t.norm(dim=0) # Norm of teacher embeddings per pixel

    norm_ratio = s_norm / (t_norm + 1e-8) # per pixel norm ratio

    stats = {
        "num_valid_pixels": int(valid.sum()),
        "global_cosine": F.cosine_similarity(s.flatten(), t.flatten(), dim=0).item(),
        "mean_spatial_cosine": cos.mean().item(),
        "std_spatial_cosine": cos.std(unbiased=False).item(),
        "min_pixel_cosine": cos.min().item(),
        "mse": mse.item(),
        "normalized_mse": (mse / teacher_ms).item(),
        "teacher_rms": teacher_ms.sqrt().item(),
        "student_rms": s.square().mean().sqrt().item(),
        "student_teacher_global_norm_ratio": (s.norm() / t.norm()).item(),
        "mean_spatial_norm_ratio": norm_ratio.mean().item(),
        "std_spatial_norm_ratio": norm_ratio.std(unbiased=False).item(),
        "min_spatial_norm_ratio": norm_ratio.min().item(),
        "max_spatial_norm_ratio": norm_ratio.max().item(),
        "teacher_channel_mean": t.mean(dim=1).tolist(),
        "teacher_channel_std": t.std(dim=1, unbiased=False).tolist(),
        "student_channel_mean": s.mean(dim=1).tolist(),
        "student_channel_std": s.std(dim=1, unbiased=False).tolist(),
    }
    return stats, cos_map


def print_stats(name, stats):
    for key, value in stats.items():
        if isinstance(value, list):
            arr = np.asarray(value)
            print(f"  {key}: min {arr.min():.4f}  mean {arr.mean():.4f}  max {arr.max():.4f}  ({len(arr)} channels, full list in JSON)")
        elif isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")


def save_cosine_image(cos_map, valid, path, title):
    data = cos_map.numpy().astype(np.float32).copy()
    data[~valid.numpy()] = np.nan  # Padding shown as blank.
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(data, cmap="viridis", vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, label="cosine similarity (clipped to [0, 1])")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def pca_rgb_images(student, teacher, valid, scale=8):
    """Project both (C, H, W) embeddings onto the top 3 joint principal components as RGB.

    The basis and color normalization are shared so teacher and student colors are comparable.
    Padding pixels are black. Returns (student_rgb, teacher_rgb) uint8 arrays upscaled by `scale`.
    """
    s = student[:, valid].T  # (N, C)
    t = teacher[:, valid].T
    both = torch.cat([s, t], dim=0)
    mean = both.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(both - mean, full_matrices=False)
    basis = vh[:3].T  # (C, 3)
    s_proj = (s - mean) @ basis
    t_proj = (t - mean) @ basis

    joint = torch.cat([s_proj, t_proj], dim=0)
    lo = torch.quantile(joint, 0.02, dim=0)
    hi = torch.quantile(joint, 0.98, dim=0)

    images = []
    for proj in (s_proj, t_proj):
        rgb = ((proj - lo) / (hi - lo).clamp_min(1e-8)).clamp(0, 1)
        canvas = torch.zeros(*valid.shape, 3)
        canvas[valid] = rgb
        canvas = canvas.repeat_interleave(scale, dim=0).repeat_interleave(scale, dim=1)
        images.append((canvas.numpy() * 255).astype(np.uint8))
    return images[0], images[1]


@torch.no_grad()
def analyze(config, model, data_loader, output_dir, device):
    analysis_dir = os.path.join(output_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    embed_shape = (config.DISTILL.EMBED_DIM, config.DISTILL.EMBED_SIZE, config.DISTILL.EMBED_SIZE)

    count = 0
    for (samples, annos), (saved_embeddings, _) in data_loader:
        images = torch.stack(samples, dim=0).to(device)
        student_batch = model(images).float().cpu()
        teacher_batch = torch.from_numpy(np.stack(saved_embeddings, axis=0)).float()
        teacher_batch = teacher_batch.view(images.size(0), *embed_shape)

        for i in range(images.size(0)):
            # pseudo_collate turns the per-sample info dict into a dict of per-batch lists.
            file_name = annos["info"].get("file_name", [None] * images.size(0))[i]
            name = os.path.splitext(file_name)[0] if file_name else f"sample_{count:05d}"

            valid = valid_pixel_mask(
                config.DATA.IMG_SIZE, annos["img_size_before_pad"][i], config.DISTILL.EMBED_SIZE
            )
            stats, cos_map = compute_stats(student_batch[i], teacher_batch[i], valid)
            stats = {"sample": name, **stats}

            print(f"[{count}] {name}")
            print_stats(name, stats)
            with open(os.path.join(analysis_dir, f"{name}.json"), "w") as f:
                json.dump(stats, f, indent=2)
            save_cosine_image(
                cos_map, valid, os.path.join(analysis_dir, f"{name}_cosine.png"),
                f"{name}  mean cos {stats['mean_spatial_cosine']:.4f}",
            )
            student_rgb, teacher_rgb = pca_rgb_images(student_batch[i], teacher_batch[i], valid)
            plt.imsave(os.path.join(analysis_dir, f"{name}_student_pca.png"), student_rgb)
            plt.imsave(os.path.join(analysis_dir, f"{name}_teacher_pca.png"), teacher_rgb)
            count += 1
    print(f"Saved analysis for {count} samples to {analysis_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", help="Output directory of a stage1_mod experiment")
    parser.add_argument(
        "--num-samples",
        "--num_samples",
        dest="num_samples",
        type=int,
        default=4,
        help="Number of samples to load (default: DATA.NUM_SAMPLES from the experiment config)",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_experiment_config(args.output_dir)
    checkpoint_path, epoch = find_latest_checkpoint(args.output_dir)
    print(f"Backbone: {config.MODEL.BACKBONE}  Head: {resolve_head_name(config)}  IMG_SIZE: {config.DATA.IMG_SIZE}")
    print(f"Latest checkpoint: {checkpoint_path} (epoch {epoch})")

    model = load_student(config, checkpoint_path, args.device)
    print(f"Loaded student with {sum(p.numel() for p in model.parameters()):,} parameters")
    num_config_samples = config.DATA.NUM_SAMPLES
    num_samples = args.num_samples if args.num_samples <= num_config_samples else num_config_samples
    dataset, data_loader = build_data_loader(config, num_samples)
    print(f"Dataset: {len(dataset)} samples, {len(data_loader)} batches (batch size {config.DATA.BATCH_SIZE})")
    analyze(config, model, data_loader, args.output_dir, args.device)
    return config, model, dataset, data_loader


if __name__ == "__main__":
    main()
