#!/usr/bin/env python3
"""Train a stage1_mod experiment, analyze its embeddings, and run SAM tests."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


STAGE1_MOD_DIR = Path(__file__).resolve().parent
PROJECT_DIR = STAGE1_MOD_DIR.parent
if str(STAGE1_MOD_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE1_MOD_DIR))

from config import _C, _update_config_from_file


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_config", type=Path, help="Stage1_mod experiment YAML config")
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=1,
        help="Number of local training processes (default: 1)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override the experiment config's OUTPUT directory",
    )
    args, train_args = parser.parse_known_args()
    return args, train_args


def resolve_output_dir(config_path: Path) -> Path:
    config = _C.clone()
    _update_config_from_file(config, str(config_path))
    output_dir = Path(config.OUTPUT) if config.OUTPUT else Path("output/stage1_mod") / config_path.stem
    if not output_dir.is_absolute():
        output_dir = PROJECT_DIR / output_dir
    return output_dir.resolve()


def run_step(name: str, command: list[str], env: dict[str, str]) -> None:
    print(f"\n==> {name}\n{shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=PROJECT_DIR, env=env, check=True)


def main() -> None:
    args, train_args = parse_args()
    config_path = args.experiment_config.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Experiment config not found: {config_path}")
    if args.nproc_per_node < 1:
        raise ValueError("--nproc-per-node must be at least 1")

    output_dir = args.output_dir or resolve_output_dir(config_path)
    if not output_dir.is_absolute():
        output_dir = PROJECT_DIR / output_dir
    output_dir = output_dir.resolve()

    env = os.environ.copy()
    pythonpath = [str(PROJECT_DIR)]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)

    train_script = STAGE1_MOD_DIR / "train_image_encoder_stage1.py"
    analyze_script = STAGE1_MOD_DIR / "analyze_embeddings.py"
    test_script = STAGE1_MOD_DIR / "test_efficient_sam.py"

    run_step(
        "Train image encoder",
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            f"--nproc_per_node={args.nproc_per_node}",
            str(train_script),
            "--cfg",
            str(config_path),
            *train_args,
            "--output",
            str(output_dir),
        ],
        env,
    )
    run_step(
        "Analyze trained embeddings",
        [sys.executable, str(analyze_script), str(output_dir)],
        env,
    )
    run_step(
        "Run EfficientSAM3 prompt tests",
        [
            sys.executable,
            str(test_script),
            "--stage1-mod-dir",
            str(output_dir),
            "--output-dir",
            str(output_dir / "sam_test_images"),
        ],
        env,
    )


if __name__ == "__main__":
    main()