#!/usr/bin/env python3
"""Run EfficientSAM3 RepViT on the local image/prompt fixtures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


STAGE1_MOD_DIR = Path(__file__).resolve().parent
PROJECT_DIR = STAGE1_MOD_DIR.parent
SAM3_PACKAGE_DIR = PROJECT_DIR / "sam3"
if str(SAM3_PACKAGE_DIR) not in sys.path:
	sys.path.insert(0, str(SAM3_PACKAGE_DIR))

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_efficientsam3_image_model


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--checkpoint",
		type=Path,
		default=PROJECT_DIR / "sam3_checkpoints" / "efficientsam3_repvit.pt",
	)
	parser.add_argument(
		"--input-dir", type=Path, default=STAGE1_MOD_DIR / "efficient_sam_tests"
	)
	parser.add_argument(
		"--sam3-checkpoint",
		type=Path,
		default=PROJECT_DIR / "sam3_checkpoints" / "sam3" / "sam3.pt",
		help="Base SAM3 checkpoint used with a multiscale student checkpoint.",
	)
	parser.add_argument(
		"--config",
		type=Path,
		default=None,
		help="Stage 1 YAML config (e.g. stage1/configs/es_tv_l.yaml). Its MODEL.BACKBONE "
		"overrides --backbone_type/--model_name.",
	)
	parser.add_argument(
		"--stage1-mod-dir",
		type=Path,
		default=None,
		help="Output directory of a stage1_mod run. The model is built from its config.json and the "
		"latest ckpt_epoch_N.pth is loaded into the student encoder of a model built from "
		"--sam3-checkpoint; --checkpoint and --config are ignored.",
	)
	parser.add_argument("--backbone_type", default="repvit")
	parser.add_argument("--model_name", default="m1.1")
	
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=PROJECT_DIR / "temp_efficient_sam_test_outputs",
	)
	parser.add_argument("--threshold", type=float, default=0.1)
	parser.add_argument(
		"--device", default="cuda" if torch.cuda.is_available() else "cpu"
	)
	return parser.parse_args()


STAGE1_BACKBONES = {
	"repvit_m0_9": ("repvit", "m0_9"),
	"repvit_m1_1": ("repvit", "m1_1"),
	"repvit_m2_3": ("repvit", "m2_3"),
	"tiny_vit_5m": ("tinyvit", "5m"),
	"tiny_vit_11m": ("tinyvit", "11m"),
	"tiny_vit_21m": ("tinyvit", "21m"),
	"efficientvit_b0": ("efficientvit", "b0"),
	"efficientvit_b1": ("efficientvit", "b1"),
	"efficientvit_b2": ("efficientvit", "b2"),
}


def stage1_config_details(config_path: Path) -> tuple[tuple[str, str], str, bool]:
	"""Resolve backbone and multiscale status, following Stage 1 BASE files."""
	import yaml

	backbone = None
	multiscale = False
	stack = [config_path]
	while stack:
		path = stack.pop()
		with path.open(encoding="utf-8") as file:
			cfg = yaml.safe_load(file) or {}
		if backbone is None:
			backbone = (cfg.get("MODEL") or {}).get("BACKBONE")
		multiscale |= "TEACHER_CHECKPOINT" in (cfg.get("DISTILL") or {})
		for base in cfg.get("BASE") or []:
			if base:
				stack.append(path.parent / base)
	if backbone not in STAGE1_BACKBONES:
		raise ValueError(f"Unsupported backbone {backbone!r} in {config_path}")
	return STAGE1_BACKBONES[backbone], backbone, multiscale


class _TrunkListWrapper(torch.nn.Module):
	"""Adapts a stage1_mod ImageStudentEncoder to the list-returning trunk the SAM3 neck expects."""

	def __init__(self, student: torch.nn.Module):
		super().__init__()
		self.model = student
		self.channel_list = [1024]

	def forward(self, x):
		return [self.model(x)]


def _setup_stage1_mod_imports() -> None:
	import sam3

	# stage1_mod/model.py imports `sam3.sam3.*`; here the inner package is already top-level `sam3`.
	sys.modules.setdefault("sam3.sam3", sam3)
	sam3.sam3 = sam3
	if str(PROJECT_DIR) not in sys.path:
		sys.path.append(PROJECT_DIR.as_posix())


def load_stage1_mod_run(run_dir: Path) -> tuple[object, Path]:
	"""Return (config, latest checkpoint path) for a stage1_mod run directory."""
	import re

	import yaml
	from yacs.config import CfgNode

	_setup_stage1_mod_imports()
	from stage1_mod.config import _C

	config_path = run_dir / "config.json"
	if not config_path.is_file():
		raise FileNotFoundError(f"No config.json in {run_dir}")
	with config_path.open(encoding="utf-8") as file:
		saved = yaml.safe_load(file)  # The trainer writes config.json as a YAML dump.
	saved.pop("BASE", None)  # Already merged into the dump.
	config = _C.clone()
	config.defrost()
	config.merge_from_other_cfg(CfgNode(saved))
	config.freeze()

	checkpoints = [
		(int(match.group(1)), path)
		for path in run_dir.iterdir()
		if (match := re.fullmatch(r"ckpt_epoch_(\d+)\.pth", path.name))
	]
	if not checkpoints:
		raise FileNotFoundError(f"No ckpt_epoch_*.pth files in {run_dir}")
	return config, max(checkpoints)[1]


def load_stage1_mod_student(
	model: torch.nn.Module, checkpoint_path: Path, config: object
) -> None:
	"""Swap the model's trunk for a stage1_mod fusion student and load its weights."""
	from stage1_mod.model import build_image_student_model

	checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
	state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else None
	if not isinstance(state, dict):
		raise ValueError(f"No model state found in stage1_mod checkpoint: {checkpoint_path}")
	state = {k.removeprefix("module."): v for k, v in state.items()}

	student = build_image_student_model(config)
	missing, unexpected = student.load_state_dict(state, strict=False)
	if missing or unexpected:
		raise RuntimeError(
			"Student architecture does not match the checkpoint: "
			f"missing {missing[:5]}, unexpected {unexpected[:5]}"
		)
	model.backbone.vision_backbone.trunk = _TrunkListWrapper(student)
	print(f"Loaded stage1_mod student from {checkpoint_path}")


class TrainingPreprocessProcessor(Sam3Processor):
	"""Preprocesses images exactly like stage1_mod training instead of stretching to a square.

	Training: resize longest side to IMG_SIZE (bilinear, antialiased), normalize with DATA.MEAN/STD,
	then zero-pad on the bottom/right. Model outputs are normalized to that padded square, so
	`original_height/width` in the state is set to the padded-square size in source pixels
	(max(H, W)); boxes then come out in source pixels and masks are cropped to the image.
	"""

	def __init__(self, model, config, **kwargs):
		super().__init__(model, resolution=int(config.DATA.IMG_SIZE), **kwargs)
		from stage1_mod.data.transforms import ResizeLongestSide

		self.img_size = int(config.DATA.IMG_SIZE)
		self.resize = ResizeLongestSide(self.img_size)
		self.pixel_mean = torch.tensor(config.DATA.MEAN).view(-1, 1, 1)
		self.pixel_std = torch.tensor(config.DATA.STD).view(-1, 1, 1)

	@torch.inference_mode()
	def set_image(self, image, state=None):
		if state is None:
			state = {}
		width, height = image.size
		img = torch.from_numpy(np.asarray(image.convert("RGB"))).permute(2, 0, 1)
		img = self.resize.apply_image_torch(img[None].float()).squeeze(0)
		img = (img - self.pixel_mean) / self.pixel_std
		pad_h = self.img_size - img.shape[-2]
		pad_w = self.img_size - img.shape[-1]
		img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h))

		side = max(width, height)
		state["image_height"] = height
		state["image_width"] = width
		state["original_height"] = side
		state["original_width"] = side
		state["backbone_out"] = self.model.backbone.forward_image(img[None].to(self.device))
		return state

	@torch.inference_mode()
	def _forward_grounding(self, state):
		state = super()._forward_grounding(state)
		height, width = state["image_height"], state["image_width"]
		state["masks_logits"] = state["masks_logits"][..., :height, :width]
		state["masks"] = state["masks"][..., :height, :width]
		return state


def draw_results(image: Image.Image, state: dict) -> Image.Image:
	annotated = image.convert("RGBA")
	masks = state.get("masks")
	boxes = state.get("boxes")
	scores = state.get("scores")

	if masks is not None:
		masks = masks.detach().cpu().numpy()
		for index, mask in enumerate(masks):
			mask = np.squeeze(mask).astype(bool)
			color = (35 + (index * 67) % 200, 210, 95 + (index * 43) % 150)
			mask_image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
			color_layer = Image.new("RGBA", annotated.size, color)
			color_layer.putalpha(mask_image.point(lambda value: value * 165 // 255))
			annotated = Image.alpha_composite(annotated, color_layer)

	draw = ImageDraw.Draw(annotated)
	if boxes is not None:
		boxes = boxes.detach().cpu().tolist()
		score_values = scores.detach().cpu().tolist() if scores is not None else []
		for index, box in enumerate(boxes):
			x0, y0, x1, y1 = box
			draw.rectangle((x0, y0, x1, y1), outline=(255, 74, 55, 255), width=3)
			label = f"{score_values[index]:.2f}" if index < len(score_values) else "object"
			draw.text((x0 + 3, max(0, y0 - 16)), label, fill=(255, 74, 55, 255))

	return annotated.convert("RGB")


def apply_prompts(processor: Sam3Processor, image: Image.Image, test: dict) -> dict:
	state = processor.set_image(image)
	for prompt in test.get("prompts", []):
		prompt_type = prompt["type"]
		if prompt_type == "text":
			state = processor.set_text_prompt(prompt=str(prompt["data"]), state=state)
		elif prompt_type == "box":
			x0, y0, x1, y1 = map(float, prompt["data"])
			# Normalize by the processor's reference size (the padded square for training-style preprocessing).
			width, height = state["original_width"], state["original_height"]
			box = [
				(x0 + x1) / (2 * width),
				(y0 + y1) / (2 * height),
				(x1 - x0) / width,
				(y1 - y0) / height,
			]
			state = processor.add_geometric_prompt(
				box=box, label=bool(prompt.get("label", 1)), state=state
			)
		else:
			raise ValueError(f"Unsupported prompt type: {prompt_type!r}")
	return state


def main() -> None:
	args = parse_args()
	if args.stage1_mod_dir is not None:
		if not args.sam3_checkpoint.is_file():
			raise FileNotFoundError(f"Checkpoint not found: {args.sam3_checkpoint}")
	elif not args.checkpoint.is_file():
		raise FileNotFoundError(f"RepViT checkpoint not found: {args.checkpoint}")

	prompt_dir = args.input_dir / "prompts"
	image_dir = args.input_dir / "images"
	prompt_files = sorted(prompt_dir.glob("*.json"))
	if not prompt_files:
		raise FileNotFoundError(f"No prompt JSON files found in {prompt_dir}")

	device = torch.device(args.device)
	multiscale = False
	student_config = None
	if args.stage1_mod_dir is not None:
		student_config, stage1_mod_checkpoint = load_stage1_mod_run(args.stage1_mod_dir)
		backbone_name = student_config.MODEL.BACKBONE
		if backbone_name not in STAGE1_BACKBONES:
			raise ValueError(f"Unsupported backbone {backbone_name!r} in {args.stage1_mod_dir}")
		args.backbone_type, args.model_name = STAGE1_BACKBONES[backbone_name]
	elif args.config is not None:
		if not args.config.is_file():
			raise FileNotFoundError(f"Config not found: {args.config}")
		(
			(args.backbone_type, args.model_name),
			backbone,
			multiscale,
		) = stage1_config_details(args.config)
	if args.stage1_mod_dir is not None:
		# SAM3 weights supply the neck, decoder and text encoder; the student trunk is overwritten below.
		model = build_efficientsam3_image_model(
			checkpoint_path=str(args.sam3_checkpoint),
			backbone_type=args.backbone_type,
			model_name=args.model_name,
			enable_segmentation=True,
			device="cpu",
		)
		load_stage1_mod_student(model, stage1_mod_checkpoint, student_config)
		model.to(device)
	elif multiscale:
		if not args.sam3_checkpoint.is_file():
			raise FileNotFoundError(f"SAM3 checkpoint not found: {args.sam3_checkpoint}")
		# This locally generated trainer checkpoint includes a YACS config object.
		checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
		student_state = checkpoint.get("model")
		if not isinstance(student_state, dict):
			raise ValueError(f"No model state found in training checkpoint: {args.checkpoint}")
		from stage1_all.efficient_sam3_multiscale import build_efficient_sam3_multiscale

		model = build_efficient_sam3_multiscale(
			backbone,
			checkpoint_path=str(args.sam3_checkpoint),
			student_state=student_state,
			device=str(device),
		)
	else:
		model = build_efficientsam3_image_model(
			checkpoint_path=str(args.checkpoint),
			backbone_type=args.backbone_type,
			model_name=args.model_name,
			enable_segmentation=True,
			device=str(device),
		)
	model.eval()
	if student_config is not None:
		processor = TrainingPreprocessProcessor(
			model, student_config, confidence_threshold=args.threshold, device=device
		)
	else:
		processor = Sam3Processor(
			model, confidence_threshold=args.threshold, device=device
		)
	args.output_dir.mkdir(parents=True, exist_ok=True)

	for prompt_file in prompt_files:
		with prompt_file.open(encoding="utf-8") as file:
			prompt_data = json.load(file)

		image_path = image_dir / prompt_data["image_name"]
		if not image_path.is_file():
			print(f"Skipping {prompt_file.name}: image not found ({image_path.name})")
			continue

		with Image.open(image_path) as opened_image:
			image = opened_image.convert("RGB")

		for test in prompt_data.get("tests", []):
			processor.reset_all_prompts({})
			state = apply_prompts(processor, image, test)
			annotated = draw_results(image, state)
			test_id = str(test.get("id", "test"))
			output_path = args.output_dir / f"{image_path.stem}_{test_id}.png"
			annotated.save(output_path)
			print(f"Saved {output_path}")


if __name__ == "__main__":
	main()
