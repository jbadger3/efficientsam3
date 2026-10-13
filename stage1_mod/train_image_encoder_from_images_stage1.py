import argparse
import datetime
import importlib
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from mmengine.dataset import pseudo_collate
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import get_config
from data.build import build_dataset
from data.sampler import MyDistributedSampler
from logger import create_logger
from lr_scheduler import build_scheduler
from model import build_image_student_model, build_image_teacher_model
from my_meter import AverageMeter
from optimizer import build_optimizer
from train_image_encoder_stage1 import (
    build_valid_mask,
    masked_cosine_loss,
    masked_mse,
    set_bn_state,
)
from utils import (
    NativeScalerWithGradNormCount,
    add_common_args,
    auto_resume_helper,
    get_git_info,
    is_main_process,
    load_checkpoint,
    save_checkpoint,
)

try:
    wandb = importlib.import_module("wandb")
except ImportError:
    wandb = None


TEACHER_CHECKPOINT = (
    Path(__file__).resolve().parents[1] / "sam3_checkpoints" / "sam3" / "sam3.pt"
)


class SAM3ViTTeacher(torch.nn.Module):
    def __init__(self, trunk, embed_size):
        super().__init__()
        self.trunk = trunk
        self.embed_size = embed_size

    def forward(self, images):
        features = self.trunk(images)[-1]
        if features.shape[-2:] != (self.embed_size, self.embed_size):
            features = torch.nn.functional.interpolate(
                features,
                size=(self.embed_size, self.embed_size),
                mode="bilinear",
                align_corners=False,
            )
        return features


def parse_option():
    parser = argparse.ArgumentParser(
        "EfficientSAM3 Stage-1 training from images", add_help=False
    )
    add_common_args(parser)
    args = parser.parse_args()
    config = get_config(args)
    return args, config


def build_image_loader(config):
    config.defrost()
    dataset_train, config.MODEL.NUM_CLASSES = build_dataset(
        is_train=True, config=config
    )
    config.freeze()

    sampler_train = MyDistributedSampler(
        dataset_train,
        shuffle=True,
        drop_last=False,
        padding=True,
        pair=False,
    )
    data_loader_train = DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
        collate_fn=pseudo_collate,
    )
    return dataset_train, data_loader_train, sampler_train


def build_vit_teacher(config, device):
    if not TEACHER_CHECKPOINT.is_file():
        raise FileNotFoundError(f"SAM3 teacher checkpoint not found: {TEACHER_CHECKPOINT}")

    teacher_config = config.clone()
    teacher_config.defrost()
    teacher_config.MODEL.RESUME = str(TEACHER_CHECKPOINT)
    teacher_config.freeze()

    full_teacher = build_image_teacher_model(teacher_config)
    vit_trunk = full_teacher.sam3.backbone.vision_backbone.trunk
    teacher = SAM3ViTTeacher(vit_trunk, config.DISTILL.EMBED_SIZE)
    del full_teacher

    for parameter in teacher.parameters():
        parameter.requires_grad = False
    teacher.eval()
    teacher.to(device)
    return teacher


def main(args, config, logger):
    dataset_train, data_loader_train, sampler_train = build_image_loader(config)
    device = (
        torch.device("cpu")
        if args.only_cpu
        else torch.device("cuda", config.LOCAL_RANK)
    )

    logger.info(f"Loading SAM3 ViT teacher from {TEACHER_CHECKPOINT}")
    teacher = build_vit_teacher(config, device)

    logger.info(f"Creating student: {config.MODEL.BACKBONE}")
    model = build_image_student_model(config).to(device)
    if args.use_sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    optimizer = build_optimizer(config, model)
    if not args.only_cpu:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[config.LOCAL_RANK],
            broadcast_buffers=False,
            find_unused_parameters=config.TRAIN.FIND_UNUSED_PARAMETERS,
        )
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    loss_scaler = NativeScalerWithGradNormCount(
        grad_scaler_enabled=config.AMP_ENABLE
    )
    lr_scheduler = build_scheduler(
        config,
        optimizer,
        len(data_loader_train) // config.TRAIN.ACCUMULATION_STEPS,
    )

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(
                    f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}"
                )
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f"Auto-resuming from {resume_file}")

    if config.MODEL.RESUME:
        load_checkpoint(
            config,
            model_without_ddp,
            optimizer,
            lr_scheduler,
            loss_scaler,
            logger,
        )
        if config.EVAL_MODE:
            return

    loss_writer = None
    if dist.get_rank() == 0:
        log_path = datetime.datetime.now().strftime("%Y-%m-%d/%H:%M:%S")
        loss_writer = SummaryWriter(f"{config.OUTPUT}/{log_path}")

    logger.info("Start online image-to-ViT distillation")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        sampler_train.set_epoch(epoch)

        train_one_epoch(
            args,
            config,
            model,
            teacher,
            data_loader_train,
            optimizer,
            epoch,
            lr_scheduler,
            loss_scaler,
            loss_writer,
            logger,
            device,
        )

        if dist.get_rank() == 0 and (
            epoch % config.SAVE_FREQ == 0
            or epoch == config.TRAIN.EPOCHS - 1
        ):
            save_checkpoint(
                config,
                epoch,
                model_without_ddp,
                0.0,
                optimizer,
                lr_scheduler,
                loss_scaler,
                logger,
            )

    total_time = time.time() - start_time
    logger.info(f"Training time {datetime.timedelta(seconds=int(total_time))}")


def train_one_epoch(
    args,
    config,
    model,
    teacher,
    data_loader,
    optimizer,
    epoch,
    lr_scheduler,
    loss_scaler,
    loss_writer,
    logger,
    device,
):
    model.train()
    teacher.eval()
    set_bn_state(config, model)
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    mse_meter = AverageMeter()
    cos_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()
    meters = defaultdict(AverageMeter)
    start = time.time()
    end = start
    data_tic = time.time()

    for idx, (samples, annos) in enumerate(data_loader):
        samples = torch.stack(samples, dim=0).to(device, non_blocking=True)
        meters["data_time"].update(time.time() - data_tic)

        with torch.cuda.amp.autocast(
            enabled=config.AMP_ENABLE and device.type == "cuda"
        ):
            with torch.no_grad():
                teacher_features = teacher(samples)
            preds = model(samples)

        valid_mask = build_valid_mask(
            config,
            annos["img_size_before_pad"],
            preds.shape,
            preds.device,
        )
        mse = masked_mse(preds, teacher_features.float(), valid_mask)
        mse_meter.update(mse.item(), samples.size(0))
        loss = mse

        with torch.no_grad():
            cos_sim = 1.0 - masked_cosine_loss(
                preds.float(), teacher_features.float(), valid_mask
            )
        cos_meter.update(cos_sim.item(), samples.size(0))

        if config.DISTILL.COSINE > 0.0:
            loss += config.DISTILL.COSINE * masked_cosine_loss(
                preds, teacher_features.float(), valid_mask
            )

        loss = loss / config.TRAIN.ACCUMULATION_STEPS
        loss_meter.update(loss.detach().item(), samples.size(0))
        grad_norm = loss_scaler(
            loss,
            optimizer,
            clip_grad=config.TRAIN.CLIP_GRAD,
            parameters=model.parameters(),
            create_graph=False,
            update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0,
        )
        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update(
                (epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS
            )

        loss_scale_value = loss_scaler.state_dict().get("scale", 1.0)
        if grad_norm is not None and not torch.isnan(grad_norm):
            norm_meter.update(grad_norm)
        scaler_meter.update(loss_scale_value)

        if device.type == "cuda":
            torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()
        data_tic = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]["lr"]
            memory_used = (
                torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                if device.type == "cuda"
                else 0.0
            )
            eta = batch_time.avg * (num_steps - idx)
            logger.info(
                f"Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]  "
                f"eta {datetime.timedelta(seconds=int(eta))}  "
                f"lr {lr:.6f}  time {batch_time.val:.4f} ({batch_time.avg:.4f})  "
                f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})  "
                f"mse {mse_meter.val:.4f} ({mse_meter.avg:.4f})  "
                f"cos_sim {cos_meter.val:.4f} ({cos_meter.avg:.4f})  "
                f"grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})  "
                f"loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})  "
                f"mem {memory_used:.0f}MB"
            )

        if loss_writer is not None:
            step = epoch * num_steps + idx
            loss_writer.add_scalar("loss/total", loss.item(), step)
            loss_writer.add_scalar("metric/cos_sim", cos_sim.item(), step)
            loss_writer.add_scalar("metric/mse", mse.item(), step)

        if getattr(config.DATA, "DEBUG", False) and idx >= 0:
            logger.info("DATA.DEBUG=True: stopping after one batch.")
            break

    logger.info(
        f"EPOCH {epoch} training takes "
        f"{datetime.timedelta(seconds=int(time.time() - start))}  "
        f"avg cos_sim {cos_meter.avg:.4f}"
    )


if __name__ == "__main__":
    args, config = parse_option()
    config.defrost()
    config.freeze()

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        rank = -1
        world_size = -1

    if args.only_cpu:
        backend = "gloo"
    else:
        torch.cuda.set_device(config.LOCAL_RANK)
        backend = "nccl"

    torch.distributed.init_process_group(
        backend=backend, init_method="env://", world_size=world_size, rank=rank
    )
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    if not args.only_cpu:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = not args.only_cpu

    linear_scaled_lr = (
        config.TRAIN.BASE_LR
        * config.DATA.BATCH_SIZE
        * dist.get_world_size()
        / 512.0
    )
    linear_scaled_warmup_lr = (
        config.TRAIN.WARMUP_LR
        * config.DATA.BATCH_SIZE
        * dist.get_world_size()
        / 512.0
    )
    linear_scaled_min_lr = (
        config.TRAIN.MIN_LR
        * config.DATA.BATCH_SIZE
        * dist.get_world_size()
        / 512.0
    )
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr *= config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr *= config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr *= config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(
        output_dir=config.OUTPUT,
        dist_rank=dist.get_rank(),
        name=f"{config.MODEL.NAME}",
    )
    if is_main_process():
        with open(os.path.join(config.OUTPUT, "config.json"), "w") as f:
            f.write(config.dump())

        config_dict = dict(config)
        config_dict["git"] = get_git_info()
        if args.use_wandb and wandb is not None:
            wandb.init(
                project="EfficientSAM3-Stage1",
                config=config_dict,
                dir=config.OUTPUT,
            )

    logger.info(f"===== git =====\n{get_git_info()}\n{config.dump()}")
    main(args, config, logger)