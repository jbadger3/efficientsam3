from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from typing import Literal

from sam3.model_builder import build_sam3_image_model
from sam3.backbones.repvit import (
    _make_divisible,
    repvit_m0_9,
    repvit_m1_1,
    repvit_m2_3,
)
from sam3.backbones.tiny_vit import (
    tiny_vit_5m_224,
    tiny_vit_11m_224,
    tiny_vit_21m_224,
)
from sam3.backbones.efficientvit import (
    efficientvit_backbone_b0,
    efficientvit_backbone_b1,
    efficientvit_backbone_b2,
)
from sam3.sam3.backbones.mobile_clip import MobileCLIPTextTransformer
from sam3.model.tokenizer_ve import SimpleTokenizer
from sam3.model.text_encoder_student import TextStudentEncoder


def resolve_head_name(config):
    if getattr(config.MODEL, "S3_ONLY", False):  # Deprecated alias used by older runs.
        return "s3_only"
    return config.MODEL.STUDENT.HEAD


def build_image_student_model(config):
    head_name = resolve_head_name(config)
    if head_name not in STUDENT_HEADS:
        raise ValueError(f"Unknown MODEL.STUDENT.HEAD {head_name!r}; choose from {sorted(STUDENT_HEADS)}")
    head_cls = STUDENT_HEADS[head_name]

    embed_size = config.DISTILL.EMBED_SIZE
    img_size = config.DATA.IMG_SIZE
    if not head_cls.resamples:
        stride = head_cls.required_strides[0]
        grid = -(-img_size // stride)
        if grid != embed_size:
            raise ValueError(
                f"Head {head_name!r} needs a {embed_size}x{embed_size} stride-{stride} grid but "
                f"DATA.IMG_SIZE={img_size} gives {grid}x{grid}. Use DATA.IMG_SIZE {embed_size * stride} "
                f"or a resampling head ({[n for n, c in STUDENT_HEADS.items() if c.resamples]})."
            )

    backbone = _build_backbone(config.MODEL.BACKBONE.lower(), img_size)
    backbone.keep_strides(head_cls.required_strides)
    head = head_cls(
        channels={s: backbone.channels[s] for s in head_cls.required_strides},
        embed_dim=config.DISTILL.EMBED_DIM,
        embed_size=embed_size,
        hidden_dim=config.MODEL.STUDENT.FUSION_DIM,
        **(
            {
                "conv_layers": config.MODEL.STUDENT.CONV_LAYERS,
                "conv_type": config.MODEL.STUDENT.CONV_TYPE,
            }
            if head_name == "efficient_head"
            else {}
        ),
    )
    return ImageStudentEncoder(backbone=backbone, head=head, embed_size=embed_size)


def build_text_student_model(config, logger=None):
    backbone = config.MODEL.BACKBONE
    
    # Default config values
    cfg = {
        "context_length": 77, # MobileCLIP default
        "vocab_size": 49408,
        "dim": 512,
        "ffn_multiplier_per_layer": 4.0,
        "n_heads_per_layer": 8,
        "n_transformer_layers": 12,
        "norm_layer": "layer_norm_fp32",
        "causal_masking": False,
        "model_name": "base",
        "embed_dropout": 0.0,
        "no_scale_embedding": False,
        "no_pos_embedding": False,
    }

    if backbone == "MobileCLIP-S0":
        cfg.update({
            "dim": 512,
            "n_transformer_layers": 4,  # Original MobileCLIP-S0 design
            "n_heads_per_layer": 8,
            "model_name": "mct",  # RepMixer architecture
            "ffn_multiplier_per_layer": 4.0,  # Original FFN multiplier
        })
    elif backbone in ["MobileCLIP-S1", "MobileCLIP2-S0", "MobileCLIP2-S2"]:
        cfg.update({
            "dim": 512,
            "n_transformer_layers": 12,
            "n_heads_per_layer": 8,
            "model_name": "base",
        })
    elif backbone == "MobileCLIP-B":
        cfg.update({
            "dim": 512,
            "n_transformer_layers": 12,
            "n_heads_per_layer": 8,
            "model_name": "base",
            "causal_masking": True,
        })
    elif backbone in ["MobileCLIP2-S3", "MobileCLIP2-S4", "MobileCLIP2-L"]:
        cfg.update({
            "dim": 768,
            "n_transformer_layers": 12,
            "n_heads_per_layer": 12,
            "model_name": "base", 
        })
        # Note: MobileCLIP2 might use "custom_text" or similar, but "base" with larger dim should work if architecture is standard transformer.
        # The subagent said "custom_text: true, no_causal_mask: true".
        # If "custom_text" implies standard transformer, then "base" is fine.
    else:
        # Default fallback
        pass

    # CONTEXT_LENGTH controls tokenization; POS_EMBED_TABLE_SIZE controls the learnable table.
    context_length = getattr(config.DISTILL, 'CONTEXT_LENGTH', 32)
    pos_embed_table_size = getattr(config.DISTILL, 'POS_EMBED_TABLE_SIZE', 0)
    if pos_embed_table_size in (None, 0):
        pos_embed_table_size = context_length
    cfg["context_length"] = pos_embed_table_size

    model = TextStudentEncoder(
        cfg=cfg,
        context_length=context_length,
        output_dim=config.DISTILL.EMBED_DIM,
    )

    if logger:
        logger.info(f"Text encoder context_length: {context_length}")
        logger.info(f"Text encoder pos_embed_table_size: {pos_embed_table_size}")

    # Load pretrained weights if specified
    if hasattr(config.MODEL, 'PRETRAINED') and config.MODEL.PRETRAINED:
        pretrained_path = config.MODEL.PRETRAINED
        if logger:
            logger.info(f"Loading pretrained text encoder from: {pretrained_path}")
        
        try:
            pretrained_state = torch.load(pretrained_path, map_location='cpu')
            
            # Handle full MobileCLIP checkpoint (contains image + text encoders)
            if 'text_encoder.embedding_layer.weight' in pretrained_state:
                if logger:
                    logger.info("Detected full MobileCLIP checkpoint, extracting text encoder...")
                # Extract text encoder weights
                text_encoder_state = {}
                for key, value in pretrained_state.items():
                    if key.startswith('text_encoder.'):
                        new_key = key.replace('text_encoder.', 'encoder.')
                        text_encoder_state[new_key] = value
                pretrained_state = text_encoder_state
            
            # Load weights with flexible matching
            missing_keys, unexpected_keys = model.load_state_dict(pretrained_state, strict=False)
            
            if logger:
                logger.info(f"Loaded pretrained weights:")
                logger.info(f"  Missing keys: {len(missing_keys)}")
                if len(missing_keys) > 0 and len(missing_keys) <= 10:
                    for key in missing_keys:
                        logger.info(f"    - {key}")
                logger.info(f"  Unexpected keys: {len(unexpected_keys)}")
                if len(unexpected_keys) > 0 and len(unexpected_keys) <= 10:
                    for key in unexpected_keys:
                        logger.info(f"    - {key}")
                
                # The projector layer should be missing (we train it from scratch)
                expected_missing = ['projector.weight', 'projector.bias']
                actual_missing_important = [k for k in missing_keys if k not in expected_missing]
                if len(actual_missing_important) > 0:
                    logger.warning(f"  Important missing keys (not projector): {actual_missing_important}")
                else:
                    logger.info(f"  ✓ All important keys loaded successfully")
        
        except Exception as e:
            if logger:
                logger.error(f"Failed to load pretrained weights: {e}")
                logger.warning("Continuing with random initialization...")
            else:
                print(f"Warning: Failed to load pretrained weights: {e}")
    
    return model


def build_image_teacher_model(config):
    checkpoint = config.MODEL.RESUME if config.MODEL.RESUME else None
    teacher = SAM3ImageTeacherEncoder(
        checkpoint_path=checkpoint,
        embed_size=config.DISTILL.EMBED_SIZE,
    )
    teacher.img_size = config.DATA.IMG_SIZE
    return teacher


def build_text_teacher_model(config):
    checkpoint = config.MODEL.RESUME if config.MODEL.RESUME else None
    context_length = getattr(config.DISTILL, 'CONTEXT_LENGTH', 32)
    teacher = SAM3TextTeacherEncoder(
        checkpoint_path=checkpoint,
        context_length=context_length,
    )
    return teacher

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        # x shape: (B, C, H, W)
        # Move channels to the end: (B, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = nn.functional.layer_norm(x, (x.shape[-1],), self.weight, self.bias, self.eps)
        # Permute back: (B, C, H, W)
        return x.permute(0, 3, 1, 2)

class DepthwiseSeparableConv(nn.Module):
    """Depthwise spatial convolution followed by a pointwise channel projection."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
    ):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
            padding_mode=padding_mode,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class StudentHead(nn.Module):
    """Maps backbone feature maps ({stride: tensor}) to a (B, embed_dim, S, S) embedding, S = embed_size.

    required_strides: strides the head consumes; the backbone drops deeper stages it does not need.
    resamples: False means the head does not resize, so the grid at required_strides[0] must already
        equal embed_size (this fixes DATA.IMG_SIZE); True means any DATA.IMG_SIZE works.
    """

    required_strides: tuple = ()
    resamples: bool = False

    def __init__(self, channels, embed_dim, embed_size, hidden_dim=256):
        super().__init__()
        self.embed_size = embed_size


class S3OnlyHead(StudentHead):
    """Stride-16 features -> 1x1 projection.

    Expects 1152x1152 input (DATA.IMG_SIZE): 1152 / 16 = 72, matching the teacher grid with no resizing.
    """

    required_strides = (16,)

    def __init__(self, channels, embed_dim, embed_size, hidden_dim=256):
        super().__init__(channels, embed_dim, embed_size, hidden_dim)
        self.output_proj = nn.Conv2d(channels[16], embed_dim, kernel_size=1)

    def forward(self, feats):
        return self.output_proj(feats[16])


class FusionS3S4Head(StudentHead):
    """Fuses stride-16 (72x72) and stride-32 (36x36, upsampled 2x) features, then projects to embed_dim.

    Expects 1152x1152 input (DATA.IMG_SIZE): s3 is 72x72 and s4 is 36x36, so the 2x upsample lines up.
    """

    required_strides = (16, 32)

    def __init__(self, channels, embed_dim, embed_size, hidden_dim=256):
        super().__init__(channels, embed_dim, embed_size, hidden_dim)
        self.s3_proj = nn.Conv2d(channels[16], hidden_dim, 1, bias=False)
        self.s4_proj = nn.Conv2d(channels[32], hidden_dim, 1, bias=False)
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )
        self.output_proj = nn.Conv2d(hidden_dim, embed_dim, kernel_size=1)

    def _align(self, s3, s4):
        return s3, F.interpolate(s4, scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, feats):
        s3, s4 = self._align(self.s3_proj(feats[16]), self.s4_proj(feats[32]))
        return self.output_proj(self.fusion(torch.cat([s3, s4], dim=1)))


class ResampledFusionS3S4Head(FusionS3S4Head):
    """Same as FusionS3S4Head, but bilinearly resizes both maps to embed_size before fusing.

    Works at any input size, e.g. 1008x1008 (s3 is 63x63, s4 is 32x32). Parameter names match
    FusionS3S4Head.
    """

    resamples = True

    def _align(self, s3, s4):
        size = (self.embed_size, self.embed_size)
        return tuple(
            F.interpolate(x, size=size, mode="bilinear", align_corners=False) for x in (s3, s4)
        )


class BaselineS4Head(StudentHead):
    """Original student head: final (stride-32) map -> conv head -> bilinear resize to embed_size.

    Works at any input size, e.g. 1008x1008 (32x32 map resized to 72x72). Hidden width is embed_dim.
    """

    required_strides = (32,)
    resamples = True

    def __init__(self, channels, embed_dim, embed_size, hidden_dim=256):
        super().__init__(channels, embed_dim, embed_size, hidden_dim)
        self.head = nn.Sequential(
            nn.Conv2d(channels[32], embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
        )

    def forward(self, feats):
        x = self.head(feats[32])
        size = (self.embed_size, self.embed_size)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

class S4InterpThenRefineHead(StudentHead):
    """Original student swapping order of operations: final (stride-32) map -> interpolate -> conv head.

    Works at any input size, e.g. 1008x1008 (32x32 map resized to 72x72). Hidden width is embed_dim.
    """

    required_strides = (32,)
    resamples = True

    def __init__(self, channels, embed_dim, embed_size, hidden_dim=256):
        super().__init__(channels, embed_dim, embed_size, hidden_dim)
        self.head = nn.Sequential(
            nn.Conv2d(channels[32], embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
        )

    def forward(self, feats):
        size = (self.embed_size, self.embed_size)
        x = F.interpolate(feats[32], size=size, mode="bilinear", align_corners=False)
        x = self.head(x)
        
        return x

class EfficientHead(StudentHead):
    """Repeatable spatial Conv/BatchNorm/GELU blocks followed by channel projections.

    Works at any input size, e.g. 1008x1008 (32x32 map resized to 72x72). Hidden width is embed_dim.
    """

    required_strides = (32,)
    resamples = True

    def __init__(self, channels, embed_dim, embed_size, hidden_dim=256, conv_layers=1, conv_type="standard"):
        super().__init__(channels, embed_dim, embed_size, hidden_dim)

        if conv_layers < 0:
            raise ValueError(f"conv_layers must be non-negative, got {conv_layers}")
        if conv_type not in {"standard", "depthwise_separable"}:
            raise ValueError(
                f"Unknown conv_type {conv_type!r}; choose 'standard' or 'depthwise_separable'"
            )

        layers = []
        for _ in range(conv_layers):
            if conv_type == "standard":
                conv = nn.Conv2d(channels[32], channels[32], kernel_size=3, padding=1, bias=False)
            else:
                conv = DepthwiseSeparableConv(
                    channels[32], channels[32], kernel_size=3, padding=1, bias=False
                )
            layers.extend((conv, LayerNorm2d(channels[32]), nn.GELU()))

        layers.extend(
            (
                nn.Conv2d(channels[32], embed_dim, kernel_size=1, bias=False),
                LayerNorm2d(embed_dim),
                nn.GELU(),
                nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=True),
            )
        )
        self.head = nn.Sequential(*layers)

    def forward(self, feats):
        size = (self.embed_size, self.embed_size)
        x = F.interpolate(feats[32], size=size, mode="bilinear", align_corners=False)
        x = self.head(x)
        
        return x

class EfficientHeadV2(StudentHead):
    """Refined spatial head with residual connections and correct normalization."""

    required_strides = (32,)
    resamples = True

    def __init__(self, channels, embed_dim, embed_size, hidden_dim=256, conv_layers=2, conv_type="standard"):
        super().__init__(channels, embed_dim, embed_size, hidden_dim)
        
        c = channels[32]
        layers = []
        for _ in range(conv_layers):
            if conv_type == "standard":
                conv = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False)
            else:
                conv = DepthwiseSeparableConv(c, c, kernel_size=3, padding=1, bias=False)
            
            # Using GroupNorm(1, c) which behaves like LayerNorm across channels, preserving H and W spatial stats
            layers.extend((
                conv, 
                nn.GroupNorm(num_groups=1, num_channels=c), 
                nn.GELU()
            ))
        
        self.spatial_refine = nn.Sequential(*layers)
        
        # Channel projection block
        self.projection = nn.Sequential(
            nn.Conv2d(c, embed_dim, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=True),
        )

    def forward(self, feats):
        x = feats[32]
        
        # 1. Refine spatial features at low-res first (with a residual skip connection)
        x = x + self.spatial_refine(x)
        
        # 2. Interpolate to target teacher size
        size = (self.embed_size, self.embed_size)
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        
        # 3. Project to teacher's embedding dimension
        x = self.projection(x)
        
        return x


class S4InterpThenRefineEfficientHead(StudentHead):
    """Original student swapping order of operations: final (stride-32) map -> interpolate -> conv head.

    Works at any input size, e.g. 1008x1008 (32x32 map resized to 72x72). Hidden width is embed_dim.
    """

    required_strides = (32,)
    resamples = True

    def __init__(self, channels, embed_dim, embed_size, hidden_dim=256):
        super().__init__(channels, embed_dim, embed_size, hidden_dim)
        self.channel_match = nn.Conv2d(channels[32], embed_dim, kernel_size=1, bias=False)
        self.head = nn.Sequential(
            DepthwiseSeparableConv(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            DepthwiseSeparableConv(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, feats):
        x = self.channel_match(feats[32])
        size = (self.embed_size, self.embed_size)
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        x = self.head(x)
        
        return x



class S3OnlyLowResInterpHead(StudentHead):
    """stride-16 map -> conv head -> bilinear resize to embed_size.

    Works at any input size, e.g. 1008x1008 (63x63 map resized to 72x72). Hidden width is embed_dim.
    """

    required_strides = (16,)
    resamples = True

    def __init__(self, channels, embed_dim, embed_size, hidden_dim=256):
        super().__init__(channels, embed_dim, embed_size, hidden_dim)
        self.head = nn.Sequential(
            nn.Conv2d(channels[16], channels[16], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels[16]),
            nn.GELU(),
            nn.Conv2d(channels[16], embed_dim, kernel_size=1),
        )

    def forward(self, feats):
        size = (self.embed_size, self.embed_size)
        x = F.interpolate(feats[16], size=size, mode="bilinear", align_corners=False)
        x = self.head(x)

        return x


# Selected by MODEL.STUDENT.HEAD.
STUDENT_HEADS = {
    "fusion_s3s4": FusionS3S4Head,  # 1152 only
    "s3_only": S3OnlyHead,  # 1152 only
    "resample_fusion_s3s4": ResampledFusionS3S4Head,  # any size, e.g. 1008
    "baseline_s4": BaselineS4Head,  # any size, e.g. 1008
    "s4_interp_then_refine": S4InterpThenRefineHead,  # any size, e.g. 1008
    "s3only_lowres_interp": S3OnlyLowResInterpHead,  # any size, e.g. 1008
    "s4_interp_then_refine_efficient": S4InterpThenRefineEfficientHead,  # any size, e.g. 1008
    "efficient_head": EfficientHead,  # any size, e.g. 1008
    "efficient_head_v2": EfficientHeadV2,  # any size, e.g. 1008
}


class ImageStudentEncoder(nn.Module):
    def __init__(self, backbone, head, embed_size=72):
        super().__init__()

        self.backbone = backbone
        self.embed_size = embed_size
        # Attribute name kept as `fusion` so checkpoints from earlier runs still load.
        self.fusion = head

    def forward(self, x):
        feats = self.fusion(self.backbone(x))

        assert feats.shape[-2:] == (
            self.embed_size,
            self.embed_size,
        )

        return feats
        

        

class SAM3ImageTeacherEncoder(nn.Module):
    def __init__(self, checkpoint_path=None, embed_size=64):
        super().__init__()
        self.embed_size = embed_size
        self.sam3 = build_sam3_image_model(
            checkpoint_path=checkpoint_path,
            load_from_HF=False if checkpoint_path else True,
            eval_mode=True,
            device="cpu",
            enable_segmentation=True,
            enable_inst_interactivity=False,
            compile=False,
            enable_text_encoder=False,
        )
        for param in self.sam3.parameters():
            param.requires_grad = False
        self.sam3.eval()
        self.img_size = 1008

    def forward(self, x):
        # Distill the raw backbone features (1024 channels) to allow
        # the student to be a drop-in replacement for the backbone.
        # Access the ViT trunk directly through the vision_backbone
        backbone_out = self.sam3.backbone.vision_backbone.trunk(x)
        # ViT returns a list of features, we want the last one
        feats = backbone_out[-1]
        
        # Interpolate if needed (though usually trunk output is already at the target resolution)
        if feats.shape[-1] != self.embed_size or feats.shape[-2] != self.embed_size:
            feats = F.interpolate(
                feats,
                size=(self.embed_size, self.embed_size),
                mode="bilinear",
                align_corners=False,
            )
        return feats


class SAM3TextTeacherEncoder(nn.Module):
    def __init__(self, checkpoint_path=None, context_length=32):
        super().__init__()
        self.context_length = context_length
        self.sam3 = build_sam3_image_model(
            checkpoint_path=checkpoint_path,
            load_from_HF=False if checkpoint_path else True,
            eval_mode=True,
            device="cpu",
            enable_segmentation=True,
            enable_inst_interactivity=False,
            compile=False,
            enable_text_encoder=True,
            enable_vision_encoder=False,
        )
        for param in self.sam3.parameters():
            param.requires_grad = False
        self.sam3.eval()

        # Override the context_length in the language backbone
        if hasattr(self.sam3.backbone, 'language_backbone'):
            self.sam3.backbone.language_backbone.context_length = context_length

    def forward(self, text, device):
        # text is a list of strings
        text_attention_mask, text_memory, text_embeds = self.sam3.backbone.language_backbone(
            text, input_boxes=None, device=device
        )
        # text_memory is [Seq, Batch, 256]
        # Truncate to context_length if needed
        if text_memory.shape[0] > self.context_length:
            text_memory = text_memory[:self.context_length]
        return text_memory


class RepViTAdapter(nn.Module):
    """Returns {16: stride-16 map, 32: stride-32 map}; the stride-32 stage is dropped if unused."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        # features[0] is patch_embed (stride 4); cfgs[i] maps to features[i + 1].
        down_idx = [i for i, cfg in enumerate(model.cfgs) if cfg[5] == 2]
        assert len(down_idx) == 3, "expected three stride-2 blocks"
        self.s4_start = down_idx[2] + 1
        self.channels = {
            16: _make_divisible(model.cfgs[down_idx[2] - 1][2], 8),
            32: _make_divisible(model.cfgs[-1][2], 8),
        }

    def keep_strides(self, strides):
        """Drop stages beyond the deepest needed stride so no parameters go without gradients."""
        if 32 not in strides:
            self.model.features = self.model.features[: self.s4_start]
            self.s4_start = len(self.model.features)
            del self.channels[32]

    def forward(self, x):
        features = self.model.features
        for layer in features[: self.s4_start]:
            x = layer(x)
        out = {16: x}
        if 32 in self.channels:
            for layer in features[self.s4_start :]:
                x = layer(x)
            out[32] = x
        return out


class TinyViTAdapter(nn.Module):
    def __init__(self, model, img_size):
        super().__init__()
        self.model = model
        self.model.head = nn.Identity()
        self.final_hw = self._compute_resolution(img_size)
        self.out_channels = self.model.norm_head.normalized_shape[0]
        # Remove norm_head to avoid DDP unused parameter error
        self.model.norm_head = nn.Identity()

    def forward(self, x):
        x = self.model.patch_embed(x)
        x = self.model.layers[0](x)
        for i in range(1, len(self.model.layers)):
            x = self.model.layers[i](x)
        B, N, C = x.shape
        H, W = self.final_hw
        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return x

    def _compute_resolution(self, img_size):
        H, W = self.model.patches_resolution
        for _ in range(self.model.num_layers - 1):
            H = (H - 1) // 2 + 1
            W = (W - 1) // 2 + 1
        return (H, W)


class EfficientViTAdapter(nn.Module):
    """Returns {16: stage3 map, 32: stage4 map}; stage 4 is dropped if unused."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.channels = {16: model.width_list[-2], 32: model.width_list[-1]}

    def keep_strides(self, strides):
        """Drop stages beyond the deepest needed stride so no parameters go without gradients."""
        if 32 not in strides:
            self.model.stages = self.model.stages[:-1]
            del self.channels[32]

    def forward(self, x):
        out = self.model(x)
        return {s: out[f"stage{3 if s == 16 else 4}"] for s in self.channels}


class EfficientSAM3VisionBackbone(nn.Module):
    def __init__(self, student_encoder, position_encoding):
        super().__init__()
        self.student_encoder = student_encoder
        self.position_encoding = position_encoding

    def forward(self, x):
        feats = self.student_encoder(x)
        sam3_out = [feats]
        sam3_pos = [self.position_encoding(feats).to(feats.dtype)]
        sam2_out = None
        sam2_pos = None
        return sam3_out, sam3_pos, sam2_out, sam2_pos


def build_efficient_sam3(config, checkpoint_path=None):
    student_encoder = build_image_student_model(config)

    # Build base SAM3 model structure
    sam3 = build_sam3_image_model(
        checkpoint_path=None,
        load_from_HF=False,
        eval_mode=True,
        device="cpu",
        enable_segmentation=True,
        enable_inst_interactivity=False,
        compile=False,
    )

    # Replace vision backbone
    original_pos_enc = sam3.backbone.vision_backbone.position_encoding
    vision_backbone = EfficientSAM3VisionBackbone(student_encoder, original_pos_enc)
    sam3.backbone.vision_backbone = vision_backbone
    
    # Disable scalping since student encoder only returns one feature map
    sam3.backbone.scalp = 0
    
    if checkpoint_path:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if "model" in state_dict:
            state_dict = state_dict["model"]
            
        # No remapping needed if checkpoint was converted with correct prefixes
        missing, unexpected = sam3.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint with {len(missing)} missing and {len(unexpected)} unexpected keys")
        
    return sam3

def _build_backbone(name, img_size):
    if name.startswith("repvit"):
        fn = {
            "repvit_m0_9": repvit_m0_9,
            "repvit_m1_1": repvit_m1_1,
            "repvit_m2_3": repvit_m2_3,
        }[name]
        model = fn(pretrained=False, num_classes=0, distillation=False)
        adapter = RepViTAdapter(model)
        return adapter

    if name.startswith("tiny_vit"):
        raise NotImplementedError("tiny_vit does not expose stride-16/32 feature maps yet")
        fn = {
            "tiny_vit_5m": tiny_vit_5m_224,
            "tiny_vit_11m": tiny_vit_11m_224,
            "tiny_vit_21m": tiny_vit_21m_224,
        }[name]
        model = fn(pretrained=False, img_size=img_size)
        adapter = TinyViTAdapter(model, img_size)
        return adapter, adapter.out_channels

    if name.startswith("efficientvit"):
        fn = {
            "efficientvit_b0": efficientvit_backbone_b0,
            "efficientvit_b1": efficientvit_backbone_b1,
            "efficientvit_b2": efficientvit_backbone_b2,
        }[name]
        model = fn()
        adapter = EfficientViTAdapter(model)
        return adapter

    raise ValueError(f"Unsupported backbone {name}")


