# Stage 1 (mod): fused stride-16 / stride-32 image student

`stage1_mod` distills the SAM3 ViT image encoder into a student that outputs a 1024x72x72 map without interpolation.

- The teacher runs at 1008x1008 (patch 14 -> 72x72) and its embeddings are saved to disk.
- The student input size is `DATA.IMG_SIZE` and its head is `MODEL.STUDENT.HEAD` (see "Student heads and image size" below). The default is **1152x1152** with `fusion_s3s4`: stage 3 (stride 16) is 72x72 and stage 4 (stride 32) is 36x36, merged by a learned head into the 1024x72x72 output.
- Supported backbones: RepViT (`es_rv_*`) and EfficientViT (`es_ev_*`). TinyViT (`es_tv_*`) is not supported yet.

Run all commands from the repository root. The SAM3 checkpoint is expected at `sam3_checkpoints/sam3/sam3.pt`.

## 1. Save teacher embeddings

Embeddings are written once per image. Use `DATA.NUM_SAMPLES` to limit how many images are saved (omit it for the whole dataset).

Single-image test set (`data/SA-1B-1P`):

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 \
  stage1_mod/save_embedding_image_stage1.py \
  --cfg stage1_mod/configs/teacher/sam_vit_huge_sa1b_5p_sam3.yaml \
  --data-path data/SA-1B-1P \
  --output output/stage1_teacher \
  --batch-size 1 \
  --opts DATA.NUM_SAMPLES 1 \
         MODEL.RESUME sam3_checkpoints/sam3/sam3.pt \
         DISTILL.TEACHER_EMBED_PATH output/stage1_teacher/embeddings/
```

Larger run (multi-GPU):

```bash
PYTHONPATH=. python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
  stage1_mod/save_embedding_image_stage1.py \
  --cfg stage1_mod/configs/teacher/sam_vit_huge_sa1b_5p_sam3.yaml \
  --data-path data/SA-1B-5P \
  --output output/stage1_teacher \
  --opts MODEL.RESUME sam3_checkpoints/sam3/sam3.pt \
         DISTILL.TEACHER_EMBED_PATH output/stage1_teacher/embeddings/
```

Optionally verify the saved embeddings against a fresh teacher forward pass by adding `--check-saved-embed` to either command.

## 2. Train the student

The effective learning rate is `TRAIN.BASE_LR * batch_size * num_gpus / 512` (same for `WARMUP_LR` and `MIN_LR`). With batch size 1 on one GPU, `BASE_LR 5.12` gives an effective LR of 1e-2.

Single-image overfit test (RepViT-M):

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 \
  stage1_mod/train_image_encoder_stage1.py \
  --cfg stage1_mod/configs/es_rv_m.yaml \
  --data-path data/SA-1B-1P \
  --output output/stage1_mod_rv_m_1img_lr1e-2 \
  --batch-size 1 \
  --opts DATA.NUM_SAMPLES 1 \
         TRAIN.EPOCHS 300 \
         TRAIN.WARMUP_EPOCHS 5 \
         SAVE_FREQ 50 \
         DISTILL.TEACHER_EMBED_PATH output/stage1_teacher/embeddings/ \
         TRAIN.BASE_LR 5.12 \
         TRAIN.WARMUP_LR 0.000512 \
         TRAIN.MIN_LR 0.00512
```

Other backbones: swap `--cfg` for `es_rv_s.yaml`, `es_rv_l.yaml`, `es_ev_s.yaml`, `es_ev_m.yaml` or `es_ev_l.yaml`.

Notes:
- `TRAIN.AUTO_RESUME` is on. Use a new `--output` directory for each run, or the trainer resumes from the latest checkpoint it finds there.
- Checkpoints from the old interpolation-based student do not load into the fusion student.
- The log prints `loss`, `cos_sim` (mean student/teacher cosine similarity over valid pixels) and `grad_norm` every `PRINT_FREQ` steps. TensorBoard logs `loss/total` and `metric/cos_sim` under `--output`.

## Student heads and image size

The image student is a backbone plus a head (`STUDENT_HEADS` in `stage1_mod/model.py`). The head turns the backbone's stride-16/32 maps into the 1024x72x72 embedding. Pick it with `MODEL.STUDENT.HEAD`; `DATA.IMG_SIZE` must suit the head:

| `MODEL.STUDENT.HEAD` | Input size | Description |
| --- | --- | --- |
| `fusion_s3s4` (default) | **1152 only** | Fuses stride-16 (72x72) and stride-32 (36x36, upsampled 2x). |
| `s3_only` | **1152 only** | Stride-16 features, 1x1 projection. |
| `resample_fusion_s3s4` | any, e.g. **1008** | Same fusion, with both maps bilinearly resized to 72x72. |
| `interp_s4` | any, e.g. **1008** | Original head: final map, conv head, bilinear resize to 72x72. |

Heads that need 1152 raise an error at build time if `DATA.IMG_SIZE` gives a different stride-16 grid (1008 gives 63x63).

Presets in `stage1_mod/configs/student/` set the size and head together. Either list one in a config's `BASE` (see `stage1_mod/configs/es_rv_m_1008.yaml`) or override on the command line:

```bash
--opts DATA.IMG_SIZE 1008 MODEL.STUDENT.HEAD interp_s4
```

To add a head: subclass `StudentHead`, set `required_strides` and `resamples`, and add it to `STUDENT_HEADS`. `MODEL.S3_ONLY` is a deprecated alias for `s3_only` so older runs still load.

## 3. Run an experiment and evaluate it

```bash
python stage1_mod/run_experiment.py \
  stage1_mod/configs/experiments/s4_interp_then_refine_efficient.yaml \
  --batch-size 1 --opts DATA.NUM_SAMPLES 1
```

The runner trains with the experiment config, analyzes the latest checkpoint, then evaluates its prompts using the fixtures in `stage1_mod/efficient_sam_tests/`. Trainer options after the config path are forwarded to `train_image_encoder_stage1.py`; `--nproc-per-node` controls local training workers (default 1). The config's merged `OUTPUT` sets the experiment directory; use `--output-dir` to override it. Embedding analysis is saved under `analysis/`, and SAM result images under `sam_test_images/` in that directory.

To only test an existing run, invoke `stage1_mod/test_efficient_sam.py` with `--stage1-mod-dir <experiment_output_dir>` and `--output-dir <experiment_output_dir>/sam_test_images`. Images are preprocessed as in training (longest side resized to `DATA.IMG_SIZE`, padded bottom/right). The "missing keys" message at start-up refers to the throwaway trunk that is replaced, and is expected.
