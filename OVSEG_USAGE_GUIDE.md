# OVSeg Repository Summary

## Overview

---

This repository implements OVSeg, an open-vocabulary semantic segmentation system built on top of Detectron2, MaskFormer, and CLIP.

The main things you can do here are:

- Run open-vocabulary semantic segmentation on arbitrary images with user-provided class names
- Evaluate pretrained OVSeg checkpoints on supported benchmark datasets
- Train or fine-tune the OVSeg segmentation model
- Fine-tune CLIP on mask-category pairs for OVSeg
- Run mask prompt tuning after CLIP fine-tuning
- Prepare builtin datasets in Detectron2 format

The main entry points are:

- `demo.py`
- `tools/web_demo.py`
- `train_net.py`
- `open_clip_training/src/training/main.py`
- `open_clip_training/src/training/main_mask_prompt_tuning.py`

## Environment Setup

This repository is intended to be managed with `uv`.

```bash
uv python install 3.9
uv sync --python 3.9
```

Optional extras:

```bash
uv sync --python 3.9 --extra clip
uv sync --python 3.9 --extra open-clip
uv sync --python 3.9 --extra demo
uv sync --python 3.9 --extra test
```

## 1. Run Open-Vocabulary Segmentation on Images

Use `demo.py` when you want to segment an image with your own class vocabulary.

Example:

```bash
uv run python demo.py \
  --config-file configs/ovseg_swinB_vitL_demo.yaml \
  --class-names 'Oculus' 'Ukulele' \
  --input ./resources/demo_samples/sample_03.jpeg \
  --output ./pred \
  --opts MODEL.WEIGHTS /path/to/ovseg_swinbase_vitL14_ft_mpt.pth
```

What it does:

- Loads the demo config
- Accepts one or more input images
- Uses your `--class-names` as the segmentation vocabulary
- Saves visualized predictions to `--output`

## 2. Run a Web Demo

Use the Gradio app if you want a browser-based UI.

```bash
uv run python tools/web_demo.py
```

What it does:

- Launches a local web UI
- Lets you upload an image
- Accepts comma-separated class names
- Returns a visualized segmentation map

## 3. Evaluate Pretrained Checkpoints

Use `train_net.py --eval-only` for benchmark evaluation.

Example: ADE20K-150 and ADE20K-847

```bash
uv run python train_net.py \
  --num-gpus 8 \
  --eval-only \
  --config-file configs/ovseg_swinB_vitL_bs32_120k.yaml \
  MODEL.WEIGHTS /path/to/ovseg_swinbase_vitL14_ft_mpt.pth \
  DATASETS.TEST '("ade20k_sem_seg_val","ade20k_full_sem_seg_val")'
```

Example: Pascal Context

```bash
uv run python train_net.py \
  --num-gpus 8 \
  --eval-only \
  --config-file configs/ovseg_swinB_vitL_bs32_120k.yaml \
  MODEL.WEIGHTS /path/to/ovseg_swinbase_vitL14_ft_mpt.pth \
  MODEL.CLIP_ADAPTER.CLIP_ENSEMBLE_WEIGHT 0.6 \
  DATASETS.TEST '("pascal_context_59_sem_seg_val","pascal_context_459_sem_seg_val")'
```

Example: Pascal VOC

```bash
uv run python train_net.py \
  --num-gpus 8 \
  --eval-only \
  --config-file configs/ovseg_swinB_vitL_bs32_120k.yaml \
  MODEL.WEIGHTS /path/to/ovseg_swinbase_vitL14_ft_mpt.pth \
  MODEL.CLIP_ADAPTER.CLIP_ENSEMBLE_WEIGHT 0.45 \
  DATASETS.TEST '("pascalvoc20_sem_seg_val")'
```

Supported builtin evaluation targets:

- `ade20k_sem_seg_val`
- `ade20k_full_sem_seg_val`
- `pascal_context_59_sem_seg_val`
- `pascal_context_459_sem_seg_val`
- `pascalvoc20_sem_seg_val`

## 4. Train the Segmentation Model

Use `train_net.py` without `--eval-only`.

Baseline training with the original CLIP:

```bash
uv run python train_net.py \
  --config-file configs/ovseg_swinB_vitL_bs32_120k.yaml \
  OUTPUT_DIR  output/mask_prompt_fwd_false \
  MODEL.CLIP_ADAPTER.MASK_PROMPT_FWD False
```

Training OVSeg with a mask-adapted CLIP checkpoint:

```bash
uv run python train_net.py \
  --config-file configs/ovseg_swinB_vitL_bs32_120k.yaml \
  OUTPUT_DIR  output/mask_adapted_clip \
  MODEL.CLIP_ADAPTER.CLIP_MODEL_NAME /path/to/mask_adapted_clip.pt
```

Notes:

- The standard training dataset is COCO-Stuff-171
- The main training entry point is `train_net.py`
- Logging is integrated with `wandb`

## 5. Run Ablations and Inference Variants

You can change inference behavior through config overrides.

Examples:

Disable mask prompt forward:

```bash
uv run python train_net.py \
  --eval-only \
  --config-file configs/ovseg_swinB_vitL_bs32_120k.yaml \
  MODEL.CLIP_ADAPTER.MASK_PROMPT_FWD False \
  MODEL.WEIGHTS /path/to/ovseg_swinbase_vitL14_ft_mpt.pth \
  DATASETS.TEST '("ade20k_sem_seg_val","ade20k_full_sem_seg_val")'
```

Disable CLIP ensemble:

```bash
uv run python train_net.py \
  --eval-only \
  --config-file configs/ovseg_swinB_vitL_bs32_120k.yaml \
  MODEL.CLIP_ADAPTER.CLIP_ENSEMBLE False \
  MODEL.WEIGHTS /path/to/ovseg_swinbase_vitL14_ft_mpt.pth \
  DATASETS.TEST '("ade20k_sem_seg_val","ade20k_full_sem_seg_val")'
```

Other configurable inference options include:

- `MODEL.CLIP_ADAPTER.CLIP_ENSEMBLE_WEIGHT`
- `MODEL.CLIP_ADAPTER.MASK_THR`
- `TEST.SLIDING_WINDOW`
- `TEST.SLIDING_TILE_SIZE`
- `TEST.SLIDING_OVERLAP`
- `TEST.DENSE_CRF`

## 6. Fine-Tune CLIP for OVSeg

The CLIP training code lives under `open_clip_training/`.

The practical execution path from the repository root is:

```bash
uv run torchrun --nproc_per_node 4 -m training.main \
  --train-data open_clip_training/openclip_data/coco_proposal_1cap.csv \
  --train-num-samples 442117 \
  --lr 5e-6 \
  --warmup 100 \
  --force-quick-gelu \
  --dataset-type csv \
  --batch-size 32 \
  --precision amp \
  --workers 4 \
  --model ViT-L-14 \
  --lock-text \
  --zeroshot-frequency 1 \
  --save-frequency 1 \
  --epochs 5 \
  --pretrained openai \
  --ade-val open_clip_training/openclip_data/ade_gt_150cls_val
```

What it does:

- Fine-tunes CLIP on collected mask-category pairs
- Writes checkpoints under the OpenCLIP logs directory
- Produces CLIP weights that can later be inserted into an OVSeg checkpoint

## 7. Run Mask Prompt Tuning

After CLIP fine-tuning, you can run mask prompt tuning.

```bash
uv run torchrun --nproc_per_node 4 -m training.main_mask_prompt_tuning \
  --train-data open_clip_training/openclip_data/coco_proposal_1cap.csv \
  --train-num-samples 442117 \
  --lr 0.05 \
  --mask_wd 0.0 \
  --warmup 100 \
  --force-quick-gelu \
  --dataset-type csv \
  --batch-size 32 \
  --precision amp \
  --workers 4 \
  --with-mask \
  --model ViT-L-14 \
  --mask-emb-depth 3 \
  --lock-text \
  --lock-image \
  --lock-image-unlocked-groups 0 \
  --zeroshot-frequency 1 \
  --save-frequency 1 \
  --epochs 5 \
  --pretrained /path/to/finetuned_clip_checkpoint.pt \
  --ade-val open_clip_training/openclip_data/ade_gt_150cls_val
```

What it does:

- Loads a fine-tuned CLIP checkpoint
- Tunes mask prompt parameters
- Produces a stronger CLIP variant for OVSeg

## 8. Merge Fine-Tuned CLIP Back Into OVSeg

There is a helper script for replacing the CLIP weights inside an OVSeg checkpoint:

```bash
uv run python tools/ovseg_replace_clip.py
```

Important:

- This helper script currently contains hard-coded absolute paths
- You will likely need to edit it before use

There is also a sanity-check script:

```bash
uv run python tools/sanity_check_ft_clip_weights.py
```

## 9. Prepare Builtin Datasets

Builtin datasets are expected under `DETECTRON2_DATASETS`, or `./datasets` by default.

Supported builtin datasets:

- COCO-Stuff-171
- ADE20K-150
- ADE20K-847
- Pascal VOC-20
- Pascal Context-59
- Pascal Context-459

Preparation scripts:

```bash
uv run python datasets/prepare_coco_stuff_sem_seg.py
uv run python datasets/prepare_ade20k_sem_seg.py
uv run python datasets/prepare_ade20k_full_sem_seg.py
uv run python datasets/prepare_voc_sem_seg.py
uv run python datasets/prepare_pascal_context.py
```

If your datasets live elsewhere:

```bash
export DETECTRON2_DATASETS=/path/to/datasets
```

## 10. Available Model Configurations

The repository includes at least these predefined configs:

- `configs/ovseg_swinB_vitL_demo.yaml`
- `configs/ovseg_swinB_vitL_bs32_120k.yaml`
- `configs/ovseg_R101c_vitB_bs32_120k.yaml`

In practice, these correspond to:

- Swin-Base backbone with CLIP ViT-L/14
- ResNet-101c backbone with CLIP ViT-B/16

## 11. What This Repository Is Not

This repository is primarily for semantic segmentation research and reproduction.

It is not structured as:

- A general object detection framework
- A generic panoptic segmentation training suite
- A polished production package

There are helper scripts, but some of them are research-oriented and may require local path fixes before use.

## Quick Start

If you want the shortest path to something useful:

1. Install the environment with `uv sync --python 3.9`
2. Download a released OVSeg checkpoint
3. Run `demo.py` for custom-class image segmentation
4. Run `train_net.py --eval-only` for benchmark evaluation
5. Use `open_clip_training` only if you need to reproduce the CLIP adaptation pipeline
