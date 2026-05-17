#!/usr/bin/env python3
"""
Reconstruct an approximate image of what actually goes into CLIP visual encoder
when mask-prompt blending is applied.

Given an image and a binary mask, this script:
1. Applies CLIP's conv1 to get patch embeddings.
2. Downsamples the mask to patch level and binarizes it.
3. Blends: inside mask -> original patch embedding, outside mask -> mask prompt embedding.
4. Uses the pseudo-inverse of conv1 to map blended patch embeddings back to pixel patches.
5. Denormalizes and saves the resulting image.

This lets you visually inspect "what the CLIP encoder sees" when the background
is replaced by mask prompt embeddings instead of black/zeros.
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# CLIP default preprocess constants
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])


def load_checkpoint(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" in ckpt:
        return ckpt["state_dict"]
    if "model" in ckpt:
        return ckpt["model"]
    return ckpt


def find_key(state_dict, substring):
    candidates = [k for k in state_dict.keys() if substring in k]
    if not candidates:
        raise RuntimeError(f"No key containing '{substring}' found.")
    # prefer shortest / most exact match
    candidates.sort(key=lambda s: len(s))
    return candidates[0]


def preprocess_image(pil_img, resolution):
    """Resize short side to resolution, center crop to (resolution, resolution), normalize."""
    w, h = pil_img.size
    scale = resolution / min(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    pil_img = pil_img.resize((new_w, new_h), Image.BICUBIC)
    # center crop
    left = (new_w - resolution) // 2
    top = (new_h - resolution) // 2
    pil_img = pil_img.crop((left, top, left + resolution, top + resolution))
    img = np.array(pil_img).astype(np.float32) / 255.0  # (H, W, 3)
    img = torch.from_numpy(img).permute(2, 0, 1)  # (3, H, W)
    # normalize
    mean = CLIP_MEAN.view(3, 1, 1)
    std = CLIP_STD.view(3, 1, 1)
    img_norm = (img - mean) / std
    return img, img_norm  # raw [0,1] and normalized


def reconstruct_patches_from_embeddings(embeddings, conv1_weight):
    """
    embeddings: (C, H, W) where H, W are grid sizes
    conv1_weight: (C, 3, ps, ps)
    Returns: (3, H*ps, W*ps) reconstructed image
    """
    C, H, W = embeddings.shape
    ps = conv1_weight.shape[-1]
    # (C, 3*ps*ps)
    Wmat = conv1_weight.view(C, -1)
    # pseudo inverse: (3*ps*ps, C)
    Wpinv = torch.linalg.pinv(Wmat)

    # flatten embeddings: (H*W, C)
    flat = embeddings.permute(1, 2, 0).reshape(-1, C)
    # reconstruct pixel patches: (H*W, 3*ps*ps)
    patches = flat @ Wpinv.T
    patches = patches.view(H, W, 3, ps, ps)
    # merge: (3, H*ps, W*ps)
    img = patches.permute(2, 0, 3, 1, 4).reshape(3, H * ps, W * ps)
    return img


def main():
    parser = argparse.ArgumentParser(
        description="Approximate reconstruction of CLIP visual input under mask-prompt blending."
    )
    parser.add_argument("--image", "-i", required=True, help="Input image path")
    parser.add_argument(
        "--mask", "-m", required=True, help="Binary mask path (white=foreground)"
    )
    parser.add_argument(
        "--checkpoint", "-c", required=True, help="Model checkpoint (.pth)"
    )
    parser.add_argument(
        "--layer", "-l", type=int, default=0, help="Mask prompt layer depth to use"
    )
    parser.add_argument(
        "--output_dir", "-o", default="recon_output", help="Output directory"
    )
    parser.add_argument(
        "--mask_is_background",
        action="store_true",
        help="If set, mask white region is treated as BACKGROUND (replace with prompt).",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load checkpoint parameters
    # ------------------------------------------------------------------
    sd = load_checkpoint(args.checkpoint)
    conv1_key = find_key(sd, "visual.conv1.weight")
    mask_emb_key = find_key(sd, "visual.mask_embedding")
    conv1_weight = sd[conv1_key]  # (C, 3, ps, ps)
    mask_embedding = sd[mask_emb_key]  # (D, G, C)

    C, _, ps, _ = conv1_weight.shape
    D, G, C2 = mask_embedding.shape
    assert C == C2, f"Channel mismatch: conv1={C}, mask_emb={C2}"
    if args.layer >= D:
        raise ValueError(
            f"Requested layer {args.layer} but mask_embedding depth is {D}"
        )
    grid_h = grid_w = int(np.sqrt(G))
    if grid_h * grid_w != G:
        common = {
            196: (14, 14),
            256: (16, 16),
            441: (21, 21),
            576: (24, 24),
            784: (28, 28),
            1024: (32, 32),
            144: (12, 12),
            100: (10, 10),
            64: (8, 8),
            49: (7, 7),
        }
        if G in common:
            grid_h, grid_w = common[G]
        else:
            raise ValueError(f"Cannot infer grid size for G={G}")
    resolution = grid_h * ps
    print(
        f"Detected: patch_size={ps}, grid={grid_h}x{grid_w}, resolution={resolution}, channels={C}"
    )
    print(f"Using mask prompt layer {args.layer} (/{D})")

    mask_emb_layer = mask_embedding[args.layer]  # (G, C)
    # reshape to spatial: (C, grid_h, grid_w)
    mask_emb_spatial = mask_emb_layer.permute(1, 0).view(C, grid_h, grid_w)

    # ------------------------------------------------------------------
    # 2. Load and preprocess image / mask
    # ------------------------------------------------------------------
    img_pil = Image.open(args.image).convert("RGB")
    img_raw, img_norm = preprocess_image(img_pil, resolution)  # both (3, H, W)

    mask_pil = Image.open(args.mask).convert("L")
    mask_pil = mask_pil.resize((resolution, resolution), Image.NEAREST)
    mask_np = np.array(mask_pil)
    # normalize mask to 0/1. threshold at 128.
    mask_bin = (mask_np > 128).astype(np.float32)
    mask_t = torch.from_numpy(mask_bin).unsqueeze(0)  # (1, H, W)

    # If user says mask white is background, invert
    if args.mask_is_background:
        mask_t = 1.0 - mask_t

    # ------------------------------------------------------------------
    # 3. Forward through conv1
    # ------------------------------------------------------------------
    x = F.conv2d(img_norm.unsqueeze(0), conv1_weight, stride=ps)  # (1, C, gh, gw)
    x = x.squeeze(0)  # (C, gh, gw)

    # downsample mask to patch grid
    m = F.avg_pool2d(mask_t, kernel_size=ps, stride=ps)  # (1, gh, gw)
    m = torch.ceil(m).squeeze(0)  # (gh, gw), binary 0/1
    # foreground (m=1) keeps original patch, background (m=0) gets prompt

    # blend
    x_blend = x * m + mask_emb_spatial * (1.0 - m)

    # ------------------------------------------------------------------
    # 4. Reconstruct images
    # ------------------------------------------------------------------
    # A) Original normalized input (for reference)
    recon_orig = reconstruct_patches_from_embeddings(x, conv1_weight)
    # B) Black-background masked (set bg patches to zero embedding -> black in recon)
    x_black = x * m  # zero out background embeddings
    recon_black = reconstruct_patches_from_embeddings(x_black, conv1_weight)
    # C) Mask-prompt blended
    recon_prompt = reconstruct_patches_from_embeddings(x_blend, conv1_weight)
    # D) Pure mask prompt (no image)
    recon_pure_prompt = reconstruct_patches_from_embeddings(
        mask_emb_spatial, conv1_weight
    )

    # ------------------------------------------------------------------
    # 5. Denormalize and save
    # ------------------------------------------------------------------
    mean = CLIP_MEAN.view(3, 1, 1)
    std = CLIP_STD.view(3, 1, 1)

    def save_tensor(t, name):
        t = t * std + mean
        t = torch.clamp(t, 0.0, 1.0)
        arr = (t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(arr).save(os.path.join(args.output_dir, name))
        print(f"Saved: {name}")

    save_tensor(recon_orig, "01_recon_original.png")
    save_tensor(recon_black, "02_recon_mask_only_black_bg.png")
    save_tensor(recon_prompt, "03_recon_mask_only_prompt_bg.png")
    save_tensor(recon_pure_prompt, "04_recon_pure_prompt.png")

    # Also save the raw preprocessed image for easy comparison
    img_raw_clamped = torch.clamp(img_raw, 0, 1)
    arr = (img_raw_clamped.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(arr).save(os.path.join(args.output_dir, "00_raw_preprocessed.png"))
    print(f"Saved: 00_raw_preprocessed.png")

    # And the mask
    mask_arr = (mask_t.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(mask_arr).save(os.path.join(args.output_dir, "00_mask.png"))
    print(f"Saved: 00_mask.png")

    print(f"\nAll outputs saved to: {args.output_dir}")
    print("Note: This is an approximation because positional embeddings, LayerNorm,")
    print("and the Transformer are ignored in the reconstruction.")


if __name__ == "__main__":
    main()
