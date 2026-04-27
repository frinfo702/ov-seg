#!/usr/bin/env python3
"""
Visualize mask prompt embeddings from a checkpoint by forcibly treating them as images.

Mask prompt shape: (mask_prompt_depth, grid_size*grid_size, width)
Example: (3, 196, 768) for ViT-B/16 with depth=3

This script produces:
1. PCA-RGB images    -> embed 768-dim patch vectors into 3-D via PCA and display as RGB
2. Norm heatmaps     -> L2 norm of each patch embedding
3. Channel stats     -> mean/std across the width dimension per patch
4. Similarity maps   -> cosine similarity between center patch and all other patches
"""

import argparse
import os
import re

import numpy as np
import torch

# matplotlib may not be available in headless env; handle gracefully
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except Exception:
    HAS_MPL = False
    print("Warning: matplotlib not available. Only raw NumPy arrays will be saved.")


def find_mask_embedding(state_dict):
    """Search state_dict keys for mask_embedding."""
    candidates = [k for k in state_dict.keys() if "mask_embedding" in k]
    if not candidates:
        raise RuntimeError("No key containing 'mask_embedding' found in checkpoint.")
    # Prefer exact match or visual.mask_embedding
    for c in candidates:
        if c.endswith("visual.mask_embedding") or c.endswith("mask_embedding"):
            print(f"Found mask_embedding key: {c}")
            return state_dict[c]
    print(f"Found mask_embedding key: {candidates[0]}")
    return state_dict[candidates[0]]


def pca_to_rgb(X, n_components=3):
    """
    X: (N, C) numpy array.
    Returns: (N, 3) numpy array in [0, 1].
    """
    X = X - X.mean(axis=0, keepdims=True)
    # Covariance-free trick via SVD
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    comps = Vt[:n_components].T  # (C, 3)
    proj = X @ comps  # (N, 3)

    # Robust normalization: clip at 1st/99th percentile then scale to [0,1]
    p_low, p_high = np.percentile(proj, [1, 99], axis=0, keepdims=True)
    proj = np.clip(proj, p_low, p_high)
    mx = proj.max(axis=0, keepdims=True)
    mn = proj.min(axis=0, keepdims=True)
    rng = mx - mn
    rng[rng == 0] = 1.0
    proj = (proj - mn) / rng
    return proj


def save_image_array(arr, path, title=""):
    """arr: (H, W, 3) RGB in [0,1] or (H, W) grayscale."""
    if not HAS_MPL:
        # Fallback: save raw npy
        np.save(path.replace(".png", ".npy"), arr)
        return
    plt.figure(figsize=(6, 6))
    if arr.ndim == 3:
        plt.imshow(arr)
    else:
        plt.imshow(arr, cmap="viridis")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def visualize_mask_embedding(tensor: torch.Tensor, out_dir: str, prefix: str = "layer"):
    """
    tensor: (D, G, C) where G = grid_h * grid_w
    """
    os.makedirs(out_dir, exist_ok=True)
    D, G, C = tensor.shape
    grid_h = grid_w = int(np.sqrt(G))
    if grid_h * grid_w != G:
        # Try common grid sizes if not perfect square (e.g. 224->14, 336->21, 384->24)
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
            # fallback: make as square as possible
            grid_w = int(np.ceil(np.sqrt(G)))
            grid_h = int(np.ceil(G / grid_w))
            print(
                f"Warning: G={G} is not a perfect square. Using ({grid_h}, {grid_w}) padding."
            )

    print(f"Processing mask_embedding: depth={D}, grid={grid_h}x{grid_w}, width={C}")

    for d in range(D):
        X = tensor[d].cpu().numpy()  # (G, C)

        # Pad if needed
        if grid_h * grid_w != G:
            pad = grid_h * grid_w - G
            X = np.concatenate([X, np.zeros((pad, C), dtype=X.dtype)], axis=0)

        # --- 1. PCA-RGB ---
        rgb = pca_to_rgb(X)  # (G, 3)
        rgb_img = rgb.reshape(grid_h, grid_w, 3)
        save_image_array(
            rgb_img,
            os.path.join(out_dir, f"{prefix}_{d}_pca_rgb.png"),
            title=f"{prefix} {d} – PCA RGB (C={C})",
        )

        # --- 2. Norm heatmap ---
        norms = np.linalg.norm(X, axis=1).reshape(grid_h, grid_w)
        save_image_array(
            norms,
            os.path.join(out_dir, f"{prefix}_{d}_norm.png"),
            title=f"{prefix} {d} – L2 Norm",
        )

        # --- 3. Mean / Std per patch ---
        mean_map = X.mean(axis=1).reshape(grid_h, grid_w)
        std_map = X.std(axis=1).reshape(grid_h, grid_w)
        save_image_array(
            mean_map,
            os.path.join(out_dir, f"{prefix}_{d}_mean.png"),
            title=f"{prefix} {d} – Mean",
        )
        save_image_array(
            std_map,
            os.path.join(out_dir, f"{prefix}_{d}_std.png"),
            title=f"{prefix} {d} – Std",
        )

        # --- 4. Cosine similarity to center patch ---
        center_idx = (grid_h // 2) * grid_w + (grid_w // 2)
        center_vec = X[center_idx]
        # safe cosine similarity
        num = X @ center_vec
        den = np.linalg.norm(X, axis=1) * np.linalg.norm(center_vec)
        den[den == 0] = 1.0
        sim = (num / den).reshape(grid_h, grid_w)
        save_image_array(
            sim,
            os.path.join(out_dir, f"{prefix}_{d}_cosine_center.png"),
            title=f"{prefix} {d} – Cosine Sim to Center",
        )

    print(f"Saved visualizations to: {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize mask prompt embeddings as images."
    )
    parser.add_argument(
        "--checkpoint", "-c", required=True, help="Path to .pth checkpoint"
    )
    parser.add_argument(
        "--output_dir", "-o", default="mask_prompt_vis", help="Output directory"
    )
    parser.add_argument(
        "--key", default=None, help="Exact state_dict key (auto-detect if omitted)"
    )
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")

    if "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif "model" in ckpt:
        sd = ckpt["model"]
    else:
        sd = ckpt

    if args.key is not None:
        if args.key not in sd:
            raise KeyError(
                f"Specified key '{args.key}' not found. Available keys:\n"
                + "\n".join([k for k in sd.keys() if "mask" in k.lower()][:20])
            )
        mask_emb = sd[args.key]
    else:
        mask_emb = find_mask_embedding(sd)

    if not isinstance(mask_emb, torch.Tensor):
        raise TypeError(f"mask_embedding is not a Tensor (type={type(mask_emb)}).")

    print(f"Raw shape: {tuple(mask_emb.shape)}")
    visualize_mask_embedding(mask_emb, args.output_dir)


if __name__ == "__main__":
    main()
