import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class SAM3ProposalGenerator(nn.Module):
    """
    SAM3 をマスクプロポーザル生成器としてラップする。

    3つの backend を自動選択:
      1. transformers.Sam3Model   (将来: >= 4.60.0)
      2. sam3 パッケージ (直接import)
      3. subprocess bridge        (.venv-sam3/bin/python 経由)
    """

    def __init__(
        self,
        model_name: str = "facebook/sam3",
        mask_threshold: float = 0.5,
        max_masks: int = 200,
    ) -> None:
        super().__init__()
        self.mask_threshold = mask_threshold
        self.max_masks = max_masks

        self.model: Optional[nn.Module] = None
        self.processor = None
        self._backend: str = self._resolve_backend(model_name)

    @staticmethod
    def _ensure_sam3_package_importable() -> bool:
        """Ensure the real sam3 package (sam3/sam3/) is importable.

        The top-level sam3/ directory (git submodule root, no __init__.py)
        creates a namespace package that shadows the editable-installed real
        package.  This method uses importlib to load the real package from
        sam3/sam3/__init__.py directly and registers it in sys.modules so
        that subsequent ``from sam3 import ...`` statements resolve correctly.
        """
        import importlib.util

        real_init = (
            Path(__file__).resolve().parent.parent.parent
            / "sam3" / "sam3" / "__init__.py"
        )
        if not real_init.exists():
            return False

        try:
            spec = importlib.util.spec_from_file_location("sam3", real_init)
            if spec is None or spec.loader is None:
                return False
            sam3_mod = importlib.util.module_from_spec(spec)
            sys.modules["sam3"] = sam3_mod
            spec.loader.exec_module(sam3_mod)
            return True
        except Exception:
            import traceback
            traceback.print_exc()
            return False

    def _resolve_backend(self, model_name: str) -> str:
        import importlib

        # 1. transformers
        if importlib.util.find_spec("transformers"):
            import transformers
            from packaging.version import Version

            if Version(transformers.__version__) >= Version("4.60.0"):
                try:
                    from transformers import Sam3Model, Sam3Processor

                    self.model = Sam3Model.from_pretrained(model_name).eval()
                    self.processor = Sam3Processor.from_pretrained(model_name)
                    for p in self.model.parameters():
                        p.requires_grad = False
                    return "transformers"
                except ImportError:
                    pass

        # 2. sam3 package
        if not self._ensure_sam3_package_importable():
            pass
        else:
            try:
                from sam3.model.sam3_image_processor import Sam3Processor
                from sam3.model_builder import build_sam3_image_model
            except ImportError:
                pass
            else:
                # Monkey-patch SAM3's MLP: perflib.fused.addmm_act casts
                # fc1 activations to bfloat16, but fc2 expects float32,
                # causing a dtype mismatch.  Replace Mlp.forward with a
                # simple linear+activation path instead of the fused kernel.
                from sam3.model import vitdet as _vitdet
                from sam3.perflib import fused as _fused

                def _mlp_forward(self, x):
                    x = self.fc1(x)
                    x = self.act(x)
                    x = self.drop1(x)
                    x = self.norm(x)
                    x = self.fc2(x)
                    x = self.drop2(x)
                    return x

                _vitdet.Mlp.forward = _mlp_forward

                m = build_sam3_image_model()
                self.model = m.eval()
                self.processor = Sam3Processor(m)
                for p in self.model.parameters():
                    p.requires_grad = False
                return "sam3_package"

        # 3. subprocess
        venv_python = self._find_sam3_python()
        if venv_python is not None:
            return "subprocess"

        raise RuntimeError(
            "SAM3 をロードできませんでした。以下のいずれかが必要:\n"
            "  1. transformers >= 4.60.0\n"
            "  2. sam3 パッケージがインストール済み\n"
            "  3. .venv-sam3/ に Python 3.12 の SAM3 環境"
        )

    @staticmethod
    def _find_sam3_python() -> Optional[str]:
        candidates = [
            Path(__file__).resolve().parent.parent.parent / ".venv-sam3" / "bin" / "python",
            Path.cwd() / ".venv-sam3" / "bin" / "python",
        ]
        for p in candidates:
            if p.is_file():
                return str(p)
        return None

    @staticmethod
    def _worker_script() -> str:
        return str(Path(__file__).resolve().parent / "sam3_worker.py")

    def forward(
        self,
        image: torch.Tensor,
        text_prompts: list[str],
        original_image_size: Optional[tuple[int, int]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            image: raw input image (C, H, W), float32 in [0, 255], RGB
            text_prompts: list of text prompts
            original_image_size: (H, W) of the input image
        Returns:
            masks: [N, H_orig, W_orig] float32 binary masks
            scores: [N] float32 confidence scores
        """
        h_orig, w_orig = original_image_size or image.shape[-2:]

        if self._backend == "transformers":
            return self._forward_transformers(image, text_prompts, h_orig, w_orig)
        elif self._backend == "sam3_package":
            return self._forward_sam3_package(image, text_prompts, h_orig, w_orig)
        else:
            return self._forward_subprocess(image, text_prompts, h_orig, w_orig)

    def _forward_transformers(
        self,
        image: torch.Tensor,
        text_prompts: list[str],
        h_orig: int,
        w_orig: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.processor is not None
        assert self.model is not None

        if image.dim() == 3:
            image = image.unsqueeze(0)

        device = next(self.model.parameters()).device
        inputs = self.processor(images=image, text=text_prompts, return_tensors="pt")
        inputs = inputs.to(device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        result = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=self.mask_threshold,
            mask_threshold=self.mask_threshold,
            target_sizes=[(h_orig, w_orig)],
        )[0]

        if result["masks"].shape[0] == 0:
            return (
                torch.zeros(0, h_orig, w_orig, dtype=torch.float32, device=device),
                torch.zeros(0, dtype=torch.float32, device=device),
            )

        masks = result["masks"].float()
        scores = result["scores"]

        if masks.shape[0] > self.max_masks:
            topk_scores, topk_indices = torch.topk(scores, self.max_masks)
            masks = masks[topk_indices]
            scores = topk_scores

        return masks, scores

    def _forward_sam3_package(
        self,
        image: torch.Tensor,
        text_prompts: list[str],
        h_orig: int,
        w_orig: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.processor is not None
        from PIL import Image as PILImage

        device = next(self.model.parameters()).device if self.model is not None else image.device

        img_np = image.detach().cpu().numpy()
        if img_np.ndim == 4:
            img_np = img_np[0]
        img_np = img_np.transpose(1, 2, 0).clip(0, 255).astype("uint8")
        pil_image = PILImage.fromarray(img_np, mode="RGB")

        inference_state = self.processor.set_image(pil_image)

        all_masks: list[torch.Tensor] = []
        all_scores: list[torch.Tensor] = []

        for prompt in text_prompts:
            output = self.processor.set_text_prompt(state=inference_state, prompt=prompt)
            masks_np: list = output["masks"]
            scores_list: list = output["scores"]
            for m, s in zip(masks_np, scores_list):
                if isinstance(m, torch.Tensor):
                    mask_t = m.float().to(device)
                else:
                    mask_t = torch.from_numpy(np.asarray(m, dtype=np.float32)).to(device)
                if isinstance(s, torch.Tensor):
                    score_t = s.to(device)
                else:
                    score_t = torch.tensor(float(s), device=device)
                all_masks.append(mask_t)
                all_scores.append(score_t)

        if len(all_masks) == 0:
            return (
                torch.zeros(0, h_orig, w_orig, dtype=torch.float32, device=device),
                torch.zeros(0, dtype=torch.float32, device=device),
            )

        masks = torch.stack(all_masks)
        scores = torch.stack(all_scores)

        if masks.shape[-2] != h_orig or masks.shape[-1] != w_orig:
            masks = F.interpolate(
                masks.unsqueeze(1), size=(h_orig, w_orig), mode="bilinear", align_corners=False
            ).squeeze(1)

        masks = (masks > self.mask_threshold).float()

        if masks.shape[0] > self.max_masks:
            topk_scores, topk_indices = torch.topk(scores, self.max_masks)
            masks = masks[topk_indices]
            scores = topk_scores

        return masks, scores

    def _forward_subprocess(
        self,
        image: torch.Tensor,
        text_prompts: list[str],
        h_orig: int,
        w_orig: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = image.device

        img_np = image.detach().cpu().numpy()
        if img_np.ndim == 4:
            img_np = img_np[0]
        img_np = img_np.transpose(1, 2, 0).clip(0, 255).astype("uint8")

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = os.path.join(tmpdir, "input.png")
            input_json = os.path.join(tmpdir, "input.json")
            output_npz = os.path.join(tmpdir, "output.npz")

            PILImage = __import__("PIL.Image", fromlist=["Image"]).Image
            PILImage.fromarray(img_np, mode="RGB").save(image_path)

            with open(input_json, "w") as f:
                json.dump({
                    "image_path": image_path,
                    "text_prompts": text_prompts,
                    "mask_threshold": self.mask_threshold,
                    "max_masks": self.max_masks,
                }, f)

            sam3_python = self._find_sam3_python()
            assert sam3_python is not None, "SAM3 subprocess python not found"

            result = subprocess.run(
                [sam3_python, self._worker_script(), "--input-json", input_json, "--output-npz", output_npz],
                capture_output=True, text=True, check=True,
            )

            if not os.path.exists(output_npz):
                raise RuntimeError(
                    f"SAM3 worker failed. stdout: {result.stdout}\nstderr: {result.stderr}"
                )

            data = np.load(output_npz)
            masks_np: np.ndarray = data["masks"]
            scores_np: np.ndarray = data["scores"]

        if masks_np.shape[0] == 0:
            return (
                torch.zeros(0, h_orig, w_orig, dtype=torch.float32, device=device),
                torch.zeros(0, dtype=torch.float32, device=device),
            )

        masks = torch.from_numpy(masks_np).to(device)
        scores = torch.from_numpy(scores_np).to(device)

        if masks.shape[-2] != h_orig or masks.shape[-1] != w_orig:
            masks = F.interpolate(
                masks.unsqueeze(1), size=(h_orig, w_orig), mode="bilinear", align_corners=False
            ).squeeze(1)

        masks = (masks > self.mask_threshold).float()
        return masks, scores
