# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Meta Platforms, Inc. All Rights Reserved
# Modified by Feng Liang from
# https://github.com/MendelXu/zsseg.baseline/blob/master/mask_former/modeling/clip_adapter/adapter.py

from typing import Any, Optional, Union

import torch
from detectron2.structures import BitMasks
from torch import nn
from torch.nn import functional as F

from .text_template import PromptExtractor
from .utils import build_clip_model, crop_with_mask

PIXEL_MEAN = (0.48145466, 0.4578275, 0.40821073)
PIXEL_STD = (0.26862954, 0.26130258, 0.27577711)

RegionBatch = Union[torch.Tensor, list[torch.Tensor]]
PreprocessOutput = tuple[
    tuple[Optional[RegionBatch], Optional[RegionBatch]],
    Optional[RegionBatch],
    torch.Tensor,
]


class ClipAdapter(nn.Module):
    def __init__(
        self,
        clip_model_name: str,
        mask_prompt_depth: int,
        text_templates: PromptExtractor,
    ) -> None:
        super().__init__()
        self.clip_model = build_clip_model(clip_model_name, mask_prompt_depth)
        self.text_templates = text_templates
        self.text_templates.init_buffer(self.clip_model)
        self.text_feature_buffer: dict[str, torch.Tensor] = {}

    def forward(self, image: torch.Tensor, text: list[str], **kwargs: Any) -> torch.Tensor:
        image = self._preprocess_image(image, **kwargs)
        text_feature = self.get_text_features(text)  # k,feat_dim
        image_features = self.get_image_features(image)
        return self.get_sim_logits(text_feature, image_features)

    def _preprocess_image(
        self,
        image: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        normalize: bool = True,
    ) -> torch.Tensor:
        del mask, normalize
        return image

    def _get_text_features(self, noun_list: list[str]) -> torch.Tensor:
        left_noun_list = [
            noun for noun in noun_list if noun not in self.text_feature_buffer
        ]
        if len(left_noun_list) > 0:
            left_text_features = self.text_templates(left_noun_list, self.clip_model)
            self.text_feature_buffer.update(
                {
                    noun: text_feature
                    for noun, text_feature in zip(left_noun_list, left_text_features)
                }
            )
        return torch.stack([self.text_feature_buffer[noun] for noun in noun_list])

    def get_text_features(self, noun_list: list[str]) -> torch.Tensor:
        return self._get_text_features(noun_list)

    def get_image_features(
        self,
        image: torch.Tensor,
        region_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del region_masks
        image_features = self.clip_model.visual(image)  # pyright: ignore[reportCallIssue]
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def get_sim_logits(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
        temperature: float = 100,
    ) -> torch.Tensor:
        return temperature * image_features @ text_features.T

    def normalize_feature(self, feat: torch.Tensor) -> torch.Tensor:
        return feat / feat.norm(dim=-1, keepdim=True)


class MaskFormerClipAdapter(ClipAdapter):
    def __init__(
        self,
        clip_model_name: str,
        text_templates: PromptExtractor,
        mask_fill: str = "mean",
        mask_expand_ratio: float = 1.0,
        mask_thr: float = 0.5,
        mask_matting: bool = False,
        region_resized: bool = True,
        mask_prompt_depth: int = 0,
        mask_prompt_fwd: bool = False,
    ) -> None:
        super().__init__(clip_model_name, mask_prompt_depth, text_templates)
        self.non_object_embedding = nn.Parameter(
            torch.empty(1, self.clip_model.text_projection.shape[-1])  # pyright: ignore[reportAttributeAccessIssue, reportIndexIssue, reportArgumentType]
        )
        nn.init.normal_(
            self.non_object_embedding.data,
            std=self.clip_model.transformer.width**-0.5,  # pyright: ignore[reportOperatorIssue, reportArgumentType]
        )
        # for test
        if mask_fill == "zero":
            self.mask_fill: tuple[float, float, float] = (0.0, 0.0, 0.0)
        elif mask_fill == "mean":
            self.mask_fill = tuple(255.0 * c for c in PIXEL_MEAN)
        else:
            raise NotImplementedError(f"Unknown mask_fill method: {mask_fill}")
        self.mask_expand_ratio = mask_expand_ratio
        self.mask_thr = mask_thr
        self.mask_matting = mask_matting
        self.region_resized = region_resized
        self.mask_prompt_fwd = mask_prompt_fwd
        self.register_buffer(
            "pixel_mean", torch.Tensor(PIXEL_MEAN).reshape(1, 3, 1, 1) * 255.0
        )
        self.register_buffer(
            "pixel_std", torch.Tensor(PIXEL_STD).reshape(1, 3, 1, 1) * 255.0
        )

    def forward(
        self,
        image: torch.Tensor,
        text: list[str],
        mask: torch.Tensor,
        normalize: bool = True,
        fwd_w_region_mask: bool = False,
    ) -> tuple[Optional[torch.Tensor], Any, torch.Tensor]:
        (regions, unnorm_regions), region_masks, valid_flag = self._preprocess_image(
            image, mask, normalize=normalize
        )
        if regions is None:
            return None, None, valid_flag
        if isinstance(regions, list):
            if self.mask_prompt_fwd or fwd_w_region_mask:
                raise NotImplementedError(
                    "Region-mask forwarding requires resized tensor regions."
                )
            image_features = torch.cat(
                [self.get_image_features(image_i) for image_i in regions], dim=0
            )
        else:
            region_mask_tensor = (
                region_masks if isinstance(region_masks, torch.Tensor) else None
            )
            if self.mask_prompt_fwd or fwd_w_region_mask:
                image_features = self.get_image_features(regions, region_mask_tensor)
            else:
                image_features = self.get_image_features(regions)
        text_feature = self.get_text_features(text)  # k,feat_dim
        return (
            self.get_sim_logits(text_feature, image_features),
            unnorm_regions,
            valid_flag,
        )

    def get_image_features(
        self,
        image: torch.Tensor,
        region_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        image_features = self.clip_model.visual(image, region_masks)  # pyright: ignore[reportCallIssue]
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def _preprocess_image(
        self, image: torch.Tensor, mask: torch.Tensor, normalize: bool = True
    ) -> PreprocessOutput:
        """crop, mask and normalize the image

        Args:
            image ([type]): [C,H,W]
            mask ([type]): [K,H,W
            normalize (bool, optional): [description]. Defaults to True.
        """
        dtype = mask.dtype
        bin_mask = mask > self.mask_thr
        valid = bin_mask.sum(dim=(-1, -2)) > 0
        bin_mask = bin_mask[valid]
        mask = mask[valid]
        if not self.mask_matting:
            mask = bin_mask
        bin_mask = BitMasks(bin_mask)
        bboxes = bin_mask.get_bounding_boxes()
        # crop,mask
        regions = []
        region_masks = []
        for bbox, single_mask in zip(bboxes, mask):
            region, region_mask = crop_with_mask(
                image.type(dtype),
                single_mask.type(dtype),
                bbox,
                fill=self.mask_fill,
                expand_ratio=self.mask_expand_ratio,
            )
            regions.append(region.unsqueeze(0))
            region_masks.append(region_mask.unsqueeze(0))
        if len(regions) == 0:
            return (None, None), None, valid
        unnorm_regions = regions
        if normalize:
            regions = [(r - self.pixel_mean) / self.pixel_std for r in regions]
        # resize
        if self.region_resized:
            regions = [
                F.interpolate(r, size=(224, 224), mode="bicubic") for r in regions
            ]
            regions = torch.cat(regions)
            region_masks = [
                F.interpolate(r, size=(224, 224), mode="nearest") for r in region_masks
            ]
            region_masks = torch.cat(region_masks)
            unnorm_regions = [
                F.interpolate(r, size=(224, 224), mode="bicubic")
                for r in unnorm_regions
            ]
            unnorm_regions = torch.cat(unnorm_regions)
        return (regions, unnorm_regions), region_masks, valid

    def get_text_features(self, noun_list: list[str]) -> torch.Tensor:
        object_text_features = self._get_text_features(noun_list)
        non_object_text_features = (
            self.non_object_embedding
            / self.non_object_embedding.norm(dim=-1, keepdim=True)
        )
        return torch.cat([object_text_features, non_object_text_features], dim=0)
