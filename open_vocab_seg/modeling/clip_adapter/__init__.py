# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

from typing import Any

from .adapter import ClipAdapter, MaskFormerClipAdapter
from .text_template import (
    ImageNetPromptExtractor,
    PredefinedPromptExtractor,
    VILDPromptExtractor,
)


def build_text_prompt(cfg: Any) -> Any:
    if cfg.TEXT_TEMPLATES == "predefined":
        text_templates = PredefinedPromptExtractor(cfg.PREDEFINED_PROMPT_TEMPLATES)
    elif cfg.TEXT_TEMPLATES == "imagenet":
        text_templates = ImageNetPromptExtractor()
    elif cfg.TEXT_TEMPLATES == "vild":
        text_templates = VILDPromptExtractor()
    else:
        raise NotImplementedError(
            f"Prompt learner {cfg.TEXT_TEMPLATES} is not supported"
        )
    return text_templates
