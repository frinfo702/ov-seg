# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

from typing import Optional

import numpy as np
import torch
from torch.nn import functional as F

try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import (
        unary_from_labels,
        unary_from_softmax,
    )
except:
    dcrf = None


def dense_crf_post_process(
    logits: torch.Tensor,
    image: np.ndarray,
    n_labels: Optional[int] = None,
    max_iters: int = 5,
    pos_xy_std: tuple[float, float] = (3, 3),
    pos_w: float = 3,
    bi_xy_std: tuple[float, float] = (80, 80),
    bi_rgb_std: tuple[float, float, float] = (13, 13, 13),
    bi_w: float = 10,
) -> torch.Tensor:
    """
    logits : [C,H,W]
    image : [3,H,W]
    """
    if dcrf is None:
        raise FileNotFoundError(
            "pydensecrf is required to perform dense crf inference."
        )
    if isinstance(logits, torch.Tensor):
        logits = F.softmax(logits, dim=0).detach().cpu().numpy()  # pyright: ignore[reportAssignmentType]
        U = unary_from_softmax(logits)
        n_labels = logits.shape[0]
    elif logits.ndim == 3:
        U = unary_from_softmax(logits)
        n_labels = logits.shape[0]
    else:
        assert n_labels is not None
        U = unary_from_labels(logits, n_labels, zero_unsure=False)

    d = dcrf.DenseCRF2D(image.shape[1], image.shape[0], n_labels)

    d.setUnaryEnergy(U)

    # This adds the color-independent term, features are the locations only.
    d.addPairwiseGaussian(
        sxy=pos_xy_std,
        compat=pos_w,
        kernel=dcrf.DIAG_KERNEL,
        normalization=dcrf.NORMALIZE_SYMMETRIC,
    )

    # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
    d.addPairwiseBilateral(
        sxy=bi_xy_std,
        srgb=bi_rgb_std,
        rgbim=image,
        compat=bi_w,
        kernel=dcrf.DIAG_KERNEL,
        normalization=dcrf.NORMALIZE_SYMMETRIC,
    )
    # Run five inference steps.
    logits = d.inference(max_iters)
    logits = np.asarray(logits).reshape((n_labels, image.shape[0], image.shape[1]))  # pyright: ignore[reportAssignmentType]
    return torch.from_numpy(logits)
