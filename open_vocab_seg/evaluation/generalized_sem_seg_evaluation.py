# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import io
import itertools
import json
import os
from collections import OrderedDict
from typing import Any, Callable, Optional

import numpy as np
import torch
from detectron2.data import MetadataCatalog
from detectron2.evaluation import SemSegEvaluator
from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.utils.file_io import PathManager
from PIL import Image


class GeneralizedSemSegEvaluator(SemSegEvaluator):
    """
    Evaluate semantic segmentation metrics.
    """

    def __init__(
        self,
        dataset_name: str,
        distributed: bool = True,
        output_dir: Optional[str] = None,
        *,
        num_classes: Optional[int] = None,
        ignore_label: Optional[int] = None,
        post_process_func: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            dataset_name,
            distributed=distributed,
            output_dir=output_dir,
            num_classes=num_classes,
            ignore_label=ignore_label,
        )
        meta = MetadataCatalog.get(dataset_name)
        try:
            self._evaluation_set = meta.evaluation_set
        except AttributeError:
            self._evaluation_set = None
        self.post_process_func = (
            post_process_func
            if post_process_func is not None
            else lambda x, **kwargs: x
        )

    def process(self, inputs: list[dict[str, Any]], outputs: list[dict[str, Any]]) -> None:
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """
        for input, output in zip(inputs, outputs):
            output = self.post_process_func(
                output["sem_seg"], image=np.array(Image.open(input["file_name"]))
            )
            output = output.argmax(dim=0).to(self._cpu_device)
            pred = np.array(output, dtype=np.int64)
            with PathManager.open(
                self.input_file_to_gt_file[input["file_name"]], "rb"
            ) as f:
                gt = np.array(Image.open(io.BytesIO(f.read())), dtype=np.int64)  # pyright: ignore[reportArgumentType]

            gt[gt == self._ignore_label] = self._num_classes

            self._conf_matrix += np.bincount(
                (self._num_classes + 1) * pred.reshape(-1) + gt.reshape(-1),
                minlength=self._conf_matrix.size,
            ).reshape(self._conf_matrix.shape)

            self._predictions.extend(self.encode_json_sem_seg(pred, input["file_name"]))

    def evaluate(self) -> OrderedDict:
        """
        Evaluates standard semantic segmentation metrics (http://cocodataset.org/#stuff-eval):

        * Mean intersection-over-union averaged across classes (mIoU)
        * Frequency Weighted IoU (fwIoU)
        * Mean pixel accuracy averaged across classes (mACC)
        * Pixel Accuracy (pACC)
        """
        if self._distributed:
            synchronize()
            conf_matrix_list = all_gather(self._conf_matrix)
            self._predictions = all_gather(self._predictions)
            self._predictions = list(itertools.chain(*self._predictions))  # pyright: ignore[reportArgumentType]
            if not is_main_process():
                return  # pyright: ignore[reportReturnType]

            self._conf_matrix = np.zeros_like(self._conf_matrix)
            for conf_matrix in conf_matrix_list:
                self._conf_matrix += conf_matrix  # pyright: ignore[reportOperatorIssue]

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "sem_seg_predictions.json")
            with PathManager.open(file_path, "wb") as f:
                f.write(json.dumps(self._predictions).encode())  # pyright: ignore[reportArgumentType]

        acc = np.full(self._num_classes, np.nan, dtype=np.float64)
        iou = np.full(self._num_classes, np.nan, dtype=np.float64)
        tp = self._conf_matrix.diagonal()[:-1].astype(np.float64)
        pos_gt = np.sum(self._conf_matrix[:-1, :-1], axis=0).astype(np.float64)
        class_weights = pos_gt / np.sum(pos_gt)
        pos_pred = np.sum(self._conf_matrix[:-1, :-1], axis=1).astype(np.float64)
        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
        iou_valid = (pos_gt + pos_pred) > 0
        union = pos_gt + pos_pred - tp
        iou[acc_valid] = tp[acc_valid] / union[acc_valid]
        macc = np.sum(acc[acc_valid]) / np.sum(acc_valid)
        miou = np.sum(iou[acc_valid]) / np.sum(iou_valid)
        fiou = np.sum(iou[acc_valid] * class_weights[acc_valid])
        pacc = np.sum(tp) / np.sum(pos_gt)

        res = {}
        res["mIoU"] = 100 * miou
        res["fwIoU"] = 100 * fiou
        for i, name in enumerate(self._class_names):
            res[f"IoU-{name}"] = 100 * iou[i]
        res["mACC"] = 100 * macc
        res["pACC"] = 100 * pacc
        for i, name in enumerate(self._class_names):
            res[f"ACC-{name}"] = 100 * acc[i]
        if self._evaluation_set is not None:
            for set_name, set_inds in self._evaluation_set.items():
                iou_list = []
                set_inds = np.array(set_inds, dtype=np.int64)
                mask = np.zeros((len(iou),), dtype=bool)
                mask[set_inds] = 1
                miou = np.sum(iou[mask][acc_valid[mask]]) / np.sum(iou_valid[mask])
                pacc = np.sum(tp[mask]) / np.sum(pos_gt[mask])
                res[f"mIoU-{set_name}"] = 100 * miou
                res[f"pAcc-{set_name}"] = 100 * pacc
                iou_list.append(miou)
                miou = np.sum(iou[~mask][acc_valid[~mask]]) / np.sum(iou_valid[~mask])
                pacc = np.sum(tp[~mask]) / np.sum(pos_gt[~mask])
                res[f"mIoU-un{set_name}"] = 100 * miou
                res[f"pAcc-un{set_name}"] = 100 * pacc
                iou_list.append(miou)
                res[f"hIoU-{set_name}"] = (
                    100 * len(iou_list) / sum([1 / iou for iou in iou_list])
                )
        if self._output_dir:
            file_path = os.path.join(self._output_dir, "sem_seg_evaluation.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(res, f)  # pyright: ignore[reportArgumentType]
        results = OrderedDict({"sem_seg": res})
        self._logger.info(results)
        return results
