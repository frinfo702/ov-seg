# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Meta Platforms, Inc. All Rights Reserved
"""
OVSAM3 Evaluation Script.

Evaluates OVSAM3 semantic segmentation on standard benchmark datasets.
Computes mIoU, mACC, pACC, fwIoU, and per-class metrics.

Usage:
    python ovsam3/train_net.py --config-file configs/ovsam_seg_swinB_vitL.yaml \\
        --eval-only MODEL.WEIGHTS /path/to/checkpoint.pth
"""

import logging
import os

import detectron2.utils.comm as comm
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetMapper, MetadataCatalog
from detectron2.data.common import DatasetFromList, MapDataset
from detectron2.data.samplers import InferenceSampler
from detectron2.engine import DefaultTrainer, default_argument_parser, launch
from detectron2.evaluation import verify_results
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

import wandb
from open_vocab_seg import add_ovseg_config
from open_vocab_seg.data.build import get_detection_dataset_dicts
from open_vocab_seg.evaluation import GeneralizedSemSegEvaluator
from open_vocab_seg.utils.events import setup_wandb
from open_vocab_seg.utils.post_process_utils import dense_crf_post_process


def _build_mapper(cfg) -> DatasetMapper:
    kwargs = DatasetMapper.from_config(cfg, is_train=False)
    return DatasetMapper(**kwargs)


def _flatten_eval_results(results, prefix="eval"):
    flat_results = {}
    for key, value in results.items():
        current_key = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat_results.update(_flatten_eval_results(value, current_key))
        elif isinstance(value, (int, float)):
            flat_results[current_key] = value
    return flat_results


class Trainer(DefaultTrainer):
    """
    Trainer subclass for OVSAM3 evaluation.
    Overrides test loader to inject class_names into each batch.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type == "sem_seg":
            return GeneralizedSemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
                post_process_func=(dense_crf_post_process if cfg.TEST.DENSE_CRF else None),
            )
        raise NotImplementedError(f"No evaluator for dataset '{dataset_name}' with type '{evaluator_type}'")

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        dataset_dicts = get_detection_dataset_dicts([dataset_name], filter_empty=False)
        dataset = DatasetFromList(dataset_dicts, copy=False)
        mapper = _build_mapper(cfg)
        dataset = MapDataset(dataset, mapper)
        sampler = InferenceSampler(len(dataset))
        batch_sampler = torch.utils.data.BatchSampler(sampler, cfg.SOLVER.TEST_IMS_PER_BATCH, drop_last=False)
        return torch.utils.data.DataLoader(
            dataset,
            num_workers=0,
            batch_sampler=batch_sampler,
            collate_fn=lambda x: x,
        )


def setup(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_ovseg_config(cfg)
    cfg.merge_from_file(args.config_file)
    opts = args.opts
    if len(opts) % 2 != 0:
        import warnings

        warnings.warn(f"opts list has odd length ({len(opts)}): {opts}. Removing trailing element to avoid error.")
        opts = opts[:-1]
    cfg.merge_from_list(opts)
    cfg.freeze()
    setup_wandb(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="ovsam3")
    return cfg


def main(args):
    cfg = setup(args)
    logger = logging.getLogger("ovsam3")
    logger.info(f"Command Line Args: {args}")

    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)

    if args.eval_only:
        res = Trainer.test(cfg, model)
        if comm.is_main_process() and wandb.run is not None:
            wandb.log(_flatten_eval_results(res))
        if comm.is_main_process():
            verify_results(cfg, res)
            for dataset_name, metrics in res.items():
                logger.info(f"Results for {dataset_name}:")
                for k, v in metrics.items():
                    logger.info(f"  {k}: {v}")
            if wandb.run is not None:
                wandb.finish()
        return res

    raise RuntimeError("Training is not implemented in this script. Use --eval-only for evaluation.")


if __name__ == "__main__":
    parser = default_argument_parser()
    args, unknown = parser.parse_known_args()
    args.opts = unknown + args.opts
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
