import sitecustomize  # noqa: F401

import argparse
import glob
import multiprocessing as mp
import os
import time

import cv2
import tqdm
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from open_vocab_seg import add_ovseg_config
from open_vocab_seg.utils import VisualizationDemo

WINDOW_NAME = "Open vocabulary segmentation (OVSAM3)"


def setup_cfg(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_ovseg_config(cfg)
    cfg.merge_from_file(args.config_file)
    opts = args.opts
    if len(opts) % 2 != 0:
        import warnings
        warnings.warn(
            f"opts list has odd length ({len(opts)}): {opts}. "
            "Removing trailing element to avoid 'Override list has odd length' error."
        )
        opts = opts[:-1]
    cfg.merge_from_list(opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(
        description="Detectron2 demo for OVSAM3 segmentation"
    )
    parser.add_argument(
        "--config-file",
        default="configs/ovsam_seg_swinB_vitL.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--class-names", nargs="+", help="A list of user-defined class_names"
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--opts",
        nargs=argparse.REMAINDER,
        default=[],
        help="Modify config options using the command-line 'KEY VALUE' pairs",
    )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args, unknown = get_parser().parse_known_args()
    args.opts = unknown + args.opts
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    if args.class_names or not cfg.DATASETS.TEST:
        dummy_name = "__unused_ovsam3_demo"
        if dummy_name not in MetadataCatalog:
            MetadataCatalog.get(dummy_name).set(
                stuff_classes=args.class_names or [],
                stuff_colors=[],
            )
        cfg.defrost()
        cfg.DATASETS.TEST = (dummy_name,)
        cfg.freeze()

    demo = VisualizationDemo(cfg)
    class_names = args.class_names
    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img, class_names)
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "finished",
                    time.time() - start_time,
                )
            )

            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    assert len(args.input) == 1, (
                        "Please specify a directory with args.output"
                    )
                    out_filename = args.output
                visualized_output.save(out_filename)
                print(f"Result saved to: {os.path.abspath(out_filename)}")
            else:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break
    else:
        raise NotImplementedError
