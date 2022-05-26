import sys
import os
import os.path as osp

sys.path.append("..")
sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])

import argparse
import logging

logger = logging.getLogger(__name__)

from tqdm import tqdm
import numpy as np

from v2x_utils import range2box, id_to_str, Evaluator
from config import add_arguments
from dataset import SUPPROTED_DATASETS
from dataset.dataset_utils import save_pkl
from models import SUPPROTED_MODELS
from models.model_utils import Channel


def eval_vic(args, dataset, model, evaluator):
    idx = 0
    for VICFrame, label, filt in tqdm(dataset):
        # if idx % 5 != 0:
        #     idx += 1
        #     continue
        pred = model(
            VICFrame,
            filt,
            None if not hasattr(dataset, "prev_inf_frame") else dataset.prev_inf_frame,
        )

        # import pdb; pdb.set_trace()
        # print(pred,label)
        # trans=VICFrame.transform('World','Vehicle_lidar')
        # # pred['boxes_3d']=trans(pred['boxes_3d'])
        # label['boxes_3d']=trans(label['boxes_3d'])
        evaluator.add_frame(pred, label)  # 计算pred和label的iou等信息
        pipe.flush()  # 对pipe清空
        pred["label"] = label["boxes_3d"]
        save_pkl(pred, osp.join(args.output, "result", id_to_str(idx) + ".pkl"))
        idx += 1
        # evaluator.print_ap("3d")
    evaluator.print_ap("3d")  # 返回AP值
    evaluator.print_ap("bev")
    print("Average Communication Cost = %.2lf Bytes" % (pipe.average_bytes()))


def eval_single(args, dataset, model, evaluator):
    for frame, label, filt in tqdm(dataset):
        pred = model(frame, filt)
        if args.sensortype == "camera":
            evaluator.add_frame(pred, label["camera"])
        elif args.sensortype == "lidar":
            evaluator.add_frame(pred, label["lidar"])
        save_pkl({"boxes_3d": label["lidar"]["boxes_3d"]}, osp.join(args.output, "result", frame.id["camera"] + ".pkl"))

    evaluator.print_ap("3d")
    evaluator.print_ap("bev")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(conflict_handler="resolve")
    add_arguments(parser)
    args, _ = parser.parse_known_args()
    # add model-specific arguments
    SUPPROTED_MODELS[args.model].add_arguments(parser)
    args = parser.parse_args()
    
    if args.quiet:
        level = logging.ERROR
    elif args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=level,
    )

    extended_range = range2box(np.array(args.extended_range))
    logger.info("loading dataset")

    dataset = SUPPROTED_DATASETS[args.dataset](
        args.input,
        args,
        split=args.split,
        sensortype=args.sensortype,
        extended_range=extended_range,
    )

    logger.info("loading evaluator")
    evaluator = Evaluator(args.pred_classes)

    logger.info("loading model")
    if args.eval_single:
        model = SUPPROTED_MODELS[args.model](args)
        eval_single(args, dataset, model, evaluator)
    else:
        pipe = Channel()
        model = SUPPROTED_MODELS[args.model](args, pipe)
        eval_vic(args, dataset, model, evaluator)
