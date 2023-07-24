# suppress output
import argparse

from haven import haven_utils as hu
from train_net import setup


EXP_GROUPS = {}

EXP_DICT_BASE = {
    "config_file": "configs/ril/panoptic-segmentation/ril1-shnv1.yaml",
    "num_gpus": 1,
    "num_machines": 1,
    "opts": [],
    "resume": False,
    "eval_only": False,
    "dist_url": "tcp://127.0.0.1:62163",
}

ARGS = argparse.Namespace(**EXP_DICT_BASE)

cfg = setup(ARGS)

# Add this transformer model
# MODEL:
#   BACKBONE:
#     NAME: "D2SwinTransformer"
#   SWIN:
#     EMBED_DIM: 128
#     DEPTHS: [2, 2, 18, 2]
#     NUM_HEADS: [4, 8, 16, 32]
#     WINDOW_SIZE: 12
#     APE: False
#     DROP_PATH_RATE: 0.3
#     PATCH_NORM: True
#     PRETRAIN_IMG_SIZE: 384
#   WEIGHTS: "swin_base_patch4_window12_384.pkl"
#   PIXEL_MEAN: [123.675, 116.280, 103.530]
#   PIXEL_STD: [58.395, 57.120, 57.375]

# Search over SOLVER.POLY_LR_POWER

EXP_GROUPS["baselines_large_scale"] = []
for lr in [0.00001, 0.0001, 0.01, 0.001]:
    for batch_size in [4]:
        dataset = "rilv7-shapenetv1"

        cfg_new = cfg.clone()
        cfg_new.defrost()
        cfg_new.SOLVER.BASE_LR = lr
        cfg_new.SOLVER.IMS_PER_BATCH = batch_size
        cfg_new.DATASETS.TRAIN = (dataset,)

        path = f"configs/configs_pkl/{hu.hash_dict({'config_name': str(cfg_new)})}.pkl"
        hu.save_pkl(path, cfg_new)

        exp_dict = {
            "config_path": path,
            "dataset": dataset,
            "lr": lr,
            "batch_size": batch_size,
            "num_gpus": 4,
        }
        EXP_GROUPS["baselines_large_scale"] += [exp_dict]


EXP_GROUPS["baselines_small_scale"] = []
for lr in [0.0001]:
    for batch_size in [2]:
        dataset = "rilv7-shapenetv1"

        cfg_new = cfg.clone()
        cfg_new.defrost()
        cfg_new.SOLVER.BASE_LR = lr
        cfg_new.SOLVER.IMS_PER_BATCH = batch_size
        cfg_new.DATASETS.TRAIN = (dataset,)

        path = f"configs/configs_pkl/{hu.hash_dict({'config_name': str(cfg_new)})}.pkl"
        hu.save_pkl(path, cfg_new)

        exp_dict = {
            "config_path": path,
            "dataset": dataset,
            "lr": lr,
            "batch_size": batch_size,
            "num_gpus": 1,
        }
        EXP_GROUPS["baselines_small_scale"] += [exp_dict]
