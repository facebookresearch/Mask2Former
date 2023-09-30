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


# Search over SOLVER.POLY_LR_POWER
ARGS = argparse.Namespace(**EXP_DICT_BASE)

cfg = setup(ARGS)


def update_cfg(lr, batch_size, train_set, test_set, model):
    cfg_new = cfg.clone()
    cfg_new.defrost()
    cfg_new.SOLVER.BASE_LR = lr
    cfg_new.SOLVER.IMS_PER_BATCH = batch_size
    cfg_new.DATASETS.TRAIN = (train_set,)
    cfg_new.DATASETS.TEST = (test_set,)

    if model == "swin":
        cfg_new.MODEL.BACKBONE.NAME = "D2SwinTransformer"
        cfg_new.MODEL.SWIN.EMBED_DIM = 128
        cfg_new.MODEL.SWIN.DEPTHS = [2, 2, 18, 2]
        cfg_new.MODEL.SWIN.NUM_HEADS = [4, 8, 16, 32]
        cfg_new.MODEL.SWIN.WINDOW_SIZE = 12
        cfg_new.MODEL.SWIN.APE = False
        cfg_new.MODEL.SWIN.DROP_PATH_RATE = 0.3
        cfg_new.MODEL.SWIN.PATCH_NORM = True
        cfg_new.MODEL.SWIN.PRETRAIN_IMG_SIZE = 384
        cfg_new.MODEL.WEIGHTS = (
            "/mnt/colab_public/digitaltwin/swin_base_patch4_window12_384_22k.pkl"
        )
        cfg_new.MODEL.PIXEL_MEAN = [123.675, 116.280, 103.530]
        cfg_new.MODEL.PIXEL_STD = [58.395, 57.120, 57.375]

    return cfg_new


EXP_GROUPS["baselines_large_scale"] = []
for lr in [0.0001]:
    for batch_size in [4]:
        for run in [1, 2, 3]:
            for model in [None, "swin"]:
                train_set = "ril"
                test_set = "ril-test"
                cfg_new = update_cfg(lr, batch_size, train_set, test_set, model=model)
                path = (
                    f"configs/configs_pkl/{hu.hash_dict({'config_name': str(cfg_new)})}.pkl"
                )
                hu.save_pkl(path, cfg_new)

                exp_list = hu.cartesian_exp_group(
                    {
                        "model": model,
                        "config_path": path,
                        "train_set": train_set,
                        "test_set": test_set,
                        "lr": lr,
                        "batch_size": batch_size,
                        "num_gpus": 4,
                        "run": run
                    },
                    remove_none=True,
                )
                EXP_GROUPS["baselines_large_scale"] += exp_list


EXP_GROUPS["baselines_small_scale"] = []
for lr in [0.0001]:
    for batch_size in [2]:
        for run in [1, 2, 3]:
            for model in [
                None,
                "swin",
            ]:
                train_set = "ril"
                test_set = "ril-test"
                cfg_new = update_cfg(lr, batch_size, train_set, test_set, model=model)

                path = f"configs/configs_pkl/{hu.hash_dict({'config_name': str(cfg_new)})}.pkl"
                hu.save_pkl(path, cfg_new)

                exp_list = hu.cartesian_exp_group(
                    {
                        "model": model,
                        "config_path": path,
                        "train_set": train_set,
                        "test_set": test_set,
                        "lr": lr,
                        "batch_size": batch_size,
                        "num_gpus": 1,
                        "run": run,
                    },
                    remove_none=True,
                )
                EXP_GROUPS["baselines_small_scale"] += exp_list
