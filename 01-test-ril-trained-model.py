import os.path

import matplotlib.pyplot as plt
from detectron2.utils.logger import setup_logger
from tqdm import trange

setup_logger()
setup_logger(name="mask2former")
import numpy as np
import cv2
import torch

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.projects.deeplab import add_deeplab_config

# import Mask2Former project
from mask2former import (
    add_maskformer2_config,
)


ril_metadata = MetadataCatalog.get("rilv7")


cfg = get_cfg()
add_deeplab_config(cfg)
add_maskformer2_config(cfg)
cfg.merge_from_file("configs/ril/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml")
# cfg.MODEL.WEIGHTS = "output/model_final.pth"
cfg.MODEL.WEIGHTS = "output/exp3-adding-noise-80kiter/model_final.pth"
cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = False
cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = True

# ======== BEGIN PLAYGROUND

# default values: 0.8 and 0.8
cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.8
cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD = 0.8

BATCH = False  # for looking at a single image (and opening a plt window)
# BATCH = True # for running this for 12 images (test/real) and writing the res to disk

# ======== END PLAYGROUND

if not torch.cuda.is_available():
    cfg.MODEL.DEVICE = "cpu"
predictor = DefaultPredictor(cfg)


def eval_on_img(img):
    outputs = predictor(im)

    # outputs["panoptic_seg"][1] # contains a human-readable list of category IDs for each part of the image,
    # e.g. [{'id': 2, 'isthing': True, 'category_id': 15}, ...]

    # outputs["panoptic_seg"][0] is a 512x512 torch.int32 tensor where each tensor cell is an int from the IDs above
    # e.g. (all cells with value `2` constitute an item of category 15)

    v = Visualizer(im[:, :, ::-1], ril_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
    panoptic_result = v.draw_panoptic_seg(outputs["panoptic_seg"][0].to("cpu"), outputs["panoptic_seg"][1]).get_image()
    return panoptic_result


if BATCH:
    for IMG in trange(10):
        img_name = str(IMG + 1).zfill(5)

        im = cv2.imread(os.path.expanduser(f"~/dev/ril-digitaltwin/scripts/imgs/512/generatorv7/{img_name}.png"))
        panoptic_result = eval_on_img(im)
        cv2.imwrite(f"output/pred-{img_name}.png", panoptic_result[:, :, ::-1])

    for i in range(2):
        im = cv2.imread(f"real-test-{i+1}.jpg")
        panoptic_result = eval_on_img(im)
        cv2.imwrite(f"output/pred-real-{i+1}.jpg", panoptic_result[:, :, ::-1])
else:
    im = cv2.imread(f"real-test-1.jpg")
    print("beginning inference...")
    panoptic_result = eval_on_img(im)
    print("...inference done")
    plt.imshow(panoptic_result[:, :, ::-1])
    plt.axis("off")
    plt.show()

print("done")
