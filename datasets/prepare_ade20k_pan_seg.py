#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
import glob
import json
import os
from collections import Counter

import numpy as np
import tqdm
from panopticapi.utils import IdGenerator, save_json
from PIL import Image

ADE20K_SEM_SEG_CATEGORIES = [
    "wall",
    "building",
    "sky",
    "floor",
    "tree",
    "ceiling",
    "road, route",
    "bed",
    "window ",
    "grass",
    "cabinet",
    "sidewalk, pavement",
    "person",
    "earth, ground",
    "door",
    "table",
    "mountain, mount",
    "plant",
    "curtain",
    "chair",
    "car",
    "water",
    "painting, picture",
    "sofa",
    "shelf",
    "house",
    "sea",
    "mirror",
    "rug",
    "field",
    "armchair",
    "seat",
    "fence",
    "desk",
    "rock, stone",
    "wardrobe, closet, press",
    "lamp",
    "tub",
    "rail",
    "cushion",
    "base, pedestal, stand",
    "box",
    "column, pillar",
    "signboard, sign",
    "chest of drawers, chest, bureau, dresser",
    "counter",
    "sand",
    "sink",
    "skyscraper",
    "fireplace",
    "refrigerator, icebox",
    "grandstand, covered stand",
    "path",
    "stairs",
    "runway",
    "case, display case, showcase, vitrine",
    "pool table, billiard table, snooker table",
    "pillow",
    "screen door, screen",
    "stairway, staircase",
    "river",
    "bridge, span",
    "bookcase",
    "blind, screen",
    "coffee table",
    "toilet, can, commode, crapper, pot, potty, stool, throne",
    "flower",
    "book",
    "hill",
    "bench",
    "countertop",
    "stove",
    "palm, palm tree",
    "kitchen island",
    "computer",
    "swivel chair",
    "boat",
    "bar",
    "arcade machine",
    "hovel, hut, hutch, shack, shanty",
    "bus",
    "towel",
    "light",
    "truck",
    "tower",
    "chandelier",
    "awning, sunshade, sunblind",
    "street lamp",
    "booth",
    "tv",
    "plane",
    "dirt track",
    "clothes",
    "pole",
    "land, ground, soil",
    "bannister, banister, balustrade, balusters, handrail",
    "escalator, moving staircase, moving stairway",
    "ottoman, pouf, pouffe, puff, hassock",
    "bottle",
    "buffet, counter, sideboard",
    "poster, posting, placard, notice, bill, card",
    "stage",
    "van",
    "ship",
    "fountain",
    "conveyer belt, conveyor belt, conveyer, conveyor, transporter",
    "canopy",
    "washer, automatic washer, washing machine",
    "plaything, toy",
    "pool",
    "stool",
    "barrel, cask",
    "basket, handbasket",
    "falls",
    "tent",
    "bag",
    "minibike, motorbike",
    "cradle",
    "oven",
    "ball",
    "food, solid food",
    "step, stair",
    "tank, storage tank",
    "trade name",
    "microwave",
    "pot",
    "animal",
    "bicycle",
    "lake",
    "dishwasher",
    "screen",
    "blanket, cover",
    "sculpture",
    "hood, exhaust hood",
    "sconce",
    "vase",
    "traffic light",
    "tray",
    "trash can",
    "fan",
    "pier",
    "crt screen",
    "plate",
    "monitor",
    "bulletin board",
    "shower",
    "radiator",
    "glass, drinking glass",
    "clock",
    "flag",  # noqa
]

PALETTE = [
    [120, 120, 120],
    [180, 120, 120],
    [6, 230, 230],
    [80, 50, 50],
    [4, 200, 3],
    [120, 120, 80],
    [140, 140, 140],
    [204, 5, 255],
    [230, 230, 230],
    [4, 250, 7],
    [224, 5, 255],
    [235, 255, 7],
    [150, 5, 61],
    [120, 120, 70],
    [8, 255, 51],
    [255, 6, 82],
    [143, 255, 140],
    [204, 255, 4],
    [255, 51, 7],
    [204, 70, 3],
    [0, 102, 200],
    [61, 230, 250],
    [255, 6, 51],
    [11, 102, 255],
    [255, 7, 71],
    [255, 9, 224],
    [9, 7, 230],
    [220, 220, 220],
    [255, 9, 92],
    [112, 9, 255],
    [8, 255, 214],
    [7, 255, 224],
    [255, 184, 6],
    [10, 255, 71],
    [255, 41, 10],
    [7, 255, 255],
    [224, 255, 8],
    [102, 8, 255],
    [255, 61, 6],
    [255, 194, 7],
    [255, 122, 8],
    [0, 255, 20],
    [255, 8, 41],
    [255, 5, 153],
    [6, 51, 255],
    [235, 12, 255],
    [160, 150, 20],
    [0, 163, 255],
    [140, 140, 200],
    [250, 10, 15],
    [20, 255, 0],
    [31, 255, 0],
    [255, 31, 0],
    [255, 224, 0],
    [153, 255, 0],
    [0, 0, 255],
    [255, 71, 0],
    [0, 235, 255],
    [0, 173, 255],
    [31, 0, 255],
    [11, 200, 200],
    [255, 82, 0],
    [0, 255, 245],
    [0, 61, 255],
    [0, 255, 112],
    [0, 255, 133],
    [255, 0, 0],
    [255, 163, 0],
    [255, 102, 0],
    [194, 255, 0],
    [0, 143, 255],
    [51, 255, 0],
    [0, 82, 255],
    [0, 255, 41],
    [0, 255, 173],
    [10, 0, 255],
    [173, 255, 0],
    [0, 255, 153],
    [255, 92, 0],
    [255, 0, 255],
    [255, 0, 245],
    [255, 0, 102],
    [255, 173, 0],
    [255, 0, 20],
    [255, 184, 184],
    [0, 31, 255],
    [0, 255, 61],
    [0, 71, 255],
    [255, 0, 204],
    [0, 255, 194],
    [0, 255, 82],
    [0, 10, 255],
    [0, 112, 255],
    [51, 0, 255],
    [0, 194, 255],
    [0, 122, 255],
    [0, 255, 163],
    [255, 153, 0],
    [0, 255, 10],
    [255, 112, 0],
    [143, 255, 0],
    [82, 0, 255],
    [163, 255, 0],
    [255, 235, 0],
    [8, 184, 170],
    [133, 0, 255],
    [0, 255, 92],
    [184, 0, 255],
    [255, 0, 31],
    [0, 184, 255],
    [0, 214, 255],
    [255, 0, 112],
    [92, 255, 0],
    [0, 224, 255],
    [112, 224, 255],
    [70, 184, 160],
    [163, 0, 255],
    [153, 0, 255],
    [71, 255, 0],
    [255, 0, 163],
    [255, 204, 0],
    [255, 0, 143],
    [0, 255, 235],
    [133, 255, 0],
    [255, 0, 235],
    [245, 0, 255],
    [255, 0, 122],
    [255, 245, 0],
    [10, 190, 212],
    [214, 255, 0],
    [0, 204, 255],
    [20, 0, 255],
    [255, 255, 0],
    [0, 153, 255],
    [0, 41, 255],
    [0, 255, 204],
    [41, 0, 255],
    [41, 255, 0],
    [173, 0, 255],
    [0, 245, 255],
    [71, 0, 255],
    [122, 0, 255],
    [0, 255, 184],
    [0, 92, 255],
    [184, 255, 0],
    [0, 133, 255],
    [255, 214, 0],
    [25, 194, 194],
    [102, 255, 0],
    [92, 0, 255],
]


if __name__ == "__main__":
    dataset_dir = os.getenv("DETECTRON2_DATASETS", "datasets")

    for name, dirname in [("train", "training"), ("val", "validation")]:
        image_dir = os.path.join(dataset_dir, f"ADEChallengeData2016/images/{dirname}/")
        semantic_dir = os.path.join(dataset_dir, f"ADEChallengeData2016/annotations/{dirname}/")
        instance_dir = os.path.join(
            dataset_dir, f"ADEChallengeData2016/annotations_instance/{dirname}/"
        )

        # folder to store panoptic PNGs
        out_folder = os.path.join(dataset_dir, f"ADEChallengeData2016/ade20k_panoptic_{name}/")
        # json with segmentations information
        out_file = os.path.join(dataset_dir, f"ADEChallengeData2016/ade20k_panoptic_{name}.json")

        if not os.path.isdir(out_folder):
            print("Creating folder {} for panoptic segmentation PNGs".format(out_folder))
            os.mkdir(out_folder)

        # json config
        config_file = "datasets/ade20k_instance_imgCatIds.json"
        with open(config_file) as f:
            config = json.load(f)

        # load catid mapping
        mapping_file = "datasets/ade20k_instance_catid_mapping.txt"
        with open(mapping_file) as f:
            map_id = {}
            for i, line in enumerate(f.readlines()):
                if i == 0:
                    continue
                ins_id, sem_id, _ = line.strip().split()
                # shift id by 1 because we want it to start from 0!
                # ignore_label becomes 255
                map_id[int(ins_id) - 1] = int(sem_id) - 1

        ADE20K_150_CATEGORIES = []
        for cat_id, cat_name in enumerate(ADE20K_SEM_SEG_CATEGORIES):
            ADE20K_150_CATEGORIES.append(
                {
                    "name": cat_name,
                    "id": cat_id,
                    "isthing": int(cat_id in map_id.values()),
                    "color": PALETTE[cat_id],
                }
            )
        categories_dict = {cat["id"]: cat for cat in ADE20K_150_CATEGORIES}

        panoptic_json_categories = ADE20K_150_CATEGORIES[:]
        panoptic_json_images = []
        panoptic_json_annotations = []

        filenames = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
        for idx, filename in enumerate(tqdm.tqdm(filenames)):
            panoptic_json_image = {}
            panoptic_json_annotation = {}

            image_id = os.path.basename(filename).split(".")[0]

            panoptic_json_image["id"] = image_id
            panoptic_json_image["file_name"] = os.path.basename(filename)

            original_format = np.array(Image.open(filename))
            panoptic_json_image["width"] = original_format.shape[1]
            panoptic_json_image["height"] = original_format.shape[0]

            pan_seg = np.zeros(
                (original_format.shape[0], original_format.shape[1], 3), dtype=np.uint8
            )
            id_generator = IdGenerator(categories_dict)

            filename_semantic = os.path.join(semantic_dir, image_id + ".png")
            filename_instance = os.path.join(instance_dir, image_id + ".png")

            sem_seg = np.asarray(Image.open(filename_semantic))
            ins_seg = np.asarray(Image.open(filename_instance))

            assert sem_seg.dtype == np.uint8
            assert ins_seg.dtype == np.uint8

            semantic_cat_ids = sem_seg - 1
            instance_cat_ids = ins_seg[..., 0] - 1
            # instance id starts from 1!
            # because 0 is reserved as VOID label
            instance_ins_ids = ins_seg[..., 1]

            segm_info = []

            # NOTE: there is some overlap between semantic and instance annotation
            # thus we paste stuffs first

            # process stuffs
            for semantic_cat_id in np.unique(semantic_cat_ids):
                if semantic_cat_id == 255:
                    continue
                if categories_dict[semantic_cat_id]["isthing"]:
                    continue
                mask = semantic_cat_ids == semantic_cat_id
                # should not have any overlap
                assert pan_seg[mask].sum() == 0

                segment_id, color = id_generator.get_id_and_color(semantic_cat_id)
                pan_seg[mask] = color

                area = np.sum(mask)  # segment area computation
                # bbox computation for a segment
                hor = np.sum(mask, axis=0)
                hor_idx = np.nonzero(hor)[0]
                x = hor_idx[0]
                width = hor_idx[-1] - x + 1
                vert = np.sum(mask, axis=1)
                vert_idx = np.nonzero(vert)[0]
                y = vert_idx[0]
                height = vert_idx[-1] - y + 1
                bbox = [int(x), int(y), int(width), int(height)]

                segm_info.append(
                    {
                        "id": int(segment_id),
                        "category_id": int(semantic_cat_id),
                        "area": int(area),
                        "bbox": bbox,
                        "iscrowd": 0,
                    }
                )

            # process things
            for thing_id in np.unique(instance_ins_ids):
                if thing_id == 0:
                    continue
                mask = instance_ins_ids == thing_id
                instance_cat_id = np.unique(instance_cat_ids[mask])
                assert len(instance_cat_id) == 1

                semantic_cat_id = map_id[instance_cat_id[0]]

                segment_id, color = id_generator.get_id_and_color(semantic_cat_id)
                pan_seg[mask] = color

                area = np.sum(mask)  # segment area computation
                # bbox computation for a segment
                hor = np.sum(mask, axis=0)
                hor_idx = np.nonzero(hor)[0]
                x = hor_idx[0]
                width = hor_idx[-1] - x + 1
                vert = np.sum(mask, axis=1)
                vert_idx = np.nonzero(vert)[0]
                y = vert_idx[0]
                height = vert_idx[-1] - y + 1
                bbox = [int(x), int(y), int(width), int(height)]

                segm_info.append(
                    {
                        "id": int(segment_id),
                        "category_id": int(semantic_cat_id),
                        "area": int(area),
                        "bbox": bbox,
                        "iscrowd": 0,
                    }
                )

            panoptic_json_annotation = {
                "image_id": image_id,
                "file_name": image_id + ".png",
                "segments_info": segm_info,
            }

            Image.fromarray(pan_seg).save(os.path.join(out_folder, image_id + ".png"))

            panoptic_json_images.append(panoptic_json_image)
            panoptic_json_annotations.append(panoptic_json_annotation)

        # save this
        d = {
            "images": panoptic_json_images,
            "annotations": panoptic_json_annotations,
            "categories": panoptic_json_categories,
        }

        save_json(d, out_file)
