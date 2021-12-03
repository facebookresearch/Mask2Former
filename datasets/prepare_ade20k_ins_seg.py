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
import pycocotools.mask as mask_util


if __name__ == "__main__":
    dataset_dir = os.getenv("DETECTRON2_DATASETS", "datasets")

    for name, dirname in [("train", "training"), ("val", "validation")]:
        image_dir = os.path.join(dataset_dir, f"ADEChallengeData2016/images/{dirname}/")
        instance_dir = os.path.join(
            dataset_dir, f"ADEChallengeData2016/annotations_instance/{dirname}/"
        )

        # img_id = 0
        ann_id = 1

        # json
        out_file = os.path.join(dataset_dir, f"ADEChallengeData2016/ade20k_instance_{name}.json")

        # json config
        instance_config_file = "datasets/ade20k_instance_imgCatIds.json"
        with open(instance_config_file) as f:
            category_dict = json.load(f)["categories"]

        # load catid mapping
        # it is important to share category id for both instance and panoptic annotations
        mapping_file = "datasets/ade20k_instance_catid_mapping.txt"
        with open(mapping_file) as f:
            map_id = {}
            for i, line in enumerate(f.readlines()):
                if i == 0:
                    continue
                ins_id, sem_id, _ = line.strip().split()
                # shift id by 1 because we want it to start from 0!
                # ignore_label becomes 255
                map_id[int(ins_id)] = int(sem_id) - 1

        for cat in category_dict:
            cat["id"] = map_id[cat["id"]]

        filenames = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))

        ann_dict = {}
        images = []
        annotations = []

        for idx, filename in enumerate(tqdm.tqdm(filenames)):
            image = {}
            image_id = os.path.basename(filename).split(".")[0]

            image["id"] = image_id
            image["file_name"] = os.path.basename(filename)

            original_format = np.array(Image.open(filename))
            image["width"] = original_format.shape[1]
            image["height"] = original_format.shape[0]

            images.append(image)

            filename_instance = os.path.join(instance_dir, image_id + ".png")
            ins_seg = np.asarray(Image.open(filename_instance))
            assert ins_seg.dtype == np.uint8

            instance_cat_ids = ins_seg[..., 0]
            # instance id starts from 1!
            # because 0 is reserved as VOID label
            instance_ins_ids = ins_seg[..., 1]

            # process things
            for thing_id in np.unique(instance_ins_ids):
                if thing_id == 0:
                    continue
                mask = instance_ins_ids == thing_id
                instance_cat_id = np.unique(instance_cat_ids[mask])
                assert len(instance_cat_id) == 1

                anno = {}
                anno['id'] = ann_id
                ann_id += 1
                anno['image_id'] = image['id']
                anno["iscrowd"] = int(0)
                anno["category_id"] = int(map_id[instance_cat_id[0]])

                inds = np.nonzero(mask)
                ymin, ymax = inds[0].min(), inds[0].max()
                xmin, xmax = inds[1].min(), inds[1].max()
                anno["bbox"] = [int(xmin), int(ymin), int(xmax - xmin + 1), int(ymax - ymin + 1)]
                # if xmax <= xmin or ymax <= ymin:
                #     continue
                rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
                rle["counts"] = rle["counts"].decode("utf-8")
                anno["segmentation"] = rle
                anno["area"] = int(mask_util.area(rle))
                annotations.append(anno)

        # save this
        ann_dict['images'] = images
        ann_dict['categories'] = category_dict
        ann_dict['annotations'] = annotations

        save_json(ann_dict, out_file)
