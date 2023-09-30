#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import json
import os
from collections import defaultdict
from tqdm import tqdm

import numpy as np
import torch

from detectron2.data import MetadataCatalog
from detectron2.data.detection_utils import read_image
from detectron2.utils.file_io import PathManager
from pycocotools import mask as maskUtils

from panopticapi.evaluation import PQStat


def default_argument_parser():
    """
    Creates a parser with some common arguments used by analysis tools.
    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(description="Evaluate PQ metric for semantic segmentation.")
    # NOTE: currently does not support Cityscapes, you need to convert
    # Cityscapes prediction format to Detectron2 prediction format.
    parser.add_argument(
        "--dataset-name",
        default="ade20k_sem_seg_val",
        choices=["ade20k_sem_seg_val", "coco_2017_test_stuff_10k_sem_seg", "ade20k_full_sem_seg_val"],
        help="dataset name you want to evaluate")
    parser.add_argument("--json-file", default="", help="path to detection json file")

    return parser


# Modified from the official panoptic api: https://github.com/cocodataset/panopticapi/blob/master/panopticapi/evaluation.py
def pq_compute_single_image(segm_gt, segm_dt, categories, ignore_label):
    pq_stat = PQStat()
    VOID = ignore_label
    OFFSET = 256 * 256 * 256

    pan_gt = segm_gt
    pan_pred = segm_dt

    gt_ann = {'segments_info': []}
    labels, labels_cnt = np.unique(segm_gt, return_counts=True)
    for cat_id, cnt in zip(labels, labels_cnt):
        if cat_id == VOID:
            continue
        gt_ann['segments_info'].append(
            {"id": cat_id, "category_id": cat_id, "area": cnt, "iscrowd": 0}
        )
    
    pred_ann = {'segments_info': []}
    for cat_id in np.unique(segm_dt):
        pred_ann['segments_info'].append({"id": cat_id, "category_id": cat_id})

    gt_segms = {el['id']: el for el in gt_ann['segments_info']}
    pred_segms = {el['id']: el for el in pred_ann['segments_info']}

    # predicted segments area calculation + prediction sanity checks
    pred_labels_set = set(el['id'] for el in pred_ann['segments_info'])
    labels, labels_cnt = np.unique(pan_pred, return_counts=True)
    for label, label_cnt in zip(labels, labels_cnt):
        if label not in pred_segms:
            if label == VOID:
                continue
            raise KeyError('In the image with ID {} segment with ID {} is presented in PNG and not presented in JSON.'.format(image_id, label))
        pred_segms[label]['area'] = label_cnt
        pred_labels_set.remove(label)
        if pred_segms[label]['category_id'] not in categories:
            raise KeyError('In the image with ID {} segment with ID {} has unknown category_id {}.'.format(image_id, label, pred_segms[label]['category_id']))
    if len(pred_labels_set) != 0:
        raise KeyError('In the image with ID {} the following segment IDs {} are presented in JSON and not presented in PNG.'.format(image_id, list(pred_labels_set)))

    # confusion matrix calculation
    pan_gt_pred = pan_gt.astype(np.uint64) * OFFSET + pan_pred.astype(np.uint64)
    gt_pred_map = {}
    labels, labels_cnt = np.unique(pan_gt_pred, return_counts=True)
    for label, intersection in zip(labels, labels_cnt):
        gt_id = label // OFFSET
        pred_id = label % OFFSET
        gt_pred_map[(gt_id, pred_id)] = intersection

    # count all matched pairs
    gt_matched = set()
    pred_matched = set()
    for label_tuple, intersection in gt_pred_map.items():
        gt_label, pred_label = label_tuple
        if gt_label not in gt_segms:
            continue
        if pred_label not in pred_segms:
            continue
        if gt_segms[gt_label]['iscrowd'] == 1:
            continue
        if gt_segms[gt_label]['category_id'] != pred_segms[pred_label]['category_id']:
            continue

        union = pred_segms[pred_label]['area'] + gt_segms[gt_label]['area'] - intersection - gt_pred_map.get((VOID, pred_label), 0)
        iou = intersection / union
        if iou > 0.5:
            pq_stat[gt_segms[gt_label]['category_id']].tp += 1
            pq_stat[gt_segms[gt_label]['category_id']].iou += iou
            gt_matched.add(gt_label)
            pred_matched.add(pred_label)

    # count false positives
    crowd_labels_dict = {}
    for gt_label, gt_info in gt_segms.items():
        if gt_label in gt_matched:
            continue
        # crowd segments are ignored
        if gt_info['iscrowd'] == 1:
            crowd_labels_dict[gt_info['category_id']] = gt_label
            continue
        pq_stat[gt_info['category_id']].fn += 1

    # count false positives
    for pred_label, pred_info in pred_segms.items():
        if pred_label in pred_matched:
            continue
        # intersection of the segment with VOID
        intersection = gt_pred_map.get((VOID, pred_label), 0)
        # plus intersection with corresponding CROWD region if it exists
        if pred_info['category_id'] in crowd_labels_dict:
            intersection += gt_pred_map.get((crowd_labels_dict[pred_info['category_id']], pred_label), 0)
        # predicted segment is ignored if more than half of the segment correspond to VOID and CROWD regions
        if intersection / pred_info['area'] > 0.5:
            continue
        pq_stat[pred_info['category_id']].fp += 1

    return pq_stat


def main():
    parser = default_argument_parser()
    args = parser.parse_args()

    _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    json_file = args.json_file

    with open(json_file) as f:
        predictions = json.load(f)

    imgToAnns = defaultdict(list)
    for pred in predictions:
        image_id = os.path.basename(pred["file_name"]).split(".")[0]
        imgToAnns[image_id].append(
            {"category_id" : pred["category_id"], "segmentation" : pred["segmentation"]}
        )

    image_ids = list(imgToAnns.keys())

    meta = MetadataCatalog.get(args.dataset_name)
    class_names = meta.stuff_classes
    num_classes = len(meta.stuff_classes)
    ignore_label = meta.ignore_label
    conf_matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)

    categories = {}
    for i in range(num_classes):
        categories[i] = {"id": i, "name": class_names[i], "isthing": 0}

    pq_stat = PQStat()
    
    for image_id in tqdm(image_ids):
        if args.dataset_name == "ade20k_sem_seg_val":
            gt_dir = os.path.join(_root, "ADEChallengeData2016", "annotations_detectron2", "validation")
            segm_gt = read_image(os.path.join(gt_dir, image_id + ".png")).copy().astype(np.int64)
        elif args.dataset_name == "coco_2017_test_stuff_10k_sem_seg":
            gt_dir = os.path.join(_root, "coco", "coco_stuff_10k", "annotations_detectron2", "test")
            segm_gt = read_image(os.path.join(gt_dir, image_id + ".png")).copy().astype(np.int64)
        elif args.dataset_name == "ade20k_full_sem_seg_val":
            gt_dir = os.path.join(_root, "ADE20K_2021_17_01", "annotations_detectron2", "validation")
            segm_gt = read_image(os.path.join(gt_dir, image_id + ".tif")).copy().astype(np.int64)
        else:
            raise ValueError(f"Unsupported dataset {args.dataset_name}")

        # get predictions
        segm_dt = np.zeros_like(segm_gt)
        anns = imgToAnns[image_id]
        for ann in anns:
            # map back category_id
            if hasattr(meta, "stuff_dataset_id_to_contiguous_id"):
                if ann["category_id"] in meta.stuff_dataset_id_to_contiguous_id:
                    category_id = meta.stuff_dataset_id_to_contiguous_id[ann["category_id"]]
            else:
                category_id = ann["category_id"]
            mask = maskUtils.decode(ann["segmentation"])
            segm_dt[mask > 0] = category_id

        # miou
        gt = segm_gt.copy()
        pred = segm_dt.copy()
        gt[gt == ignore_label] = num_classes
        conf_matrix += np.bincount(
            (num_classes + 1) * pred.reshape(-1) + gt.reshape(-1),
            minlength=conf_matrix.size,
        ).reshape(conf_matrix.shape)

        # pq
        pq_stat_single = pq_compute_single_image(segm_gt, segm_dt, categories, meta.ignore_label)
        pq_stat += pq_stat_single

    metrics = [("All", None), ("Stuff", False)]
    results = {}
    for name, isthing in metrics:
        results[name], per_class_results = pq_stat.pq_average(categories, isthing=isthing)
        if name == 'All':
            results['per_class'] = per_class_results
    print("{:10s}| {:>5s}  {:>5s}  {:>5s} {:>5s}".format("", "PQ", "SQ", "RQ", "N"))
    print("-" * (10 + 7 * 4))

    for name, _isthing in metrics:
        print("{:10s}| {:5.1f}  {:5.1f}  {:5.1f} {:5d}".format(
            name,
            100 * results[name]['pq'],
            100 * results[name]['sq'],
            100 * results[name]['rq'],
            results[name]['n'])
        )

    # calculate miou
    acc = np.full(num_classes, np.nan, dtype=np.float64)
    iou = np.full(num_classes, np.nan, dtype=np.float64)
    tp = conf_matrix.diagonal()[:-1].astype(np.float64)
    pos_gt = np.sum(conf_matrix[:-1, :-1], axis=0).astype(np.float64)
    pos_pred = np.sum(conf_matrix[:-1, :-1], axis=1).astype(np.float64)
    acc_valid = pos_gt > 0
    acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
    iou_valid = (pos_gt + pos_pred) > 0
    union = pos_gt + pos_pred - tp
    iou[acc_valid] = tp[acc_valid] / union[acc_valid]
    miou = np.sum(iou[acc_valid]) / np.sum(iou_valid)

    print("")
    print(f"mIoU: {miou}")


if __name__ == '__main__':
    main()
