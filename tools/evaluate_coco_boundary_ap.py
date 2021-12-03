#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modified by Bowen Cheng from: https://github.com/bowenc0221/boundary-iou-api/blob/master/tools/coco_instance_evaluation.py

"""
Evaluation for COCO val2017:
python ./tools/coco_instance_evaluation.py \
    --gt-json-file COCO_GT_JSON \
    --dt-json-file COCO_DT_JSON
"""
import argparse
import json

from boundary_iou.coco_instance_api.coco import COCO
from boundary_iou.coco_instance_api.cocoeval import COCOeval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-json-file", default="")
    parser.add_argument("--dt-json-file", default="")
    parser.add_argument("--iou-type", default="boundary")
    parser.add_argument("--dilation-ratio", default="0.020", type=float)
    args = parser.parse_args()
    print(args)

    annFile = args.gt_json_file
    resFile = args.dt_json_file
    dilation_ratio = args.dilation_ratio
    if args.iou_type == "boundary":
        get_boundary = True
    else:
        get_boundary = False
    cocoGt = COCO(annFile, get_boundary=get_boundary, dilation_ratio=dilation_ratio)
    
    # remove box predictions
    resFile = json.load(open(resFile))
    for c in resFile:
        c.pop("bbox", None)

    cocoDt = cocoGt.loadRes(resFile)
    cocoEval = COCOeval(cocoGt, cocoDt, iouType=args.iou_type, dilation_ratio=dilation_ratio)
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


if __name__ == '__main__':
    main()
