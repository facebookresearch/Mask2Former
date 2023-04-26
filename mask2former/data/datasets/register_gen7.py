import json
import os.path

from detectron2.data import DatasetCatalog, MetadataCatalog

## MINI DS
# PATH_IMAGES = os.path.expanduser("~/dev/ril-digitaltwin/scripts/imgs/512/generatorv7")
# PATH_PANOPT = os.path.expanduser("~/dev/ril-digitaltwin/scripts/imgs/512/generatorv7_panoptic")
# PATH_SEMSEG = os.path.expanduser("~/dev/ril-digitaltwin/scripts/imgs/512/generatorv7_semseg")

## FULL DS
PATH_IMAGES = os.path.expanduser("~/data/ril-digitaltwin/gen7/generatorv7")
PATH_PANOPT = os.path.expanduser("~/data/ril-digitaltwin/gen7/generatorv7_panoptic")
PATH_SEMSEG = PATH_PANOPT  # FIXME this is wrong but inconsequential
DATA_JSON = os.path.join(PATH_PANOPT, "00000_dsinfo.json")

DATASET_NAME = "rilv7"

data_json = json.load(open(DATA_JSON, "r"))
categories = data_json["categories"]


def convert_category_id(segment_info, meta):
    if segment_info["category_id"] in meta["thing_dataset_id_to_contiguous_id"]:
        segment_info["category_id"] = meta["thing_dataset_id_to_contiguous_id"][segment_info["category_id"]]
        segment_info["isthing"] = True
    else:
        segment_info["category_id"] = meta["stuff_dataset_id_to_contiguous_id"][segment_info["category_id"]]
        segment_info["isthing"] = False
    return segment_info


def adjust_meta_for_vis(segment_info, meta):
    if segment_info["category_id"] in meta["thing_dataset_id_to_contiguous_id"].values():
        segment_info["category_id"] += 1
    return segment_info


def replace_paths(info, path_inputs, path_panoptic, path_semseg, metadata):
    out = []
    for x in info["annotations"]:
        x["pan_seg_file_name"] = f"{path_panoptic}/{x['pan_seg_file_name']}"
        del x["sem_seg_file_name"]
        # x["sem_seg_file_name"] = f"{path_semseg}/{x['file_name']}"  # FIXME
        x["file_name"] = f"{path_inputs}/{x['file_name']}"
        x["segments_info"] = [convert_category_id(y, metadata) for y in x["segments_info"]]
        out.append(x)
    return out


# TODO integrate this and basically redo this file from scratch with new knawledge
def get_metadata():
    meta = {}
    thing_classes = [k["name"] for k in categories]
    # thing_classes = [k["name"] for k in categories if k["isthing"] == 1]
    stuff_classes = [k["name"] for k in categories]

    meta["thing_classes"] = thing_classes
    meta["stuff_classes"] = stuff_classes

    thing_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id = {}

    for i, cat in enumerate(categories):
        if cat["isthing"]:
            thing_dataset_id_to_contiguous_id[cat["id"]] = i
        # else:
        #     stuff_dataset_id_to_contiguous_id[cat["id"]] = i

        # in order to use sem_seg evaluator
        stuff_dataset_id_to_contiguous_id[cat["id"]] = i

    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id

    return meta


metadata = get_metadata()
data = replace_paths(data_json, PATH_IMAGES, PATH_PANOPT, PATH_SEMSEG, metadata)


def get_data():  # this is stupid -.-'
    return data


DatasetCatalog.register(DATASET_NAME, get_data)
MetadataCatalog.get(DATASET_NAME).set(
    panoptic_root=PATH_PANOPT,
    image_root=PATH_IMAGES,
    evaluator_type="ril_panoptic",
    ignore_label=1,
    label_divisor=1000,
    panoptic_json=DATA_JSON,
    **metadata,
)


# TODO then add this to toolkit
# todo install mask2former on toolkit
# TODO see if we can call the yaml file with the train_net script

# python train_net.py --config-file ../configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml --num-gpus 2 SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0001

if __name__ == "__main__":
    from pprint import pprint

    data = get_data()
    print(len(data))
    print(data[2])
    import random
    from detectron2.utils.visualizer import Visualizer
    import cv2

    meta = MetadataCatalog.get(DATASET_NAME)

    for d in random.sample(data, 3):
        # d["segments_info"] = [adjust_meta_for_vis(x, metadata) for x in d["segments_info"]]
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=meta, scale=2)
        vis = visualizer.draw_dataset_dict(d)
        cv2.imshow(f"img {d['file_name']}", vis.get_image()[:, :, ::-1])
        cv2.waitKey(-1)
