import json
import os.path

from detectron2.data import DatasetCatalog, MetadataCatalog

CATEGORIES_JSON = os.path.expanduser("~/dev/ril-digitaltwin/scripts/ril-test-categories.json")
PATH_IMAGES = os.path.expanduser("~/dev/ril-digitaltwin/scripts/imgs/512/generatorv7")
PATH_PANOPT = os.path.expanduser("~/dev/ril-digitaltwin/scripts/imgs/512/generatorv7_panoptic")
PATH_SEMSEG = os.path.expanduser("~/dev/ril-digitaltwin/scripts/imgs/512/generatorv7_semseg")
DATA_JSON = os.path.join(PATH_PANOPT, "00000_dsinfo.json")
DATASET_NAME = "rilv7"

data_json = json.load(open(DATA_JSON, "r"))


# this is stupid but I need this somewhere else too
def get_categories():
    categories = json.load(open(CATEGORIES_JSON, "r"))
    return categories


# todo remove this and replace with the internal cats json
categories = get_categories()


def replace_paths(info, path_inputs, path_panoptic, path_semseg):
    out = []
    for x in info["annotations"]:
        x["pan_seg_file_name"] = f"{path_panoptic}/{x['pan_seg_file_name']}"
        x["sem_seg_file_name"] = f"{path_semseg}/{x['file_name']}"  # FIXME
        x["file_name"] = f"{path_inputs}/{x['file_name']}"
        out.append(x)
    return out


data = replace_paths(data_json, PATH_IMAGES, PATH_PANOPT, PATH_SEMSEG)


def get_data():  # this is stupid -.-'
    return data


thing_ids = [x["id"] for x in categories if x["isthing"]]
thing_names = [x["name"] for x in categories if x["isthing"]]
# thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}

DatasetCatalog.register(DATASET_NAME, get_data)
MetadataCatalog.get(DATASET_NAME).set(
    panoptic_root=PATH_PANOPT,
    image_root=PATH_IMAGES,
    # json_file=instances_json,
    evaluator_type="ril_panoptic",
    ignore_label=255,
    label_divisor=1000,
    stuff_classes=["background"],  # this is stupid but it seems to work
    thing_dataset_id_to_contiguous_id=thing_dataset_id_to_contiguous_id,
    thing_classes=thing_names,
    stuff_dataset_id_to_contiguous_id={0: 0},
    panoptic_json=DATA_JSON,
)


# TODO then add this to toolkit
# todo install mask2former on toolkit
# TODO see if we can call the yaml file with the train_net script

# python train_net.py --config-file ../configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml --num-gpus 2 SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0001

if __name__ == "__main__":
    data = get_data()
    print(len(data))
    print(data[2])
    import random
    from detectron2.utils.visualizer import Visualizer
    import cv2

    meta = MetadataCatalog.get(DATASET_NAME)

    for d in random.sample(data, 3):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=meta, scale=2)
        vis = visualizer.draw_dataset_dict(d)
        cv2.imshow(f"img {d['file_name']}", vis.get_image()[:, :, ::-1])
        cv2.waitKey(-1)
