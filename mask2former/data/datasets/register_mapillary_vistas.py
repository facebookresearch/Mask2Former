# Copyright (c) Facebook, Inc. and its affiliates.
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

MAPILLARY_VISTAS_SEM_SEG_CATEGORIES = [
    {
        "color": [165, 42, 42],
        "instances": True,
        "readable": "Bird",
        "name": "animal--bird",
        "evaluate": True,
    },
    {
        "color": [0, 192, 0],
        "instances": True,
        "readable": "Ground Animal",
        "name": "animal--ground-animal",
        "evaluate": True,
    },
    {
        "color": [196, 196, 196],
        "instances": False,
        "readable": "Curb",
        "name": "construction--barrier--curb",
        "evaluate": True,
    },
    {
        "color": [190, 153, 153],
        "instances": False,
        "readable": "Fence",
        "name": "construction--barrier--fence",
        "evaluate": True,
    },
    {
        "color": [180, 165, 180],
        "instances": False,
        "readable": "Guard Rail",
        "name": "construction--barrier--guard-rail",
        "evaluate": True,
    },
    {
        "color": [90, 120, 150],
        "instances": False,
        "readable": "Barrier",
        "name": "construction--barrier--other-barrier",
        "evaluate": True,
    },
    {
        "color": [102, 102, 156],
        "instances": False,
        "readable": "Wall",
        "name": "construction--barrier--wall",
        "evaluate": True,
    },
    {
        "color": [128, 64, 255],
        "instances": False,
        "readable": "Bike Lane",
        "name": "construction--flat--bike-lane",
        "evaluate": True,
    },
    {
        "color": [140, 140, 200],
        "instances": True,
        "readable": "Crosswalk - Plain",
        "name": "construction--flat--crosswalk-plain",
        "evaluate": True,
    },
    {
        "color": [170, 170, 170],
        "instances": False,
        "readable": "Curb Cut",
        "name": "construction--flat--curb-cut",
        "evaluate": True,
    },
    {
        "color": [250, 170, 160],
        "instances": False,
        "readable": "Parking",
        "name": "construction--flat--parking",
        "evaluate": True,
    },
    {
        "color": [96, 96, 96],
        "instances": False,
        "readable": "Pedestrian Area",
        "name": "construction--flat--pedestrian-area",
        "evaluate": True,
    },
    {
        "color": [230, 150, 140],
        "instances": False,
        "readable": "Rail Track",
        "name": "construction--flat--rail-track",
        "evaluate": True,
    },
    {
        "color": [128, 64, 128],
        "instances": False,
        "readable": "Road",
        "name": "construction--flat--road",
        "evaluate": True,
    },
    {
        "color": [110, 110, 110],
        "instances": False,
        "readable": "Service Lane",
        "name": "construction--flat--service-lane",
        "evaluate": True,
    },
    {
        "color": [244, 35, 232],
        "instances": False,
        "readable": "Sidewalk",
        "name": "construction--flat--sidewalk",
        "evaluate": True,
    },
    {
        "color": [150, 100, 100],
        "instances": False,
        "readable": "Bridge",
        "name": "construction--structure--bridge",
        "evaluate": True,
    },
    {
        "color": [70, 70, 70],
        "instances": False,
        "readable": "Building",
        "name": "construction--structure--building",
        "evaluate": True,
    },
    {
        "color": [150, 120, 90],
        "instances": False,
        "readable": "Tunnel",
        "name": "construction--structure--tunnel",
        "evaluate": True,
    },
    {
        "color": [220, 20, 60],
        "instances": True,
        "readable": "Person",
        "name": "human--person",
        "evaluate": True,
    },
    {
        "color": [255, 0, 0],
        "instances": True,
        "readable": "Bicyclist",
        "name": "human--rider--bicyclist",
        "evaluate": True,
    },
    {
        "color": [255, 0, 100],
        "instances": True,
        "readable": "Motorcyclist",
        "name": "human--rider--motorcyclist",
        "evaluate": True,
    },
    {
        "color": [255, 0, 200],
        "instances": True,
        "readable": "Other Rider",
        "name": "human--rider--other-rider",
        "evaluate": True,
    },
    {
        "color": [200, 128, 128],
        "instances": True,
        "readable": "Lane Marking - Crosswalk",
        "name": "marking--crosswalk-zebra",
        "evaluate": True,
    },
    {
        "color": [255, 255, 255],
        "instances": False,
        "readable": "Lane Marking - General",
        "name": "marking--general",
        "evaluate": True,
    },
    {
        "color": [64, 170, 64],
        "instances": False,
        "readable": "Mountain",
        "name": "nature--mountain",
        "evaluate": True,
    },
    {
        "color": [230, 160, 50],
        "instances": False,
        "readable": "Sand",
        "name": "nature--sand",
        "evaluate": True,
    },
    {
        "color": [70, 130, 180],
        "instances": False,
        "readable": "Sky",
        "name": "nature--sky",
        "evaluate": True,
    },
    {
        "color": [190, 255, 255],
        "instances": False,
        "readable": "Snow",
        "name": "nature--snow",
        "evaluate": True,
    },
    {
        "color": [152, 251, 152],
        "instances": False,
        "readable": "Terrain",
        "name": "nature--terrain",
        "evaluate": True,
    },
    {
        "color": [107, 142, 35],
        "instances": False,
        "readable": "Vegetation",
        "name": "nature--vegetation",
        "evaluate": True,
    },
    {
        "color": [0, 170, 30],
        "instances": False,
        "readable": "Water",
        "name": "nature--water",
        "evaluate": True,
    },
    {
        "color": [255, 255, 128],
        "instances": True,
        "readable": "Banner",
        "name": "object--banner",
        "evaluate": True,
    },
    {
        "color": [250, 0, 30],
        "instances": True,
        "readable": "Bench",
        "name": "object--bench",
        "evaluate": True,
    },
    {
        "color": [100, 140, 180],
        "instances": True,
        "readable": "Bike Rack",
        "name": "object--bike-rack",
        "evaluate": True,
    },
    {
        "color": [220, 220, 220],
        "instances": True,
        "readable": "Billboard",
        "name": "object--billboard",
        "evaluate": True,
    },
    {
        "color": [220, 128, 128],
        "instances": True,
        "readable": "Catch Basin",
        "name": "object--catch-basin",
        "evaluate": True,
    },
    {
        "color": [222, 40, 40],
        "instances": True,
        "readable": "CCTV Camera",
        "name": "object--cctv-camera",
        "evaluate": True,
    },
    {
        "color": [100, 170, 30],
        "instances": True,
        "readable": "Fire Hydrant",
        "name": "object--fire-hydrant",
        "evaluate": True,
    },
    {
        "color": [40, 40, 40],
        "instances": True,
        "readable": "Junction Box",
        "name": "object--junction-box",
        "evaluate": True,
    },
    {
        "color": [33, 33, 33],
        "instances": True,
        "readable": "Mailbox",
        "name": "object--mailbox",
        "evaluate": True,
    },
    {
        "color": [100, 128, 160],
        "instances": True,
        "readable": "Manhole",
        "name": "object--manhole",
        "evaluate": True,
    },
    {
        "color": [142, 0, 0],
        "instances": True,
        "readable": "Phone Booth",
        "name": "object--phone-booth",
        "evaluate": True,
    },
    {
        "color": [70, 100, 150],
        "instances": False,
        "readable": "Pothole",
        "name": "object--pothole",
        "evaluate": True,
    },
    {
        "color": [210, 170, 100],
        "instances": True,
        "readable": "Street Light",
        "name": "object--street-light",
        "evaluate": True,
    },
    {
        "color": [153, 153, 153],
        "instances": True,
        "readable": "Pole",
        "name": "object--support--pole",
        "evaluate": True,
    },
    {
        "color": [128, 128, 128],
        "instances": True,
        "readable": "Traffic Sign Frame",
        "name": "object--support--traffic-sign-frame",
        "evaluate": True,
    },
    {
        "color": [0, 0, 80],
        "instances": True,
        "readable": "Utility Pole",
        "name": "object--support--utility-pole",
        "evaluate": True,
    },
    {
        "color": [250, 170, 30],
        "instances": True,
        "readable": "Traffic Light",
        "name": "object--traffic-light",
        "evaluate": True,
    },
    {
        "color": [192, 192, 192],
        "instances": True,
        "readable": "Traffic Sign (Back)",
        "name": "object--traffic-sign--back",
        "evaluate": True,
    },
    {
        "color": [220, 220, 0],
        "instances": True,
        "readable": "Traffic Sign (Front)",
        "name": "object--traffic-sign--front",
        "evaluate": True,
    },
    {
        "color": [140, 140, 20],
        "instances": True,
        "readable": "Trash Can",
        "name": "object--trash-can",
        "evaluate": True,
    },
    {
        "color": [119, 11, 32],
        "instances": True,
        "readable": "Bicycle",
        "name": "object--vehicle--bicycle",
        "evaluate": True,
    },
    {
        "color": [150, 0, 255],
        "instances": True,
        "readable": "Boat",
        "name": "object--vehicle--boat",
        "evaluate": True,
    },
    {
        "color": [0, 60, 100],
        "instances": True,
        "readable": "Bus",
        "name": "object--vehicle--bus",
        "evaluate": True,
    },
    {
        "color": [0, 0, 142],
        "instances": True,
        "readable": "Car",
        "name": "object--vehicle--car",
        "evaluate": True,
    },
    {
        "color": [0, 0, 90],
        "instances": True,
        "readable": "Caravan",
        "name": "object--vehicle--caravan",
        "evaluate": True,
    },
    {
        "color": [0, 0, 230],
        "instances": True,
        "readable": "Motorcycle",
        "name": "object--vehicle--motorcycle",
        "evaluate": True,
    },
    {
        "color": [0, 80, 100],
        "instances": False,
        "readable": "On Rails",
        "name": "object--vehicle--on-rails",
        "evaluate": True,
    },
    {
        "color": [128, 64, 64],
        "instances": True,
        "readable": "Other Vehicle",
        "name": "object--vehicle--other-vehicle",
        "evaluate": True,
    },
    {
        "color": [0, 0, 110],
        "instances": True,
        "readable": "Trailer",
        "name": "object--vehicle--trailer",
        "evaluate": True,
    },
    {
        "color": [0, 0, 70],
        "instances": True,
        "readable": "Truck",
        "name": "object--vehicle--truck",
        "evaluate": True,
    },
    {
        "color": [0, 0, 192],
        "instances": True,
        "readable": "Wheeled Slow",
        "name": "object--vehicle--wheeled-slow",
        "evaluate": True,
    },
    {
        "color": [32, 32, 32],
        "instances": False,
        "readable": "Car Mount",
        "name": "void--car-mount",
        "evaluate": True,
    },
    {
        "color": [120, 10, 10],
        "instances": False,
        "readable": "Ego Vehicle",
        "name": "void--ego-vehicle",
        "evaluate": True,
    },
    {
        "color": [0, 0, 0],
        "instances": False,
        "readable": "Unlabeled",
        "name": "void--unlabeled",
        "evaluate": False,
    },
]


def _get_mapillary_vistas_meta():
    stuff_classes = [k["readable"] for k in MAPILLARY_VISTAS_SEM_SEG_CATEGORIES if k["evaluate"]]
    assert len(stuff_classes) == 65

    stuff_colors = [k["color"] for k in MAPILLARY_VISTAS_SEM_SEG_CATEGORIES if k["evaluate"]]
    assert len(stuff_colors) == 65

    ret = {
        "stuff_classes": stuff_classes,
        "stuff_colors": stuff_colors,
    }
    return ret


def register_all_mapillary_vistas(root):
    root = os.path.join(root, "mapillary_vistas")
    meta = _get_mapillary_vistas_meta()
    for name, dirname in [("train", "training"), ("val", "validation")]:
        image_dir = os.path.join(root, dirname, "images")
        gt_dir = os.path.join(root, dirname, "labels")
        name = f"mapillary_vistas_sem_seg_{name}"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="jpg")
        )
        MetadataCatalog.get(name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=65,  # different from other datasets, Mapillary Vistas sets ignore_label to 65
            **meta,
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_mapillary_vistas(_root)
