# Copyright (c) Facebook, Inc. and its affiliates.
import json
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.file_io import PathManager


MAPILLARY_VISTAS_SEM_SEG_CATEGORIES = [
    {'color': [165, 42, 42],
    'id': 1,
    'isthing': 1,
    'name': 'Bird',
    'supercategory': 'animal--bird'},
    {'color': [0, 192, 0],
    'id': 2,
    'isthing': 1,
    'name': 'Ground Animal',
    'supercategory': 'animal--ground-animal'},
    {'color': [196, 196, 196],
    'id': 3,
    'isthing': 0,
    'name': 'Curb',
    'supercategory': 'construction--barrier--curb'},
    {'color': [190, 153, 153],
    'id': 4,
    'isthing': 0,
    'name': 'Fence',
    'supercategory': 'construction--barrier--fence'},
    {'color': [180, 165, 180],
    'id': 5,
    'isthing': 0,
    'name': 'Guard Rail',
    'supercategory': 'construction--barrier--guard-rail'},
    {'color': [90, 120, 150],
    'id': 6,
    'isthing': 0,
    'name': 'Barrier',
    'supercategory': 'construction--barrier--other-barrier'},
    {'color': [102, 102, 156],
    'id': 7,
    'isthing': 0,
    'name': 'Wall',
    'supercategory': 'construction--barrier--wall'},
    {'color': [128, 64, 255],
    'id': 8,
    'isthing': 0,
    'name': 'Bike Lane',
    'supercategory': 'construction--flat--bike-lane'},
    {'color': [140, 140, 200],
    'id': 9,
    'isthing': 1,
    'name': 'Crosswalk - Plain',
    'supercategory': 'construction--flat--crosswalk-plain'},
    {'color': [170, 170, 170],
    'id': 10,
    'isthing': 0,
    'name': 'Curb Cut',
    'supercategory': 'construction--flat--curb-cut'},
    {'color': [250, 170, 160],
    'id': 11,
    'isthing': 0,
    'name': 'Parking',
    'supercategory': 'construction--flat--parking'},
    {'color': [96, 96, 96],
    'id': 12,
    'isthing': 0,
    'name': 'Pedestrian Area',
    'supercategory': 'construction--flat--pedestrian-area'},
    {'color': [230, 150, 140],
    'id': 13,
    'isthing': 0,
    'name': 'Rail Track',
    'supercategory': 'construction--flat--rail-track'},
    {'color': [128, 64, 128],
    'id': 14,
    'isthing': 0,
    'name': 'Road',
    'supercategory': 'construction--flat--road'},
    {'color': [110, 110, 110],
    'id': 15,
    'isthing': 0,
    'name': 'Service Lane',
    'supercategory': 'construction--flat--service-lane'},
    {'color': [244, 35, 232],
    'id': 16,
    'isthing': 0,
    'name': 'Sidewalk',
    'supercategory': 'construction--flat--sidewalk'},
    {'color': [150, 100, 100],
    'id': 17,
    'isthing': 0,
    'name': 'Bridge',
    'supercategory': 'construction--structure--bridge'},
    {'color': [70, 70, 70],
    'id': 18,
    'isthing': 0,
    'name': 'Building',
    'supercategory': 'construction--structure--building'},
    {'color': [150, 120, 90],
    'id': 19,
    'isthing': 0,
    'name': 'Tunnel',
    'supercategory': 'construction--structure--tunnel'},
    {'color': [220, 20, 60],
    'id': 20,
    'isthing': 1,
    'name': 'Person',
    'supercategory': 'human--person'},
    {'color': [255, 0, 0],
    'id': 21,
    'isthing': 1,
    'name': 'Bicyclist',
    'supercategory': 'human--rider--bicyclist'},
    {'color': [255, 0, 100],
    'id': 22,
    'isthing': 1,
    'name': 'Motorcyclist',
    'supercategory': 'human--rider--motorcyclist'},
    {'color': [255, 0, 200],
    'id': 23,
    'isthing': 1,
    'name': 'Other Rider',
    'supercategory': 'human--rider--other-rider'},
    {'color': [200, 128, 128],
    'id': 24,
    'isthing': 1,
    'name': 'Lane Marking - Crosswalk',
    'supercategory': 'marking--crosswalk-zebra'},
    {'color': [255, 255, 255],
    'id': 25,
    'isthing': 0,
    'name': 'Lane Marking - General',
    'supercategory': 'marking--general'},
    {'color': [64, 170, 64],
    'id': 26,
    'isthing': 0,
    'name': 'Mountain',
    'supercategory': 'nature--mountain'},
    {'color': [230, 160, 50],
    'id': 27,
    'isthing': 0,
    'name': 'Sand',
    'supercategory': 'nature--sand'},
    {'color': [70, 130, 180],
    'id': 28,
    'isthing': 0,
    'name': 'Sky',
    'supercategory': 'nature--sky'},
    {'color': [190, 255, 255],
    'id': 29,
    'isthing': 0,
    'name': 'Snow',
    'supercategory': 'nature--snow'},
    {'color': [152, 251, 152],
    'id': 30,
    'isthing': 0,
    'name': 'Terrain',
    'supercategory': 'nature--terrain'},
    {'color': [107, 142, 35],
    'id': 31,
    'isthing': 0,
    'name': 'Vegetation',
    'supercategory': 'nature--vegetation'},
    {'color': [0, 170, 30],
    'id': 32,
    'isthing': 0,
    'name': 'Water',
    'supercategory': 'nature--water'},
    {'color': [255, 255, 128],
    'id': 33,
    'isthing': 1,
    'name': 'Banner',
    'supercategory': 'object--banner'},
    {'color': [250, 0, 30],
    'id': 34,
    'isthing': 1,
    'name': 'Bench',
    'supercategory': 'object--bench'},
    {'color': [100, 140, 180],
    'id': 35,
    'isthing': 1,
    'name': 'Bike Rack',
    'supercategory': 'object--bike-rack'},
    {'color': [220, 220, 220],
    'id': 36,
    'isthing': 1,
    'name': 'Billboard',
    'supercategory': 'object--billboard'},
    {'color': [220, 128, 128],
    'id': 37,
    'isthing': 1,
    'name': 'Catch Basin',
    'supercategory': 'object--catch-basin'},
    {'color': [222, 40, 40],
    'id': 38,
    'isthing': 1,
    'name': 'CCTV Camera',
    'supercategory': 'object--cctv-camera'},
    {'color': [100, 170, 30],
    'id': 39,
    'isthing': 1,
    'name': 'Fire Hydrant',
    'supercategory': 'object--fire-hydrant'},
    {'color': [40, 40, 40],
    'id': 40,
    'isthing': 1,
    'name': 'Junction Box',
    'supercategory': 'object--junction-box'},
    {'color': [33, 33, 33],
    'id': 41,
    'isthing': 1,
    'name': 'Mailbox',
    'supercategory': 'object--mailbox'},
    {'color': [100, 128, 160],
    'id': 42,
    'isthing': 1,
    'name': 'Manhole',
    'supercategory': 'object--manhole'},
    {'color': [142, 0, 0],
    'id': 43,
    'isthing': 1,
    'name': 'Phone Booth',
    'supercategory': 'object--phone-booth'},
    {'color': [70, 100, 150],
    'id': 44,
    'isthing': 0,
    'name': 'Pothole',
    'supercategory': 'object--pothole'},
    {'color': [210, 170, 100],
    'id': 45,
    'isthing': 1,
    'name': 'Street Light',
    'supercategory': 'object--street-light'},
    {'color': [153, 153, 153],
    'id': 46,
    'isthing': 1,
    'name': 'Pole',
    'supercategory': 'object--support--pole'},
    {'color': [128, 128, 128],
    'id': 47,
    'isthing': 1,
    'name': 'Traffic Sign Frame',
    'supercategory': 'object--support--traffic-sign-frame'},
    {'color': [0, 0, 80],
    'id': 48,
    'isthing': 1,
    'name': 'Utility Pole',
    'supercategory': 'object--support--utility-pole'},
    {'color': [250, 170, 30],
    'id': 49,
    'isthing': 1,
    'name': 'Traffic Light',
    'supercategory': 'object--traffic-light'},
    {'color': [192, 192, 192],
    'id': 50,
    'isthing': 1,
    'name': 'Traffic Sign (Back)',
    'supercategory': 'object--traffic-sign--back'},
    {'color': [220, 220, 0],
    'id': 51,
    'isthing': 1,
    'name': 'Traffic Sign (Front)',
    'supercategory': 'object--traffic-sign--front'},
    {'color': [140, 140, 20],
    'id': 52,
    'isthing': 1,
    'name': 'Trash Can',
    'supercategory': 'object--trash-can'},
    {'color': [119, 11, 32],
    'id': 53,
    'isthing': 1,
    'name': 'Bicycle',
    'supercategory': 'object--vehicle--bicycle'},
    {'color': [150, 0, 255],
    'id': 54,
    'isthing': 1,
    'name': 'Boat',
    'supercategory': 'object--vehicle--boat'},
    {'color': [0, 60, 100],
    'id': 55,
    'isthing': 1,
    'name': 'Bus',
    'supercategory': 'object--vehicle--bus'},
    {'color': [0, 0, 142],
    'id': 56,
    'isthing': 1,
    'name': 'Car',
    'supercategory': 'object--vehicle--car'},
    {'color': [0, 0, 90],
    'id': 57,
    'isthing': 1,
    'name': 'Caravan',
    'supercategory': 'object--vehicle--caravan'},
    {'color': [0, 0, 230],
    'id': 58,
    'isthing': 1,
    'name': 'Motorcycle',
    'supercategory': 'object--vehicle--motorcycle'},
    {'color': [0, 80, 100],
    'id': 59,
    'isthing': 0,
    'name': 'On Rails',
    'supercategory': 'object--vehicle--on-rails'},
    {'color': [128, 64, 64],
    'id': 60,
    'isthing': 1,
    'name': 'Other Vehicle',
    'supercategory': 'object--vehicle--other-vehicle'},
    {'color': [0, 0, 110],
    'id': 61,
    'isthing': 1,
    'name': 'Trailer',
    'supercategory': 'object--vehicle--trailer'},
    {'color': [0, 0, 70],
    'id': 62,
    'isthing': 1,
    'name': 'Truck',
    'supercategory': 'object--vehicle--truck'},
    {'color': [0, 0, 192],
    'id': 63,
    'isthing': 1,
    'name': 'Wheeled Slow',
    'supercategory': 'object--vehicle--wheeled-slow'},
    {'color': [32, 32, 32],
    'id': 64,
    'isthing': 0,
    'name': 'Car Mount',
    'supercategory': 'void--car-mount'},
    {'color': [120, 10, 10],
    'id': 65,
    'isthing': 0,
    'name': 'Ego Vehicle',
    'supercategory': 'void--ego-vehicle'}
]


def load_mapillary_vistas_panoptic_json(json_file, image_dir, gt_dir, semseg_dir, meta):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/coco/train2017".
        gt_dir (str): path to the raw annotations. e.g., "~/coco/panoptic_train2017".
        json_file (str): path to the json file. e.g., "~/coco/annotations/panoptic_train2017.json".
    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """

    def _convert_category_id(segment_info, meta):
        if segment_info["category_id"] in meta["thing_dataset_id_to_contiguous_id"]:
            segment_info["category_id"] = meta["thing_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
            segment_info["isthing"] = True
        else:
            segment_info["category_id"] = meta["stuff_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
            segment_info["isthing"] = False
        return segment_info

    with PathManager.open(json_file) as f:
        json_info = json.load(f)

    ret = []
    for ann in json_info["annotations"]:
        image_id = ann["image_id"]
        # TODO: currently we assume image and label has the same filename but
        # different extension, and images have extension ".jpg" for COCO. Need
        # to make image extension a user-provided argument if we extend this
        # function to support other COCO-like datasets.
        image_file = os.path.join(image_dir, os.path.splitext(ann["file_name"])[0] + ".jpg")
        label_file = os.path.join(gt_dir, ann["file_name"])
        sem_label_file = os.path.join(semseg_dir, ann["file_name"])
        segments_info = [_convert_category_id(x, meta) for x in ann["segments_info"]]
        ret.append(
            {
                "file_name": image_file,
                "image_id": image_id,
                "pan_seg_file_name": label_file,
                "sem_seg_file_name": sem_label_file,
                "segments_info": segments_info,
            }
        )
    assert len(ret), f"No images found in {image_dir}!"
    assert PathManager.isfile(ret[0]["file_name"]), ret[0]["file_name"]
    assert PathManager.isfile(ret[0]["pan_seg_file_name"]), ret[0]["pan_seg_file_name"]
    assert PathManager.isfile(ret[0]["sem_seg_file_name"]), ret[0]["sem_seg_file_name"]
    return ret


def register_mapillary_vistas_panoptic(
    name, metadata, image_root, panoptic_root, semantic_root, panoptic_json, instances_json=None
):
    """
    Register a "standard" version of ADE20k panoptic segmentation dataset named `name`.
    The dictionaries in this registered dataset follows detectron2's standard format.
    Hence it's called "standard".
    Args:
        name (str): the name that identifies a dataset,
            e.g. "ade20k_panoptic_train"
        metadata (dict): extra metadata associated with this dataset.
        image_root (str): directory which contains all the images
        panoptic_root (str): directory which contains panoptic annotation images in COCO format
        panoptic_json (str): path to the json panoptic annotation file in COCO format
        sem_seg_root (none): not used, to be consistent with
            `register_coco_panoptic_separated`.
        instances_json (str): path to the json instance annotation file
    """
    panoptic_name = name
    DatasetCatalog.register(
        panoptic_name,
        lambda: load_mapillary_vistas_panoptic_json(
            panoptic_json, image_root, panoptic_root, semantic_root, metadata
        ),
    )
    MetadataCatalog.get(panoptic_name).set(
        panoptic_root=panoptic_root,
        image_root=image_root,
        panoptic_json=panoptic_json,
        json_file=instances_json,
        evaluator_type="mapillary_vistas_panoptic_seg",
        ignore_label=65,  # different from other datasets, Mapillary Vistas sets ignore_label to 65
        label_divisor=1000,
        **metadata,
    )


_PREDEFINED_SPLITS_ADE20K_PANOPTIC = {
    "mapillary_vistas_panoptic_train": (
        "mapillary_vistas/training/images",
        "mapillary_vistas/training/panoptic",
        "mapillary_vistas/training/panoptic/panoptic_2018.json",
        "mapillary_vistas/training/labels",
    ),
    "mapillary_vistas_panoptic_val": (
        "mapillary_vistas/validation/images",
        "mapillary_vistas/validation/panoptic",
        "mapillary_vistas/validation/panoptic/panoptic_2018.json",
        "mapillary_vistas/validation/labels",
    ),
}


def get_metadata():
    meta = {}
    # The following metadata maps contiguous id from [0, #thing categories +
    # #stuff categories) to their names and colors. We have to replica of the
    # same name and color under "thing_*" and "stuff_*" because the current
    # visualization function in D2 handles thing and class classes differently
    # due to some heuristic used in Panoptic FPN. We keep the same naming to
    # enable reusing existing visualization functions.
    thing_classes = [k["name"] for k in MAPILLARY_VISTAS_SEM_SEG_CATEGORIES]
    thing_colors = [k["color"] for k in MAPILLARY_VISTAS_SEM_SEG_CATEGORIES]
    stuff_classes = [k["name"] for k in MAPILLARY_VISTAS_SEM_SEG_CATEGORIES]
    stuff_colors = [k["color"] for k in MAPILLARY_VISTAS_SEM_SEG_CATEGORIES]

    meta["thing_classes"] = thing_classes
    meta["thing_colors"] = thing_colors
    meta["stuff_classes"] = stuff_classes
    meta["stuff_colors"] = stuff_colors

    # Convert category id for training:
    #   category id: like semantic segmentation, it is the class id for each
    #   pixel. Since there are some classes not used in evaluation, the category
    #   id is not always contiguous and thus we have two set of category ids:
    #       - original category id: category id in the original dataset, mainly
    #           used for evaluation.
    #       - contiguous category id: [0, #classes), in order to train the linear
    #           softmax classifier.
    thing_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id = {}

    for i, cat in enumerate(MAPILLARY_VISTAS_SEM_SEG_CATEGORIES):
        if cat["isthing"]:
            thing_dataset_id_to_contiguous_id[cat["id"]] = i
        # else:
        #     stuff_dataset_id_to_contiguous_id[cat["id"]] = i

        # in order to use sem_seg evaluator
        stuff_dataset_id_to_contiguous_id[cat["id"]] = i

    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id

    return meta


def register_all_mapillary_vistas_panoptic(root):
    metadata = get_metadata()
    for (
        prefix,
        (image_root, panoptic_root, panoptic_json, semantic_root),
    ) in _PREDEFINED_SPLITS_ADE20K_PANOPTIC.items():
        # The "standard" version of COCO panoptic segmentation dataset,
        # e.g. used by Panoptic-DeepLab
        register_mapillary_vistas_panoptic(
            prefix,
            metadata,
            os.path.join(root, image_root),
            os.path.join(root, panoptic_root),
            os.path.join(root, semantic_root),
            os.path.join(root, panoptic_json),
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_mapillary_vistas_panoptic(_root)
