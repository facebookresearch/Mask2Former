import copy, pickle
import json
import os.path

from detectron2.data import DatasetCatalog, MetadataCatalog


def get_dataset():
    which_folder = 1

    if which_folder == 0:
        savedir_base = "~/data/ril-digitaltwin"
    elif which_folder == 1:
        # savedir_base_json = "/mnt/home/projects/digitaltwin/data/generatorv7-small"
        savedir_base = "/mnt/colab_public/digitaltwin"

    ## MINI DS
    # PATH_IMAGES = os.path.expanduser("~/dev/ril-digitaltwin/scripts/imgs/512/generatorv7")
    # PATH_PANOPT = os.path.expanduser("~/dev/ril-digitaltwin/scripts/imgs/512/generatorv7_panoptic")
    # PATH_SEMSEG = os.path.expanduser("~/dev/ril-digitaltwin/scripts/imgs/512/generatorv7_semseg")
    DATASET_NAME_RAW = "rilv7"
    DATASET_NAME_SHN = "rilv7-shapenetv1"
    DATASET_NAME_TEST = "rilv7-test"

    PATH_MetadataCatalog = os.path.expanduser(
        f"{savedir_base}/{DATASET_NAME_RAW}/metadata.pkl"
    )
    PATH_DatasetCatalog = os.path.expanduser(
        f"{savedir_base}/{DATASET_NAME_RAW}/datasetcatalog.pkl"
    )
    if False:
        with open(PATH_MetadataCatalog, "rb") as f:
            MetadataCatalog2 = pickle.load(f)
        # with open(PATH_DatasetCatalog, "rb") as f:
        #     DatasetCatalog = pickle.load(f)

        MetadataCatalog2 = MetadataCatalog
        # DatasetCatalog = DatasetCatalog

    else:
        ## FULL DS
        PATH_IMAGES = os.path.expanduser(
            f"{savedir_base}/gen7panoptic/gen7/generatorv7"
        )

        PATH_IMAGES_SHAPENET = os.path.expanduser(
            f"{savedir_base}/gen7-shapenet/generatorv7_shapenetv1"
        )
        PATH_PANOPT = os.path.expanduser(
            f"{savedir_base}/gen7panoptic/gen7/generatorv7_panoptic"
        )
        # PATH_SEMSEG = PATH_PANOPT  # FIXME this is wrong but inconsequential
        DATA_JSON = os.path.join(PATH_PANOPT, "00000_dsinfo.json")

        TEST_SPLIT = 0.1  # 10 %

        data_json = json.load(open(DATA_JSON, "r"))
        categories = data_json["categories"]
        # dj_shn = copy.deepcopy(data_json)
        dj_shn = json.load(open(DATA_JSON, "r"))
        len_data = len(data_json["annotations"])
        len_test = int(len_data * TEST_SPLIT)
        len_train = len_data - len_test

        data_json_train = data_json.copy()
        data_json_train["annotations"] = data_json_train["annotations"][:len_train]
        data_json_train_path = os.path.join(PATH_PANOPT, "00000_dsinfo_train.json")
        if not os.path.exists(data_json_train_path):
            json.dump(data_json_train, open(data_json_train_path, "w"))

        data_json_test = data_json.copy()
        data_json_test["annotations"] = data_json_test["annotations"][len_train:]
        data_json_test_path = os.path.join(PATH_PANOPT, "00000_dsinfo_test.json")
        if not os.path.exists(data_json_test_path):
            json.dump(data_json_test, open(data_json_test_path, "w"))

        def convert_category_id(segment_info, meta):
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

        def adjust_meta_for_vis(segment_info, meta):
            if (
                segment_info["category_id"]
                in meta["thing_dataset_id_to_contiguous_id"].values()
            ):
                segment_info["category_id"] += 1
            return segment_info

        def replace_paths(info, path_inputs, path_panoptic, metadata, start, end):
            out = []
            for x in info["annotations"][start:end]:
                x["pan_seg_file_name"] = f"{path_panoptic}/{x['pan_seg_file_name']}"
                if "sem_seg_file_name" in x:
                    del x["sem_seg_file_name"]
                # x["sem_seg_file_name"] = f"{path_semseg}/{x['file_name']}"  # FIXME
                x["file_name"] = f"{path_inputs}/{x['file_name']}"
                x["segments_info"] = [
                    convert_category_id(y, metadata) for y in x["segments_info"]
                ]
                out.append(x)
            return out

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

            meta[
                "thing_dataset_id_to_contiguous_id"
            ] = thing_dataset_id_to_contiguous_id
            meta[
                "stuff_dataset_id_to_contiguous_id"
            ] = stuff_dataset_id_to_contiguous_id

            return meta

        metadata = get_metadata()
        data_train_raw = replace_paths(
            data_json, PATH_IMAGES, PATH_PANOPT, metadata, 0, len_train
        )
        data_train_shapenet = replace_paths(
            dj_shn, PATH_IMAGES_SHAPENET, PATH_PANOPT, metadata, 0, len_train
        )
        data_test = replace_paths(
            data_json, PATH_IMAGES, PATH_PANOPT, metadata, len_train, len_data
        )

        def get_data_train_raw():  # this is stupid -.-'
            return data_train_raw

        def get_data_train_shn():  # this is stupid -.-'
            return data_train_shapenet

        def get_data_test():  # this is stupid -.-'
            return data_test

        DatasetCatalog.register(DATASET_NAME_RAW, get_data_train_raw)
        DatasetCatalog.register(DATASET_NAME_SHN, get_data_train_shn)
        DatasetCatalog.register(DATASET_NAME_TEST, get_data_test)

        full_metadata = {
            "panoptic_root": PATH_PANOPT,
            "image_root": PATH_IMAGES,
            "evaluator_type": "ril_panoptic",
            "ignore_label": 1,
            "label_divisor": 1000,
            # "panoptic_json": DATA_JSON,
        }
        full_metadata.update(metadata)

        MetadataCatalog.get(DATASET_NAME_RAW).set(
            panoptic_json=data_json_train_path, **full_metadata
        )
        MetadataCatalog.get(DATASET_NAME_SHN).set(
            panoptic_json=data_json_train_path, **full_metadata
        )
        MetadataCatalog.get(DATASET_NAME_TEST).set(
            panoptic_json=data_json_test_path, **full_metadata
        )

        os.makedirs(os.path.dirname(PATH_MetadataCatalog), exist_ok=True)
        with open(PATH_MetadataCatalog, "wb") as f:
            pickle.dump(MetadataCatalog, f)
        # with open(PATH_DatasetCatalog, "wb") as f:
        #     pickle.dump(DatasetCatalog, f)


get_dataset()

# python train_net.py --config-file ../configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml --num-gpus 2 SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0001

if __name__ == "__main__":
    from pprint import pprint

    data = get_data_train_shn()
    print(len(data))
    print(data[2])
    import random
    from detectron2.utils.visualizer import Visualizer
    import cv2

    meta = MetadataCatalog.get(DATASET_NAME_SHN)

    for d in random.sample(data, 3):
        # d["segments_info"] = [adjust_meta_for_vis(x, metadata) for x in d["segments_info"]]
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=meta, scale=2)
        vis = visualizer.draw_dataset_dict(d)
        cv2.imshow(f"img {d['file_name']}", vis.get_image()[:, :, ::-1])
        cv2.waitKey(-1)
