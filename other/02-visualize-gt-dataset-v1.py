from matplotlib.patches import Rectangle

from mask2former.data.datasets.register_gen7 import get_data, get_categories
import matplotlib.pyplot as plt

data = get_data()
print(data)

cats = get_categories()
print(cats)

IDX = 0

d = data[IDX]
print(d.keys())

im = plt.imread(d["file_name"])
print(im.shape)
# plt.imshow(im)
# plt.show()


def get_cat(cat):
    for x in cats:
        if x["id"] == cat:
            return x["name"]
    return "NAN"


thing_ids = [x["id"] for x in cats if x["isthing"]]
thing_names = [x["name"] for x in cats if x["isthing"]]
thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}

for k, v in thing_dataset_id_to_contiguous_id.items():
    print(k, v, thing_names[v], thing_names[k])

# for x in d["segments_info"]:
#     cat_name = get_cat(x["category_id"])
#     bbox = x["bbox"]
#     plt.imshow(im)
#     plt.gca().add_patch(Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor="r", facecolor="none"))
#     plt.title(cat_name)
#     plt.show()
