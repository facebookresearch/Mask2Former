from detectron2.data import transforms as T
import numpy as np

# number of masks
PATCHES_AMOUNT = (5, 40)
# mask size
PATCH_SIZE = (10, 25)
# mask types
PATCH_TYPE = ["black", "white", "gaussian"]
PATCH_STRIP_MAX = 100


class RandomPatches(T.Augmentation):
    def get_transform(self, image):
        # print(type(image), image.shape, image.dtype, image.max()) # <class 'numpy.ndarray'> (800, 800, 3) uint8 255
        h, w = image.shape[:2]
        img_copy = image.copy()
        patches = np.random.randint(*PATCHES_AMOUNT)
        for _ in range(patches):
            if np.random.rand() < 0.5:  # coin toss
                # box
                sizex = np.random.randint(*PATCH_SIZE)
                sizey = np.random.randint(*PATCH_SIZE)
            else:
                # strip
                if np.random.rand() < 0.5:
                    # wider than tall
                    sizex = np.random.randint(PATCH_SIZE[0], PATCH_STRIP_MAX)
                    sizey = np.random.randint(*PATCH_SIZE)
                else:
                    sizey = np.random.randint(PATCH_SIZE[0], PATCH_STRIP_MAX)
                    sizex = np.random.randint(*PATCH_SIZE)

            mtype = np.random.choice(PATCH_TYPE)
            patch = np.ones((sizey, sizex, 3), dtype=np.float32)  # mtype == white
            if mtype == "black":
                patch *= 0
            elif mtype == "gaussian":
                patch = np.random.random((sizey, sizex, 3))
            posx = np.random.randint(0, w - sizex)
            posy = np.random.randint(0, h - sizey)
            img_copy[posy : posy + sizey, posx : posx + sizex] = (patch * 255).astype(np.uint8)

        return T.BlendTransform(img_copy, 1, 0)
