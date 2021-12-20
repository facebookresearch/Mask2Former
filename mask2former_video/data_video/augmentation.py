# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/sukjunhwang/IFC

import numpy as np
import logging
import sys
from fvcore.transforms.transform import (
    HFlipTransform,
    NoOpTransform,
    VFlipTransform,
)
from PIL import Image

from detectron2.data import transforms as T


class ResizeShortestEdge(T.Augmentation):
    """
    Scale the shorter edge to the given size, with a limit of `max_size` on the longer edge.
    If `max_size` is reached, then downscale so that the longer edge does not exceed max_size.
    """

    def __init__(
        self, short_edge_length, max_size=sys.maxsize, sample_style="range", interp=Image.BILINEAR, clip_frame_cnt=1
    ):
        """
        Args:
            short_edge_length (list[int]): If ``sample_style=="range"``,
                a [min, max] interval from which to sample the shortest edge length.
                If ``sample_style=="choice"``, a list of shortest edge lengths to sample from.
            max_size (int): maximum allowed longest edge length.
            sample_style (str): either "range" or "choice".
        """
        super().__init__()
        assert sample_style in ["range", "choice", "range_by_clip", "choice_by_clip"], sample_style

        self.is_range = ("range" in sample_style)
        if isinstance(short_edge_length, int):
            short_edge_length = (short_edge_length, short_edge_length)
        if self.is_range:
            assert len(short_edge_length) == 2, (
                "short_edge_length must be two values using 'range' sample style."
                f" Got {short_edge_length}!"
            )
        self._cnt = 0
        self._init(locals())

    def get_transform(self, image):
        if self._cnt % self.clip_frame_cnt == 0:
            if self.is_range:
                self.size = np.random.randint(self.short_edge_length[0], self.short_edge_length[1] + 1)
            else:
                self.size = np.random.choice(self.short_edge_length)
            if self.size == 0:
                return NoOpTransform()

            self._cnt = 0   # avoiding overflow
        self._cnt += 1

        h, w = image.shape[:2]

        scale = self.size * 1.0 / min(h, w)
        if h < w:
            newh, neww = self.size, scale * w
        else:
            newh, neww = scale * h, self.size
        if max(newh, neww) > self.max_size:
            scale = self.max_size * 1.0 / max(newh, neww)
            newh = newh * scale
            neww = neww * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return T.ResizeTransform(h, w, newh, neww, self.interp)


class RandomFlip(T.Augmentation):
    """
    Flip the image horizontally or vertically with the given probability.
    """

    def __init__(self, prob=0.5, *, horizontal=True, vertical=False, clip_frame_cnt=1):
        """
        Args:
            prob (float): probability of flip.
            horizontal (boolean): whether to apply horizontal flipping
            vertical (boolean): whether to apply vertical flipping
        """
        super().__init__()

        if horizontal and vertical:
            raise ValueError("Cannot do both horiz and vert. Please use two Flip instead.")
        if not horizontal and not vertical:
            raise ValueError("At least one of horiz or vert has to be True!")
        self._cnt = 0

        self._init(locals())

    def get_transform(self, image):
        if self._cnt % self.clip_frame_cnt == 0:
            self.do = self._rand_range() < self.prob
            self._cnt = 0   # avoiding overflow
        self._cnt += 1

        h, w = image.shape[:2]

        if self.do:
            if self.horizontal:
                return HFlipTransform(w)
            elif self.vertical:
                return VFlipTransform(h)
        else:
            return NoOpTransform()


def build_augmentation(cfg, is_train):
    logger = logging.getLogger(__name__)
    aug_list = []
    if is_train:
        # Crop
        if cfg.INPUT.CROP.ENABLED:
            aug_list.append(T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))

        # Resize
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        ms_clip_frame_cnt = cfg.INPUT.SAMPLING_FRAME_NUM if "by_clip" in cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING else 1
        aug_list.append(ResizeShortestEdge(min_size, max_size, sample_style, clip_frame_cnt=ms_clip_frame_cnt))

        # Flip
        if cfg.INPUT.RANDOM_FLIP != "none":
            if cfg.INPUT.RANDOM_FLIP == "flip_by_clip":
                flip_clip_frame_cnt = cfg.INPUT.SAMPLING_FRAME_NUM
            else:
                flip_clip_frame_cnt = 1

            aug_list.append(
                # NOTE using RandomFlip modified for the support of flip maintenance
                RandomFlip(
                    horizontal=(cfg.INPUT.RANDOM_FLIP == "horizontal") or (cfg.INPUT.RANDOM_FLIP == "flip_by_clip"),
                    vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
                    clip_frame_cnt=flip_clip_frame_cnt,
                )
            )

        # Additional augmentations : brightness, contrast, saturation, rotation
        augmentations = cfg.INPUT.AUGMENTATIONS
        if "brightness" in augmentations:
            aug_list.append(T.RandomBrightness(0.9, 1.1))
        if "contrast" in augmentations:
            aug_list.append(T.RandomContrast(0.9, 1.1))
        if "saturation" in augmentations:
            aug_list.append(T.RandomSaturation(0.9, 1.1))
        if "rotation" in augmentations:
            aug_list.append(
                T.RandomRotation(
                    [-15, 15], expand=False, center=[(0.4, 0.4), (0.6, 0.6)], sample_style="range"
                )
            )
    else:
        # Resize
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
        aug_list.append(T.ResizeShortestEdge(min_size, max_size, sample_style))

    return aug_list
