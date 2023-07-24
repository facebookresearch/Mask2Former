# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/sukjunhwang/IFC

from . import builtin  # ensure the builtin datasets are registered

__all__ = [k for k in globals().keys() if "builtin" not in k and not k.startswith("_")]
