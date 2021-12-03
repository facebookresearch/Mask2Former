## Advanced Usage of Mask2Former

This document provides a brief intro of the advanced usage of Mask2Former for research purpose.

Mask2Former is highly modulized, it consists of three components: a backbone, a pixel decoder and a Transformer decoder.
You can easily replace each of these three components with your own implementation.

### Test Mask2Former with your own backbone

1. Define and register your backbone under `mask2former/modeling/backbone`. You can follow the Swin Transformer as an example.
2. Change the config file accordingly.

### Test Mask2Former with your own pixel decoder

1. Define and register your pixel decoder under `mask2former/modeling/pixel_decoder`.
2. Change the config file accordingly.

Note that, your pixel decoder must have a `self.forward_features(features)` methods that returns three values:
1. `mask_features`, which is the per-pixel embeddings with resolution 1/4 of the original image. This is used to produce binary masks.
2. `None`, you can simply return `None` for the second value.
3. `multi_scale_features`, which is the multi-scale inputs to the Transformer decoder. This must be a list with length 3.
We use resolution 1/32, 1/16, and 1/8 but you can use arbitrary resolutions here.

Example config to use a Transformer-encoder enhanced FPN instead of MSDeformAttn:
```
MODEL:
  SEM_SEG_HEAD:
    # pixel decoder
    PIXEL_DECODER_NAME: "TransformerEncoderPixelDecoder"
    IN_FEATURES: ["res2", "res3", "res4", "res5"]
    COMMON_STRIDE: 4
    TRANSFORMER_ENC_LAYERS: 6
```

### Build a new Transformer decoder.

Transformer decoders are defined under `mask2former/modeling/transformer_decoder`.
