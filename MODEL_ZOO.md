# Mask2Former Model Zoo and Baselines

## Introduction

This file documents a collection of models reported in our paper.
All numbers were obtained on [Big Basin](https://engineering.fb.com/data-center-engineering/introducing-big-basin-our-next-generation-ai-hardware/)
servers with 8 NVIDIA V100 GPUs & NVLink (except Swin-L models are trained with 16 NVIDIA V100 GPUs).

#### How to Read the Tables
* The "Name" column contains a link to the config file. Running `train_net.py --num-gpus 8` with this config file
  will reproduce the model (except Swin-L models are trained with 16 NVIDIA V100 GPUs with distributed training on two nodes).
* The *model id* column is provided for ease of reference.
  To check downloaded file integrity, any model on this page contains its md5 prefix in its file name.

#### Detectron2 ImageNet Pretrained Models

It's common to initialize from backbone models pre-trained on ImageNet classification tasks. The following backbone models are available:

* [R-50.pkl (torchvision)](https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/torchvision/R-50.pkl): converted copy of [torchvision's ResNet-50](https://pytorch.org/docs/stable/torchvision/models.html#torchvision.models.resnet50) model.
  More details can be found in [the conversion script](tools/convert-torchvision-to-d2.py).
* [R-103.pkl](https://dl.fbaipublicfiles.com/detectron2/DeepLab/R-103.pkl): a ResNet-101 with its first 7x7 convolution replaced by 3 3x3 convolutions. This modification has been used in most semantic segmentation papers (a.k.a. ResNet101c in our paper). We pre-train this backbone on ImageNet using the default recipe of [pytorch examples](https://github.com/pytorch/examples/tree/master/imagenet).

Note: below are available pretrained models in Detectron2 that we do not use in our paper.
* [R-50.pkl](https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-50.pkl): converted copy of [MSRA's original ResNet-50](https://github.com/KaimingHe/deep-residual-networks) model.
* [R-101.pkl](https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-101.pkl): converted copy of [MSRA's original ResNet-101](https://github.com/KaimingHe/deep-residual-networks) model.
* [X-101-32x8d.pkl](https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/FAIR/X-101-32x8d.pkl): ResNeXt-101-32x8d model trained with Caffe2 at FB.

#### Third-party ImageNet Pretrained Models

Our paper also uses ImageNet pretrained models that are not part of Detectron2, please refer to [tools](https://github.com/facebookresearch/MaskFormer/tree/master/tools) to get those pretrained models.

#### License

All models available for download through this document are licensed under the
[Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/).

## COCO Model Zoo

### Panoptic Segmentation

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">epochs</th>
<th valign="bottom">PQ</th>
<th valign="bottom">AP</th>
<th valign="bottom">mIoU</th>
<th valign="bottom">model id</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: maskformer2_R50_bs16_50ep -->
 <tr><td align="left"><a href="configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml">Mask2Former</a></td>
<td align="center">R50</td>
<td align="center">50</td>
<td align="center">51.9</td>
<td align="center">41.7</td>
<td align="center">61.7</td>
<td align="center">47430278_4</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/panoptic/maskformer2_R50_bs16_50ep/model_final_94dc52.pkl">model</a></td>
</tr>
<!-- ROW: maskformer2_R101_bs16_50ep -->
 <tr><td align="left"><a href="configs/coco/panoptic-segmentation/maskformer2_R101_bs16_50ep.yaml">Mask2Former</a></td>
<td align="center">R101</td>
<td align="center">50</td>
<td align="center">52.6</td>
<td align="center">42.6</td>
<td align="center">62.4</td>
<td align="center">47992113_1</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/panoptic/maskformer2_R101_bs16_50ep/model_final_b807bd.pkl">model</a></td>
</tr>
<!-- ROW: maskformer2_swin_tiny_bs16_50ep -->
 <tr><td align="left"><a href="configs/coco/panoptic-segmentation/swin/maskformer2_swin_tiny_bs16_50ep.yaml">Mask2Former</a></td>
<td align="center">Swin-T</td>
<td align="center">50</td>
<td align="center">53.2</td>
<td align="center">43.3</td>
<td align="center">63.2</td>
<td align="center">48558700_1</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/panoptic/maskformer2_swin_tiny_bs16_50ep/model_final_9fd0ae.pkl">model</a></td>
</tr>
<!-- ROW: maskformer2_swin_small_bs16_50ep -->
 <tr><td align="left"><a href="configs/coco/panoptic-segmentation/swin/maskformer2_swin_small_bs16_50ep.yaml">Mask2Former</a></td>
<td align="center">Swin-S</td>
<td align="center">50</td>
<td align="center">54.6</td>
<td align="center">44.7</td>
<td align="center">64.2</td>
<td align="center">48558700_3</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/panoptic/maskformer2_swin_small_bs16_50ep/model_final_a407fd.pkl">model</a></td>
</tr>
<!-- ROW: maskformer2_swin_base_384_bs16_50ep -->
 <tr><td align="left"><a href="configs/coco/panoptic-segmentation/swin/maskformer2_swin_base_384_bs16_50ep.yaml">Mask2Former</a></td>
<td align="center">Swin-B</td>
<td align="center">50</td>
<td align="center">55.1</td>
<td align="center">45.2</td>
<td align="center">65.1</td>
<td align="center">48558700_5</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/panoptic/maskformer2_swin_base_384_bs16_50ep/model_final_9d7f02.pkl">model</a></td>
</tr>
<!-- ROW: maskformer2_swin_base_IN21k_384_bs16_50ep -->
 <tr><td align="left"><a href="configs/coco/panoptic-segmentation/swin/maskformer2_swin_base_IN21k_384_bs16_50ep.yaml">Mask2Former</a></td>
<td align="center">Swin-B (IN21k)</td>
<td align="center">50</td>
<td align="center">56.4</td>
<td align="center">46.3</td>
<td align="center">67.1</td>
<td align="center">48558700_7</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/panoptic/maskformer2_swin_base_IN21k_384_bs16_50ep/model_final_54b88a.pkl">model</a></td>
</tr>
<!-- ROW: maskformer2_swin_large_IN21k_384_bs16_100ep -->
 <tr><td align="left"><a href="configs/coco/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml">Mask2Former (200 queries)</a></td>
<td align="center">Swin-L (IN21k)</td>
<td align="center">100</td>
<td align="center">57.8</td>
<td align="center">48.6</td>
<td align="center">67.4</td>
<td align="center">47429163_0</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/panoptic/maskformer2_swin_large_IN21k_384_bs16_100ep/model_final_f07440.pkl">model</a></td>
</tr>
</tbody></table>


### Instance Segmentation

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">epochs</th>
<th valign="bottom">AP</th>
<th valign="bottom">Boundary AP</th>
<th valign="bottom">model id</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: maskformer2_R50_bs16_50ep -->
 <tr><td align="left"><a href="configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml">Mask2Former</a></td>
<td align="center">R50</td>
<td align="center">50</td>
<td align="center">43.7</td>
<td align="center">30.6</td>
<td align="center">47430277_2</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_R50_bs16_50ep/model_final_3c8ec9.pkl">model</a></td>
</tr>
<!-- ROW: maskformer2_R101_bs16_50ep -->
 <tr><td align="left"><a href="configs/coco/instance-segmentation/maskformer2_R101_bs16_50ep.yaml">Mask2Former</a></td>
<td align="center">R101</td>
<td align="center">50</td>
<td align="center">44.2</td>
<td align="center">31.1</td>
<td align="center">47992113_0</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_R101_bs16_50ep/model_final_eba159.pkl">model</a></td>
</tr>
<!-- ROW: maskformer2_swin_tiny_bs16_50ep -->
 <tr><td align="left"><a href="configs/coco/instance-segmentation/swin/maskformer2_swin_tiny_bs16_50ep.yaml">Mask2Former</a></td>
<td align="center">Swin-T</td>
<td align="center">50</td>
<td align="center">45.0</td>
<td align="center">31.8</td>
<td align="center">48558700_0</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_swin_tiny_bs16_50ep/model_final_86143f.pkl">model</a></td>
</tr>
<!-- ROW: maskformer2_swin_small_bs16_50ep -->
 <tr><td align="left"><a href="configs/coco/instance-segmentation/swin/maskformer2_swin_small_bs16_50ep.yaml">Mask2Former</a></td>
<td align="center">Swin-S</td>
<td align="center">50</td>
<td align="center">46.3</td>
<td align="center">32.9</td>
<td align="center">48558700_2</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_swin_small_bs16_50ep/model_final_1e7f22.pkl">model</a></td>
</tr>
<!-- ROW: maskformer2_swin_base_384_bs16_50ep -->
 <tr><td align="left"><a href="configs/coco/instance-segmentation/swin/maskformer2_swin_base_384_bs16_50ep.yaml">Mask2Former</a></td>
<td align="center">Swin-B</td>
<td align="center">50</td>
<td align="center">46.7</td>
<td align="center">33.2</td>
<td align="center">48558700_4</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_swin_base_384_bs16_50ep/model_final_f6e0f6.pkl">model</a></td>
</tr>
<!-- ROW: maskformer2_swin_base_IN21k_384_bs16_50ep -->
 <tr><td align="left"><a href="configs/coco/instance-segmentation/swin/maskformer2_swin_base_IN21k_384_bs16_50ep.yaml">Mask2Former</a></td>
<td align="center">Swin-B (IN21k)</td>
<td align="center">50</td>
<td align="center">48.1</td>
<td align="center">34.4</td>
<td align="center">48558700_6</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_swin_base_IN21k_384_bs16_50ep/model_final_83d103.pkl">model</a></td>
</tr>
<!-- ROW: maskformer2_swin_large_IN21k_384_bs16_100ep -->
 <tr><td align="left"><a href="configs/coco/instance-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml">Mask2Former (200 queries)</a></td>
<td align="center">Swin-L (IN21k)</td>
<td align="center">100</td>
<td align="center">50.1</td>
<td align="center">36.2</td>
<td align="center">48235555</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_swin_large_IN21k_384_bs16_100ep/model_final_e5f453.pkl">model</a></td>
</tr>
</tbody></table>


## Cityscapes Model Zoo

### Panoptic Segmentation

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">iterations</th>
<th valign="bottom">PQ</th>
<th valign="bottom">AP</th>
<th valign="bottom">mIoU</th>
<th valign="bottom">model id</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: maskformer2_R50_bs16_90k -->
 <tr><td align="left"><a href="configs/cityscapes/panoptic-segmentation/maskformer2_R50_bs16_90k.yaml">Mask2Former</a></td>
<td align="center">R50</td>
<td align="center">90k</td>
<td align="center">62.1</td>
<td align="center">37.3</td>
<td align="center">77.5</td>
<td align="center">48267400_0</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/maskformer/mask2former/cityscapes/panoptic/maskformer2_R50_bs16_90k/model_final_4ab90c.pkl">model</a></td>
</tr>
<!-- ROW: maskformer2_R101_bs16_90k -->
 <tr><td align="left"><a href="configs/cityscapes/panoptic-segmentation/maskformer2_R101_bs16_90k.yaml">Mask2Former</a></td>
<td align="center">R101</td>
<td align="center">90k</td>
<td align="center">62.4</td>
<td align="center">37.7</td>
<td align="center">78.6</td>
<td align="center">48267400_11</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/maskformer/mask2former/cityscapes/panoptic/maskformer2_R101_bs16_90k/model_final_04d286.pkl">model</a></td>
</tr>
<!-- ROW: maskformer2_swin_tiny_bs16_90k -->
 <tr><td align="left"><a href="configs/cityscapes/panoptic-segmentation/swin/maskformer2_swin_tiny_bs16_90k.yaml">Mask2Former</a></td>
<td align="center">Swin-T</td>
<td align="center">90k</td>
<td align="center">63.9</td>
<td align="center">39.1</td>
<td align="center">80.5</td>
<td align="center">48333144_2</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/maskformer/mask2former/cityscapes/panoptic/maskformer2_swin_tiny_bs16_90k/model_final_ceba0f.pkl">model</a></td>
</tr>
<!-- ROW: maskformer2_swin_small_bs16_90k -->
 <tr><td align="left"><a href="configs/cityscapes/panoptic-segmentation/swin/maskformer2_swin_small_bs16_90k.yaml">Mask2Former</a></td>
<td align="center">Swin-S</td>
<td align="center">90k</td>
<td align="center">64.8</td>
<td align="center">40.7</td>
<td align="center">81.8</td>
<td align="center">48381916</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/maskformer/mask2former/cityscapes/panoptic/maskformer2_swin_small_bs16_90k/model_final_23efb7.pkl">model</a></td>
</tr>
<!-- ROW: maskformer2_swin_base_IN21k_384_bs16_90k -->
 <tr><td align="left"><a href="configs/cityscapes/panoptic-segmentation/swin/maskformer2_swin_base_IN21k_384_bs16_90k.yaml">Mask2Former</a></td>
<td align="center">Swin-B (IN21k)</td>
<td align="center">90k</td>
<td align="center">66.1</td>
<td align="center">42.8</td>
<td align="center">82.7</td>
<td align="center">48333157_2</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/maskformer/mask2former/cityscapes/panoptic/maskformer2_swin_base_IN21k_384_bs16_90k/model_final_fa840f.pkl">model</a></td>
</tr>
<!-- ROW: maskformer2_swin_large_IN21k_384_bs16_90k -->
 <tr><td align="left"><a href="configs/cityscapes/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_90k.yaml">Mask2Former (200 queries)</a></td>
<td align="center">Swin-L (IN21k)</td>
<td align="center">90k</td>
<td align="center">66.6</td>
<td align="center">43.6</td>
<td align="center">82.9</td>
<td align="center">48318254_2</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/maskformer/mask2former/cityscapes/panoptic/maskformer2_swin_large_IN21k_384_bs16_90k/model_final_064788.pkl">model</a></td>
</tr>
</tbody></table>


### Instance Segmentation

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">iterations</th>
<th valign="bottom">AP</th>
<th valign="bottom">AP50</th>
<th valign="bottom">model id</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: maskformer2_R50_bs16_90k -->
 <tr><td align="left"><a href="configs/cityscapes/instance-segmentation/maskformer2_R50_bs16_90k.yaml">Mask2Former</a></td>
<td align="center">R50</td>
<td align="center">90k</td>
<td align="center">37.4</td>
<td align="center">61.9</td>
<td align="center">48267400_8</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/maskformer/mask2former/cityscapes/instance/maskformer2_R50_bs16_90k/model_final_01a8ed.pkl">model</a></td>
</tr>
<!-- ROW: maskformer2_R101_bs16_90k -->
 <tr><td align="left"><a href="configs/cityscapes/instance-segmentation/maskformer2_R101_bs16_90k.yaml">Mask2Former</a></td>
<td align="center">R101</td>
<td align="center">90k</td>
<td align="center">38.5</td>
<td align="center">63.9</td>
<td align="center">48267400_16</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/maskformer/mask2former/cityscapes/instance/maskformer2_R101_bs16_90k/model_final_c2b8c8.pkl">model</a></td>
</tr>
<!-- ROW: maskformer2_swin_tiny_bs16_90k -->
 <tr><td align="left"><a href="configs/cityscapes/instance-segmentation/swin/maskformer2_swin_tiny_bs16_90k.yaml">Mask2Former</a></td>
<td align="center">Swin-T</td>
<td align="center">90k</td>
<td align="center">39.7</td>
<td align="center">66.9</td>
<td align="center">48333144_4</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/maskformer/mask2former/cityscapes/instance/maskformer2_swin_tiny_bs16_90k/model_final_6a63f8.pkl">model</a></td>
</tr>
<!-- ROW: maskformer2_swin_small_bs16_90k -->
 <tr><td align="left"><a href="configs/cityscapes/instance-segmentation/swin/maskformer2_swin_small_bs16_90k.yaml">Mask2Former</a></td>
<td align="center">Swin-S</td>
<td align="center">90k</td>
<td align="center">41.8</td>
<td align="center">70.4</td>
<td align="center">48333149_4</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/maskformer/mask2former/cityscapes/instance/maskformer2_swin_small_bs16_90k/model_final_974824.pkl">model</a></td>
</tr>
<!-- ROW: maskformer2_swin_base_IN21k_384_bs16_90k -->
 <tr><td align="left"><a href="configs/cityscapes/instance-segmentation/swin/maskformer2_swin_base_IN21k_384_bs16_90k.yaml">Mask2Former</a></td>
<td align="center">Swin-B (IN21k)</td>
<td align="center">90k</td>
<td align="center">42.0</td>
<td align="center">68.8</td>
<td align="center">48333157_4</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/maskformer/mask2former/cityscapes/instance/maskformer2_swin_base_IN21k_384_bs16_90k/model_final_7a3b6d.pkl">model</a></td>
</tr>
<!-- ROW: maskformer2_swin_large_IN21k_384_bs16_90k -->
 <tr><td align="left"><a href="configs/cityscapes/instance-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_90k.yaml">Mask2Former (200 queries)</a></td>
<td align="center">Swin-L (IN21k)</td>
<td align="center">90k</td>
<td align="center">43.7</td>
<td align="center">71.4</td>
<td align="center">49111004_2</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/maskformer/mask2former/cityscapes/instance/maskformer2_swin_large_IN21k_384_bs16_90k/model_final_dfa862.pkl">model</a></td>
</tr>
</tbody></table>


### Semantic Segmentation

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">iterations</th>
<th valign="bottom">mIoU</th>
<th valign="bottom">mIoU (ms+flip)</th>
<th valign="bottom">model id</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: maskformer2_R50_bs16_90k -->
 <tr><td align="left"><a href="configs/cityscapes/semantic-segmentation/maskformer2_R50_bs16_90k.yaml">Mask2Former</a></td>
<td align="center">R50</td>
<td align="center">90k</td>
<td align="center">79.4</td>
<td align="center">82.2</td>
<td align="center">48267400_4</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/maskformer/mask2former/cityscapes/semantic/maskformer2_R50_bs16_90k/model_final_cc1b1f.pkl">model</a></td>
</tr>
<!-- ROW: maskformer2_R101_bs16_90k -->
 <tr><td align="left"><a href="configs/cityscapes/semantic-segmentation/maskformer2_R101_bs16_90k.yaml">Mask2Former</a></td>
<td align="center">R101</td>
<td align="center">90k</td>
<td align="center">80.1</td>
<td align="center">81.9</td>
<td align="center">48267400_13</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/maskformer/mask2former/cityscapes/semantic/maskformer2_R101_bs16_90k/model_final_257ce8.pkl">model</a></td>
</tr>
<!-- ROW: maskformer2_swin_tiny_bs16_90k -->
 <tr><td align="left"><a href="configs/cityscapes/semantic-segmentation/swin/maskformer2_swin_tiny_bs16_90k.yaml">Mask2Former</a></td>
<td align="center">Swin-T</td>
<td align="center">90k</td>
<td align="center">82.1</td>
<td align="center">83.0</td>
<td align="center">48333144_3</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/maskformer/mask2former/cityscapes/semantic/maskformer2_swin_tiny_bs16_90k/model_final_2d58d4.pkl">model</a></td>
</tr>
<!-- ROW: maskformer2_swin_small_bs16_90k -->
 <tr><td align="left"><a href="configs/cityscapes/semantic-segmentation/swin/maskformer2_swin_small_bs16_90k.yaml">Mask2Former</a></td>
<td align="center">Swin-S</td>
<td align="center">90k</td>
<td align="center">82.6</td>
<td align="center">83.6</td>
<td align="center">48333149_3</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/maskformer/mask2former/cityscapes/semantic/maskformer2_swin_small_bs16_90k/model_final_fa26ae.pkl">model</a></td>
</tr>
<!-- ROW: maskformer2_swin_base_IN21k_384_bs16_90k -->
 <tr><td align="left"><a href="configs/cityscapes/semantic-segmentation/swin/maskformer2_swin_base_IN21k_384_bs16_90k.yaml">Mask2Former</a></td>
<td align="center">Swin-B (IN21k)</td>
<td align="center">90k</td>
<td align="center">83.3</td>
<td align="center">84.5</td>
<td align="center">48333157_3</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/maskformer/mask2former/cityscapes/semantic/maskformer2_swin_base_IN21k_384_bs16_90k/model_final_1c6b65.pkl">model</a></td>
</tr>
<!-- ROW: maskformer2_swin_large_IN21k_384_bs16_90k -->
 <tr><td align="left"><a href="configs/cityscapes/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_90k.yaml">Mask2Former</a></td>
<td align="center">Swin-L (IN21k)</td>
<td align="center">90k</td>
<td align="center">83.3</td>
<td align="center">84.3</td>
<td align="center">48318254_5</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/maskformer/mask2former/cityscapes/semantic/maskformer2_swin_large_IN21k_384_bs16_90k/model_final_17c1ee.pkl">model</a></td>
</tr>
</tbody></table>


## ADE20K Model Zoo

### Panoptic Segmentation

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">iterations</th>
<th valign="bottom">PQ</th>
<th valign="bottom">AP</th>
<th valign="bottom">mIoU</th>
<th valign="bottom">model id</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: maskformer2_R50_bs16_160k -->
 <tr><td align="left"><a href="configs/ade20k/panoptic-segmentation/maskformer2_R50_bs16_160k.yaml">Mask2Former</a></td>
<td align="center">R50</td>
<td align="center">160k</td>
<td align="center">39.7</td>
<td align="center">26.5</td>
<td align="center">46.1</td>
<td align="center">48243028_0</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/maskformer/mask2former/ade20k/panoptic/maskformer2_R50_bs16_160k/model_final_5c90d4.pkl">model</a></td>
</tr>
<!-- ROW: maskformer2_swin_large_IN21k_384_bs16_160k -->
 <tr><td align="left"><a href="configs/ade20k/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k.yaml">Mask2Former (200 queries)</a></td>
<td align="center">Swin-L (IN21k)</td>
<td align="center">160k</td>
<td align="center">48.1</td>
<td align="center">34.2</td>
<td align="center">54.5</td>
<td align="center">48267279</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/maskformer/mask2former/ade20k/panoptic/maskformer2_swin_large_IN21k_384_bs16_160k/model_final_e0c58e.pkl">model</a></td>
</tr>
</tbody></table>


### Instance Segmentation

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">iterations</th>
<th valign="bottom">AP</th>
<th valign="bottom">model id</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: maskformer2_R50_bs16_160k -->
 <tr><td align="left"><a href="configs/ade20k/instance-segmentation/maskformer2_R50_bs16_160k.yaml">Mask2Former</a></td>
<td align="center">R50</td>
<td align="center">160k</td>
<td align="center">26.4</td>
<td align="center">47429167_7</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/maskformer/mask2former/ade20k/instance/maskformer2_R50_bs16_160k/model_final_67e945.pkl">model</a></td>
</tr>
<!-- ROW: maskformer2_swin_large_IN21k_384_bs16_160k -->
 <tr><td align="left"><a href="configs/ade20k/instance-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k.yaml">Mask2Former (200 queries)</a></td>
<td align="center">R50</td>
<td align="center">160k</td>
<td align="center">34.9</td>
<td align="center">49040271_0</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/maskformer/mask2former/ade20k/instance/maskformer2_swin_large_IN21k_384_bs16_160k/model_final_92dae9.pkl">model</a></td>
</tr>
</tbody></table>


### Semantic Segmentation

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">iterations</th>
<th valign="bottom">mIoU</th>
<th valign="bottom">mIoU (ms+flip)</th>
<th valign="bottom">model id</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: maskformer2_R50_bs16_160k -->
 <tr><td align="left"><a href="configs/ade20k/semantic-segmentation/maskformer2_R50_bs16_160k.yaml">Mask2Former</a></td>
<td align="center">R50</td>
<td align="center">160k</td>
<td align="center">47.2</td>
<td align="center">49.2</td>
<td align="center">47429167_5</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/maskformer/mask2former/ade20k/semantic/maskformer2_R50_bs16_160k/model_final_500878.pkl">model</a></td>
</tr>
<!-- ROW: maskformer2_R101_bs16_90k -->
 <tr><td align="left"><a href="configs/ade20k/semantic-segmentation/maskformer2_R101_bs16_90k.yaml">Mask2Former</a></td>
<td align="center">R101</td>
<td align="center">160k</td>
<td align="center">47.8</td>
<td align="center">50.1</td>
<td align="center">48243040_0</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/maskformer/mask2former/ade20k/semantic/maskformer2_R101_bs16_90k/model_final_0d68be.pkl">model</a></td>
</tr>
<!-- ROW: maskformer2_swin_tiny_bs16_160k -->
 <tr><td align="left"><a href="configs/ade20k/semantic-segmentation/swin/maskformer2_swin_tiny_bs16_160k.yaml">Mask2Former</a></td>
<td align="center">Swin-T</td>
<td align="center">160k</td>
<td align="center">47.7</td>
<td align="center">49.6</td>
<td align="center">48333144_5</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/maskformer/mask2former/ade20k/semantic/maskformer2_swin_tiny_bs16_160k/model_final_5274a6.pkl">model</a></td>
</tr>
<!-- ROW: maskformer2_swin_small_bs16_160k -->
 <tr><td align="left"><a href="configs/ade20k/semantic-segmentation/swin/maskformer2_swin_small_bs16_160k.yaml">Mask2Former</a></td>
<td align="center">Swin-S</td>
<td align="center">160k</td>
<td align="center">51.3</td>
<td align="center">52.4</td>
<td align="center">48333149_5</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/maskformer/mask2former/ade20k/semantic/maskformer2_swin_small_bs16_160k/model_final_011c6d.pkl">model</a></td>
</tr>
<!-- ROW: maskformer2_swin_base_384_bs16_160k_res640 -->
 <tr><td align="left"><a href="configs/ade20k/semantic-segmentation/swin/maskformer2_swin_base_384_bs16_160k_res640.yaml">Mask2Former</a></td>
<td align="center">Swin-B</td>
<td align="center">160k</td>
<td align="center">52.4</td>
<td align="center">53.7</td>
<td align="center">48333153_5</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/maskformer/mask2former/ade20k/semantic/maskformer2_swin_base_384_bs16_160k_res640/model_final_503e96.pkl">model</a></td>
</tr>
<!-- ROW: maskformer2_swin_base_IN21k_384_bs16_160k_res640 -->
 <tr><td align="left"><a href="configs/ade20k/semantic-segmentation/swin/maskformer2_swin_base_IN21k_384_bs16_160k_res640.yaml">Mask2Former</a></td>
<td align="center">Swin-B (IN21k)</td>
<td align="center">160k</td>
<td align="center">53.9</td>
<td align="center">55.1</td>
<td align="center">48333157_5</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/maskformer/mask2former/ade20k/semantic/maskformer2_swin_base_IN21k_384_bs16_160k_res640/model_final_7e47bf.pkl">model</a></td>
</tr>
<!-- ROW: maskformer2_swin_large_IN21k_384_bs16_160k_res640 -->
 <tr><td align="left"><a href="configs/ade20k/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k_res640.yaml">Mask2Former</a></td>
<td align="center">Swin-L (IN21k)</td>
<td align="center">160k</td>
<td align="center">56.1</td>
<td align="center">57.3</td>
<td align="center">48004474_0</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/maskformer/mask2former/ade20k/semantic/maskformer2_swin_large_IN21k_384_bs16_160k_res640/model_final_6b4a3a.pkl">model</a></td>
</tr>
</tbody></table>


## Mapillary Vistas Model Zoo

### Panoptic Segmentation

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">iterations</th>
<th valign="bottom">PQ</th>
<th valign="bottom">mIoU</th>
<th valign="bottom">model id</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: maskformer_R50_bs16_300k -->
 <tr><td align="left"><a href="configs/mapillary-vistas/panoptic-segmentation/maskformer_R50_bs16_300k.yaml">Mask2Former</a></td>
<td align="center">R50</td>
<td align="center">300k</td>
<td align="center">36.3</td>
<td align="center">50.7</td>
<td align="center">49392417_0</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/maskformer/mask2former/mapillary_vistas/panoptic/maskformer_R50_bs16_300k/model_final_4e9874.pkl">model</a></td>
</tr>
<!-- ROW: maskformer2_swin_large_IN21k_384_bs16_300k -->
 <tr><td align="left"><a href="configs/mapillary-vistas/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_300k.yaml">Mask2Former (200 queries)</a></td>
<td align="center">Swin-L (IN21k)</td>
<td align="center">300k</td>
<td align="center">45.5</td>
<td align="center">60.8</td>
<td align="center">48267065_4</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/maskformer/mask2former/mapillary_vistas/panoptic/maskformer2_swin_large_IN21k_384_bs16_300k/model_final_132c71.pkl">model</a></td>
</tr>
</tbody></table>


### Semantic Segmentation

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">iterations</th>
<th valign="bottom">mIoU</th>
<th valign="bottom">mIoU (ms+flip)</th>
<th valign="bottom">model id</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: maskformer_R50_bs16_300k -->
 <tr><td align="left"><a href="configs/mapillary-vistas/semantic-segmentation/maskformer_R50_bs16_300k.yaml">Mask2Former</a></td>
<td align="center">R50</td>
<td align="center">300k</td>
<td align="center">57.4</td>
<td align="center">59.0</td>
<td align="center">49189528_1</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/maskformer/mask2former/mapillary_vistas/semantic/maskformer_R50_bs16_300k/model_final_6c66d0.pkl">model</a></td>
</tr>
<!-- ROW: maskformer2_swin_large_IN21k_384_bs16_300k -->
 <tr><td align="left"><a href="configs/mapillary-vistas/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_300k.yaml">Mask2Former</a></td>
<td align="center">Swin-L (IN21k)</td>
<td align="center">300k</td>
<td align="center">63.2</td>
<td align="center">64.7</td>
<td align="center">49189528_0</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/maskformer/mask2former/mapillary_vistas/semantic/maskformer2_swin_large_IN21k_384_bs16_300k/model_final_90ee2d.pkl">model</a></td>
</tr>
</tbody></table>


## Video Instance Segmentation
### YouTubeVIS 2019

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">iterations</th>
<th valign="bottom">AP</th>
<th valign="bottom">model id</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: maskformer2_R50_bs16_50ep -->
 <tr><td align="left"><a href="configs/youtubevis_2019/video_maskformer2_R50_bs16_8ep.yaml">Mask2Former</a></td>
<td align="center">R50</td>
<td align="center">6k</td>
<td align="center">46.4</td>
<td align="center">51130652_3</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/maskformer/video_mask2former/ytvis_2019/video_maskformer2_R50_bs16_8ep/model_final_34112b.pkl">model</a></td>
</tr>
<!-- ROW: maskformer2_R101_bs16_50ep -->
 <tr><td align="left"><a href="configs/youtubevis_2019/video_maskformer2_R101_bs16_8ep.yaml">Mask2Former</a></td>
<td align="center">R101</td>
<td align="center">6k</td>
<td align="center">49.2</td>
<td align="center">50897581_1</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/maskformer/video_mask2former/ytvis_2019/video_maskformer2_R101_bs16_8ep/model_final_a34dca.pkl">model</a></td>
</tr>
<!-- ROW: maskformer2_swin_tiny_bs16_50ep -->
 <tr><td align="left"><a href="configs/youtubevis_2019/swin/video_maskformer2_swin_tiny_bs16_8ep.yaml">Mask2Former</a></td>
<td align="center">Swin-T</td>
<td align="center">6k</td>
<td align="center">51.5</td>
<td align="center">50897611_3</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/maskformer/video_mask2former/ytvis_2019/video_maskformer2_swin_tiny_bs16_8ep/model_final_26fffe.pkl">model</a></td>
</tr>
<!-- ROW: maskformer2_swin_small_bs16_50ep -->
 <tr><td align="left"><a href="configs/youtubevis_2019/swin/video_maskformer2_swin_small_bs16_8ep.yaml">Mask2Former</a></td>
<td align="center">Swin-S</td>
<td align="center">6k</td>
<td align="center">54.3</td>
<td align="center">50897661_2</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/maskformer/video_mask2former/ytvis_2019/video_maskformer2_swin_small_bs16_8ep/model_final_4a5174.pkl">model</a></td>
</tr>
<!-- ROW: maskformer2_swin_base_IN21k_384_bs16_50ep -->
 <tr><td align="left"><a href="configs/youtubevis_2019/swin/video_maskformer2_swin_base_IN21k_384_bs16_8ep.yaml">Mask2Former</a></td>
<td align="center">Swin-B (IN21k)</td>
<td align="center">6k</td>
<td align="center">59.5</td>
<td align="center">50897733_2</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/maskformer/video_mask2former/ytvis_2019/video_maskformer2_swin_base_IN21k_384_bs16_8ep/model_final_221a8a.pkl">model</a></td>
</tr>
<!-- ROW: maskformer2_swin_large_IN21k_384_bs16_100ep -->
 <tr><td align="left"><a href="configs/youtubevis_2019/swin/video_maskformer2_swin_large_IN21k_384_bs16_8ep.yaml">Mask2Former (200 queries)</a></td>
<td align="center">Swin-L (IN21k)</td>
<td align="center">6k</td>
<td align="center">60.4</td>
<td align="center">50908813_0</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/maskformer/video_mask2former/ytvis_2019/video_maskformer2_swin_large_IN21k_384_bs16_8ep/model_final_c5c739.pkl">model</a></td>
</tr>
</tbody></table>


### YouTubeVIS 2021

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">iterations</th>
<th valign="bottom">AP</th>
<th valign="bottom">model id</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: maskformer2_R50_bs16_50ep -->
 <tr><td align="left"><a href="configs/youtubevis_2021/video_maskformer2_R50_bs16_8ep.yaml">Mask2Former</a></td>
<td align="center">R50</td>
<td align="center">8k</td>
<td align="center">40.6</td>
<td align="center">51130652_7</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/maskformer/video_mask2former/ytvis_2021/video_maskformer2_R50_bs16_8ep/model_final_b8aae2.pkl">model</a></td>
</tr>
<!-- ROW: maskformer2_R101_bs16_50ep -->
 <tr><td align="left"><a href="configs/youtubevis_2021/video_maskformer2_R101_bs16_8ep.yaml">Mask2Former</a></td>
<td align="center">R101</td>
<td align="center">8k</td>
<td align="center">42.4</td>
<td align="center">50897581_8</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/maskformer/video_mask2former/ytvis_2021/video_maskformer2_R101_bs16_8ep/model_final_6efd7a.pkl">model</a></td>
</tr>
<!-- ROW: maskformer2_swin_tiny_bs16_50ep -->
 <tr><td align="left"><a href="configs/youtubevis_2021/swin/video_maskformer2_swin_tiny_bs16_8ep.yaml">Mask2Former</a></td>
<td align="center">Swin-T</td>
<td align="center">8k</td>
<td align="center">45.9</td>
<td align="center">50897611_7</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/maskformer/video_mask2former/ytvis_2021/video_maskformer2_swin_tiny_bs16_8ep/model_final_965185.pkl">model</a></td>
</tr>
<!-- ROW: maskformer2_swin_small_bs16_50ep -->
 <tr><td align="left"><a href="configs/youtubevis_2021/swin/video_maskformer2_swin_small_bs16_8ep.yaml">Mask2Former</a></td>
<td align="center">Swin-S</td>
<td align="center">8k</td>
<td align="center">48.6</td>
<td align="center">50897661_7</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/maskformer/video_mask2former/ytvis_2021/video_maskformer2_swin_small_bs16_8ep/model_final_0dec91.pkl">model</a></td>
</tr>
<!-- ROW: maskformer2_swin_base_IN21k_384_bs16_50ep -->
 <tr><td align="left"><a href="configs/youtubevis_2021/swin/video_maskformer2_swin_base_IN21k_384_bs16_8ep.yaml">Mask2Former</a></td>
<td align="center">Swin-B (IN21k)</td>
<td align="center">8k</td>
<td align="center">52.0</td>
<td align="center">50897733_9</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/maskformer/video_mask2former/ytvis_2021/video_maskformer2_swin_base_IN21k_384_bs16_8ep/model_final_a9b925.pkl">model</a></td>
</tr>
<!-- ROW: maskformer2_swin_large_IN21k_384_bs16_100ep -->
 <tr><td align="left"><a href="configs/youtubevis_2021/swin/video_maskformer2_swin_large_IN21k_384_bs16_8ep.yaml">Mask2Former (200 queries)</a></td>
<td align="center">Swin-L (IN21k)</td>
<td align="center">8k</td>
<td align="center">52.6</td>
<td align="center">50908813_6</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/maskformer/video_mask2former/ytvis_2021/video_maskformer2_swin_large_IN21k_384_bs16_8ep/model_final_4da256.pkl">model</a></td>
</tr>
</tbody></table>
