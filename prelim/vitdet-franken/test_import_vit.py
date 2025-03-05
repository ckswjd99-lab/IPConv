import pickle

import torch
import torch.nn as nn
from functools import partial

import cv2

from mini_det.modeling.backbone.vit import ViT
from mini_det.modeling.backbone.vit import SimpleFeaturePyramid
from mini_det.modeling.backbone.fpn import LastLevelMaxPool, ShapeSpec
from mini_det.modeling.meta_arch import GeneralizedRCNN

from mini_det.layers import ShapeSpec
from mini_det.modeling.meta_arch import GeneralizedRCNN
from mini_det.modeling.anchor_generator import DefaultAnchorGenerator
from mini_det.modeling.backbone.fpn import LastLevelMaxPool
from mini_det.modeling.box_regression import Box2BoxTransform
from mini_det.modeling.matcher import Matcher
from mini_det.modeling.poolers import ROIPooler
from mini_det.modeling.proposal_generator import RPN, StandardRPNHead
from mini_det.modeling.roi_heads import (
    StandardROIHeads,
    FastRCNNOutputLayers,
    MaskRCNNConvUpsampleHead,
    FastRCNNConvFCHead,
)

constants = dict(
    imagenet_rgb256_mean=[123.675, 116.28, 103.53],
    imagenet_rgb256_std=[58.395, 57.12, 57.375],
)

embed_dim, depth, num_heads, dp = 768, 12, 12, 0.1
num_classes = 80

# create model
backbone = SimpleFeaturePyramid(
    net = ViT(
        img_size=1024,
        patch_size=16,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        drop_path_rate=dp,
        window_size=14,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        window_block_indexes=[
            # 2, 5, 8 11 for global attention
            0,
            1,
            3,
            4,
            6,
            7,
            9,
            10,
        ],
        residual_block_indexes=[],
        use_rel_pos=True,
        out_feature="last_feat",
    ),
    in_feature="last_feat",
    out_channels=256,
    scale_factors=(4.0, 2.0, 1.0, 0.5),
    top_block=LastLevelMaxPool(),
    norm="LN",
    square_pad=1024,
)

model = GeneralizedRCNN(
    backbone=backbone,
    proposal_generator = RPN(
        in_features=["p2", "p3", "p4", "p5", "p6"],
        head=StandardRPNHead(in_channels=256, num_anchors=3, conv_dims=[-1, -1]),
        anchor_generator=DefaultAnchorGenerator(
            sizes=[[32], [64], [128], [256], [512]],
            aspect_ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64],
            offset=0.0,
        ),
        anchor_matcher=Matcher(
            thresholds=[0.3, 0.7], labels=[0, -1, 1], allow_low_quality_matches=True
        ),
        box2box_transform=Box2BoxTransform(weights=[1.0, 1.0, 1.0, 1.0]),
        batch_size_per_image=256,
        positive_fraction=0.5,
        pre_nms_topk=(2000, 1000),
        post_nms_topk=(1000, 1000),
        nms_thresh=0.7,
    ),
    roi_heads=StandardROIHeads(
        num_classes=num_classes,
        batch_size_per_image=512,
        positive_fraction=0.25,
        proposal_matcher=Matcher(
            thresholds=[0.5], labels=[0, 1], allow_low_quality_matches=False
        ),
        box_in_features=["p2", "p3", "p4", "p5"],
        box_pooler=ROIPooler(
            output_size=7,
            scales=(1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32),
            sampling_ratio=0,
            pooler_type="ROIAlignV2",
        ),
        box_head=FastRCNNConvFCHead(
            input_shape=ShapeSpec(channels=256, height=7, width=7),
            conv_dims=[256, 256, 256, 256],
            fc_dims=[1024],
            conv_norm="LN"
        ),
        box_predictor=FastRCNNOutputLayers(
            input_shape=ShapeSpec(channels=1024),
            test_score_thresh=0.05,
            box2box_transform=Box2BoxTransform(weights=(10, 10, 5, 5)),
            num_classes=num_classes,
        ),
        mask_in_features=["p2", "p3", "p4", "p5"],
        mask_pooler=ROIPooler(
            output_size=14,
            scales=(1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32),
            sampling_ratio=0,
            pooler_type="ROIAlignV2",
        ),
        mask_head=MaskRCNNConvUpsampleHead(
            input_shape=ShapeSpec(channels=256, width=14, height=14),
            num_classes=num_classes,
            conv_dims=[256, 256, 256, 256, 256],
            conv_norm="LN",
        ),
    ),
    pixel_mean=constants["imagenet_rgb256_mean"],
    pixel_std=constants["imagenet_rgb256_std"],
    input_format="RGB",
)

# load weight
weight_pkl = './model_final_61ccd1.pkl'
with open(weight_pkl, 'rb') as f:
    weights = pickle.load(f)['model']

for name, param in model.named_parameters():
    if name in weights:
        param.data.copy_(torch.tensor(weights[name]))
    else:
        print(f"Parameter {name} not found in weights")

# load image
image_path = './000000000139.jpg'
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = torch.tensor(img, dtype=torch.uint8).permute(2, 0, 1)

# inference
model.eval()
temp_input = [{"image": img, "file_name": image_path, "height": img.shape[1], "width": img.shape[2]}]
output = model(temp_input)
print(output)
