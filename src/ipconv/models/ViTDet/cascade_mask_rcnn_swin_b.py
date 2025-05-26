import torch
import torch.nn as nn
import numpy as np
from .structures.image_list import ImageList

from .modeling.backbone import (SwinTransformer, FPN)
from .modeling.backbone.fpn import LastLevelMaxPool, ShapeSpec
from .modeling.backbone.utils import get_abs_pos, window_partition, window_unpartition, add_decomposed_rel_pos
from .modeling.meta_arch import GeneralizedRCNN
from .modeling.proposal_generator import RPN, StandardRPNHead
from .modeling.anchor_generator import DefaultAnchorGenerator
from .modeling.matcher import Matcher
from .modeling.poolers import ROIPooler
from .modeling.box_regression import Box2BoxTransform
from .modeling.roi_heads import (
    StandardROIHeads,
    FastRCNNOutputLayers,
    MaskRCNNConvUpsampleHead,
    FastRCNNConvFCHead,
    CascadeROIHeads
)

import pickle
import torch.nn.functional as F

from typing import Dict

def make_cascade_mask_rcnn_swin_b():
    num_classes = 80

    swin_b = SwinTransformer(
        depths=[2, 2, 18, 2],
        drop_path_rate=0.4,
        embed_dim=128,
        num_heads=[4, 8, 16, 32],
    )

    model = GeneralizedRCNN(
        backbone=FPN(
            bottom_up=swin_b,
            in_features=("p0", "p1", "p2", "p3"),
            out_channels=256,
            top_block=LastLevelMaxPool(),
            square_pad=1024,
            norm="LN"
        ),
        proposal_generator=RPN(
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
        roi_heads=CascadeROIHeads(
            num_classes=80,
            batch_size_per_image=512,
            positive_fraction=0.25,
            box_in_features=["p2", "p3", "p4", "p5"],
            box_pooler=ROIPooler(
                output_size=7,
                scales=(1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32),
                sampling_ratio=0,
                pooler_type="ROIAlignV2",
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
            box_heads=[
                FastRCNNConvFCHead(
                    input_shape=ShapeSpec(channels=256, height=7, width=7),
                    conv_dims=[256, 256, 256, 256],
                    fc_dims=[1024],
                    conv_norm="LN",
                )
                for _ in range(3)
            ],
            box_predictors=[
                FastRCNNOutputLayers(
                    input_shape=ShapeSpec(channels=1024),
                    test_score_thresh=0.05,
                    box2box_transform=Box2BoxTransform(weights=(w1, w1, w2, w2)),
                    cls_agnostic_bbox_reg=True,
                    num_classes=num_classes,
                )
                for (w1, w2) in [(10, 5), (20, 10), (30, 15)]
            ],
            proposal_matchers=[
                Matcher(thresholds=[th], labels=[0, 1], allow_low_quality_matches=False)
                for th in [0.5, 0.6, 0.7]
            ],

        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
        input_format="BGR",
    )

    return model

class CascadeMaskRCNN_Swin_B_Contexted(nn.Module):
    def __init__(self, num_classes=80, device="cuda"):
        super(CascadeMaskRCNN_Swin_B_Contexted, self).__init__()
        self.base_model = make_cascade_mask_rcnn_swin_b().to(device)
        self.num_classes = num_classes
        self.device = device

        self.COCO_LABELS_LIST = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
            'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

    def load_weight(self, weight_pkl_path='./model_final_246a82.pkl'):
        with open(weight_pkl_path, 'rb') as f:
            weights = pickle.load(f)['model']

        for name, param in self.base_model.named_parameters():
            if name in weights:
                param.data.copy_(torch.tensor(weights[name]))
            else:
                print(f"Parameter {name} not found in weights")

    def forward(self, image_ndarray: np.ndarray):
        image_tensor = torch.tensor(image_ndarray, dtype=torch.uint8, device=self.device).permute(2, 0, 1)
        input = [{"image": image_tensor, "height": image_tensor.shape[-2], "width": image_tensor.shape[-1]}]
        
        detections = self.base_model(input)

        predictions = detections[0]
        boxes = predictions["instances"].pred_boxes.tensor.cpu().numpy()
        labels = predictions["instances"].pred_classes.cpu().numpy()
        scores = predictions["instances"].scores.cpu().numpy()

        return boxes, labels, scores

    def forward_analyzed(self, image_ndarray: np.ndarray):
        return self.forward(image_ndarray)
    
    def forward_contexted(
        self, 
        image_ndarray: np.ndarray,
        anchor_features: Dict[str, torch.Tensor] = {},
        dirtiness_map: torch.Tensor = torch.ones(1, 64, 64, 1, device="cuda"),
        only_backbone: bool = False,
    ):
        new_cache_feature = {}

        # convert to tensor
        image_tensor = torch.tensor(image_ndarray, dtype=torch.uint8).permute(2, 0, 1).to(self.device).half()
        input = [{"image": image_tensor, "height": image_tensor.shape[-2], "width": image_tensor.shape[-1]}]
        
        # preprocess
        images = [self.base_model._move_to_current_device(x["image"]) for x in input]
        images = [(x - self.base_model.pixel_mean) / self.base_model.pixel_std for x in images]
        images = ImageList.from_tensors(
            images,
            self.base_model.backbone.size_divisibility,
            padding_constraints=self.base_model.backbone.padding_constraints,
        )

        # inference: backbone
        backbone = self.base_model.backbone
        net = backbone.bottom_up

        # Swin forward
        x = net.patch_embed(images.tensor)  # B, C, H, W -> B, embed_dim, H/4, W/4
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # B, H, W, C
        x = x.reshape(B, H * W, C)  # Convert to BLC format

        if net.ape:
            x = x + net.absolute_pos_embed
        x = net.pos_drop(x)

        # Store intermediate features for FPN
        features = {}
        stage_idx = 0

        # Process through Swin layers
        for i, layer in enumerate(net.layers):
            # Calculate attention mask for SW-MSA
            window_size = layer.window_size
            shift_size = layer.shift_size
            H_pad = H + (window_size - H % window_size) % window_size
            W_pad = W + (window_size - W % window_size) % window_size
            img_mask = torch.zeros((1, H_pad, W_pad, 1), device=x.device)
            h_slices = (slice(0, -window_size),
                       slice(-window_size, -shift_size),
                       slice(-shift_size, None))
            w_slices = (slice(0, -window_size),
                       slice(-window_size, -shift_size),
                       slice(-shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows, pad_hw = window_partition(img_mask, window_size)
            mask_windows = mask_windows.view(-1, window_size * window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

            # Process blocks in the layer
            for block in layer.blocks:
                block.H, block.W = H, W
                #x = block(x, attn_mask)  # x is in BLC format
                
                B, L, C = x.shape
                H, W = block.H, block.W
                assert L == H * W, "input feature has wrong size"
                shortcut = x
                x = block.norm1(x)
                x = x.view(B, H, W, C)

                # pad feature maps to multiples of window size
                pad_l = pad_t = 0
                pad_r = (block.window_size - W % block.window_size) % block.window_size
                pad_b = (block.window_size - H % block.window_size) % block.window_size
                x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
                _, Hp, Wp, _ = x.shape

                # cyclic shift
                if block.shift_size > 0:
                    shifted_x = torch.roll(x, shifts=(-block.shift_size, -block.shift_size), dims=(1, 2))
                    attn_mask = attn_mask
                else:
                    shifted_x = x
                    attn_mask = None

                # partition windows
                x_windows, pad_hw = window_partition(shifted_x, block.window_size)  # nW*B, window_size, window_size, C
                x_windows = x_windows.view(-1, block.window_size * block.window_size, C)  # nW*B, window_size*window_size, C

                # W-MSA/SW-MSA
                attn_windows = block.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

                # merge windows
                attn_windows = attn_windows.view(-1, block.window_size, block.window_size, C)
                shifted_x = window_unpartition(attn_windows, block.window_size, pad_hw, (H, W))  # B H' W' C

                # reverse cyclic shift
                if block.shift_size > 0:
                    x = torch.roll(shifted_x, shifts=(block.shift_size, block.shift_size), dims=(1, 2))
                else:
                    x = shifted_x

                if pad_r > 0 or pad_b > 0:
                    x = x[:, :H, :W, :].contiguous()

                x = x.view(B, H * W, C)

                # FFN
                x = shortcut + block.drop_path(x)
                x = x + block.drop_path(block.mlp(block.norm2(x)))

            # Apply stage normalization and store features
            if hasattr(net, f'norm{stage_idx}'):
                norm = getattr(net, f'norm{stage_idx}')
                x_out = norm(x)  # x_out is in BLC format
                
                # Convert to BCHW format for FPN
                expected_channels = self.base_model.backbone.bottom_up.num_features[stage_idx]
                out = x_out.view(B, H, W, expected_channels).permute(0, 3, 1, 2).contiguous()
                features[f'p{stage_idx}'] = out
                stage_idx += 1

            # Apply downsample if exists
            if layer.downsample is not None:
                # Keep in BLC format for downsample
                x = layer.downsample(x, H, W)
                H, W = H // 2, W // 2
                C = C * 2

        # FPN forward
        results = []
        feature_map_to_stage = {'p0': '2', 'p1': '3', 'p2': '4', 'p3': '5'}  # Map feature names to stage numbers
        
        for f in backbone.in_features:
            if f in features:
                x = features[f]  # Already in BCHW format
                stage_num = feature_map_to_stage[f]
                
                # Apply lateral connection (1x1 conv)
                lateral = getattr(backbone, f'fpn_lateral{stage_num}')(x)
                
                # Apply output conv (3x3 conv)
                out = getattr(backbone, f'fpn_output{stage_num}')(lateral)
                results.append(out)

        # Apply top block if exists
        if backbone.top_block is not None:
            if backbone.top_block.in_feature in features:
                top_block_in_feature = features[backbone.top_block.in_feature]
            else:
                top_block_in_feature = results[backbone._out_features.index(backbone.top_block.in_feature)]
            results.extend(backbone.top_block(top_block_in_feature))
            
        features = {f: res for f, res in zip(backbone._out_features, results)}

        # Process predictions
        predictions = self.base_model([{"image": image_tensor, "height": image_tensor.shape[-2], "width": image_tensor.shape[-1]}])
        boxes = predictions[0]["instances"].pred_boxes.tensor.cpu().numpy()
        labels = predictions[0]["instances"].pred_classes.cpu().numpy()
        scores = predictions[0]["instances"].scores.cpu().numpy()

        return (boxes, labels, scores), new_cache_feature