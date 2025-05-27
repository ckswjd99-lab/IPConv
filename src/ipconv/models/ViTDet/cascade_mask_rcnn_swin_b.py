import torch
import torch.nn as nn
import numpy as np
from .structures.image_list import ImageList

from .modeling.backbone import (SwinTransformer, FPN)
from .modeling.backbone.fpn import LastLevelMaxPool, ShapeSpec
from .modeling.backbone.utils import get_abs_pos, window_partition, window_unpartition, add_decomposed_rel_pos, window_reverse
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
import torch.utils.checkpoint as checkpoint

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
        dirtiness_map: torch.Tensor = torch.ones(1, 256, 256, 1, device="cuda"),
        only_backbone: bool = False,
    ):
        new_cache_feature = {}

        # convert to tensor
        image_tensor = torch.tensor(image_ndarray, dtype=torch.uint8).permute(2, 0, 1).to(self.device).half()
        batched_inputs = [{"image": image_tensor, "height": image_tensor.shape[-2], "width": image_tensor.shape[-1]}]

        # Preprocess image
        base_model = self.base_model
        images = base_model.preprocess_image(batched_inputs)
        
        # Backbone
        backbone = base_model.backbone
        swin_model = backbone.bottom_up
        x = images.tensor
        
        x = swin_model.patch_embed(x)

        Wh, Ww = x.size(2), x.size(3)

        ## projection without APE
        x = x.flatten(2).transpose(1, 2)
        x = swin_model.pos_drop(x)

        outs = {}
        for i in range(swin_model.num_layers):
            # Swin Transformer Layer
            layer = swin_model.layers[i]
            LH, LW = Wh, Ww
            
            ## downsample the dirtiness map to the current layer's feature map size
            dmap_layer = F.interpolate(dirtiness_map, size=(LH, LW), mode="area")
            dmap_layer = (dmap_layer > 0).float()

            Hp = int(np.ceil(LH / layer.window_size)) * layer.window_size
            Wp = int(np.ceil(LW / layer.window_size)) * layer.window_size
            img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
            h_slices = (
                slice(0, -layer.window_size),
                slice(-layer.window_size, -layer.shift_size),
                slice(-layer.shift_size, None),
            )
            w_slices = (
                slice(0, -layer.window_size),
                slice(-layer.window_size, -layer.shift_size),
                slice(-layer.shift_size, None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows, _ = window_partition(
                img_mask, layer.window_size
            )  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, layer.window_size * layer.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
                attn_mask == 0, float(0.0)
            )

            for bidx, block in enumerate(layer.blocks):
                # Swin Transformer Block
                block.H, block.W = LH, LW
                Block_B, Block_L, Block_C = x.shape
                Block_H, Block_W = block.H, block.W

                shortcut = x
                x = block.norm1(x)
                x = x.view(Block_B, Block_H, Block_W, Block_C)

                # pad feature maps to multiples of window size
                pad_l = pad_t = 0
                pad_r = (block.window_size - Block_W % block.window_size) % block.window_size
                pad_b = (block.window_size - Block_H % block.window_size) % block.window_size
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
                x_windows, _ = window_partition(
                    shifted_x, block.window_size
                )  # nW*B, window_size, window_size, C
                x_windows = x_windows.view(
                    -1, block.window_size * block.window_size, Block_C
                )  # nW*B, window_size*window_size, C

                # W-MSA/SW-MSA
                ATTN_B_, ATTN_N, ATTN_C = x_windows.shape
                qkv = (
                    block.attn.qkv(x_windows)
                    .reshape(ATTN_B_, ATTN_N, 3, block.attn.num_heads, ATTN_C // block.attn.num_heads)
                    .permute(2, 0, 3, 1, 4)
                )
                q, k, v = qkv[0], qkv[1], qkv[2]

                q = q * block.attn.scale
                attn = q @ k.transpose(-2, -1)

                relative_position_bias = block.attn.relative_position_bias_table[
                    block.attn.relative_position_index.view(-1)
                ].view(
                    block.attn.window_size[0] * block.attn.window_size[1], block.attn.window_size[0] * block.attn.window_size[1], -1
                )  # Wh*Ww,Wh*Ww,nH
                relative_position_bias = relative_position_bias.permute(
                    2, 0, 1
                ).contiguous()  # nH, Wh*Ww, Wh*Ww
                attn = attn + relative_position_bias.unsqueeze(0)

                if attn_mask is not None:
                    nW = attn_mask.shape[0]
                    attn = attn.view(ATTN_B_ // nW, nW, block.attn.num_heads, ATTN_N, ATTN_N) + attn_mask.unsqueeze(1).unsqueeze(0)
                    attn = attn.view(-1, block.attn.num_heads, ATTN_N, ATTN_N)
                    attn = block.attn.softmax(attn)
                else:
                    attn = block.attn.softmax(attn)

                attn = block.attn.attn_drop(attn)

                x_windows = (attn @ v).transpose(1, 2).reshape(ATTN_B_, ATTN_N, ATTN_C)
                x_windows = block.attn.proj(x_windows)
                x_windows = block.attn.proj_drop(x_windows)

                attn_windows = x_windows

                # merge windows
                attn_windows = attn_windows.view(-1, block.window_size, block.window_size, Block_C)
                shifted_x = window_reverse(attn_windows, block.window_size, Hp, Wp)  # B H' W' C

                # reverse cyclic shift
                if block.shift_size > 0:
                    x = torch.roll(shifted_x, shifts=(block.shift_size, block.shift_size), dims=(1, 2))
                else:
                    x = shifted_x

                if pad_r > 0 or pad_b > 0:
                    x = x[:, :Block_H, :Block_W, :].contiguous()

                x = x.view(Block_B, Block_H * Block_W, Block_C)

                # FFN
                x = shortcut + block.drop_path(x)
                x = x + block.drop_path(block.mlp(block.norm2(x)))
            
            if layer.downsample is not None:
                x_down = layer.downsample(x, LH, LW)
                Wh, Ww = (LH + 1) // 2, (LW + 1) // 2
                x_out, H, W, x, Wh, Ww = x, LH, LW, x_down, Wh, Ww
            else:
                x_out, H, W, x, Wh, Ww = x, LH, LW, x, LH, LW

            if i in swin_model.out_indices:
                norm_layer = getattr(swin_model, f"norm{i}")
                x_out = norm_layer(x_out)

                out = x_out.view(-1, H, W, swin_model.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs["p{}".format(i)] = out

        bottom_up_features = outs

        # FPN
        results = []
        prev_features = backbone.lateral_convs[0](bottom_up_features[backbone.in_features[-1]])
        results.append(backbone.output_convs[0](prev_features))

        ## reverse feature maps into top-down order (from low to high resolution)
        for idx, (lateral_conv, output_conv) in enumerate(
            zip(backbone.lateral_convs, backbone.output_convs)
        ):
            ## Slicing of ModuleList is not supported https://github.com/pytorch/pytorch/issues/47336
            ## Therefore we loop over all modules but skip the first one
            if idx > 0:
                features = backbone.in_features[-idx - 1]
                features = bottom_up_features[features]
                top_down_features = F.interpolate(prev_features, scale_factor=2.0, mode="nearest")
                lateral_features = lateral_conv(features)
                prev_features = lateral_features + top_down_features
                if backbone._fuse_type == "avg":
                    prev_features /= 2
                results.insert(0, output_conv(prev_features))

        if backbone.top_block is not None:
            if backbone.top_block.in_feature in bottom_up_features:
                top_block_in_feature = bottom_up_features[backbone.top_block.in_feature]
            else:
                top_block_in_feature = results[backbone._out_features.index(backbone.top_block.in_feature)]
            results.extend(backbone.top_block(top_block_in_feature))
        
        features = {f: res for f, res in zip(backbone._out_features, results)}
        
        # Post-process features
        proposals, _ = base_model.proposal_generator(images, features, None)
        results, _ = base_model.roi_heads(images, features, proposals, None)

        predictions = GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)

        # Process predictions
        boxes = predictions[0]["instances"].pred_boxes.tensor.cpu().numpy()
        labels = predictions[0]["instances"].pred_classes.cpu().numpy()
        scores = predictions[0]["instances"].scores.cpu().numpy()

        boxes[:, :] -= 128

        return (boxes, labels, scores), new_cache_feature