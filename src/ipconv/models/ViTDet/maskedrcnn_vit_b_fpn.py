import pickle

import torch
import torch.nn as nn
from functools import partial

import numpy as np
import os
import json
import cv2
from tqdm import tqdm

from .modeling.backbone.vit import ViT
from .modeling.backbone.vit import SimpleFeaturePyramid
from .modeling.backbone.fpn import LastLevelMaxPool, ShapeSpec
from .modeling.meta_arch import GeneralizedRCNN

from .layers import ShapeSpec
from .modeling.meta_arch import GeneralizedRCNN
from .modeling.anchor_generator import DefaultAnchorGenerator
from .modeling.backbone.fpn import LastLevelMaxPool
from .modeling.box_regression import Box2BoxTransform
from .modeling.matcher import Matcher
from .modeling.poolers import ROIPooler
from .modeling.proposal_generator import RPN, StandardRPNHead
from .modeling.roi_heads import (
    StandardROIHeads,
    FastRCNNOutputLayers,
    MaskRCNNConvUpsampleHead,
    FastRCNNConvFCHead,
)

from ..proc_image import calculate_multi_iou, calculate_iou, visualize_detection
from ..constants import COCO_LABELS_LIST


class MaskedRCNN_ViT_B_FPN_Contexted(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()

        self.device = device

        # constants
        constants = dict(
            imagenet_rgb256_mean=[123.675, 116.28, 103.53],
            imagenet_rgb256_std=[58.395, 57.12, 57.375],
        )

        embed_dim, depth, num_heads, dp = 768, 12, 12, 0.1
        num_classes = 80

        # backbone
        self.backbone = SimpleFeaturePyramid(
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
        ).to(self.device)

        # model
        self.base_model = GeneralizedRCNN(
            backbone=self.backbone,
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
        ).to(self.device)
    
    def load_weight(self, weight_pkl_path='./model_final_61ccd1.pkl'):
        with open(weight_pkl_path, 'rb') as f:
            weights = pickle.load(f)['model']

        for name, param in self.base_model.named_parameters():
            if name in weights:
                param.data.copy_(torch.tensor(weights[name]))
            else:
                print(f"Parameter {name} not found in weights")

    def forward(self, image_ndarray: np.ndarray):
        # image_ndarray: (H, W, C)
        image_tensor = torch.tensor(image_ndarray, dtype=torch.uint8).permute(2, 0, 1).to(self.device)
        input = [{"image": image_tensor, "height": image_tensor.shape[-2], "width": image_tensor.shape[-1]}]
        
        detections = self.base_model(input)

        predictions = detections[0]
        boxes = predictions["instances"].pred_boxes.tensor.cpu().numpy()
        labels = predictions["instances"].pred_classes.cpu().numpy()
        scores = predictions["instances"].scores.cpu().numpy()

        return boxes, labels, scores
    
    @torch.no_grad()
    def validate_DAVIS_plain(self, sequence_name, data_root="/data/DAVIS", output_root="./output/maskedrcnn_vit_b_fpn", leave=False):
        self.base_model.eval()

        sequence_path = os.path.join(data_root, "JPEGImages/480p", sequence_name)
        frames = sorted(os.listdir(sequence_path))

        annotations_path = os.path.join(data_root, "Annotations_bbox/480p", f"{sequence_name}.json")
        with open(annotations_path, "r") as f:
            annotations = json.load(f)

        output_path = os.path.join(output_root, "plain_inference", sequence_name)
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(os.path.join(output_path, "temp"), exist_ok=True)

        inference_results = {}
        IoU_results = []

        pbar = tqdm(range(len(frames)), leave=leave)
        for i in pbar:
            basename = os.path.splitext(frames[i])[0]
            target_image = cv2.imread(os.path.join(sequence_path, frames[i]))
            annotation = annotations.get(basename, [])  # List of bounding boxes, each box is in a format of {'x_min': 431, 'y_min': 230, 'x_max': 460, 'y_max': 260, 'label': '14'}

            boxes_gt = [[float(box['x_min']), float(box['y_min']), float(box['x_max']), float(box['y_max'])] for box in annotation]
            labels_gt = [-1 for box in annotation]
            scores_gt = [1.0 for _ in annotation]

            boxes_pred, labels_pred, scores_pred = self.forward(target_image)

            inference_results[basename] = (boxes_pred, labels_pred, scores_pred)

            ious = calculate_multi_iou(boxes_gt, labels_gt, boxes_pred, labels_pred)
            iou = np.mean(ious) if len(ious) > 0 else 0.0
            IoU_results.append(iou if not np.isnan(iou) else 0.0)

            pbar.set_description(f"Processing {basename}, IoU: {iou:.4f}")

            image_bbox_gt = visualize_detection(target_image, boxes_gt, labels_gt, scores_gt, colors=np.array([[0, 0, 255] for _ in range(len(COCO_LABELS_LIST))]))
            image_bbox = visualize_detection(image_bbox_gt, boxes_pred, labels_pred, scores_pred)
            cv2.imwrite(os.path.join(output_path, "temp", f"{basename}.jpg"), image_bbox)
        
        avg_iou = np.mean(IoU_results)

        video_path = os.path.join(output_root, f"plain_inference/{sequence_name}", f"plain.mp4")
        os.system(f"ffmpeg -y -r 10 -i {output_path}/temp/%05d.jpg -c:v libx264 -vf fps=25 -pix_fmt yuv420p {video_path} > /dev/null 2>&1")
        os.system(f"rm -rf {output_path}/temp")
        
        return avg_iou, inference_results
