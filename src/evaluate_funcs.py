import argparse
import os
import sys
import time
import cv2
import numpy as np
import json
from tqdm import tqdm
from pathlib import Path
import math

from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Dict, Any, Tuple

from datasets.vid import VIDResize, VID
from ipconv.models import (
    ViTDeT_b_Imagenet_Contexted, MaskedRCNN_ViT_B_FPN_Contexted, MaskedRCNN_ViT_L_FPN_Contexted, MaskedRCNN_ViT_H_FPN_Contexted,
    CascadeMaskRCNN_Swin_B_Contexted, DINO_4Scale_Swin_Contexted, DINO_5Scale_Swin_Contexted,
    LWDETR_xLarge_Contexted
)


def prepare_environment(args) -> Tuple[Any, Dict[str, List[Tuple[torch.Tensor, Dict[str, int]]]]]:
    '''
    Prepare model and dataset for evaluation.
    '''

    # Prepare model
    models_dict = {
        "vitdet-b": ViTDeT_b_Imagenet_Contexted,
        "vitdet-l": MaskedRCNN_ViT_L_FPN_Contexted,
        "vitdet-h": MaskedRCNN_ViT_H_FPN_Contexted,
        "dino-swin4": DINO_4Scale_Swin_Contexted,
        "lwdetr": LWDETR_xLarge_Contexted,
    }

    models_weight_dict = {
        "vitdet-b": "./ipconv/models/frcnn_vitdet_final.pth",
        "vitdet-l": "./ipconv/models/model_final_6146ed.pkl",
        "vitdet-h": "./ipconv/models/model_final_7224f1.pkl"
    }

    models_settings_dict = {
        "vitdet-b": {
            "input_img_size": (1024, 1024),
            "block_size": 16,
            "background_color": (123.675, 116.28, 103.53)
        },
        "vitdet-l": {
            "input_img_size": (1024, 1024),
            "block_size": 16,
            "background_color": (123.675, 116.28, 103.53)
        },
        "vitdet-h": {
            "input_img_size": (1024, 1024),
            "block_size": 16,
            "background_color": (123.675, 116.28, 103.53)
        },
        "dino-swin4": {
            "input_img_size": (1024, 1024),
            "block_size": 16,
            "background_color": (0, 0, 0)
        },
        "lwdetr": {
            "input_img_size": (1024, 1024),
            "block_size": 16,
            "background_color": (0, 0, 0)
        }
    }

    if args.model not in models_dict:
        raise ValueError(f"Unknown model: {args.model}")

    model = models_dict[args.model]()

    if args.model in models_weight_dict:
        weight_path = models_weight_dict[args.model]
        weights = torch.load(weight_path, map_location='cpu')  # 또는 'cuda' 필요 시

        model_state = model.state_dict()        
        weights_state = weights["model"] if "model" in weights else weights
        adjusted_weights_state = {f"base_model.{k}": v for k, v in weights_state.items()}

        filtered_ckpt = {k: v for k, v in adjusted_weights_state.items() if k in model_state}

        model.load_state_dict(filtered_ckpt, strict=False)
        print(f"✅ Loaded {len(filtered_ckpt)} keys")

    settings_dict = models_settings_dict[args.model] if args.model in models_settings_dict else {}
    
    # Prepare dataset
    img_max_size = int(1024 * 0.8) // 2 * 2

    if args.dataset == "davis":
        data_root = "/data/DAVIS/"
        frames_path = os.path.join(data_root, "JPEGImages/480p")

        dataset_dict = {}

        if args.sequence is None:
            sequences = [seq for seq in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, seq))]
        else:
            sequences = args.sequence
        
        for sequence_name in sequences:
            sequence_path = f"{frames_path}/{sequence_name}"
            annotations_path = os.path.join(data_root, "Annotations_bbox/480p", f"{sequence_name}.json")

            seq_images = []

            for img_name in sorted(os.listdir(sequence_path)):
                img_loaded = cv2.imread(os.path.join(sequence_path, img_name))
                
                img_scaled = cv2.resize(
                    img_loaded, 
                    dsize=None,
                    fx=img_max_size/max(img_loaded.shape[:2]),
                    fy=img_max_size/max(img_loaded.shape[:2]),
                    interpolation=cv2.INTER_LINEAR
                )
                seq_images.append(img_scaled)
            
            with open(annotations_path, "r") as f:
                annotations = json.load(f)
                annotations = [annotations[frame_name] for frame_name in sorted(annotations.keys())]

            dataset_dict[sequence_name] = list(zip(seq_images, annotations))

    if args.dataset == "imnet-vid":
        dataset_dict = VID(
        Path("/home/nxclab/data", "vid"),
        split="vid_val",
        tar_path=Path("/home/nxclab/data", "vid", "vid_data.tar"),
        combined_transform=VIDResize(
            short_edge_length=640, max_size=1024
        ),
        )
        

    return model, dataset_dict, settings_dict


def estimate_affine(prev_nd, curr_nd):
    MAX_PTS, LK_WIN   = 400, (15, 15)
    QUALITY, RANSAC_RE = 0.01, 3.0
    DOWNSCALE         = 0.5

    prev_g = cv2.resize(cv2.cvtColor(prev_nd, cv2.COLOR_BGR2GRAY), None, fx=DOWNSCALE, fy=DOWNSCALE, interpolation=cv2.INTER_AREA)
    curr_g = cv2.resize(cv2.cvtColor(curr_nd, cv2.COLOR_BGR2GRAY), None, fx=DOWNSCALE, fy=DOWNSCALE, interpolation=cv2.INTER_AREA)

    p0 = cv2.goodFeaturesToTrack(prev_g, MAX_PTS, QUALITY, 7)
    if p0 is None:  return np.eye(2, 3, np.float32)
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_g, curr_g, p0, None,
                                         winSize=LK_WIN, maxLevel=3)
    ok = st.squeeze() == 1
    if ok.sum() < 6: return np.eye(2, 3, np.float32)
    T, _ = cv2.estimateAffinePartial2D(p1[ok], p0[ok], method=cv2.LMEDS)

    T[0,2] /= DOWNSCALE;  T[1,2] /= DOWNSCALE
    
    return T.astype(np.float32) if T is not None else None


def refresh_placing_matrix(placing_matrix, img_H, img_W, input_img_size, block_size):
    """
    Refresh the placing matrix based on the image size and input image size.
    """

    points = np.array([[0, 0], [0, img_H], [img_W, 0], [img_W, img_H]], dtype=np.float32).reshape(-1, 1, 2)
    frame_points = cv2.transform(points, placing_matrix)

    shift_x, shift_y = 0, 0
    
    if np.any(frame_points < 0) and np.any(frame_points > input_img_size[0]):
        return True, placing_matrix, (shift_x, shift_y)
    elif np.any(frame_points < 0) or np.any(frame_points > input_img_size[0]):
        shift_x_minus = math.floor(min(0, frame_points[:, 0, 0].min() / block_size))
        shift_x_plus = math.ceil(max(0, (frame_points[:, 0, 0].max() - input_img_size[0]) / block_size))
        shift_y_minus = math.floor(min(0, frame_points[:, 0, 1].min() / block_size))
        shift_y_plus = math.ceil(max(0, (frame_points[:, 0, 1].max() - input_img_size[0]) / block_size))
        shift_x = int(shift_x_minus + shift_x_plus)
        shift_y = int(shift_y_minus + shift_y_plus)

        placing_matrix[0, 2] -= shift_x * block_size
        placing_matrix[1, 2] -= shift_y * block_size
    
    return False, placing_matrix, (shift_x, shift_y)


def shift_anchor_features(anchor_features: dict, shift_x: int, shift_y: int, ape: torch.Tensor = None) -> dict:
    """
    anchor_features의 모든 qkv/out 텐서를 shift_x, shift_y만큼 블록 단위로 이동시킴.
    """
    for key, value in anchor_features.items():
        if "qkv" in key:
            num_windows = value.shape[1]
            num_hw = value.shape[3]

            sqrt_num_windows = int(math.sqrt(num_windows))
            sqrt_num_hw = int(math.sqrt(num_hw))

            key_reshaped = value.view(
                value.shape[0], sqrt_num_windows, sqrt_num_windows, value.shape[2],
                sqrt_num_hw, sqrt_num_hw, value.shape[4]
            )
            key_reshaped = key_reshaped.permute(0, 1, 4, 2, 5, 3, 6).contiguous().view(
                value.shape[0], sqrt_num_windows * sqrt_num_hw, sqrt_num_windows * sqrt_num_hw, value.shape[2], value.shape[4]
            )
            key_reshaped = key_reshaped.roll(shifts=(-shift_y, -shift_x), dims=(1, 2))
            key_reshaped = key_reshaped.view(
                value.shape[0], sqrt_num_windows, sqrt_num_hw, sqrt_num_windows, sqrt_num_hw, value.shape[2], value.shape[4]
            )
            key_reshaped = key_reshaped.permute(0, 1, 3, 5, 2, 4, 6).contiguous()
            key_reshaped = key_reshaped.view(*value.shape)
            anchor_features[key] = key_reshaped
        if "out" in key:
            # value: (B, H, W, C)
            if ape is not None:
                value -= ape
            value = value.roll(shifts=(-shift_y, -shift_x), dims=(1, 2))
            if ape is not None:
                value += ape
            anchor_features[key] = value
    
    return anchor_features


def create_sensitivity_map(
    boxes: List[List[float]],
    scores: List[float],
    map_size: Tuple[int, int] = (1024, 1024),
) -> np.ndarray:
    """
    Create a sensitivity map based on bounding boxes and scores.

    Arguments:
        boxes (List[List[float]]): List of bounding boxes, each box is in a format of [x_min, y_min, x_max, y_max].
        scores (List[float]): List of scores corresponding to each bounding box.

    Returns:
        np.ndarray: Sensitivity map of shape (64, 64).
    """
    # Create a blank sensitivity map
    sensitivity_map = np.zeros(map_size, dtype=np.float32)

    # Iterate through each bounding box and its corresponding score
    for box, score in zip(boxes, scores):
        x_min, y_min, x_max, y_max = map(int, box)
        # Create a mask for the current bounding box
        mask = np.zeros(map_size, dtype=np.float32)
        mask[y_min:y_max, x_min:x_max] = score
        # Add the mask to the sensitivity map
        sensitivity_map += mask

    # Expand the sensitivity map
    sensitivity_map = cv2.GaussianBlur(sensitivity_map, (63, 63), 1.5) * 255 * 255

    # Min-max normalization
    min_val = np.min(sensitivity_map)
    max_val = np.max(sensitivity_map)
    if max_val - min_val > 0:
        sensitivity_map = (sensitivity_map - min_val) / (max_val - min_val)
    else:
        sensitivity_map = np.zeros_like(sensitivity_map)

    return sensitivity_map

def create_dirtiness_map(
    anchor_image: np.ndarray, 
    current_image: np.ndarray,
    block_size: int = 16,
    dirty_thres: int = 30,
    chromakey: np.ndarray = np.array([123.675, 116.28, 103.53], dtype=np.uint8),
    sensi_map: np.ndarray = None,
) -> torch.Tensor:
    residual = cv2.absdiff(anchor_image, current_image)
    
    # inside current_image, if there is any pixel with chromakey color, set the residual as 0
    # chromakey_mask = np.all(current_image == chromakey, axis=-1)
    # residual[chromakey_mask] = 0

    dirtiness_map = cv2.cvtColor(residual, cv2.COLOR_BGR2GRAY)

    image_H, image_W = residual.shape[:2]
    
    dirtiness_map = cv2.GaussianBlur(dirtiness_map, (15, 15), 1.5)
    if sensi_map is None:
        dirtiness_map = (dirtiness_map > dirty_thres).astype(np.float32)
    else:
        dirtiness_map = (dirtiness_map > dirty_thres * (1 - sensi_map)).astype(np.float32)

    dirtiness_map = cv2.GaussianBlur(dirtiness_map, (15, 15), 1.5)
    dirtiness_map = cv2.resize(dirtiness_map, (image_W // block_size, image_H // block_size), interpolation=cv2.INTER_LINEAR)
    dirtiness_map = (dirtiness_map > 0).astype(np.float32)

    dirtiness_map = torch.from_numpy(dirtiness_map)
    dirtiness_map = dirtiness_map.unsqueeze(0).unsqueeze(-1)

    if dirtiness_map.sum() == 0:
        dirtiness_map[0, 0, 0, 0] = 1

    return dirtiness_map

def expand_mask_neighbors(mask_4d: torch.Tensor) -> torch.Tensor:
    mask_4d = mask_4d.permute(0, 3, 1, 2)  # (1, 1, 64, 64)
    kernel = torch.ones((1, 1, 3, 3), device=mask_4d.device, dtype=mask_4d.dtype)
    
    expanded = F.conv2d(mask_4d, kernel, padding=1)
    expanded = (expanded > 0).float()
    expanded = expanded.permute(0, 2, 3, 1)
    
    return expanded