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
import pickle

from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch

from typing import List, Dict, Any, Tuple

from datasets.vid import VIDResize, VID
from ipconv.models import (
    MaskedRCNN_ViT_B_FPN_Contexted, MaskedRCNN_ViT_L_FPN_Contexted, MaskedRCNN_ViT_H_FPN_Contexted,
    CascadeMaskRCNN_Swin_B_Contexted, CascadeMaskRCNN_Swin_L_Contexted,
    # CascadeMaskRCNN_MViT_B_Contexted,
    # DINO_4Scale_Swin_Contexted, DINO_5Scale_Swin_Contexted,
    # LWDETR_xLarge_Contexted
)
from ipconv.models.ViTDet.modeling.backbone.utils import window_reverse, window_partition


def prepare_environment(args) -> Tuple[Any, Dict[str, List[Tuple[torch.Tensor, Dict[str, int]]]]]:
    '''
    Prepare model and dataset for evaluation.
    '''

    # Prepare model
    models_dict = {
        # "vitdet-b-imnetvid": ViTDeT_b_Imagenet_Contexted,
        "vitdet-b": MaskedRCNN_ViT_B_FPN_Contexted,
        "vitdet-l": MaskedRCNN_ViT_L_FPN_Contexted,
        "vitdet-h": MaskedRCNN_ViT_H_FPN_Contexted,
        # "dino-swin4": DINO_4Scale_Swin_Contexted,
        # "lwdetr": LWDETR_xLarge_Contexted,
        "swin-b": CascadeMaskRCNN_Swin_B_Contexted,
        "swin-l": CascadeMaskRCNN_Swin_L_Contexted,
        # "mvit-b": CascadeMaskRCNN_MViT_B_Contexted,
    }

    models_weight_dict = {
        # "vitdet-b-imnetvid": "./weights/frcnn_vitdet_final.pth",  # 이 부분!
        "vitdet-b": "./weights/model_final_61ccd1.pkl",  # 이 부분!
        "vitdet-l": "./weights/model_final_6146ed.pkl",
        "vitdet-h": "./weights/model_final_7224f1.pkl",
        "swin-b": "./weights/model_final_246a82.pkl",
        "swin-l": "./weights/model_final_7c897e.pkl",
        "mvit-b": "./weights/model_final_8c3da3.pkl",
    }

    models_settings_dict = {
        # "vitdet-b-imnetvid": {
        #     "input_img_size": (1024, 1024),
        #     "block_size": 16,
        #     "background_color": (123.675, 116.28, 103.53)
        # },
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
        },
        "mvit-b": {
            "input_img_size": (1024, 1024),
            "block_size": 16,
            "background_color": (123.675, 116.28, 103.53)
        },
    }

    if args.model not in models_dict:
        raise ValueError(f"Unknown model: {args.model}")

    model = models_dict[args.model](args.device)

    if args.model in models_weight_dict:
        weight_path = models_weight_dict[args.model]
        ext = os.path.splitext(weight_path)[-1].lower()

        if ext == '.pkl':
            with open(weight_path, 'rb') as f:
                weights = pickle.load(f)
        else:
            weights = torch.load(weight_path, map_location='cpu', weights_only=False)

        model_state = model.state_dict()

        if isinstance(weights, dict) and "model" in weights:
            weights_state = weights["model"]
        else:
            weights_state = weights

        adjusted_weights_state = {
            f"base_model.{k}": torch.tensor(v) if isinstance(v, np.ndarray) else v
            for k, v in weights_state.items()
        }

        filtered_ckpt = {
            k: v for k, v in adjusted_weights_state.items() if k in model_state
        }

        model.load_state_dict(filtered_ckpt, strict=False)
        print(f"✅ Loaded {len(filtered_ckpt)} keys (converted from numpy if needed)")

        model = model.to(args.device)


    settings_dict = models_settings_dict[args.model] if args.model in models_settings_dict else {}
    
    # Prepare dataset
    img_max_size = int(1024 * 0.8) // 2 * 2

    if args.dataset == "DAVIS2017_trainval":
        data_root = "data/DAVIS2017_trainval/"
        frames_path = os.path.join(data_root, "JPEGImages/480p")

        dataset_dict = {}

        if args.sequence is None:
            sequences = [seq for seq in os.listdir(frames_path) if os.path.isdir(os.path.join(frames_path, seq))]
        else:
            sequences = args.sequence

        for sequence_name in sequences:
            sequence_path = os.path.join(frames_path, sequence_name)
            annotations_path = os.path.join(data_root, "Annotations/480p", sequence_name)

            seq_images = []
            seq_masks = []

            # 이미지와 마스크 파일을 정렬하여 짝맞춤
            img_names = sorted(os.listdir(sequence_path))
            mask_names = sorted(os.listdir(annotations_path))

            for img_name, mask_name in zip(img_names, mask_names):
                img_path = os.path.join(sequence_path, img_name)
                mask_path = os.path.join(annotations_path, mask_name)

                img_loaded = cv2.imread(img_path)
                if img_loaded is None:
                    print(f"[Warning] Failed to load image: {img_path}")
                    continue

                img_scaled = cv2.resize(
                    img_loaded,
                    dsize=None,
                    fx=img_max_size / max(img_loaded.shape[:2]),
                    fy=img_max_size / max(img_loaded.shape[:2]),
                    interpolation=cv2.INTER_LINEAR
                )
                seq_images.append(img_scaled)
                seq_masks.append(mask_path)

            dataset_dict[sequence_name] = list(zip(seq_images, seq_masks))
    
    elif args.dataset == "DAVIS2019_challenge" or args.dataset == "DAVIS2019_testdev":
        data_root = f"data/{args.dataset}/"
        frames_path = os.path.join(data_root, "JPEGImages/480p")

        dataset_dict = {}

        if args.sequence is None:
            sequences = [seq for seq in os.listdir(frames_path) if os.path.isdir(os.path.join(frames_path, seq))]
        else:
            sequences = args.sequence

        for sequence_name in sequences:
            sequence_path = os.path.join(frames_path, sequence_name)

            seq_images = []
            seq_masks = []

            # 이미지와 마스크 파일을 정렬하여 짝맞춤
            img_names = sorted(os.listdir(sequence_path))

            for img_name in img_names:
                img_path = os.path.join(sequence_path, img_name)

                img_loaded = cv2.imread(img_path)
                if img_loaded is None:
                    print(f"[Warning] Failed to load image: {img_path}")
                    continue

                img_scaled = cv2.resize(
                    img_loaded,
                    dsize=None,
                    fx=img_max_size / max(img_loaded.shape[:2]),
                    fy=img_max_size / max(img_loaded.shape[:2]),
                    interpolation=cv2.INTER_LINEAR
                )
                seq_images.append(img_scaled)
                seq_masks.append(None)

            dataset_dict[sequence_name] = list(zip(seq_images, seq_masks))


    if args.dataset == "imnet-vid":
        dataset_dict = VID(
        Path("data", "vid"),
        # split="vid_val",
        split="vid_cocoval",
        tar_path=Path("data", "vid", "vid_data.tar"),
        combined_transform=VIDResize(
            short_edge_length=640, max_size=int(1024 * 0.9)
        ),
        )
    
    # dataset_dict["name"] = args.dataset

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
    if ok.sum() < 6: return np.eye(2, 3, dtype=np.float32)
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


def shift_anchor_features(
        anchor_features: dict, 
        shift_x: int, 
        shift_y: int, 
        ape: torch.Tensor = None,
        k_pe: Dict[str, torch.Tensor] = None,
        v_pe: Dict[str, torch.Tensor] = None
) -> dict:
    """
    anchor_features의 모든 qkv/out 텐서를 shift_x, shift_y만큼 블록 단위로 이동시킴.
    """
    for key, value in anchor_features.items():
        if "qkv" in key and "qkvpe" not in key:
            bidx = int(key.split("block")[-1].split("_")[0])
            x_std = anchor_features.get(f"block{bidx}_std", None).mean()
            qkvpe = anchor_features.get(f"block{bidx}_qkvpe", None)

            num_windows = value.shape[1]
            num_hw = value.shape[3]

            sqrt_num_windows = int(math.sqrt(num_windows))
            sqrt_num_hw = int(math.sqrt(num_hw))

            key_reshaped = value.view(
                value.shape[0], sqrt_num_windows, sqrt_num_windows, value.shape[2],
                sqrt_num_hw, sqrt_num_hw, value.shape[4]
            )   # (3, sqrt_num_windows, sqrt_num_windows, num_heads, H, W, C)
            key_reshaped = key_reshaped.permute(0, 1, 4, 2, 5, 3, 6).contiguous().view(
                value.shape[0], sqrt_num_windows * sqrt_num_hw, sqrt_num_windows * sqrt_num_hw, value.shape[2], value.shape[4]
            )   # (3, real_H, real_W, num_heads, C)

            pe_reshaped = qkvpe.view(
                qkvpe.shape[0], sqrt_num_windows, sqrt_num_windows, qkvpe.shape[2],
                sqrt_num_hw, sqrt_num_hw, qkvpe.shape[4]
            ).permute(0, 1, 4, 2, 5, 3, 6).contiguous().view(
                value.shape[0], sqrt_num_windows * sqrt_num_hw, sqrt_num_windows * sqrt_num_hw, value.shape[2], value.shape[4]
            )

            key_reshaped -= pe_reshaped / x_std
            key_reshaped = key_reshaped.roll(shifts=(-shift_y, -shift_x), dims=(1, 2))
            key_reshaped += pe_reshaped / x_std

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

def shift_anchor_features_swin(
        anchor_features: dict, 
        shift_x: int, 
        shift_y: int, 
) -> dict:
    """
    anchor_features의 모든 qkv/out 텐서를 shift_x, shift_y만큼 블록 단위로 이동시킴.
    """
    for key, value in anchor_features.items():
        if "qkv" in key and "meta" not in key:
            lidx = int(key.split("layer")[-1].split("_")[0])
            bidx = int(key.split("block")[-1].split("_")[0])

            metadata = anchor_features.get(f"{key}_meta", None)
            ATTN_B_ = metadata["ATTN_B_"]
            ATTN_N = metadata["ATTN_N"]
            ATTN_C = metadata["ATTN_C"]
            window_size = metadata["window_size"]
            Hp = metadata["Hp"]
            Wp = metadata["Wp"]
            sqrt_n = int(math.sqrt(ATTN_N))

            shift_x_block = shift_x >> lidx
            shift_y_block = shift_y >> lidx

            qkv = value.reshape(ATTN_B_, sqrt_n, sqrt_n, ATTN_C * 3)
            qkv_unwin = window_reverse(qkv, window_size, Hp, Wp)

            qkv_unwin = qkv_unwin.roll(shifts=(-shift_y_block, -shift_x_block), dims=(1, 2))

            qkv, _ = window_partition(qkv_unwin, window_size)
            qkv = qkv.reshape(ATTN_B_, -1)

            anchor_features[key] = qkv
        if "out" in key:
            lidx = int(key.split("layer")[-1].split("_")[0])
            
            # value: (B, H, W, C)
            BHW, C = value.shape
            sqrt_n = int(math.sqrt(value.shape[0]))
            shift_x_block = shift_x >> lidx
            shift_y_block = shift_y >> lidx

            value = value.reshape(1, sqrt_n, sqrt_n, -1)
            value = value.roll(shifts=(-shift_x, -shift_y), dims=(1, 2))
            value = value.reshape(-1, C)
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
    dmap_type: str = "threshold",
    dirty_thres: int = 30,
    dirty_topk: int = 100,
    chromakey: np.ndarray = None,
    sensi_map: np.ndarray = None,
) -> torch.Tensor:
    residual = cv2.absdiff(anchor_image, current_image)
    
    # inside current_image, if there is any pixel with chromakey color, set the residual as 0
    if chromakey is not None:
        chromakey_mask = np.all(current_image == chromakey, axis=-1)
        residual[chromakey_mask] = 0

    dirtiness_map = cv2.cvtColor(residual, cv2.COLOR_BGR2GRAY)

    image_H, image_W = residual.shape[:2]
    
    dirtiness_map = cv2.GaussianBlur(dirtiness_map, (15, 15), 1.5)

    if sensi_map is not None:
        dirtiness_map = dirtiness_map / (1 - sensi_map + 1e-6)

    if dmap_type == "threshold":
        dirtiness_map = (dirtiness_map > dirty_thres).astype(np.float32)
        dirtiness_map = cv2.GaussianBlur(dirtiness_map, (15, 15), 1.5)
        dirtiness_map = cv2.resize(dirtiness_map, (image_W // block_size, image_H // block_size), interpolation=cv2.INTER_LINEAR)
        dirtiness_map = (dirtiness_map > 0).astype(np.float32)

    elif dmap_type == "topk":
        dirtiness_map = cv2.resize(dirtiness_map, (image_W // block_size, image_H // block_size), interpolation=cv2.INTER_AREA)
        # make strictly top k elements in dirtiness_map to 1, others to 0
        flat_map = dirtiness_map.flatten()
        
        k = min(dirty_topk, len(flat_map))
        if k > 0:
            topk_indices = np.argpartition(flat_map, -k)[-k:]
            new_map = np.zeros_like(flat_map)
            new_map[topk_indices] = 1.0
            dirtiness_map = new_map.reshape(dirtiness_map.shape).astype(np.float32)
        else:
            dirtiness_map = np.zeros_like(dirtiness_map).astype(np.float32)
    
    dirtiness_map = torch.from_numpy(dirtiness_map)
    dirtiness_map = dirtiness_map.unsqueeze(0).unsqueeze(-1)

    if dirtiness_map.sum() == 0:
        dirtiness_map[0, 0, 0, 0] = 1

    return dirtiness_map

def create_reference_map(
    anchor_image: np.ndarray, 
    current_image: np.ndarray,
    dirtiness_map: torch.Tensor,
    block_size: int = 16,
    refmap_type: str = "threshold",
    similar_thres: int = 30,
    similar_topk: int = 100,
):
    """
    Creates a reference map containing the index of the most similar patch in the anchor_image 
    within a 5x5 window for each patch in the current_image.
    """
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda:0')
    
    # --- 1. PREPARE INPUTS & MASKS ---
    device = torch.device('cuda:0')
    
    # Handle dirtiness_map early to skip computation if no patches are dirty
    if isinstance(dirtiness_map, np.ndarray):
        dmap_t = torch.from_numpy(dirtiness_map).to(device).bool()
    else:
        dmap_t = dirtiness_map.to(device).bool()
        
    dmap_t = dmap_t.squeeze() # Shape (64, 64)
    
    num_blocks_h = current_image.shape[0] // block_size
    num_blocks_w = current_image.shape[1] // block_size
    
    # If no patches are dirty, return early
    if not dmap_t.any():
        updated_dmap = torch.zeros(1, num_blocks_h, num_blocks_w, 1, device=device)
        refmap_out = torch.tensor(-1, device=device, dtype=torch.long).repeat(1, num_blocks_h, num_blocks_w, 1)
        return updated_dmap, refmap_out
        
    # Get indices of dirty patches
    # Shapes: (N,)
    dirty_y, dirty_x = torch.where(dmap_t)
    N = len(dirty_y)

    anchor_t = torch.from_numpy(anchor_image).to(device).float().permute(2, 0, 1).unsqueeze(0) # (1, C, H, W)
    current_t = torch.from_numpy(current_image).to(device).float().permute(2, 0, 1).unsqueeze(0) # (1, C, H, W)
    
    B, C, H, W = anchor_t.shape
    
    # --- 2. EXTRACT DIRTY PATCHES (Current) ---
    # curr_patches: (1, C, num_blocks_h, block_size, num_blocks_w, block_size) -> (C, num_blocks_h, num_blocks_w, block_size, block_size)
    curr_view = current_t.squeeze(0).view(C, num_blocks_h, block_size, num_blocks_w, block_size)
    curr_view = curr_view.permute(1, 3, 0, 2, 4) # (num_blocks_h, num_blocks_w, C, block_size, block_size)
    
    # Extract only the dirty patches: (N, C, block_size, block_size) -> (N, C * block_size^2)
    dirty_curr_patches = curr_view[dirty_y, dirty_x].reshape(N, -1)
    
    # --- 3. EXTRACT WINDOWS COMPARING TO ANCHOR (5x5) ---
    search_window_blocks = 5
    padding_blocks = search_window_blocks // 2  # 2 blocks padding
    
    # We will build indices for the 5x5 window (25 candidates) around each dirty patch
    # Coordinates of the candidates relative to the anchor image
    dy_grid, dx_grid = torch.meshgrid(
        torch.arange(-padding_blocks, padding_blocks + 1, device=device),
        torch.arange(-padding_blocks, padding_blocks + 1, device=device),
        indexing='ij'
    )
    dy_flat = dy_grid.reshape(-1)  # (25,)
    dx_flat = dx_grid.reshape(-1)  # (25,)
    
    # Target window block indices for each dirty patch
    # Shape: (N, 25)
    cand_y = torch.clamp(dirty_y.unsqueeze(1) + dy_flat, 0, num_blocks_h - 1)
    cand_x = torch.clamp(dirty_x.unsqueeze(1) + dx_flat, 0, num_blocks_w - 1)
    
    # Pad anchor to view it block-wise easily without border issues if we used unfold,
    # but here we can just use the calculated clamped indices.
    anchor_view = anchor_t.squeeze(0).view(C, num_blocks_h, block_size, num_blocks_w, block_size)
    anchor_view = anchor_view.permute(1, 3, 0, 2, 4) # (num_blocks_h, num_blocks_w, C, block_size, block_size)
    
    # Gather candidates: (N, 25, C, block_size, block_size) -> (N, 25, C * block_size^2)
    cand_patches = anchor_view[cand_y, cand_x].reshape(N, 25, -1)
    
    # --- 4. CALCULATE SSD ---
    # dirty_curr_patches: (N, 1, C*block_size^2)
    # cand_patches:       (N, 25, C*block_size^2)
    diff = dirty_curr_patches.unsqueeze(1) - cand_patches
    ssd = torch.sum(diff ** 2, dim=-1) # Shape: (N, 25)
    
    # Find minimum SSD for each dirty patch
    min_ssd_values, best_match_local_idx = torch.min(ssd, dim=-1) # Shape: (N,)
    
    # --- 5. EVALUATE THRESHOLDS / TOP-K ---
    mask = torch.zeros(N, dtype=torch.bool, device=device)
    
    if refmap_type == "topk":
        k = min(similar_topk, N)
        if k > 0:
            _, topk_indices = torch.topk(-min_ssd_values, k)
            mask[topk_indices] = True
    elif refmap_type == "threshold":
        # Any candidate with SSD <= similar_thres is a valid reference match
        mask = min_ssd_values <= similar_thres
        
    # --- 6. ASSEMBLE REFMAP & UPDATED_DMAP ---
    # Global flat index of the matched candidates
    # cand_y and cand_x shape: (N, 25)
    # best_match_local_idx shape: (N,)
    # Select the target Y, X for the minimum SSD
    best_target_y = cand_y[torch.arange(N, device=device), best_match_local_idx]
    best_target_x = cand_x[torch.arange(N, device=device), best_match_local_idx]
    matched_ref_map = best_target_y * num_blocks_w + best_target_x # (N,)
    
    # Output maps
    refmap_out = torch.tensor(-1, device=device, dtype=torch.long).repeat(num_blocks_h, num_blocks_w)
    updated_dmap = dmap_t.clone()
    
    # Only for the patches that passed the mask condition (subset of N)
    valid_mask_idx = torch.nonzero(mask, as_tuple=True)[0]
    
    y_valid = dirty_y[valid_mask_idx]
    x_valid = dirty_x[valid_mask_idx]
    
    # These patches have a valid reference, so we don't need to recompute them
    refmap_out[y_valid, x_valid] = matched_ref_map[valid_mask_idx]
    updated_dmap[y_valid, x_valid] = False # No longer dirty => No recompute
    
    # --- 7. FORMAT OUTPUTS ---
    # Expand dims to match original usages: (1, 64, 64, 1)
    refmap_out = refmap_out.unsqueeze(0).unsqueeze(-1)
    updated_dmap = updated_dmap.unsqueeze(0).unsqueeze(-1).float()
    
    return updated_dmap, refmap_out

def update_cache_with_refmap(cached_features_dict: dict, refmap: torch.Tensor) -> dict:
    """
    Updates the cached features based on the reference map.
    For patches that have a valid reference index (not -1), their features are 
    replaced by the features from the matched reference index.
    """
    if not cached_features_dict:
        return cached_features_dict

    refmap_flat = refmap.view(-1).long()
    needs_ref = refmap_flat != -1

    if not needs_ref.any():
        return cached_features_dict

    ref_indices = refmap_flat[needs_ref]
    needs_ref_indices = torch.nonzero(needs_ref).squeeze(-1) # indices in 4096

    for key, value in cached_features_dict.items():
        if "qkv" in key and "qkvpe" not in key:
            num_windows = value.shape[1]
            num_hw = value.shape[3]
            
            if num_windows == 1 and num_hw >= 4096:
                # Global
                old_values = value[:, :, :, ref_indices, :].clone()
                value[:, :, :, needs_ref_indices, :] = old_values
                cached_features_dict[key] = value
            elif num_windows > 1:
                # Windowed
                sqrt_num_windows = int(math.sqrt(num_windows))
                sqrt_num_hw = int(math.sqrt(num_hw))
                
                key_reshaped = value.view(
                    value.shape[0], sqrt_num_windows, sqrt_num_windows, value.shape[2],
                    sqrt_num_hw, sqrt_num_hw, value.shape[4]
                )
                key_reshaped = key_reshaped.permute(0, 1, 4, 2, 5, 3, 6).contiguous().view(
                    value.shape[0], sqrt_num_windows * sqrt_num_hw, sqrt_num_windows * sqrt_num_hw, value.shape[2], value.shape[4]
                )
                
                # Apply refmap on the top-left 64x64 region.
                region_64x64 = key_reshaped[:, :64, :64, :, :].reshape(value.shape[0], 4096, value.shape[2], value.shape[4])
                old_values = region_64x64[:, ref_indices, :, :].clone()
                region_64x64[:, needs_ref_indices, :, :] = old_values
                key_reshaped[:, :64, :64, :, :] = region_64x64.view(value.shape[0], 64, 64, value.shape[2], value.shape[4])
                
                # Repartition
                key_reshaped = key_reshaped.view(
                    value.shape[0], sqrt_num_windows, sqrt_num_hw, sqrt_num_windows, sqrt_num_hw, value.shape[2], value.shape[4]
                )
                key_reshaped = key_reshaped.permute(0, 1, 3, 5, 2, 4, 6).contiguous()
                key_reshaped = key_reshaped.view(*value.shape)
                cached_features_dict[key] = key_reshaped

        elif "attn_v" in key:
            # value shape: (B * num_heads, HW, C)
            old_values = value[:, ref_indices, :].clone()
            value[:, needs_ref_indices, :] = old_values
            cached_features_dict[key] = value

        elif "attn" in key:
            # value shape: (B * num_heads, HW, HW)
            old_values = value[:, ref_indices, :].clone()
            value[:, needs_ref_indices, :] = old_values
            
            old_values_key = value[:, :, ref_indices].clone()
            value[:, :, needs_ref_indices] = old_values_key
            cached_features_dict[key] = value

        elif "out" in key:
            # value shape: (B, H, W, C)
            B, H, W, C = value.shape
            value_flat = value.view(B, H * W, C)
            old_values = value_flat[:, ref_indices, :].clone()
            value_flat[:, needs_ref_indices, :] = old_values
            cached_features_dict[key] = value_flat.view(B, H, W, C)

    return cached_features_dict


def expand_mask_neighbors(mask_4d: torch.Tensor, expansion: int = 1) -> torch.Tensor:
    if mask_4d.dim() == 2:
        mask_4d = mask_4d.unsqueeze(0).unsqueeze(-1)
    elif mask_4d.dim() == 3:
        mask_4d = mask_4d.unsqueeze(-1)
    mask_4d = mask_4d.permute(0, 3, 1, 2)  # (1, 1, 64, 64)
    ksize = 2 * expansion + 1
    kernel = torch.ones((1, 1, ksize, ksize), device=mask_4d.device, dtype=mask_4d.dtype)
    
    expanded = F.conv2d(mask_4d, kernel, padding=expansion)
    expanded = (expanded > 0).float()
    expanded = expanded.permute(0, 2, 3, 1)
    
    return expanded

def objdet_coco_to_imvid(boxes, labels, scores):
    COCO_LABELS_LIST = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 
        'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 
        'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 
        'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 
        'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 
        'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 
        'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 
        'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 
        'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    VID_LABELS_LIST = [
        "airplane", "antelope", "bear", "bicycle", "bird", "bus", "car", 
        "cattle", "dog", "domestic_cat", "elephant", "fox", "giant_panda", 
        "hamster", "horse", "lion", "lizard", "monkey", "motorcycle", "rabbit", 
        "red_panda", "sheep", "snake", "squirrel", "tiger", "train", "turtle", 
        "watercraft", "whale", "zebra"
    ]

    common_class_names = set(COCO_LABELS_LIST) & set(VID_LABELS_LIST)
    common_coco_indices = {i for i, name in enumerate(COCO_LABELS_LIST) if name in common_class_names}

    coco_name_to_idx = {name: i for i, name in enumerate(COCO_LABELS_LIST)}
    vid_name_to_idx = {name: i for i, name in enumerate(VID_LABELS_LIST)}
    
    coco_idx_to_vid_idx = {
        coco_name_to_idx[name]: vid_name_to_idx[name] for name in common_class_names
    }

    mask = np.isin(labels, list(common_coco_indices))

    filtered_boxes = boxes[mask]
    filtered_coco_labels = labels[mask]
    filtered_scores = scores[mask]

    if filtered_coco_labels.size > 0:
        remapped_labels = np.array([coco_idx_to_vid_idx[coco_idx] for coco_idx in filtered_coco_labels])
    else:
        remapped_labels = filtered_coco_labels

    return filtered_boxes, remapped_labels, filtered_scores