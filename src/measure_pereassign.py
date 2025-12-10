import torch
import numpy as np

from ipconv.models import (
    MaskedRCNN_ViT_B_FPN_Contexted, MaskedRCNN_ViT_L_FPN_Contexted, MaskedRCNN_ViT_H_FPN_Contexted,
    CascadeMaskRCNN_Swin_B_Contexted, CascadeMaskRCNN_Swin_L_Contexted
)
from ipconv.models.ViTDet.modeling.backbone.utils import get_abs_pos

from evaluate_funcs import shift_anchor_features, shift_anchor_features_swin

import torch
import math
from typing import Dict, Union

def shift_anchor_features_optimized(
    model,
    anchor_features: Dict[str, torch.Tensor], 
    shift_x: int, 
    shift_y: int, 
    ape: torch.Tensor = None,
) -> Dict[str, torch.Tensor]:
    for key, value in anchor_features.items():
        if "qkv" in key and "qkvpe" not in key:
            bidx = int(key.split("block")[-1].split("_")[0])
            qkvpe = anchor_features.get(f"block{bidx}_qkvpe", None)
            x_std = anchor_features.get(f"block{bidx}_std", None).mean()
            QKV_DIM, NumWindows, NumHeads, HW, C = value.shape
            sqrt_num_windows = int(math.sqrt(NumWindows))

            if bidx in model.window_block_indexes:
                continue

            value_RPE_removed = value - qkvpe / x_std

            shifted_value = value_RPE_removed.view(
                QKV_DIM, sqrt_num_windows, sqrt_num_windows, NumHeads, HW, C
            )
            
            shifted_value = shifted_value.roll(
                shifts=(-shift_y, -shift_x), 
                dims=(1, 2)
            )

            shifted_value = shifted_value.view(*value.shape)
            shifted_value += qkvpe / x_std
            
            anchor_features[key] = shifted_value

        if "out" in key:
            if ape is not None:
                value -= ape

            shifted_value = value.roll(shifts=(-shift_y, -shift_x), dims=(1, 2))

            if ape is not None:
                shifted_value += ape
                
            anchor_features[key] = shifted_value
            
    return anchor_features

@torch.no_grad()
def measure_vit_shift(model="base"):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WARMUP = 20
    NUM_REPEATS = 20

    if model == "base":
        model = MaskedRCNN_ViT_B_FPN_Contexted()
    elif model == "large":
        model = MaskedRCNN_ViT_L_FPN_Contexted()
    elif model == "huge":
        model = MaskedRCNN_ViT_H_FPN_Contexted()
    else:
        print("Unknown model type")
    model.to(DEVICE)
    model.eval()

    input_img_size = (1024, 1024)
    block_size = 16
    dummy_input = np.random.rand(input_img_size[0], input_img_size[1], 3).astype(np.uint8)
    ape = get_abs_pos(
        model.backbone.net.pos_embed,
        model.backbone.net.pretrain_use_cls_token,
        (input_img_size[0] // block_size, input_img_size[1] // block_size)
    )

    _, cached_features = model.forward_contexted(dummy_input, only_backbone=True)

    for _ in range(NUM_WARMUP):
        orig = shift_anchor_features(cached_features, 0, 0, ape)
        opted = shift_anchor_features_optimized(model, cached_features, 0, 0, ape)

        # compare correctness
        for key in orig.keys():
            if not torch.allclose(orig[key], opted[key], atol=1e-5):
                print(f"Mismatch found in key: {key}")
                return

    ts_start = torch.cuda.Event(enable_timing=True)
    ts_end = torch.cuda.Event(enable_timing=True)

    ts_start.record()
    for _ in range(NUM_REPEATS):
        shift_anchor_features_optimized(model, cached_features, 0, 0, ape)
    ts_end.record()

    torch.cuda.synchronize()
    elapsed_time = ts_start.elapsed_time(ts_end)
    print(f"ViT-B FPN shift time: {elapsed_time / NUM_REPEATS:.3f} ms")

@torch.no_grad()
def measure_swin_shift(model="base"):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WARMUP = 20
    NUM_REPEATS = 20

    if model == "base":
        model = CascadeMaskRCNN_Swin_B_Contexted()
    elif model == "large":
        model = CascadeMaskRCNN_Swin_L_Contexted()
    else:
        print("Unknown model type")
    model.to(DEVICE)
    model.eval()

    input_img_size = (1024, 1024)
    block_size = 16
    dummy_input = np.random.rand(input_img_size[0], input_img_size[1], 3).astype(np.uint8)

    _, cached_features, _ = model.forward_contexted(dummy_input, only_backbone=True)

    for _ in range(NUM_WARMUP):
        orig = shift_anchor_features_swin(cached_features, 0, 0)
        # opted = shift_anchor_features_optimized(model, cached_features, 0, 0, ape)

        # compare correctness
        # for key in orig.keys():
        #     if not torch.allclose(orig[key], opted[key], atol=1e-5):
        #         print(f"Mismatch found in key: {key}")
        #         return

    ts_start = torch.cuda.Event(enable_timing=True)
    ts_end = torch.cuda.Event(enable_timing=True)

    ts_start.record()
    for _ in range(NUM_REPEATS):
        shift_anchor_features_swin(cached_features, 0, 0)
        # shift_anchor_features_optimized(model, cached_features, 0, 0, ape)
    ts_end.record()

    torch.cuda.synchronize()
    elapsed_time = ts_start.elapsed_time(ts_end)
    print(f"Swin-B FPN shift time: {elapsed_time / NUM_REPEATS:.3f} ms")

if __name__ == "__main__":
    # measure_vit_shift("base")
    # measure_vit_shift("large")
    # measure_vit_shift("huge")
    # measure_swin_shift()
    measure_swin_shift("large")