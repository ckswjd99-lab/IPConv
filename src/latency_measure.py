import numpy as np
import time
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Dict, Any, Tuple

from ipconv.models import (
    MaskedRCNN_ViT_B_FPN_Contexted, MaskedRCNN_ViT_L_FPN_Contexted, MaskedRCNN_ViT_H_FPN_Contexted,
)

def measure_latency_memory(
    model: nn.Module,
    patch_keep_rate: float,
    method: str
):
    model.eval()
    dummy_input = np.zeros((1024, 1024, 3))

    dmap = torch.zeros((1, 64, 64, 1), dtype=torch.float32, device="cuda")
    num_patches = 64 * 64
    num_keep = int(num_patches * patch_keep_rate)
    idx_rand = torch.randperm(num_patches)[:num_keep]
    dmap.view(-1)[idx_rand] = 1.0

    num_warmup = 5
    num_repeats = 10

    # inference_func = model.forward_contexted if method == "ours" else model.forward_eventful
    if method == "vanilla":
        inference_func = model.forward
    elif method == "ours":
        inference_func = model.forward_contexted
    elif method == "eventful":
        inference_func = model.forward_eventful
    elif method == "maskvd":
        inference_func = model.forward_maskvd
    else:
        raise ValueError(f"Unknown method: {method}")

    for _ in range(num_warmup):
        output = inference_func(dummy_input, dirtiness_map=dmap, only_backbone=True)

    start_time = time.time()
    for _ in tqdm(range(num_repeats), leave=False):
        inference_func(dummy_input, dirtiness_map=dmap, only_backbone=True)
    end_time = time.time()

    cache_size = 0

    if method != "vanilla":
        cache = output[1]

        for key, value in cache.items():
            if isinstance(value, torch.Tensor):
                size = value.element_size() * value.numel()
                cache_size += size
            

    latency = (end_time - start_time) / num_repeats
    return latency, cache_size


@torch.no_grad()
def main():
    models_dict = {
        "ViT-base": MaskedRCNN_ViT_B_FPN_Contexted,
        # "ViT-large": MaskedRCNN_ViT_L_FPN_Contexted,
        # "ViT-huge": MaskedRCNN_ViT_H_FPN_Contexted,
    }

    keep_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # keep_rates = [0.1]

    # methods = ["ours", "eventful", "maskvd", "vanilla"]
    methods = ["maskvd"]

    for mname, model_class in models_dict.items():
        model = model_class("cuda")
        model.eval()

        for method in methods:
            for keep_rate in keep_rates:
                latency, cache_size = measure_latency_memory(model, keep_rate, method)
                print(f"Model: {mname}, Method: {method}, Patch Keep Rate: {keep_rate}, Latency: {latency:.4f} seconds, Cache Size: {cache_size / (1024 * 1024):.2f} MB")

if __name__ == "__main__":
    main()
    print("Latency measurement completed.")