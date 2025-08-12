import numpy as np
import torch
from tqdm import tqdm

from ipconv.models import (
    MaskedRCNN_ViT_B_FPN_Contexted,
    MaskedRCNN_ViT_L_FPN_Contexted,
    MaskedRCNN_ViT_H_FPN_Contexted
)

def create_random_dmap(dirty_rate):
    dmap = torch.zeros((1, 64, 64, 1), dtype=torch.float32)

    indices = torch.randperm(64 * 64)[:int(64 * 64 * dirty_rate)]
    dmap.view(-1)[indices] = 1.0

    return dmap

@torch.no_grad()
def measure_latency(model, dmap, num_iterations=10):
    model.eval()

    # warmup
    for _ in range(5):
        input_ndarray = np.random.rand(1024, 1024, 3).astype(np.float32)
        _ = model.forward_contexted(input_ndarray, dirtiness_map=dmap, only_backbone=True)

    total_time = 0.0
    for _ in tqdm(range(num_iterations), leave=False):
        input_ndarray = np.random.rand(1024, 1024, 3).astype(np.float32)
        dmap = dmap.cuda()

        torch.cuda.synchronize()  # Ensure all previous operations are complete
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)

        start_time.record()
        _ = model.forward_contexted(input_ndarray, dirtiness_map=dmap, only_backbone=True)
        end_time.record()

        torch.cuda.synchronize()  # Wait for the end time to be recorded
        total_time += start_time.elapsed_time(end_time)

    average_latency = total_time / num_iterations
    return average_latency

if __name__ == "__main__":
    dirty_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    models = [
        MaskedRCNN_ViT_B_FPN_Contexted,
        MaskedRCNN_ViT_L_FPN_Contexted,
        MaskedRCNN_ViT_H_FPN_Contexted
    ]

    mnames = [
        "MaskedRCNN_ViT_B_FPN_Contexted",
        "MaskedRCNN_ViT_L_FPN_Contexted",
        "MaskedRCNN_ViT_H_FPN_Contexted"
    ]

    for mname, model in zip(mnames, models):
        model = model("cuda")

        for dirty_rate in dirty_rates:
            dmap = create_random_dmap(dirty_rate)
            latency = measure_latency(model, dmap)
            print(f"Model: {mname}, Dirty Rate: {dirty_rate:.1f}, Latency: {latency:.2f} ms")