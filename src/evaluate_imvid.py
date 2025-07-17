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

import torch
import torch.nn as nn
from ipconv.models.ViTDet.eventful_transformer.base import dict_csv_header, dict_csv_line, dict_string
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from typing import List, Dict, Any, Tuple

from evaluate_funcs import (
    prepare_environment,
    estimate_affine,
    refresh_placing_matrix,
    shift_anchor_features,
    refresh_placing_matrix,
    create_dirtiness_map,
    create_sensitivity_map,
    expand_mask_neighbors
)
from ipconv.models.ViTDet.modeling.backbone.utils import get_abs_pos

outputs = []
labels = []

def evaluate_sequence(
    model: nn.Module,
    sequence_name: str,
    sequence_data: List[Tuple[torch.Tensor, Dict[str, int]]],
    frame_rate: int,
    **kwargs: Any
):
    """
    Evaluate the model on a single sequence of images.
    """
    
    def safe_tensor(array, shape, dtype):
        return (
            torch.from_numpy(array).reshape(shape).type(dtype)
            if array.size > 0
            else torch.empty(*shape, dtype=dtype)
        )
    
    pbar = enumerate(sequence_data)
    img_sample = sequence_data[0][0]
    img_H, img_W = img_sample.shape[1:]
    input_img_size = (1024, 1024)
    block_size = 16
    background_color = kwargs.get("background_color", (0, 0, 0))
    
    centering_vector = np.array([(input_img_size[1] - img_W) / 2, (input_img_size[0] - img_H) / 2])
    centering_matrix = np.array([
        [1.0, 0.0, centering_vector[0]],
        [0.0, 1.0, centering_vector[1]],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    ape = get_abs_pos(
        model.backbone.net.pos_embed,
        model.backbone.net.pretrain_use_cls_token,
        (input_img_size[0] // block_size, input_img_size[1] // block_size)
    )

    # variables
    frames_until_refresh = 0

    ref_frame = None
    ref_frame_aligned = None
    cached_features_dict = {}

    cum_shift_x, cum_shift_y = 0, 0
    placing_matrix = centering_matrix.copy()
    sensitivity_map = None

    for idx, (image, annotations) in pbar:
        image: np.ndarray
        annotations: Dict[str, int]

        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).cpu().numpy()

        ## REFRESH CHECK ##
        refresh = False
        
        # > frame rate
        if frames_until_refresh <= 0:
            refresh = True
        
        # > frame drift out
        shift_x, shift_y = 0, 0
        if not refresh:
            affine_matrix = estimate_affine(ref_frame, image)
            placing_matrix = placing_matrix @ np.vstack([affine_matrix, [0, 0, 1]])

            refresh_pmat, placing_matrix, (shift_x, shift_y) = refresh_placing_matrix(
                placing_matrix, img_H, img_W, input_img_size, block_size
            )

            refresh |= refresh_pmat
        cum_shift_x += shift_x
        cum_shift_y += shift_y

        # > scale check
        if not refresh:
            scaling_factor = np.sqrt(np.linalg.det(placing_matrix[:2, :2]))
            #print(f"Scaling factor: {scaling_factor:.2f}")
            if scaling_factor < 0.8 or scaling_factor > 1.2:
                refresh = True


        ## PREPROCESS ##
        if refresh:
            placing_matrix = centering_matrix.copy()
            frames_until_refresh = frame_rate
            cached_features_dict = {}
            shift_x, shift_y = 0, 0

        # > Place the image in the input
        image_placed = np.zeros((input_img_size[1], input_img_size[0], 3), dtype=np.uint8)
        image_placed = cv2.warpAffine(
            image, 
            placing_matrix[:2, :],
            dsize=input_img_size,
            dst=image_placed,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=background_color
        )

        # > Shift cached features and reference frame
        if shift_x != 0 or shift_y != 0:
            cached_features_dict = shift_anchor_features(
                cached_features_dict, shift_x, shift_y, ape
            )
            if ref_frame_aligned is not None:
                ref_frame_aligned = np.roll(ref_frame_aligned, shift=(-shift_y * block_size, -shift_x * block_size), axis=(0, 1))
            if sensitivity_map is not None:
                sensitivity_map = np.roll(sensitivity_map, shift=(-shift_y * block_size, -shift_x * block_size), axis=(0, 1))
        
        # > Create dirtiness map and sensitivity map
        dmap = create_dirtiness_map(
            anchor_image=ref_frame_aligned,
            current_image=image_placed,
            block_size=block_size
        ).to("cuda") if not refresh else torch.ones(1, 64, 64, 1, device="cuda")
        dmap_ndarray = dmap.squeeze().cpu().numpy()
        dmap_ndarray = cv2.resize(dmap_ndarray, (input_img_size[0], input_img_size[1]), interpolation=cv2.INTER_NEAREST)
        dmap_ndarray = np.repeat(dmap_ndarray[:, :, np.newaxis], 3, axis=2)

        # > Update the placed image with the dirtiness map
        if ref_frame_aligned is not None:
            image_placed = image_placed * dmap_ndarray + ref_frame_aligned * (1 - dmap_ndarray)
            image_placed = np.clip(image_placed, 0, 255).astype(np.uint8)

        # > Expand the sensitive area
        if sensitivity_map is not None:
            dmap_expanded = expand_mask_neighbors(dmap).cpu().numpy().squeeze(0).squeeze(-1)
            sensi_map_downsized = cv2.resize(sensitivity_map, (input_img_size[0] // block_size, input_img_size[1] // block_size), interpolation=cv2.INTER_AREA)
            sensi_map_downsized = (sensi_map_downsized > 0.5).astype(np.float32)
            dmap_expanded = dmap_expanded * sensi_map_downsized + dmap.squeeze().cpu().numpy() * (1 - sensi_map_downsized)
            dmap_recompute = torch.from_numpy(dmap_expanded).unsqueeze(0).unsqueeze(-1).to("cuda")
        else:
            dmap_recompute = dmap


        ## INFERENCE ##
        (boxes_cont, labels_cont, scores_cont), cached_features_dict = model.forward_contexted(image_placed, cached_features_dict, dmap_recompute)

        
        ## POSTPROCESS ##
        # > Create sensitivity map
        sensitivity_map = create_sensitivity_map(boxes_cont, scores_cont, input_img_size)
        
        
        ## VISUALIZE ##
        # > Draw the full border
        vis_image = image_placed.copy()
        cv2.rectangle(vis_image, (0, 0), (input_img_size[0], input_img_size[1]), (0, 255, 255), 2)

        # > Boost the dirtiness map
        dmap_recompute = dmap_recompute.squeeze().cpu().numpy()
        dmap_recompute = cv2.resize(dmap_recompute, (input_img_size[0], input_img_size[1]), interpolation=cv2.INTER_NEAREST)
        vis_image[:, :, 1] = np.clip(vis_image[:, :, 1] + dmap_recompute * 30, 0, 255)

        # > Draw boxes and labels on the placed image
        for box, label, score in zip(boxes_cont, labels_cont, scores_cont):
            if score < 0.5:
                continue
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis_image, f"{label} {score:.2f}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # rolling shift the image with cum_shift_x and cum_shift_y
        vis_image = np.roll(vis_image, shift=(cum_shift_y * block_size, cum_shift_x * block_size), axis=(0, 1))
        
        if ref_frame_aligned is not None:
            cv2.imwrite(f"temp/{sequence_name}_{idx:04d}_ref.jpg", ref_frame_aligned)
        cv2.imwrite(f"temp/{sequence_name}_{idx:04d}.jpg", vis_image)

        #print(f"Processed frame {idx} of sequence {sequence_name}, boxes: {len(boxes_cont)}")

        ref_frame = image.copy()
        ref_frame_aligned = image_placed.copy()
        frames_until_refresh -= 1
        
        # affine predicted bounding box
        def inverse_affine_boxes(transformed_boxes, placing_matrix):
            inverse_affine = np.linalg.inv(placing_matrix)[:2, :]

            restored_boxes = []
            for box in transformed_boxes:
                x1, y1, x2, y2 = box

                point_lt = np.array([x1, y1], dtype=np.float32).reshape(-1, 1, 2)
                point_rt = np.array([x2, y1], dtype=np.float32).reshape(-1, 1, 2)
                point_lb = np.array([x1, y2], dtype=np.float32).reshape(-1, 1, 2)
                point_rb = np.array([x2, y2], dtype=np.float32).reshape(-1, 1, 2)

                src_pts = np.concatenate([point_lt, point_rt, point_lb, point_rb], axis=0)
                dst_pts = cv2.transform(src_pts, inverse_affine)

                x_min = int(np.mean(dst_pts[[0, 2], 0, 0]))
                y_min = int(np.mean(dst_pts[[0, 1], 0, 1]))
                x_max = int(np.mean(dst_pts[[1, 3], 0, 0]))
                y_max = int(np.mean(dst_pts[[2, 3], 0, 1]))

                restored_boxes.append([x_min, y_min, x_max, y_max])
            return restored_boxes
        
        boxes_affined = inverse_affine_boxes(boxes_cont, placing_matrix)
        boxes_affined = np.array(boxes_affined, dtype=np.float32)


        result = {
            "boxes": safe_tensor(boxes_affined, (-1, 4), torch.float32),
            "labels": safe_tensor(labels_cont, (-1,), torch.int64),
            "scores": safe_tensor(scores_cont, (-1,), torch.float32)
        }

        outputs.append(result)
        gt_boxes = annotations["boxes"].reshape(-1, 4)
        gt_labels = annotations["labels"].reshape(-1)
        labels.append({"boxes": gt_boxes, "labels": gt_labels})

    os.system(f"ffmpeg -framerate {frame_rate} -i temp/{sequence_name}_%04d.jpg -c:v libx264 -pix_fmt yuv420p temp/{sequence_name}_{frame_rate}fps.mp4 -y")



def evaluate(
    model, 
    dataset: Dict[str, List[Tuple[torch.Tensor, Dict[str, int]]]],
    frame_rates: List[int],
    **kwargs: Any
):
    """
    Evaluate the model on the dataset at specified frame rates.
    """
    model.eval()
    model.counting()
    model.clear_counts()
    sequence_name = 0
    n_frames = 0
    for sequence_data in dataset:
        for frame_rate in frame_rates:

            print(f"Evaluating sequence: {sequence_name}, frame rate: {frame_rate} fps")

            evaluate_sequence(model, sequence_name, sequence_data, frame_rate, **kwargs)
            model.reset()
            n_frames += len(sequence_data)

        sequence_name += 1

    mean_ap = MeanAveragePrecision(box_format='xyxy')
    mean_ap.update(outputs, labels)
    metrics = mean_ap.compute()

    counts = model.total_counts() / n_frames
    model.clear_counts()
    return {"metrics": metrics, "counts": counts}


@torch.no_grad()
def main(args):

    def tee_print(s, file, flush=True):
        print(s, flush=flush)
        print(s, file=file, flush=flush)

    def save_csv_results(results, output_dir, first_run=False):
        for key, val in results.items():
            with open(output_dir / f"{key}.csv", "a") as csv_file:
                if first_run:
                    print(dict_csv_header(val), file=csv_file)
                print(dict_csv_line(val), file=csv_file)

    def do_evaluation(title, results):
        with open(output_dir / "output.txt", "a") as tee_file:

            # Print and save results.
            tee_print(title, tee_file)
            if isinstance(results, dict):
                save_csv_results(results, output_dir, first_run=(len(completed) == 0))
                for key, val in results.items():
                    tee_print(key.capitalize(), tee_file)
                    tee_print(dict_string(val), tee_file)
            else:
                tee_print(results, tee_file)
            tee_print("", tee_file)
            completed.append(title)

            # Save pred_outputs.pt
            save_path = Path("output/pred_outputs.pt")
            cpu_outputs = []
            for d in outputs:
                cpu_outputs.append({
                    "boxes":  d["boxes"].cpu(),   # shape (N,4)
                    "labels": d["labels"].cpu(),  # shape (N,)
                    "scores": d["scores"].cpu()   # shape (N,)
                })
            torch.save(cpu_outputs, save_path)
            print(f"Saved {len(cpu_outputs)} predictions to {save_path}")
                
    model, dataset, settings_dict = prepare_environment(args)

    results = evaluate(model, dataset, args.frame_rates, **settings_dict)

    completed = []
    output_dir = Path("output")

    do_evaluation("Vanilla", results)
    

def parse_int_list(value):
    """Parse comma-separated integers into a list."""
    return [int(x.strip()) for x in value.split(',')]

def parse_str_list(value):
    """Parse comma-separated strings into a list."""
    return [x.strip() for x in value.split(',')]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model on a dataset.")
    parser.add_argument("--model", type=str, default="vitdet-b", help="Model to use for evaluation.",
        choices=["vitdet-b", "vitdet-l", "vitdet-h", "dino-swin4", "lwdetr"],
    )
    parser.add_argument("--dataset", type=str, default="imnet-vid", help="Dataset to evaluate on.",
        choices=["davis", "imnet-vid"],
    )
    parser.add_argument("--frame-rates", type=parse_int_list, default=[100], 
                       help="Frame rate(s) for evaluation. Comma-separated integers (e.g., 1,6,100).")
    parser.add_argument("--sequence", type=parse_str_list, default=["bear"], 
                       help="Specific sequence(s) to evaluate on. Comma-separated strings (e.g., bear,camel). If None, evaluates on all sequences.")
    args = parser.parse_args()

    main(args)

    