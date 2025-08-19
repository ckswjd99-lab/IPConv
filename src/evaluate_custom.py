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
import re
import time
import csv

import torch
import torch.nn as nn

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
from dds_utils import Results, read_results_dict, evaluate, cleanup, Region, compute_regions_size, compress_and_get_size

outputs = Results()
labels = Results()
global_fid = 1
dmap_dict = {}


def save_dirty_patches(image_placed, dmap_recompute, save_dir, fid):
    """
    dmap_recompute == 1 인 영역만 남긴 frame 이미지를 저장
    """
    os.makedirs(save_dir, exist_ok=True)

    # dmap_recompute: [1, H/block, W/block, 1] (torch)
    dmap_mask = dmap_recompute.squeeze().cpu().numpy()  # [H/block, W/block]
    H, W = image_placed.shape[:2]

    # upsample mask to full resolution
    dmap_mask_up = cv2.resize(dmap_mask, (W, H), interpolation=cv2.INTER_NEAREST)
    dmap_mask_up = (dmap_mask_up > 0.5).astype(np.uint8)

    # 3-channel mask
    dmap_mask_3c = np.repeat(dmap_mask_up[:, :, None], 3, axis=2)

    # apply mask (dirty 영역만 남김)
    dirty_only = image_placed * dmap_mask_3c

    out_path = os.path.join(save_dir, f"frame_{fid:05d}.png")
    cv2.imwrite(out_path, dirty_only)

    return out_path


def compute_bandwidth_for_dirty_regions(images_path, dmap_dict, start_id, end_id, qp=None, resolution=None):
    """
    dmap_dict: {fid: dmap_recompute tensor} 형태로 저장된 dict
    """
    temp_dir = os.path.join(images_path, "dirty_frames")
    os.makedirs(temp_dir, exist_ok=True)

    for fid in range(start_id, end_id):
        if fid not in dmap_dict:
            continue
        image_path = os.path.join(images_path, f"frame_{fid:05d}.png")
        image = cv2.imread(image_path)
        if image is None:
            continue

        dmap_recompute = dmap_dict[fid]
        save_dirty_patches(image, dmap_recompute, temp_dir, fid)

    size = compress_and_get_size(temp_dir, start_id, end_id, qp, resolution=resolution)

    return size

def pad_to_divisible(image: np.ndarray, size_divisibility: int = 32):
    h, w = image.shape[:2]
    new_h = (h + size_divisibility - 1) // size_divisibility * size_divisibility
    new_w = (w + size_divisibility - 1) // size_divisibility * size_divisibility

    pad_h = new_h - h
    pad_w = new_w - w

    padded = np.pad(
        image,
        pad_width=((0, pad_h), (0, pad_w), (0, 0)),
        mode='constant',
        constant_values=0,
    )
    return padded, h, w 

def evaluate_sequence(
    image_dir: str,
    model: nn.Module,
    sequence_name: str,
    sequence_data: List[Tuple[torch.Tensor, Dict[str, int]]],
    frame_rate: int,
    dmap_type: str = "threshold",
    dirty_thres: int = 30,
    dirty_topk: int = 100,
    sensi_expansion: int = 1,
    **kwargs: Any
):
    """
    Evaluate the model on a single sequence of images.
    """
    
    def safe_tensor(array, shape, dtype):
        if array.size > 0:
            return torch.from_numpy(array).reshape(shape).type(dtype)
        else:
            # shape 내 -1을 0으로 바꿔서 empty tensor를 안전하게 생성
            safe_shape = tuple(0 if s == -1 else s for s in shape)
            return torch.empty(*safe_shape, dtype=dtype)
        
    first_img_path = os.path.join(image_dir, sequence_data[0])
    img_sample = cv2.imread(first_img_path)  # shape: (H, W, C)

    if img_sample is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {first_img_path}")

    img_H, img_W = img_sample.shape[:2]

    input_img_size = (1024, 1024)
    block_size = 16
    background_color = kwargs.get("background_color", (0, 0, 0))

    centering_vector = np.array([
        (input_img_size[1] - img_W) / 2,
        (input_img_size[0] - img_H) / 2
    ])
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

    pbar = enumerate(sequence_data)
    seq_pred = Results()
    
    for idx, fname in pbar:
        image_path = os.path.join(image_dir, fname)
        image = cv2.imread(image_path)
        image, orig_height, orig_width = pad_to_divisible(image)

        if "png" not in fname:
            continue

        t1=time.time()
        fid = int(re.split(r'[_.]', fname)[1])

        image: np.ndarray

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
        if not refresh:
            dmap_raw = create_dirtiness_map(
                anchor_image=ref_frame_aligned,
                current_image=image_placed,
                block_size=block_size,
                dmap_type=dmap_type,
                dirty_thres=dirty_thres,
                dirty_topk=dirty_topk
            )

            if isinstance(dmap_raw, np.ndarray):
                dmap = torch.from_numpy(dmap_raw).to("cuda")
            elif isinstance(dmap_raw, torch.Tensor):
                dmap = dmap_raw.to("cuda")
            else:
                raise TypeError("Unsupported type for dirtiness map")
        else:
            dmap = torch.ones(1, 64, 64, 1, device="cuda")

        
        dmap_ndarray = dmap.squeeze().cpu().numpy()
        dmap_ndarray = cv2.resize(dmap_ndarray, (input_img_size[0], input_img_size[1]), interpolation=cv2.INTER_NEAREST)
        dmap_ndarray = np.repeat(dmap_ndarray[:, :, np.newaxis], 3, axis=2)

        # > Update the placed image with the dirtiness map
        if ref_frame_aligned is not None:
            image_placed = image_placed * dmap_ndarray + ref_frame_aligned * (1 - dmap_ndarray)
            image_placed = np.clip(image_placed, 0, 255).astype(np.uint8)

        # > Expand the sensitive area
        if sensitivity_map is not None:
            dmap_expanded = expand_mask_neighbors(dmap, sensi_expansion).cpu().numpy().squeeze(0).squeeze(-1)
            sensi_map_downsized = cv2.resize(sensitivity_map, (input_img_size[0] // block_size, input_img_size[1] // block_size), interpolation=cv2.INTER_AREA)
            sensi_map_downsized = (sensi_map_downsized > 0.5).astype(np.float32)
            dmap_expanded = dmap_expanded * sensi_map_downsized + dmap.squeeze().cpu().numpy() * (1 - sensi_map_downsized)
            dmap_recompute = torch.from_numpy(dmap_expanded).unsqueeze(0).unsqueeze(-1).to("cuda")
        else:
            dmap_recompute = dmap

        dmap_dict[fid] = dmap_recompute.cpu()


        ## INFERENCE ##
        (boxes_cont, labels_cont, scores_cont), cached_features_dict, pred_masks = model.forward_contexted(image_placed, cached_features_dict, dmap_recompute)

        
        ## POSTPROCESS ##
        # > Create sensitivity map
        sensitivity_map = create_sensitivity_map(boxes_cont, scores_cont, input_img_size)

        '''
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
            cv2.imwrite(f"temp/{sequence_name}_{idx:04d}_ref.jpg", ref_frame_aligned[:, :, ::-1])
        cv2.imwrite(f"temp/{sequence_name}_{idx:04d}.jpg", vis_image[:, :, ::-1])

        #print(f"Processed frame {idx} of sequence {sequence_name}, boxes: {len(boxes_cont)}")
        '''
        
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

        frame_with_no_results = True

        boxes = result["boxes"].cpu().numpy()   # shape: (N, 4)
        labels = result["labels"].cpu().numpy() # shape: (N,)
        scores = result["scores"].cpu().numpy() # shape: (N,)

        for (x1, y1, x2, y2), label, score in zip(boxes, labels, scores):
            
            w = (x2 - x1) / orig_width
            h = (y2 - y1) / orig_height
            x = x1 / orig_width
            y = y1 / orig_height

            if label in [2, 5, 6, 7]:
                label_str = "vehicle"
            elif label in [0, 1, 3]:
                label_str = "persons"
            elif label in [9, 10, 11, 12]:
                label_str = "roadside-objects"
            else:
                continue
            
            r = Region(
                fid,               # frame id
                float(x),         # x
                float(y),         # y
                float(w),          # width
                float(h),          # height
                float(score),      # confidence
                label_str,        # label (정수 → 문자열)
                1,                 # resolution/scale factor
                origin="mpeg"      # origin 정보
            )
            seq_pred.append(r)
            frame_with_no_results = False

        if frame_with_no_results:
            seq_pred.append(
                Region(fid, 0, 0, 0, 0, 0.1, "no obj", 1)
            )


        if (fid+1)%100==0:
            print('detect fid',fid)

    #os.system(f"ffmpeg -framerate {frame_rate} -i temp/{sequence_name}_%04d.jpg -c:v libx264 -pix_fmt yuv420p temp/{sequence_name}_{frame_rate}fps.mp4 -y")

    return seq_pred

def load_gt_as_results(csv_path):
    gt_results = Results()
    max_fid = 0
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            fid = int(row[0]); max_fid = max(max_fid, fid)
            x = float(row[1]); y = float(row[2]); w = float(row[3]); h = float(row[4])
            label = row[5]; conf = float(row[6]); resolution = float(row[7]); origin = row[8]
            r = Region(fid, x, y, w, h, conf, label, resolution, origin)
            gt_results.append(r)
    return gt_results, max_fid

def evaluate_custom(
    model, 
    dataset: str,
    sequence_number: int,
    frame_rates: List[int],
    dmap_type: str = "threshold",
    dirty_thres: int = 30,
    dirty_topk: int = 100,
    sensi_expansion: int = 1,
    **kwargs: Any
):
    bw = 0
    """
    Evaluate the model on the dataset at specified frame rates.
    """
    global global_fid

    for i in range (sequence_number):
        images_direc = os.path.join(dataset, str(i), "frames_png")
        sequence_data = sorted(os.listdir(images_direc))
        print(f"Evaluating sequence {i}")
        model.eval()
        seq_pred = evaluate_sequence(images_direc, model, str(i), sequence_data, frame_rates[0], dmap_type, dirty_thres, dirty_topk, sensi_expansion, **kwargs)

        out_dir = f"/home/nxc/sooyoung7896/IPConv/results/{args.dataset}/{args.model}"
        os.makedirs(out_dir, exist_ok=True)
        seq_pred.write(os.path.join(out_dir, "predictions"))

        for fid, dets in sorted(seq_pred.regions_dict.items()):
            outputs.regions_dict[global_fid] = dets
            global_fid += 1

        # GT merge
        labels_direc = os.path.join(dataset, str(i), "labels")
        gt_results, max_fid = load_gt_as_results(labels_direc)
        for fid, dets in sorted(gt_results.regions_dict.items()):
            labels.regions_dict[global_fid - max_fid + fid - 1] = dets

        bw_size = compute_bandwidth_for_dirty_regions(
            images_path=images_direc,
            dmap_dict=dmap_dict,
            start_id=1,
            end_id=len(sequence_data),
            qp=26,
            resolution=None
        )
        print("Bandwidth (bytes):", bw_size)
        bw += bw_size

        
    total_max_fid = max(labels.regions_dict.keys())

    tp, fp, fn, count, precision, recall, f1, f1_list, mAP = evaluate(
        total_max_fid,
        outputs.regions_dict,
        labels.regions_dict,
        args.gt_confid_thresh,
        args.mpeg_confid_thresh,
        args.max_area_thresh_gt,
        args.max_area_thresh_mpeg,
        iou_thresh=args.iou_thresh
    )

    output_str = (
        f"\n[Overall Evaluation - {args.dataset}/{args.model}]\n"
        f"Precision: {precision:.4f}\n"
        f"Recall   : {recall:.4f}\n"
        f"F1-score : {f1:.4f}\n"
        f"mAP50    : {mAP:.4f}\n"
        f"Bandwidth-. : {bw:.4f}\n"
    )

    print(output_str)

    out_dir = f"output/{args.dataset}/{args.model}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.frame_rates}+{args.dirty_thres}.txt")

    with open(out_path, "w") as f:
        f.write(output_str)


@torch.no_grad()
def main(args):

    model, dataset, settings_dict = prepare_environment(args)
    evaluate_custom(model, dataset, args.sequence_number, args.frame_rates, args.dmap_type, args.dirty_thres, args.dirty_topk, args.sensi_expansion, **settings_dict)
    

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
    parser.add_argument("--dataset", type=str, default="highway", help="Dataset to evaluate on.",
        choices=["highway", "city_drive"],
    )
    parser.add_argument("--frame-rates", type=parse_int_list, default=[100], 
                       help="Frame rate(s) for evaluation. Comma-separated integers (e.g., 1,6,100).")
    parser.add_argument("--sequence-number", type=int, default=5, 
                       help="Specific sequence(s) to evaluate on. Comma-separated strings (e.g., bear,camel). If None, evaluates on all sequences.")
    parser.add_argument("--dmap_type", type=str, choices=["threshold", "topk"], default="threshold",
                       help="Type of dirtiness map to use. 'threshold' for thresholding, 'topk' for top-k dirtiness.")
    parser.add_argument("--dirty_thres", type=int, default=30, nargs="?",
                       help="Dirtiness threshold for the dirtiness map. Default is 30.")
    parser.add_argument("--dirty-topk", type=int, default=100, nargs="?",
                       help="Top-k dirtiness for the dirtiness map. Default is 100.")
    parser.add_argument("--sensi-expansion", type=int, default=1,
                       help="Expansion factor for the sensitivity map. Default is 1.")
    parser.add_argument("--gt_confid_thresh", type=float, default=0.5)
    parser.add_argument("--mpeg_confid_thresh", type=float, default=0.5)
    parser.add_argument("--max_area_thresh_gt", type=float, default=0.4)
    parser.add_argument("--max_area_thresh_mpeg", type=float, default=0.4)
    parser.add_argument("--iou_thresh", type=float, default=0.5)
    args = parser.parse_args()

    main(args)

    