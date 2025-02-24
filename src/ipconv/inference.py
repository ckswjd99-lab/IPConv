import cv2
import torch
import torchvision
import numpy as np

from torchvision.models.detection import FasterRCNN
from typing import Tuple, List, OrderedDict, Dict
from collections.abc import Callable

from .constants import COCO_LABELS_LIST, COCO_COLORS_ARRAY, COCO_LABELS_MAP
from .proc_image import apply_dirtiness_map, refine_images, shift_features_dict


@torch.no_grad()
def inference(
    model_detector: FasterRCNN,
    image_ndarray: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Convert image to tensor
    image_tensor = torchvision.transforms.functional.to_tensor(image_ndarray).to("cuda").unsqueeze(0)
    original_image_sizes: List[Tuple[int, int]] = [(image_ndarray.shape[0], image_ndarray.shape[1])]

    # Preprocess the input image
    images, _ = model_detector.transform(image_tensor, None)

    # Extract features
    x = images.tensors

    # Get the outputs of all layers
    conv1_out = model_detector.backbone.body.conv1(x)
    conv1_out = model_detector.backbone.body.bn1(conv1_out)
    conv1_out = model_detector.backbone.body.relu(conv1_out)
    conv1_out = model_detector.backbone.body.maxpool(conv1_out)

    layer1_out = model_detector.backbone.body.layer1(conv1_out)
    layer2_out = model_detector.backbone.body.layer2(layer1_out)
    layer3_out = model_detector.backbone.body.layer3(layer2_out)
    layer4_out = model_detector.backbone.body.layer4(layer3_out)

    # Use all layer outputs as features
    raw_features = OrderedDict([
        ("0", layer1_out),
        ("1", layer2_out),
        ("2", layer3_out),
        ("3", layer4_out),
    ])

    features = model_detector.backbone.fpn(raw_features)
    
    if isinstance(features, torch.Tensor):
        features = OrderedDict([("0", features)])  # Handle single tensor case

    # Detect objects
    proposals, _ = model_detector.rpn(images, features, None)  # targets are None for inference
    detections, _ = model_detector.roi_heads(features, proposals, images.image_sizes, None)  # targets are None for inference

    detections = model_detector.transform.postprocess(detections, images.image_sizes, original_image_sizes)

    prediction = detections[0]  # Get the prediction for the first image in the batch

    # Extract boxes, labels, and scores
    boxes = prediction["boxes"].cpu().numpy()
    labels = prediction["labels"].cpu().numpy()
    scores = prediction["scores"].cpu().numpy()

    return boxes, labels, scores


@torch.no_grad()
def inference_contexted(
    model_detector: FasterRCNN,
    image_ndarray: np.ndarray,
    cache_features: Dict[str, torch.Tensor] = {},
    dirtiness_map: torch.Tensor = torch.zeros(1, 1, 1, 1).to("cuda"),
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Dict[str, torch.Tensor]]:
    new_cache_features = {}

    # Convert image to tensor
    image_tensor = torchvision.transforms.functional.to_tensor(image_ndarray).to("cuda").unsqueeze(0)
    original_image_sizes: List[Tuple[int, int]] = [(image_ndarray.shape[0], image_ndarray.shape[1])]

    # Preprocess the input image
    images, _ = model_detector.transform(image_tensor, None)

    # Extract features
    x = images.tensors

    # Get the outputs of all layers
    fname = "input"
    new_cache_features[fname] = x
    x, dirtiness_map = apply_dirtiness_map(fname, x, cache_features, dirtiness_map)

    conv1_out = model_detector.backbone.body.conv1(x)
    conv1_out = model_detector.backbone.body.bn1(conv1_out)
    conv1_out = model_detector.backbone.body.relu(conv1_out)
    conv1_out = model_detector.backbone.body.maxpool(conv1_out)
    fname = "conv1"
    new_cache_features[fname] = conv1_out
    conv1_out, dirtiness_map = apply_dirtiness_map(fname, conv1_out, cache_features, dirtiness_map)

    layer1_out = model_detector.backbone.body.layer1(conv1_out)
    fname = "layer1"
    new_cache_features[fname] = layer1_out
    layer1_out, dirtiness_map = apply_dirtiness_map(fname, layer1_out, cache_features, dirtiness_map)

    layer2_out = model_detector.backbone.body.layer2(layer1_out)
    fname = "layer2"
    new_cache_features[fname] = layer2_out
    layer2_out, dirtiness_map = apply_dirtiness_map(fname, layer2_out, cache_features, dirtiness_map)

    layer3_out = model_detector.backbone.body.layer3(layer2_out)
    fname = "layer3"
    new_cache_features[fname] = layer3_out
    layer3_out, dirtiness_map = apply_dirtiness_map(fname, layer3_out, cache_features, dirtiness_map)

    layer4_out = model_detector.backbone.body.layer4(layer3_out)
    fname = "layer4"
    new_cache_features[fname] = layer4_out
    layer4_out, dirtiness_map = apply_dirtiness_map(fname, layer4_out, cache_features, dirtiness_map)

    # Use all layer outputs as features
    raw_features = OrderedDict([
        ("0", layer1_out),
        ("1", layer2_out),
        ("2", layer3_out),
        ("3", layer4_out),
    ])

    features = model_detector.backbone.fpn(raw_features)
    
    if isinstance(features, torch.Tensor):
        features = OrderedDict([("0", features)])  # Handle single tensor case

    # Detect objects
    proposals, _ = model_detector.rpn(images, features, None)  # targets are None for inference
    detections, _ = model_detector.roi_heads(features, proposals, images.image_sizes, None)  # targets are None for inference

    detections = model_detector.transform.postprocess(detections, images.image_sizes, original_image_sizes)

    prediction = detections[0]  # Get the prediction for the first image in the batch

    # Extract boxes, labels, and scores
    boxes = prediction["boxes"].cpu().numpy()
    labels = prediction["labels"].cpu().numpy()
    scores = prediction["scores"].cpu().numpy()

    return (boxes, labels, scores), new_cache_features


@torch.no_grad()
def visualize_detection(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.9,
    colors: np.ndarray = COCO_COLORS_ARRAY,
) -> np.ndarray:
    for i in range(len(boxes)):
        if scores[i] > threshold:
            color = colors[labels[i]]
            x0, y0, x1, y1 = map(int, boxes[i])
            cv2.rectangle(image, (x0, y0), (x1, y1), (color * 255).astype(int).tolist(), 2)
            if labels[i] != -1:
                cv2.putText(
                    image,
                    f"{COCO_LABELS_LIST[labels[i]]}: {scores[i]:.2f}",
                    (x0, y0 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (color * 255).astype(int).tolist(),
                    2,
                    cv2.LINE_AA,
                )

    return image


@torch.no_grad()
def features_absdiff(
    features1: Dict[str, torch.Tensor],
    features2: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    return {k: torch.abs(features1[k] - features2[k]) for k in features1.keys()}


@torch.no_grad()
def visualize_features(
    features: Dict[str, torch.Tensor]
) -> Dict[str, np.ndarray]:
    ret_val = {}
    for k, v in features.items():
        v = v.mean(dim=1).squeeze().cpu().numpy()
        if v.max() > 0:
            v = v / v.max()
            v = (v * 255).astype(np.uint8)
        ret_val[k] = v

    return ret_val
    

if __name__ == "__main__":
    import os

    demo_output_dir = "output/inference"
    os.makedirs(demo_output_dir, exist_ok=True)
    os.makedirs(os.path.join(demo_output_dir, "features_diff"), exist_ok=True)

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    ).to("cuda")
    model.eval()

    # DEMO 1: Simple Inference
    image_path = "/data/DAVIS/JPEGImages/480p/bear/00000.jpg"
    image_ndarray = cv2.imread(image_path)
    image_tensor = torchvision.transforms.functional.to_tensor(image_ndarray).to("cuda")

    boxes, labels, scores = inference(model, image_ndarray)

    image_ndarray = visualize_detection(image_ndarray, boxes, labels, scores)

    cv2.imwrite(os.path.join(demo_output_dir, "inference_result.jpg"), image_ndarray)

    # DEMO 2: Contexted Inference
    sequence = "bear"
    image1_path = f"/data/DAVIS/JPEGImages/480p/{sequence}/00000.jpg"
    image2_path = f"/data/DAVIS/JPEGImages/480p/{sequence}/00010.jpg"

    image1_ndarray = cv2.imread(image1_path)
    image2_ndarray = cv2.imread(image2_path)
    
    image_W, image_H = image1_ndarray.shape[1], image1_ndarray.shape[0]

    image1_affined, tvec = refine_images(image1_ndarray, image2_ndarray)
    residual = cv2.absdiff(image2_ndarray, image1_affined)
    residual = cv2.cvtColor(residual, cv2.COLOR_BGR2GRAY)
    
    block_size = 16
    dirty_thres = 15
    dirtiness_map = residual

    dirtiness_map = cv2.GaussianBlur(dirtiness_map, (7, 7), 1.5)
    dirtiness_map = (dirtiness_map > dirty_thres).astype(np.float32)

    dirtiness_map = cv2.GaussianBlur(dirtiness_map, (15, 15), 1.5)
    dirtiness_map = cv2.resize(dirtiness_map, (image_W // block_size, image_H // block_size), interpolation=cv2.INTER_LINEAR)
    dirtiness_map = (dirtiness_map > 0).astype(np.float32)
    
    dirtiness_map = torch.tensor(dirtiness_map).unsqueeze(0).unsqueeze(0).to("cuda")
    
    recompute_percentage = dirtiness_map.sum().item() / dirtiness_map.numel()
    print(f"Recompute Percentage: {recompute_percentage * 100:.2f}%")

    image1_tensor = torchvision.transforms.functional.to_tensor(image1_ndarray).to("cuda")
    image2_tensor = torchvision.transforms.functional.to_tensor(image2_ndarray).to("cuda")


    (boxes1, labels1, scores1), cache_features_1 = inference_contexted(model, image1_ndarray)
    
    image1_ndarray = visualize_detection(image1_ndarray, boxes1, labels1, scores1)
    cv2.imwrite(os.path.join(demo_output_dir, "contexted_inference_result1.jpg"), image1_ndarray)

    cache_features_1 = shift_features_dict(cache_features_1, (image_W, image_H), tvec)
    (boxes2, labels2, scores2), cache_features_2_contexted = inference_contexted(model, image2_ndarray, cache_features_1, dirtiness_map)
    
    (boxes2_gt, labels2_gt, scores2_gt), cache_features_2 = inference_contexted(model, image2_ndarray)

    # print the boxes, labels, and scores
    print("[Ground Truth]")
    for i in range(len(boxes2_gt)):
        if scores2_gt[i] > 0.9:
            print(f"{COCO_LABELS_LIST[labels2_gt[i]]} {scores2_gt[i]:.2f}: \t{boxes2_gt[i]}")
    print()
    print("[Feature Reused]")
    for i in range(len(boxes2)):
        if scores2[i] > 0.9:
            print(f"{COCO_LABELS_LIST[labels2[i]]} {scores2[i]:.2f}: \t{boxes2[i]}")

    image2_ndarray = visualize_detection(image2_ndarray, boxes2, labels2, scores2)
    image2_gt_ndarray = visualize_detection(image2_ndarray.copy(), boxes2_gt, labels2_gt, scores2_gt, colors=np.array([[0, 0, 255] for _ in range(len(COCO_LABELS_LIST))]))

    features_diff = features_absdiff(cache_features_2, cache_features_2_contexted)
    features_diff = visualize_features(features_diff)

    cv2.imwrite(os.path.join(demo_output_dir, "contexted_inference_result2.jpg"), image2_gt_ndarray)
    
    dirtiness_map = dirtiness_map.squeeze().cpu().numpy()
    dirtiness_map = cv2.resize(dirtiness_map, (image_W, image_H), interpolation=cv2.INTER_NEAREST)
    residual = cv2.cvtColor(residual, cv2.COLOR_GRAY2BGR).astype(np.uint16)
    residual[..., 1] = np.clip(residual[..., 1] + dirtiness_map * 30, 0, 255)
    residual = residual.astype(np.uint8)
    cv2.imwrite(os.path.join(demo_output_dir, "residual.jpg"), residual)
    
    for k, v in features_diff.items():
        v = v.astype(np.uint8)
        cv2.imwrite(os.path.join(demo_output_dir, "features_diff", f"{k}.jpg"), v)
    


