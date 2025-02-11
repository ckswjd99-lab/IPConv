import cv2
import torch
import torchvision
import numpy as np

from constants import COCO_LABELS_LIST, COCO_COLORS_ARRAY, COCO_LABELS_MAP

from torchvision.models.detection import FasterRCNN
from typing import Tuple, List, OrderedDict

@torch.no_grad()
def inference(
    model_detector: FasterRCNN,
    image: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Convert image to tensor
    image_tensor = torchvision.transforms.functional.to_tensor(image).to("cuda").unsqueeze(0)
    original_image_sizes: List[Tuple[int, int]] = [(image.shape[0], image.shape[1])]

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
def visualize_detection(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.9,
) -> np.ndarray:
    for i in range(len(boxes)):
        if scores[i] > threshold:
            x0, y0, x1, y1 = map(int, boxes[i])
            cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(
                image,
                f"{COCO_LABELS_LIST[labels[i]]}: {scores[i]:.2f}",
                (x0, y0 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

    return image


if __name__ == "__main__":
    import os

    image_path = "/data/DAVIS/JPEGImages/480p/bear/00000.jpg"
    image_ndarray = cv2.imread(image_path)
    image_tensor = torchvision.transforms.functional.to_tensor(image_ndarray).to("cuda")

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    ).to("cuda")
    model.eval()

    boxes, labels, scores = inference(model, image_ndarray)

    image_ndarray = visualize_detection(image_ndarray, boxes, labels, scores)

    os.makedirs("output", exist_ok=True)
    cv2.imwrite("output/inference_result.jpg", image_ndarray)