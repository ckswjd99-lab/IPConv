from ipconv.models.ViTDet.cascade_mask_rcnn_swin_b import CascadeMaskRCNN_Swin_B_Contexted

import torch
import cv2
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def main():

    model = CascadeMaskRCNN_Swin_B_Contexted().to(DEVICE)
    model.eval()
    model.load_weight("./ipconv/models/model_final_246a82.pkl")

    print(model)

    # constants
    COCO_LABELS_LIST = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
        'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    np.random.seed(42)
    COCO_COLORS_ARRAY = np.random.randint(256, size=(91, 3)) / 255
    COCO_LABELS_MAP = {k: v for v, k in enumerate(COCO_LABELS_LIST)}

    # load image
    image_path = "/data/DAVIS/JPEGImages/480p/bear/00000.jpg"
    image_ndarray = cv2.imread(image_path)

    boxes, labels, scores = model(image_ndarray)

    # visualize
    def visualize(image, boxes, labels, scores):
        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = box
            color = COCO_COLORS_ARRAY[label]
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color.tolist(), 2)
            cv2.putText(image, f"{COCO_LABELS_LIST[label]}: {score:.2f}", (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 2)
        return image

    image_ndarray = visualize(image_ndarray, boxes, labels, scores)

    cv2.imwrite("output.jpg", image_ndarray)


if __name__ == "__main__":
    main()