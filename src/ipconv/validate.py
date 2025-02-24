import torch, torchvision
import os, cv2, json
import numpy as np

from tqdm import tqdm

from .inference import inference_contexted, visualize_detection
from .proc_image import shift_features_dict, refine_images
from .constants import COCO_LABELS_LIST


def create_dirtiness_map(
    anchor_image: np.ndarray, 
    current_image: np.ndarray,
    block_size: int = 16,
    dirty_thres: int = 30
) -> torch.Tensor:
    residual = cv2.absdiff(anchor_image, current_image)
    dirtiness_map = cv2.cvtColor(residual, cv2.COLOR_BGR2GRAY)

    image_H, image_W = residual.shape[:2]
    
    dirtiness_map = cv2.GaussianBlur(dirtiness_map, (7, 7), 1.5)
    dirtiness_map = (dirtiness_map > dirty_thres).astype(np.float32)

    dirtiness_map = cv2.GaussianBlur(dirtiness_map, (15, 15), 1.5)
    dirtiness_map = cv2.resize(dirtiness_map, (image_W // block_size, image_H // block_size), interpolation=cv2.INTER_LINEAR)
    dirtiness_map = (dirtiness_map > 0).astype(np.float32)
    
    dirtiness_map = torch.tensor(dirtiness_map).unsqueeze(0).unsqueeze(0).to("cuda")

    return dirtiness_map


def calculate_iou(target_box, infer_box):
    x1_gt, y1_gt, x2_gt, y2_gt = target_box
    x1, y1, x2, y2 = infer_box

    xA = max(x1, x1_gt)
    yA = max(y1, y1_gt)
    xB = min(x2, x2_gt)
    yB = min(y2, y2_gt)

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (x2 - x1 + 1) * (y2 - y1 + 1)
    boxBArea = (x2_gt - x1_gt + 1) * (y2_gt - y1_gt + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou


def calculate_multi_iou(target_boxes, target_labels, infer_boxes, infer_labels):
    iou_results = []
    for target_box, target_label in zip(target_boxes, target_labels):
        iou_list = [calculate_iou(target_box, infer_box) for infer_box, infer_label in zip(infer_boxes, infer_labels) if target_label == infer_label or target_label == -1]
        iou_results.append(max(iou_list) if len(iou_list) > 0 else 0)

    return iou_results


def validate_DAVIS(model, sequence_name, gop, data_root="/data/DAVIS", output_root="./output", leave=False):
    model.eval()

    sequence_path = os.path.join(data_root, "JPEGImages/480p", sequence_name)
    frames = sorted(os.listdir(sequence_path))

    annotations_path = os.path.join(data_root, "Annotations_bbox/480p", f"{sequence_name}.json")
    with open(annotations_path, "r") as f:
        annotations = json.load(f)

    output_path = os.path.join(output_root, "contexted_inference", sequence_name)
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, "temp"), exist_ok=True)

    anchor_image = None
    anchor_features_dict = None

    compute_rates = []
    inference_results = {}
    IoU_results = []

    pbar = tqdm(range(len(frames)), leave=leave)
    for i in pbar:
        basename = os.path.splitext(frames[i])[0]
        target_image = cv2.imread(os.path.join(sequence_path, frames[i]))
        annotation = annotations.get(basename, [])  # List of bounding boxes, each box is in a format of {'x_min': 431, 'y_min': 230, 'x_max': 460, 'y_max': 260, 'label': '14'}

        boxes_gt = [[float(box['x_min']), float(box['y_min']), float(box['x_max']), float(box['y_max'])] for box in annotation]
        labels_gt = [-1 for box in annotation]
        scores_gt = [1.0 for _ in annotation]

        try:
            if i % gop == 0:
                (boxes, labels, scores), features = inference_contexted(model, target_image)

                anchor_image = target_image
                anchor_features_dict = features

                recompute_rate = 1
            else:
                aligned_image, shift_vector = refine_images(anchor_image, target_image)
                aligned_features = shift_features_dict(
                    anchor_features_dict, 
                    (aligned_image.shape[1], aligned_image.shape[0]), 
                    shift_vector
                )

                dirtiness_map = create_dirtiness_map(anchor_image, target_image, block_size=16, dirty_thres=30)
                recompute_rate = torch.mean(dirtiness_map).item()

                (boxes, labels, scores), features = inference_contexted(model, target_image, aligned_features, dirtiness_map)
                inference_results[basename] = (boxes, labels, scores)

                dirtiness_map = torch.nn.functional.interpolate(dirtiness_map, size=target_image.shape[:2], mode='nearest')
                dirtiness_map = dirtiness_map.squeeze(0).squeeze(0).cpu().numpy()
                dirtiness_map = np.stack([dirtiness_map] * 3, axis=-1)
                
                anchor_image = (target_image * dirtiness_map + aligned_image * (1 - dirtiness_map)).astype(np.uint8)
                anchor_features_dict = features
        except:
            recompute_rate = 1
            boxes, labels, scores = [], [], []
            inference_results[basename] = (boxes, labels, scores)

        # Calculate IoU of the boxes
        iou = np.mean(calculate_multi_iou(boxes_gt, labels_gt, boxes, labels))
        IoU_results.append(iou if iou > 0 else 0)

        compute_rates.append(recompute_rate)
        pbar.set_description(f"Processing {basename}, recomp: {recompute_rate:.3f}, IoU: {iou:.3f}")

        image_bbox_gt = visualize_detection(target_image, boxes_gt, labels_gt, scores_gt, colors=np.array([[0, 0, 255] for _ in range(len(COCO_LABELS_LIST))]))
        image_bbox = visualize_detection(image_bbox_gt, boxes, labels, scores)
        cv2.imwrite(os.path.join(output_path, "temp", f"{basename}.jpg"), image_bbox)

    # Make video of the results
    avg_compute_rate = np.mean(compute_rates)
    avg_iou = np.mean(IoU_results)

    video_path = os.path.join(output_root, f"contexted_inference/{sequence_name}", f"gop{gop}_rcr{int(avg_compute_rate*1e4):04d}_iou{int(avg_iou*1e4):04d}.mp4")
    os.system(f"ffmpeg -y -r 10 -i {output_path}/temp/%05d.jpg -c:v libx264 -vf fps=25 -pix_fmt yuv420p {video_path} > /dev/null 2>&1")
    os.system(f"rm -rf {output_path}/temp")


    return avg_compute_rate, avg_iou, inference_results


def main():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    ).to("cuda")
    model.eval()

    sequence_name = "sheep"
    gop = 1

    avg_compute_rate, avg_iou, inference_results = validate_DAVIS(model, sequence_name, gop)
    print(f"Image sequence: {sequence_name}")
    print(f"GOP: {gop}")
    print(f"Average recompute rate: {avg_compute_rate}")
    print(f"Average IoU: {avg_iou}")
        

if __name__ == "__main__":
    main()