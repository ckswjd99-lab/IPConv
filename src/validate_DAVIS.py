import torchvision
import os

from ipconv.validate import validate_DAVIS

def main():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    ).to("cuda")
    model.eval()

    # sequence_names = os.listdir("/data/DAVIS/JPEGImages/480p")
    # gops = [1, 6, 30, 100]

    sequence_names = ['bear']
    gops = [1, 6, 30, 100]

    for sequence_name in sequence_names:
        # if os.path.exists(f"./output/contexted_inference/{sequence_name}"):
        #     continue
        for gop in gops:
            print(f"Processing sequence: {sequence_name}, GOP: {gop}")
            avg_compute_rate, avg_iou, inference_results = validate_DAVIS(model, sequence_name, gop)
            print(f"  - Average recompute rate: {avg_compute_rate}")
            print(f"  - Average IoU: {avg_iou}")
            print()
        

if __name__ == "__main__":
    main()
        