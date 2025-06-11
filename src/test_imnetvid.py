from datasets.vid import VIDResize, VID
from pathlib import Path
from torch.utils.data import DataLoader

from ipconv.models import MaskedRCNN_ViT_B_FPN_Contexted

from torchmetrics.detection.mean_ap import MeanAveragePrecision

import torch
import cv2
import numpy as np
import os

from tqdm import tqdm


def dict_to_device(x, device):
    return {key: value.to(device) for key, value in x.items()}

def squeeze_dict(x, dim=None):
    return {key: value.squeeze(dim=dim) for key, value in x.items()}


@torch.no_grad()
def main():
    device = "cuda"

    model = MaskedRCNN_ViT_B_FPN_Contexted(device)
    model.load_weight("./ipconv/models/model_final_61ccd1.pkl")
    model.eval()

    long_edge = 1024 - 128
    dataset_full = VID(
        Path("data", "vid"),
        split="vid_val",
        tar_path=Path("data", "vid", "data.tar"),
        combined_transform=VIDResize(
            short_edge_length=640 * long_edge // 1024, max_size=long_edge
        ),
    )

    outputs = []
    labels = []

    bomb_time = 1
    for seq_idx, sequence in enumerate(dataset_full):
        seq_dataset = DataLoader(sequence, batch_size=1)
        
        pbar = tqdm(enumerate(seq_dataset), desc=f"Seq {seq_idx}/{len(dataset_full)}", leave=False, total=len(seq_dataset))
        for fidx, (frame, annotations) in pbar:
            frame_ndarray = frame.squeeze(0).permute(1, 2, 0).numpy()

            frame_ndarray = cv2.cvtColor(frame_ndarray, cv2.COLOR_BGR2RGB)
            
            boxes_pred, labels_pred, scores_pred = model(frame_ndarray)
            output = {
                'boxes': torch.tensor(boxes_pred, device=device),
                'labels': torch.tensor(labels_pred, device=device).fill_(0),
                'scores': torch.tensor(scores_pred, device=device)
            }

            annotations['labels'] = torch.tensor(np.zeros_like(annotations['labels']), device=device)

            pbar.set_postfix(num_boxes=len(output['boxes']))

            outputs.append(output)
            labels.append(squeeze_dict(dict_to_device(annotations, device), dim=0))

            # visualize annotations
            for box, label in zip(annotations['boxes'][0], annotations['labels'][0]):
                cv2.rectangle(frame_ndarray, 
                              (int(box[0].item()), int(box[1].item())), 
                              (int(box[2].item()), int(box[3].item())), 
                              (255, 0, 0), 2)
                cv2.putText(frame_ndarray, f"{label.item()}", 
                            (int(box[0].item()), int(box[1].item()) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # visualize bbox with score > 0.5
            for box, label, score in zip(output['boxes'], output['labels'], output['scores']):
                if score > 0.5:
                    cv2.rectangle(frame_ndarray, 
                                  (int(box[0]), int(box[1])), 
                                  (int(box[2]), int(box[3])), 
                                  (0, 255, 0), 2)
                    cv2.putText(frame_ndarray, f"{label.item()} {score.item():.2f}", 
                                (int(box[0]), int(box[1]) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imwrite(f"temp/frame{fidx:03d}.jpg", frame_ndarray)

        # make jpg into video
        if not os.path.exists("output/imnet_vid"):
            os.makedirs("output/imnet_vid")
        os.system(f'ffmpeg -framerate 30 -i temp/frame%03d.jpg -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -pix_fmt yuv420p output/imnet_vid/seq{seq_idx}.mp4 -y')

        # clean up temp directory
        for f in os.listdir("temp"):
            os.remove(os.path.join("temp", f))

        bomb_time -= 1
        if bomb_time <= 0:
            break
    
    mean_ap = MeanAveragePrecision(iou_type="bbox", iou_thresholds=None)
    mean_ap.update(outputs, labels)
    stats = mean_ap.compute()

    print(f"mAP@[0.5:0.95]: {stats['map']*100:.1f}")
    print(f"AP50         : {stats['map_50']*100:.1f}")
    print(f"AP75         : {stats['map_75']*100:.1f}")


    print("Metrics:", stats)
    


if __name__ == "__main__":
    main()
