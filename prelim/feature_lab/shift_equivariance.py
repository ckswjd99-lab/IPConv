import torchvision
import torch
import os
import numpy as np
import cv2
from tqdm import tqdm

from utils import extract_features, shift_by_float, refine_images, generate_recompute_mask

DATASET_PATH = '/data/DAVIS/JPEGImages/480p/'
VIDEO_NAME = 'bear'
VIDEO_PATH = os.path.join(DATASET_PATH, VIDEO_NAME)
OUTPUT_DIR = f'./output/{VIDEO_NAME}'
DEVICE = 'cuda:3'

frames = sorted(os.listdir(VIDEO_PATH))
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1
).to(DEVICE)
model.eval()

@torch.no_grad()
def main():
    for i in tqdm(range(1, len(frames))):
        max_video_size = 1344, 768

        prev_frame = cv2.resize(cv2.imread(os.path.join(VIDEO_PATH, frames[i - 1])), max_video_size)
        curr_frame = cv2.resize(cv2.imread(os.path.join(VIDEO_PATH, frames[i])), max_video_size)

        prev_frame_refined, translation_vector = refine_images(curr_frame, prev_frame)

        residual_mask, recompute_mask = generate_recompute_mask(
            prev_frame_refined, 
            curr_frame,
            threshold_value=30,
            blur_kernel_size=(5, 5),
            blur_sigma=1.5,
            grid_size=(64, 64),
            grid_threshold_ratio=0.05
        )

        prev_rfeats, prev_feats = extract_features(model, prev_frame, device=DEVICE)
        curr_rfeats, curr_feats = extract_features(model, curr_frame, device=DEVICE)

        for fname in prev_rfeats:
            prev_feat = prev_rfeats[fname]
            curr_feat = curr_rfeats[fname]

            # feature scaler
            original_img_w, original_img_h = prev_frame.shape[1], prev_frame.shape[0]
            feat_w, feat_h = prev_feat.shape[3], prev_feat.shape[2]

            translation_vector_resized = translation_vector * np.array([feat_w / original_img_w, feat_h / original_img_h])

            # calculate difference between features
            diff = torch.abs(prev_feat - curr_feat)
            diff = diff.squeeze(0).mean(dim=0).cpu().numpy()

            # shift features by translation vector
            prev_feat_shifted = shift_by_float(prev_feat, translation_vector_resized)
            diff_shifted = torch.abs(prev_feat_shifted - curr_feat)
            diff_shifted = diff_shifted.squeeze(0).mean(dim=0).cpu().numpy()

            # save difference as grayscale image
            norm_max = max(diff.max(), diff_shifted.max())

            diff = (diff / norm_max * 255).astype('uint8')
            diff_shifted = (diff_shifted / norm_max * 255).astype('uint8')

            # make diff_shifted from 1 channel to 3 channel
            diff_shifted = cv2.merge([diff_shifted, diff_shifted, diff_shifted]).astype('uint8')

            # apply mask
            residual_mask_resized = cv2.resize(residual_mask.astype('uint8') * 255, (feat_w, feat_h), interpolation=cv2.INTER_NEAREST) > 0
            recompute_mask_resized = cv2.resize(recompute_mask.astype('uint8') * 255, (feat_w, feat_h), interpolation=cv2.INTER_NEAREST) > 0

            # diff_shifted[residual_mask_resized, 2] = 255
            diff_shifted[recompute_mask_resized, 1] += 50
            diff_shifted = np.clip(diff_shifted, 0, 255).astype('uint8')

            os.makedirs(os.path.join(OUTPUT_DIR, f'{fname}'), exist_ok=True)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f'{fname}/{i:05d}.png'), diff)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f'{fname}/{i:05d}_shifted.png'), diff_shifted)




if __name__ == '__main__':
    main()