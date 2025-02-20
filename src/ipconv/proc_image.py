import torch
import torchvision
import numpy as np
import cv2

from torch.nn import functional as F

from typing import Tuple, Dict


def apply_dirtiness_map(fname, feature, cache_features, dirtiness_map: torch.Tensor):
    if fname in cache_features:
        dirtiness_map_temp = torch.nn.functional.interpolate(dirtiness_map.clone(), size=feature.shape[-2:], mode='nearest')
        feature_new = cache_features[fname] * (1 - dirtiness_map_temp) + feature * dirtiness_map_temp

        feature = feature_new

    return feature, dirtiness_map


def shift_by_float(tensor: torch.Tensor, tvec: tuple[float, float]) -> torch.Tensor:
    B, C, H, W = tensor.shape
    tx, ty = tvec
    
    # Normalize translation to [-1, 1] range (grid_sample expects normalized coordinates)
    tx /= W / 2
    ty /= H / 2
    
    # Construct affine transformation matrix for translation
    theta = torch.tensor([[1, 0, -tx], [0, 1, -ty]], dtype=torch.float, device=tensor.device)
    theta = theta.unsqueeze(0).expand(B, -1, -1)  # Expand for batch processing
    
    # Generate sampling grid
    grid = F.affine_grid(theta, [B, C, H, W], align_corners=True)
    
    # Perform grid sampling (bilinear interpolation)
    shifted_tensor = F.grid_sample(tensor, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    
    return shifted_tensor


def shift_features_dict(
    features: Dict[str, torch.Tensor],
    ref_size: Tuple[int, int],
    shift_vector: Tuple[float, float],
) -> Dict[str, torch.Tensor]:
    shifted_features = {}
    for fname, feature in features.items():
        if feature is not None:
            scaled_shift_vector = (shift_vector[0] * feature.shape[-1] / ref_size[0],
                                   shift_vector[1] * feature.shape[-2] / ref_size[1])
            shifted_features[fname] = shift_by_float(feature, scaled_shift_vector)
        else:
            shifted_features[fname] = None

    return shifted_features


def refine_images(
    anchor_image_ndarray: np.ndarray,
    target_image_ndarray: np.ndarray,
) -> Tuple[np.ndarray, Tuple[float, float]]:

    gray1 = cv2.cvtColor(anchor_image_ndarray, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(target_image_ndarray, cv2.COLOR_BGR2GRAY)

    # Shi-Tomasi corner detection
    corners = cv2.goodFeaturesToTrack(gray1, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    # Lucas-Kanade optical flow calculation
    p1, st, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, corners, None, winSize=(15, 15), maxLevel=2,
                                           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    # Select good points only
    if p1 is not None and st is not None:
        good_new = p1[st == 1]
        good_old = corners[st == 1]

        # Outlier removal using RANSAC
        matrix, inliers = cv2.estimateAffinePartial2D(good_old, good_new, method=cv2.RANSAC, ransacReprojThreshold=3)
        inlier_old = good_old[inliers.flatten() == 1]
        inlier_new = good_new[inliers.flatten() == 1]

        # Calculate translation vector based on median (Translation Only)
        # translation_vector = np.median(inlier_new - inlier_old, axis=0)
        translation_vector = np.mean(inlier_new - inlier_old, axis=0)
        matrix = np.array([[1, 0, translation_vector[0]],
                           [0, 1, translation_vector[1]]], dtype=np.float32)

        # Apply image transformation
        aligned_image = cv2.warpAffine(anchor_image_ndarray, matrix, (anchor_image_ndarray.shape[1], anchor_image_ndarray.shape[0]))

        return aligned_image, tuple(translation_vector)
    else:
        return np.zeros_like(anchor_image_ndarray), (0, 0)
    

if __name__ == "__main__":
    import os

    # DEMO 1: Test refine_images
    anchor_image = cv2.imread('/data/DAVIS/JPEGImages/480p/bear/00000.jpg')
    target_image = cv2.imread('/data/DAVIS/JPEGImages/480p/bear/00001.jpg')

    refined_image, translation_vector = refine_images(anchor_image, target_image)
    print(translation_vector)

    residual = cv2.absdiff(target_image, refined_image)
    residual = cv2.cvtColor(residual, cv2.COLOR_BGR2GRAY)

    os.makedirs('output', exist_ok=True)
    cv2.imwrite('output/residual.jpg', residual)