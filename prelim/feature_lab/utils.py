import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def extract_features(model, image_ndarray, device='cuda'):
    # model: torch.nn.Module
    # image_ndarray: numpy.ndarray (loaded with cv2, BGR format)

    # Convert image to tensor
    image_tensor = torch.from_numpy(image_ndarray).float().permute(2, 0, 1)
    batch_images = [image_tensor.to(device)]

    targets = None

    batch_images, targets = model.transform(batch_images, targets)
    raw_features = model.backbone.body(batch_images.tensors)
    features = model.backbone.fpn(raw_features)

    return raw_features, features


def shift_by_float(tensor: torch.Tensor, tvec: tuple[float, float]) -> torch.Tensor:
    """
    Shift a 4D image tensor by a sub-pixel amount using bilinear interpolation.
    
    Args:
        tensor (torch.Tensor): Input image tensor of shape (B, C, H, W)
        tvec (tuple(float, float)): Translation vector (tx, ty) in pixels
    
    Returns:
        torch.Tensor: Shifted image tensor of shape (B, C, H, W)
    """
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


def refine_images(
    ndimage_target: np.ndarray,
    ndimage_source: np.ndarray,
) -> tuple[np.ndarray, tuple[float, float]]:
    gray1 = cv2.cvtColor(ndimage_source, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(ndimage_target, cv2.COLOR_BGR2GRAY)

    # Shi-Tomasi 코너 검출
    corners = cv2.goodFeaturesToTrack(gray1, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    # 루카스-카나데 광류 계산
    p1, st, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, corners, None, winSize=(15, 15), maxLevel=2,
                                           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    # 좋은 점만 선택
    if p1 is not None and st is not None:
        good_new = p1[st == 1]
        good_old = corners[st == 1]

        # RANSAC을 통한 이상치 제거
        matrix, inliers = cv2.estimateAffinePartial2D(good_old, good_new, method=cv2.RANSAC, ransacReprojThreshold=3)
        inlier_old = good_old[inliers.flatten() == 1]
        inlier_new = good_new[inliers.flatten() == 1]

        # 중앙값 기반 평행 이동(Translation Only) 계산
        translation_vector = np.median(inlier_new - inlier_old, axis=0)
        matrix = np.array([[1, 0, translation_vector[0]],
                           [0, 1, translation_vector[1]]], dtype=np.float32)
    
        # 이미지 변환 적용
        aligned_img = cv2.warpAffine(ndimage_source, matrix, (ndimage_source.shape[1], ndimage_source.shape[0]))

        return aligned_img, translation_vector
    
    return ndimage_source, (0, 0)


def generate_recompute_mask(
    ndimage_curr: np.ndarray,
    ndimage_prev: np.ndarray,
    threshold_value: int = 30,
    blur_kernel_size: tuple = (5, 5),
    blur_sigma: float = 1.5,
    grid_size: tuple = (32, 32),
    grid_threshold_ratio: float = 0.05
) -> np.ndarray:
    gray1 = cv2.cvtColor(ndimage_prev, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(ndimage_curr, cv2.COLOR_BGR2GRAY)
    residual = cv2.absdiff(gray1, gray2)

    blurred_residual = cv2.GaussianBlur(residual, blur_kernel_size, blur_sigma)
    mask = blurred_residual > threshold_value

    height, width = mask.shape
    grid_mask = np.zeros_like(mask, dtype=bool)

    for y in range(0, height, grid_size[1]):
        for x in range(0, width, grid_size[0]):
            grid_x_start = x
            grid_y_start = y
            grid_x_end = min(x + grid_size[0], width)
            grid_y_end = min(y + grid_size[1], height)

            grid_area = mask[grid_y_start:grid_y_end, grid_x_start:grid_x_end]
            grid_change_ratio = np.sum(grid_area) / grid_area.size

            if grid_change_ratio > grid_threshold_ratio:
                grid_mask[grid_y_start:grid_y_end, grid_x_start:grid_x_end] = True

    return mask, grid_mask
