import cv2
import numpy as np
import os

IMAGE_PATHS = [
    'data/DAVIS/JPEGImages/480p/bmx-bumps/00000.jpg',
    'data/DAVIS/JPEGImages/480p/bmx-bumps/00001.jpg'
]

def naive_residuals(
    image1: np.ndarray,
    image2: np.ndarray,
):
    """
    Compute the naive residuals between two images.
    
    Args:
        image1 (np.ndarray): First image.
        image2 (np.ndarray): Second image.
    
    Returns:
        np.ndarray: The absolute difference between the two images.
    """
    diff = cv2.absdiff(image1, image2)
    diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    return diff

def estimate_affine(img1, img2):
    orb = cv2.ORB_create(5000)
    k1,d1 = orb.detectAndCompute(img1,None)
    k2,d2 = orb.detectAndCompute(img2,None)
    if d1 is None or d2 is None: return None
    matches = sorted(cv2.BFMatcher(
        cv2.NORM_HAMMING, crossCheck=True).match(d1,d2),
        key=lambda m:m.distance)[:300]
    if len(matches) < 6: return None
    src = np.float32([k1[m.queryIdx].pt for m in matches])
    dst = np.float32([k2[m.trainIdx].pt for m in matches])
    M,_ = cv2.estimateAffinePartial2D(src,dst,method=cv2.LMEDS)
    return M                              # 2×3

def true_residuals(
    image_curr: np.ndarray,
    image_prev: np.ndarray,
):
    """
    Compute the motion compensated residuals between two images.
    
    Args:
        image_curr (np.ndarray): Current image.
        image_prev (np.ndarray): Previous image.
    
    Returns:
        np.ndarray: Motion compensated residuals between the two images.
    """
    # Place image_curr in a canvas
    CANVAS_SIZE = (1024, 1024)
    canvas = np.zeros((CANVAS_SIZE[0], CANVAS_SIZE[1], 3), dtype=np.uint8)
    h, w = image_curr.shape[:2]
    canvas[:h, :w] = image_curr
    
    # Detect keypoints and compute descriptors using ORB
    # orb = cv2.ORB_create()
    # kp1, des1 = orb.detectAndCompute(image_prev, None)
    # kp2, des2 = orb.detectAndCompute(canvas, None)
    
    # # Match features using BFMatcher
    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # matches = bf.match(des1, des2)
    # matches = sorted(matches, key=lambda x: x.distance)

    # # Select good matches
    # good_matches = matches[:min(len(matches), 1000)]

    # # Extract coordinates of matching points
    # src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    # dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # # Calculate Affine Transform
    # affine_matrix, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, cv2.LMEDS, maxIters=5000, confidence=0.999, refineIters=10)

    affine_matrix = estimate_affine(image_prev, canvas)

    # Apply the transformation to the previous image
    prev_affined = cv2.warpAffine(image_prev, affine_matrix, (canvas.shape[1], canvas.shape[0]))[:h, :w]
    
    # Compute the residuals
    residuals = cv2.absdiff(image_curr, prev_affined)
    residuals = cv2.cvtColor(residuals, cv2.COLOR_BGR2GRAY)
    
    return residuals


def main():
    # Load images
    image1 = cv2.imread(IMAGE_PATHS[0])
    image2 = cv2.imread(IMAGE_PATHS[1])
    
    if image1 is None or image2 is None:
        print("Error loading images.")
        return
    
    # Compute naive residuals
    naive_res = naive_residuals(image1, image2)
    
    # Compute true residuals
    true_res = true_residuals(image2, image1)
    
    # Save the results
    os.makedirs('figs', exist_ok=True)
    cv2.imwrite('figs/naive_residuals.jpg', naive_res)
    cv2.imwrite('figs/true_residuals.jpg', true_res)


if __name__ == "__main__":
    main()