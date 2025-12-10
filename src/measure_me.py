import cv2
import numpy as np
import os
import glob
import time
import torch
import gc

# --- 1. CPU Dirtiness Map (Original) ---
def create_dirtiness_map_cpu(
    anchor_image: np.ndarray, 
    current_image: np.ndarray,
    block_size: int = 16,
    dmap_type: str = "threshold",
    dirty_thres: int = 30,
    dirty_topk: int = 100,
) -> torch.Tensor:
    # 1. Residual
    residual = cv2.absdiff(anchor_image, current_image)
    dirtiness_map = cv2.cvtColor(residual, cv2.COLOR_BGR2GRAY)

    image_H, image_W = residual.shape[:2]
    
    # 2. Blur 1
    dirtiness_map = cv2.GaussianBlur(dirtiness_map, (15, 15), 1.5)

    if dmap_type == "threshold":
        # 3. Threshold 1
        dirtiness_map = (dirtiness_map > dirty_thres).astype(np.float32)
        # 4. Blur 2
        dirtiness_map = cv2.GaussianBlur(dirtiness_map, (15, 15), 1.5)
        # 5. Resize
        dirtiness_map = cv2.resize(dirtiness_map, (image_W // block_size, image_H // block_size), interpolation=cv2.INTER_LINEAR)
        # 6. Threshold 2
        dirtiness_map = (dirtiness_map > 0).astype(np.float32)

    elif dmap_type == "topk":
        dirtiness_map = cv2.resize(dirtiness_map, (image_W // block_size, image_H // block_size), interpolation=cv2.INTER_AREA)
        flat_map = dirtiness_map.flatten()
        topk_indices = np.argpartition(flat_map, -dirty_topk)[-dirty_topk:]
        topk_values = flat_map[topk_indices]
        threshold = topk_values.min()
        dirtiness_map = (dirtiness_map >= threshold).astype(np.float32)
    
    # 7. To Tensor
    dirtiness_map = torch.from_numpy(dirtiness_map)
    dirtiness_map = dirtiness_map.unsqueeze(0).unsqueeze(-1)

    if dirtiness_map.sum() == 0:
        dirtiness_map[0, 0, 0, 0] = 1

    return dirtiness_map

# --- 2. CUDA Dirtiness Map (Accelerated) ---
def create_dirtiness_map_cuda(
    anchor_gpu, 
    current_gpu,
    block_size=16,
    dmap_type="threshold",
    dirty_thres=30,
    dirty_topk=100
) -> torch.Tensor:
    """
    CUDA accelerated version of create_dirtiness_map.
    Inputs are cv2.cuda_GpuMat objects.
    """
    # 1. Residual (AbsDiff)
    residual_gpu = cv2.cuda.absdiff(anchor_gpu, current_gpu)
    
    # 2. Convert to Grayscale
    dirtiness_gpu = cv2.cuda.cvtColor(residual_gpu, cv2.COLOR_BGR2GRAY)
    
    # 3. Blur 1
    # createGaussianFilter returns a filter object, then apply it
    gauss_filter = cv2.cuda.createGaussianFilter(cv2.CV_8UC1, cv2.CV_8UC1, (15, 15), 1.5)
    dirtiness_gpu = gauss_filter.apply(dirtiness_gpu)

    if dmap_type == "threshold":
        # 4. Threshold 1 (> dirty_thres)
        # cv2.cuda.threshold outputs 0 or maxval (255). Equivalent to (x > thres)
        _, dirtiness_gpu = cv2.cuda.threshold(dirtiness_gpu, dirty_thres, 255, cv2.THRESH_BINARY)
        
        # 5. Blur 2
        dirtiness_gpu = gauss_filter.apply(dirtiness_gpu)
        
        # 6. Resize (Downscale)
        w, h = dirtiness_gpu.size()
        new_w, new_h = w // block_size, h // block_size
        dirtiness_gpu = cv2.cuda.resize(dirtiness_gpu, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # 7. Threshold 2 (> 0)
        # We want 1.0 for True, 0.0 for False.
        # We use 1 as maxval, so output is 0 or 1 (uint8)
        _, dirtiness_gpu = cv2.cuda.threshold(dirtiness_gpu, 0, 1, cv2.THRESH_BINARY)
        
        # 8. Download (Small size now)
        dirtiness_map = dirtiness_gpu.download()
        
        # Convert to float32 (to match CPU output type)
        dirtiness_map = dirtiness_map.astype(np.float32)

    elif dmap_type == "topk":
        # TopK is complex on GPU with OpenCV.
        # Strategy: Resize on GPU -> Download -> TopK on CPU
        w, h = dirtiness_gpu.size()
        new_w, new_h = w // block_size, h // block_size
        dirtiness_gpu = cv2.cuda.resize(dirtiness_gpu, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        dirtiness_map = dirtiness_gpu.download()
        
        # CPU logic for sorting
        flat_map = dirtiness_map.flatten()
        topk_indices = np.argpartition(flat_map, -dirty_topk)[-dirty_topk:]
        topk_values = flat_map[topk_indices]
        threshold = topk_values.min()
        dirtiness_map = (dirtiness_map >= threshold).astype(np.float32)

    # 9. To Tensor
    dirtiness_map = torch.from_numpy(dirtiness_map)
    dirtiness_map = dirtiness_map.unsqueeze(0).unsqueeze(-1)

    if dirtiness_map.sum() == 0:
        dirtiness_map[0, 0, 0, 0] = 1

    return dirtiness_map


# --- CPU Estimate & Process ---
def estimate_affine_cpu(prev_nd, curr_nd):
    MAX_PTS, LK_WIN, QUALITY, DOWNSCALE = 400, (15, 15), 0.01, 0.5
    
    prev_g = cv2.resize(cv2.cvtColor(prev_nd, cv2.COLOR_BGR2GRAY), None, fx=DOWNSCALE, fy=DOWNSCALE, interpolation=cv2.INTER_AREA)
    curr_g = cv2.resize(cv2.cvtColor(curr_nd, cv2.COLOR_BGR2GRAY), None, fx=DOWNSCALE, fy=DOWNSCALE, interpolation=cv2.INTER_AREA)

    p0 = cv2.goodFeaturesToTrack(prev_g, MAX_PTS, QUALITY, 7)
    if p0 is None: return np.eye(2, 3, dtype=np.float32)
    
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_g, curr_g, p0, None, winSize=LK_WIN, maxLevel=3)
    ok = st.squeeze() == 1
    if ok.sum() < 6: return np.eye(2, 3, dtype=np.float32)
    
    T, _ = cv2.estimateAffinePartial2D(p1[ok], p0[ok], method=cv2.LMEDS)
    if T is not None:
        T[0,2] /= DOWNSCALE; T[1,2] /= DOWNSCALE
        return T.astype(np.float32)
    return None

def process_frame_pair_cpu(prev_frame, curr_frame):
    h, w = prev_frame.shape[:2]
    input_img_size = (w, h)
    background_color = (0, 0, 0)

    # 1. Estimate
    t0 = time.perf_counter()
    transform = estimate_affine_cpu(prev_frame, curr_frame)
    t1 = time.perf_counter()
    est_time = t1 - t0

    warp_time = 0.0
    dirty_time = 0.0
    success = False

    if transform is not None:
        # 2. Warp
        t2 = time.perf_counter()
        warped_frame = cv2.warpAffine(prev_frame, transform, dsize=input_img_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=background_color)
        t3 = time.perf_counter()
        warp_time = t3 - t2

        # 3. Dirtiness (CPU)
        t4 = time.perf_counter()
        _ = create_dirtiness_map_cpu(warped_frame, curr_frame)
        t5 = time.perf_counter()
        dirty_time = t5 - t4
        
        success = True
    
    return success, est_time, warp_time, dirty_time


# --- CUDA Estimate & Process ---
def estimate_affine_cuda(prev_gpu, curr_gpu):
    MAX_PTS, LK_WIN, QUALITY, DOWNSCALE, MIN_DIST = 400, (15, 15), 0.01, 0.5, 7

    prev_g_gpu = cv2.cuda.resize(cv2.cuda.cvtColor(prev_gpu, cv2.COLOR_BGR2GRAY), (0, 0), fx=DOWNSCALE, fy=DOWNSCALE, interpolation=cv2.INTER_AREA)
    curr_g_gpu = cv2.cuda.resize(cv2.cuda.cvtColor(curr_gpu, cv2.COLOR_BGR2GRAY), (0, 0), fx=DOWNSCALE, fy=DOWNSCALE, interpolation=cv2.INTER_AREA)

    detector = cv2.cuda.createGoodFeaturesToTrackDetector(cv2.CV_8UC1, MAX_PTS, qualityLevel=QUALITY, minDistance=MIN_DIST)
    p0_gpu = detector.detect(prev_g_gpu)
    if p0_gpu is None or p0_gpu.empty(): return np.eye(2, 3, dtype=np.float32)

    lk_optical_flow = cv2.cuda.SparsePyrLKOpticalFlow_create(winSize=LK_WIN, maxLevel=3)
    p1_gpu, status_gpu, _ = lk_optical_flow.calc(prev_g_gpu, curr_g_gpu, p0_gpu, None)

    p0 = p0_gpu.download().reshape(-1, 2)
    p1 = p1_gpu.download().reshape(-1, 2)
    st = status_gpu.download().reshape(-1)

    ok = st == 1
    if ok.sum() < 6: return np.eye(2, 3, dtype=np.float32)

    T, _ = cv2.estimateAffinePartial2D(p1[ok], p0[ok], method=cv2.LMEDS)
    if T is not None:
        T[0,2] /= DOWNSCALE; T[1,2] /= DOWNSCALE
        return T.astype(np.float32)
    return None

def process_frame_pair_cuda(prev_frame, curr_frame):
    h, w = prev_frame.shape[:2]
    input_img_size = (w, h)
    background_color = (0, 0, 0, 0)

    # 1. Estimate
    t0 = time.perf_counter()
    prev_gpu = cv2.cuda_GpuMat(); prev_gpu.upload(prev_frame)
    curr_gpu = cv2.cuda_GpuMat(); curr_gpu.upload(curr_frame)
    
    transform = estimate_affine_cuda(prev_gpu, curr_gpu)
    t1 = time.perf_counter()
    est_time = t1 - t0

    warp_time = 0.0
    dirty_time = 0.0
    success = False

    if transform is not None:
        # 2. Warp (GPU)
        t2 = time.perf_counter()
        warped_gpu = cv2.cuda.warpAffine(prev_gpu, transform, dsize=input_img_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=background_color)
        cv2.cuda.Stream_Null().waitForCompletion() # Sync
        t3 = time.perf_counter()
        warp_time = t3 - t2

        # 3. Dirtiness (GPU)
        # We pass GpuMats directly to avoid full image download
        t4 = time.perf_counter()
        _ = create_dirtiness_map_cuda(warped_gpu, curr_gpu)
        t5 = time.perf_counter()
        dirty_time = t5 - t4
        
        success = True
    
    return success, est_time, warp_time, dirty_time


# --- Runner ---
def process_video_file(video_path, process_func):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  Error: {video_path}")
        return [], [], [], [], (0, 0)

    # Get resolution
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    resolution = (width, height)

    est_lat, warp_lat, dirty_lat, total_lat = [], [], [], []
    ret, prev_frame = cap.read()
    if not ret: cap.release(); return [], [], [], [], resolution

    # Warmup
    ret_warmup, warmup_frame = cap.read()
    if ret_warmup:
        for _ in range(3): process_func(prev_frame, warmup_frame)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, prev_frame = cap.read()

    while True:
        ret, curr_frame = cap.read()
        if not ret: break 

        success, t_est, t_warp, t_dirty = process_func(prev_frame, curr_frame)

        if success:
            est_lat.append(t_est * 1000)
            warp_lat.append(t_warp * 1000)
            dirty_lat.append(t_dirty * 1000)
            total_lat.append((t_est + t_warp + t_dirty) * 1000)

        prev_frame = curr_frame

    cap.release()
    return est_lat, warp_lat, dirty_lat, total_lat, resolution

def run_benchmark_all_videos(video_folder):
    # Auto-detect CUDA
    use_cuda = False
    try:
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            use_cuda = True
            print(">>> CUDA Mode Enabled")
    except:
        print(">>> CPU Mode Enabled")

    process_func = process_frame_pair_cuda if use_cuda else process_frame_pair_cpu

    video_paths = sorted(glob.glob(os.path.join(video_folder, "*.mp4")))
    if not video_paths: print("No videos found."); return

    print(f"Found {len(video_paths)} videos.")
    print("-" * 90)

    all_est, all_warp, all_dirty, all_total = [], [], [], []

    for i, vid_path in enumerate(video_paths):
        filename = os.path.basename(vid_path)
        print(f"[{i+1}/{len(video_paths)}] {filename} ... ", end="", flush=True)
        
        est, warp, dirty, total, (w, h) = process_video_file(vid_path, process_func)
        
        if total:
            # Resolution added to log
            print(f"({w}x{h}) | Avg: {np.mean(total):.2f} ms")
            all_est.extend(est); all_warp.extend(warp); all_dirty.extend(dirty); all_total.extend(total)
        else:
            print(f"({w}x{h}) | Skipped (No valid pairs)")

    if all_total:
        print("=" * 40)
        print(f"Mode            : {'CUDA' if use_cuda else 'CPU'}")
        print(f"Total Frames    : {len(all_total)}")
        print("-" * 40)
        print(f"{'Metric':<15} | {'Avg (ms)':<10} | {'Min (ms)':<10} | {'Max (ms)':<10}")
        print("-" * 40)
        print(f"{'Estimate':<15} | {np.mean(all_est):<10.4f} | {np.min(all_est):<10.4f} | {np.max(all_est):<10.4f}")
        print(f"{'Warp':<15} | {np.mean(all_warp):<10.4f} | {np.min(all_warp):<10.4f} | {np.max(all_warp):<10.4f}")
        print(f"{'Dirtiness':<15} | {np.mean(all_dirty):<10.4f} | {np.min(all_dirty):<10.4f} | {np.max(all_dirty):<10.4f}")
        print("-" * 40)
        print(f"{'Total':<15} | {np.mean(all_total):<10.4f} | {np.min(all_total):<10.4f} | {np.max(all_total):<10.4f}")
        print("-" * 40)
        print(f"Overall FPS     : {(1000 / np.mean(all_total)):.2f} FPS")
        print("=" * 40)

if __name__ == "__main__":
    # video_dir = "./data/DAVIS2017_trainval/MP4Videos_me"
    video_dir = "./data/imnetvid_val/videos_me"
    if os.path.exists(video_dir):
        run_benchmark_all_videos(video_dir)
    else:
        print(f"Path not found: {video_dir}")