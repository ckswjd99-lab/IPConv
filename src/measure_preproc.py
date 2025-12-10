import subprocess
import numpy as np
import cv2
import time
import io
import csv
import os
from typing import Optional, Tuple

# =========================================================================
# 1. Affine 추정 로직 (Python/NumPy)
# =========================================================================

def rigid_from_mvs(mvs: np.ndarray) -> Optional[np.ndarray]:
    """
    모션 벡터 배열(N, 5)을 기반으로 2x3 Affine Partial (rigid) 변환을 추정합니다.
    
    mvs columns: [dst_x, dst_y, motion_x, motion_y, motion_scale]
    """
    if mvs is None or mvs.shape[0] < 3 or mvs.shape[1] != 5:
        # MV가 너무 적거나 형식이 잘못된 경우
        return None
        
    # dst: 현재 프레임의 MV 위치 (Col 0, 1)
    dst = mvs[:, 0:2].astype(np.float32) 
    
    # src: 참조 프레임의 위치 = dst + (motion / scale)
    # Col 2, 3: motion_x, motion_y / Col 4: motion_scale
    src_motion = mvs[:, 2:4].astype(np.float32) / mvs[:, 4:5].astype(np.float32)
    src = dst + src_motion
    
    if dst.shape[0] < 3 or src.shape[0] < 3:
        return None
        
    # RANSAC을 사용하여 변환 행렬 추정
    # estimateAffinePartial2D는 rigid (회전 + 이동) 변환을 추정합니다.
    M, _ = cv2.estimateAffinePartial2D(dst, src,
                                       method=cv2.RANSAC,
                                       ransacReprojThreshold=2.0,
                                       confidence=0.995,
                                       refineIters=10)
    return M # shape (2,3) or None

# =========================================================================
# 2. 실행 및 데이터 파싱 로직
# =========================================================================

def run_mv_extractor(video_path: str) -> Tuple[np.ndarray, float]:
    """
    외부 바이너리를 실행하고 모션 벡터를 추출합니다.

    :param video_path: 인코딩된 비디오 파일 경로.
    :return: (모션 벡터 배열 (N x 5), 바이너리 실행 시간(초))
    """
    
    # NOTE: 바이너리가 현재 디렉토리 또는 PATH에 있다고 가정
    cmd = ["./custom_codec/extract_mvs", video_path]
    
    start_time = time.perf_counter()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=300)
        
    except subprocess.CalledProcessError as e:
        # 바이너리 실행 실패 (파일 없음, 권한 없음 등)
        print(f"  [ERROR] Binary failed with code {e.returncode}. Stderr: {e.stderr.strip()}")
        return np.zeros((0, 5), dtype=np.int32), 0.0
    except subprocess.TimeoutExpired:
        print("  [ERROR] MV Extraction timed out (over 300s).")
        return np.zeros((0, 5), dtype=np.int32), 0.0

    extractor_time = time.perf_counter() - start_time
    
    # CSV 파싱 (바이너리는 표준 출력(stdout)으로 CSV를 내보냅니다)
    csv_data = io.StringIO(result.stdout)
    reader = csv.reader(csv_data, delimiter=',')
    
    # 헤더 행 스킵
    header = next(reader, None) 
    if header is None or header[0] != 'framenum':
        print(f"  [ERROR] Invalid output format. Header not found.")
        return np.zeros((0, 5), dtype=np.int32), 0.0

    mv_list = []
    
    # 6, 7, 9, 10, 11번 인덱스만 추출하여 [dst_x, dst_y, motion_x, motion_y, motion_scale] 순서로 저장
    for row in reader:
        if len(row) >= 12:
            try:
                mv_list.append([
                    int(row[0]),  # framenum
                    int(row[6]),  # dstx -> 0: dst_x
                    int(row[7]),  # dsty -> 1: dst_y
                    int(row[9]),  # motion_x -> 2: motion_x
                    int(row[10]), # motion_y -> 3: motion_y
                    int(row[11])  # motion_scale -> 4: motion_scale
                ])
            except ValueError:
                # 숫자 변환 실패 시 해당 행 스킵
                continue

    mvs_array = np.array(mv_list, dtype=np.int32)
    return mvs_array, extractor_time

# =========================================================================
# 3. 메인 실행
# =========================================================================

def motion_estimate_video(video_path):
    # 1. MV 추출 (바이너리 실행) 시간 측정
    print(f"Running MV Extraction on: {os.path.basename(video_path)}...", end=" ")
    mvs_array, extractor_time = run_mv_extractor(video_path)

    latency_list = []

    # Split for each frame
    for frame_num in np.unique(mvs_array[:, 0]):
        frame_mvs = mvs_array[mvs_array[:, 0] == frame_num][:, 1:]  # framenum 제외

        if frame_mvs.shape[0] == 0:
            print("❌ 실패: 추출된 모션 벡터가 없거나(0개) 바이너리 실행에 실패했습니다.")
            return

        # 2. Affine 추정 (NumPy/OpenCV) 시간 측정
        # print(f"2. Extracted {frame_mvs.shape[0]} MVs. Starting Affine Estimation...")
        
        start_time_est = time.perf_counter()
        transform_matrix = rigid_from_mvs(frame_mvs)
        estimation_time = time.perf_counter() - start_time_est

        latency_list.append(estimation_time)

    # 3. 결과 출력
    avg_latency = np.mean(latency_list) if latency_list else 0.0
    print(f"{avg_latency*1000:.4f} ms/frame")

    return avg_latency

if __name__ == '__main__':
    # VIDEO_DIR = "./data/DAVIS2017_trainval/MP4Videos_me"
    VIDEO_DIR = "./data/imnetvid_val/videos_me"

    eta_me = []
    for video_file in os.listdir(VIDEO_DIR):
        if not video_file.lower().endswith('.mp4'):
            continue

        video_path = os.path.join(VIDEO_DIR, video_file)
        avg_latency = motion_estimate_video(video_path)
        eta_me.append(avg_latency)
    
    overall_avg_latency = np.mean(eta_me) if eta_me else 0.0
    print(f"\nOverall Average Motion Estimation Latency: {overall_avg_latency*1000:.4f} ms")

    