import csv
import os
import subprocess
import statistics
from pathlib import Path

# 현재 스크립트 위치를 기준으로 프로젝트 루트 디렉토리 (IPConv/) 설정
# (스크립트가 src/measure_encode.py 에 있다고 가정)
# HINT: Path(__file__).resolve().parent를 사용하면 src/ 폴더로 설정됨
BASE_DIR = Path(__file__).resolve().parent

# =========================================================================
# 1. 데이터셋 경로 및 정보 설정
# =========================================================================

FPS=100

DATASETS_INFO = [
    {
        'name': 'DAVIS2017_trainval',
        'csv_path': BASE_DIR / 'data' / 'DAVIS2017_trainval' / f'compression_benchmark_f{FPS}.csv',
        'video_base_path': BASE_DIR / 'data' / 'DAVIS2017_trainval' / 'MP4Videos_me',
        'resolution_filter': None, # DAVIS는 해상도 필터링 없음
    },
    # {
    #     'name': 'imnetvid_val',
    #     'csv_path': BASE_DIR / 'data' / 'imnetvid_val' / f'compression_benchmark_f{FPS}.csv',
    #     'video_base_path': BASE_DIR / 'data' / 'imnetvid_val' / 'videos_me',
    #     'resolution_filter': None
    # },
]

# =========================================================================
# 2. 유틸리티 함수 (수정됨)
# =========================================================================

def parse_time_to_seconds(time_str: str) -> float:
    """GStreamer 로그 시간 문자열 (0:00:00.xxx)을 초 단위 float로 변환합니다."""
    try:
        # H:MM:SS.ns 형식을 초로 변환
        parts = time_str.split(':')
        if len(parts) == 3:
            h, m, s = map(float, parts)
        else:
            s = float(time_str) # ns만 있는 경우나 기타 단일 값 처리
        
        return h * 3600 + m * 60 + s
    except Exception as e:
        # print(f"  [ERROR] Time parsing failed for '{time_str}': {e}") # 로그 과다 방지
        return 0.0

def get_video_info(video_path: Path) -> dict:
    """ffprobe를 호출하여 동영상 파일의 총 프레임 수, 너비, 높이를 가져옵니다."""
    info = {'nb_frames': 0, 'width': 0, 'height': 0}
    try:
        # ffprobe 명령어 (프레임 수, 너비, 높이 추출)
        cmd = [
            'ffprobe', 
            '-v', 'error', 
            '-select_streams', 'v:0', 
            '-show_entries', 'stream=nb_frames,width,height', 
            '-of', 'csv=p=0', # CSV 형식으로 출력, 헤더 없음
            str(video_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output_parts = result.stdout.strip().split(',')
        
        if len(output_parts) == 3:
            info['nb_frames'] = int(output_parts[2]) if output_parts[2].isdigit() else 0
            info['width'] = int(output_parts[1]) if output_parts[1].isdigit() else 0
            info['height'] = int(output_parts[0]) if output_parts[0].isdigit() else 0
            
    except subprocess.CalledProcessError as e:
        # print(f"  [ERROR] ffprobe failed for {video_path.name}: {e.stderr.strip()}")
        # print(f"          -> HINT: Please ensure 'ffprobe' is installed and accessible.")
        pass # 에러는 발생하더라도 로직에 치명적이지 않다면 무시
    except Exception as e:
        # print(f"  [ERROR] Video info determination failed: {e}")
        pass
        
    return info

# =========================================================================
# 3. 메인 처리 로직 (수정됨)
# =========================================================================

def measure_average_encoding_time(datasets_info):
    """각 데이터셋별 평균 인코딩 시간을 계산하고 결과를 출력합니다."""
    
    total_summary = {}

    for dataset in datasets_info:
        csv_path = dataset['csv_path']
        video_base_path = dataset['video_base_path']
        resolution_filter = dataset['resolution_filter']
        
        # 총 누적 값
        total_frames_sum = 0
        total_time_sec_sum = 0.0
        # 개별 프레임당 시간 (통계 계산용 리스트)
        individual_time_per_frame_ms = []
        processed_videos_count = 0
        
        if not csv_path.exists():
            print(f"Dataset: {dataset['name']} - CSV file not found at {csv_path}. Skipping.")
            total_summary[dataset['name']] = "CSV Not Found"
            continue

        print(f"Processing Dataset: {dataset['name']} from {csv_path.name} (Filter: {resolution_filter})...")

        with open(csv_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            
            for row in reader:
                filename = row['Filename'].strip()
                time_str = row['Execution_Time'].strip()
                
                # 1. 경로 조립 및 비디오 정보 가져오기
                video_full_path = video_base_path / filename
                video_info = get_video_info(video_full_path)
                frame_count = video_info['nb_frames']
                time_sec = parse_time_to_seconds(time_str)
                
                # 1.5. 해상도 필터링 적용 (IMNET-VID 전용)
                if resolution_filter:
                    if (video_info['width'], video_info['height']) != resolution_filter:
                        # print(f"  [SKIP] {filename}: Resolution {video_info['width']}x{video_info['height']} != {resolution_filter}") # 로그 과다 방지
                        continue # 필터 조건에 맞지 않으면 스킵
                
                
                # 2. 유효성 검증 및 누적
                if frame_count > 0 and time_sec > 0:
                    # 프레임당 인코딩 시간 (ms) 계산 및 리스트에 추가
                    time_per_frame_ms = (time_sec / frame_count) * 1000
                    individual_time_per_frame_ms.append(time_per_frame_ms)
                    
                    total_frames_sum += frame_count
                    total_time_sec_sum += time_sec
                    processed_videos_count += 1
                else:
                    print(f"  [SKIP] {filename}: Insufficient data (Frames: {frame_count} / Time: {time_sec:.3f}s)") # 로그 과다 방지
                    pass
        
        # 3. 데이터셋별 통계 계산
        if processed_videos_count > 1:
            # 총합 기반 평균 (가중 평균)
            avg_time_overall_ms = (total_time_sec_sum / total_frames_sum) * 1000 
            # 개별 영상의 평균 시간들의 평균
            avg_time_mean_ms = statistics.mean(individual_time_per_frame_ms) 
            # 표준 편차
            std_time_ms = statistics.stdev(individual_time_per_frame_ms)

            total_summary[dataset['name']] = {
                'total_videos': processed_videos_count,
                'total_frames': total_frames_sum,
                'total_time_sec': total_time_sec_sum,
                'avg_time_overall_ms': avg_time_overall_ms,
                'std_time_ms': std_time_ms,
                'avg_fps': total_frames_sum / total_time_sec_sum 
            }
        elif processed_videos_count == 1:
            # 영상이 하나일 경우 표준편차 계산 불가
            total_summary[dataset['name']] = {
                'total_videos': processed_videos_count,
                'total_frames': total_frames_sum,
                'total_time_sec': total_time_sec_sum,
                'avg_time_overall_ms': (total_time_sec_sum / total_frames_sum) * 1000,
                'std_time_ms': 'N/A (Single Video)',
                'avg_fps': total_frames_sum / total_time_sec_sum
            }
        else:
            total_summary[dataset['name']] = "No Valid Data Processed"
        
        print("-" * 50)

        # break  # 현재는 첫 번째 데이터셋만 처리하도록 설정 (테스트 용이)


    # 4. 최종 결과 출력
    print("\n" + "=" * 60)
    print(" 🎬 최종 인코딩 성능 분석 결과 (프레임당 평균 시간 및 편차) ")
    print("=" * 60)
    
    RESULTS_FILE = BASE_DIR / 'encoding_summary.txt'
    with open(RESULTS_FILE, 'w') as f:
        f.write("Encoding Performance Summary\n")
        
        for name, result in total_summary.items():
            if isinstance(result, dict):
                output_str = (
                    f"--- Dataset: {name} ---\n"
                    f"  Processed Videos: {result['total_videos']}\n"
                    f"  Total Frames: {result['total_frames']} frames\n"
                    f"  Total Encoding Time: {result['total_time_sec']:.3f} seconds\n"
                    f"  Avg Time Per Frame (ms) [Overall]: {result['avg_time_overall_ms']:.3f} ms\n"
                    f"  Standard Deviation (ms): {result['std_time_ms']:.3f}\n"
                    f"  Equivalent Avg FPS: {result['avg_fps']:.1f} FPS\n"
                )
                print(output_str)
                f.write(output_str + "\n")
            else:
                print(f"--- Dataset: {name} --- Status: {result}")
                f.write(f"--- Dataset: {name} --- Status: {result}\n\n")

    print(f"\nSummary results also saved to: {RESULTS_FILE}")

# 스크립트 실행
if __name__ == "__main__":
    measure_average_encoding_time(DATASETS_INFO)