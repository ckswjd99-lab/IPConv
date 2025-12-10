import numpy as np
import time
import os
import sys
import threading
import torch
import torch.nn as nn
from tqdm import tqdm

# 모델 import
try:
    from ipconv.models import (
        MaskedRCNN_ViT_B_FPN_Contexted, MaskedRCNN_ViT_L_FPN_Contexted, MaskedRCNN_ViT_H_FPN_Contexted,
        CascadeMaskRCNN_Swin_B_Contexted, CascadeMaskRCNN_Swin_L_Contexted,
    )
except ImportError:
    print("[Warning] ipconv.models not found. Please check your python path.")
    # 테스트를 위해 강제 종료하지 않고 진행하려면 아래 줄 주석 처리
    sys.exit(1)

# --- 1. PowerEstimator (Updated: Average Power Approach) ---
class PowerEstimator:
    def __init__(self, idle_load_duration=5, idle_load_samples=10, sampling_rate=30):
        # 경로 저장 변수
        self.board_rail = None
        self.proc_rail = None
        self.soc_rail = None
        
        # 확인된 경로: hwmon4 (Jetson Orin 계열 가정)
        base_path = '/sys/class/hwmon/hwmon4/'
        print(f"[PowerEstimator] Scanning sensors in {base_path}...")
        
        if not os.path.exists(base_path):
            print(f"[Error] {base_path} does not exist. Check hardware.")

        # 채널 스캔
        for i in range(1, 20):
            l_path = os.path.join(base_path, f"in{i}_label")
            v_path = os.path.join(base_path, f"in{i}_input")
            c_path = os.path.join(base_path, f"curr{i}_input")
            
            if os.path.exists(l_path) and os.path.exists(v_path) and os.path.exists(c_path):
                try:
                    with open(l_path, 'r') as f:
                        label = f.read().strip() 
                    
                    if label == "VDD_IN":
                        print(f"  - [Board] Found: {label} (Channel {i})")
                        self.board_rail = (v_path, c_path)
                    elif label == "VDD_CPU_GPU_CV":
                        print(f"  - [Processors] Found: {label} (Channel {i})")
                        self.proc_rail = (v_path, c_path)
                    elif label == "VDD_SOC":
                        print(f"  - [SOC] Found: {label} (Channel {i})")
                        self.soc_rail = (v_path, c_path)
                except:
                    pass

        # 예외 처리
        if not self.board_rail: print("[Warning] VDD_IN missing.")
        if not self.proc_rail: print("[Warning] VDD_CPU_GPU_CV missing.")
        if not self.soc_rail: print("[Warning] VDD_SOC missing.")

        # Idle Power 설정 (User provided fixed values)
        self.idle_board = 5283.5
        self.idle_proc = 810.0
        self.idle_soc = 1606.3
        
        print(f"[Power] Using pre-measured Idle values")
        print(f"[Power] Idle -> Board:{self.idle_board:.1f}mW, Proc:{self.idle_proc:.1f}mW, SOC:{self.idle_soc:.1f}mW")
        
        self.sampling_interval = 1.0 / sampling_rate
        
        # Thread control
        self.monitoring = False
        self.thread = None
        self.results = {}

    def _get_instant_powers(self):
        """ (Board, Proc, SOC) in mW """
        b_p, p_p, s_p = 0.0, 0.0, 0.0
        
        # Board
        if self.board_rail:
            try:
                with open(self.board_rail[0], "r") as fv: v = float(fv.read())
                with open(self.board_rail[1], "r") as fc: c = float(fc.read())
                b_p = (v * c) / 1000.0
            except: pass
        
        # Proc
        if self.proc_rail:
            try:
                with open(self.proc_rail[0], "r") as fv: v = float(fv.read())
                with open(self.proc_rail[1], "r") as fc: c = float(fc.read())
                p_p = (v * c) / 1000.0
            except: pass
            
        # SOC
        if self.soc_rail:
            try:
                with open(self.soc_rail[0], "r") as fv: v = float(fv.read())
                with open(self.soc_rail[1], "r") as fc: c = float(fc.read())
                s_p = (v * c) / 1000.0
            except: pass
            
        return b_p, p_p, s_p

    def _monitor_loop(self):
        # 누적 변수 (mW 단위)
        acc_b, acc_p, acc_s = 0.0, 0.0, 0.0
        sample_count = 0
        
        # 실제 모니터링 시작 시간 측정
        start_time = time.time()
        
        while self.monitoring:
            cycle_start = time.time()
            
            # 1. Read Power (mW)
            curr_b, curr_p, curr_s = self._get_instant_powers()
            
            # 2. Accumulate Raw Power (뺄셈 없이 그냥 더함)
            acc_b += curr_b
            acc_p += curr_p
            acc_s += curr_s
            sample_count += 1
            
            # 3. Sleep remainder
            elapsed = time.time() - cycle_start
            sleep_time = max(0, self.sampling_interval - elapsed)
            time.sleep(sleep_time)

        # 모니터링 종료 시간
        end_time = time.time()
        total_duration = end_time - start_time

        # 4. Calculate Average Power & Dynamic Energy
        if sample_count > 0:
            avg_b = acc_b / sample_count
            avg_p = acc_p / sample_count
            avg_s = acc_s / sample_count
        else:
            avg_b, avg_p, avg_s = 0.0, 0.0, 0.0

        # Dynamic Power = Average Power - Idle Power 
        # (음수 방지용 max(0, ...)는 마지막에 한 번만 적용)
        dyn_p_b = max(0, avg_b - self.idle_board)
        dyn_p_p = max(0, avg_p - self.idle_proc)
        dyn_p_s = max(0, avg_s - self.idle_soc)

        # Total Dynamic Energy (mJ) = Dynamic Power (mW) * Duration (s)
        dyn_e_b = dyn_p_b * total_duration
        dyn_e_p = dyn_p_p * total_duration
        dyn_e_s = dyn_p_s * total_duration

        # Debug info
        # print(f"[Power Summary] Duration: {total_duration:.2f}s, Samples: {sample_count}")
        # print(f"  - Board: Avg {avg_b:.1f}mW - Idle {self.idle_board:.1f}mW = Dyn {dyn_p_b:.1f}mW -> {dyn_e_b:.2f} mJ")

        # Save results (mJ 단위)
        self.results = {
            "board_mj": dyn_e_b,
            "proc_mj": dyn_e_p,
            "soc_mj": dyn_e_s
        }

    def start(self):
        """Start background monitoring thread"""
        self.monitoring = True
        self.results = {}
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.start()

    def stop(self):
        """Stop monitoring and return energy (mJ)"""
        self.monitoring = False
        if self.thread:
            self.thread.join()
        return self.results

# --- 2. 측정 함수 ---
def measure_latency_memory_energy(
    model: nn.Module,
    patch_keep_rate: float,
    method: str,
    input_size: int,
    p_est: PowerEstimator
):
    model.eval()
    dummy_input = np.zeros((input_size, input_size, 3))
    block_size = 16
    num_blocks_sqrt = input_size // block_size

    dmap = torch.zeros((1, num_blocks_sqrt, num_blocks_sqrt, 1), dtype=torch.float32, device="cuda")
    num_patches = num_blocks_sqrt * num_blocks_sqrt
    num_keep = int(num_patches * patch_keep_rate)
    
    if num_keep > 0:
        idx_rand = torch.randperm(num_patches)[:num_keep]
        dmap.view(-1)[idx_rand] = 1.0

    num_warmup = 10
    num_repeats = 10

    if method == "vanilla": inference_func = model.forward
    elif method == "ours": inference_func = model.forward_contexted
    elif method == "eventful": inference_func = model.forward_eventful
    elif method == "maskvd": inference_func = model.forward_maskvd
    elif method == "stgt": inference_func = model.forward_stgt
    else: raise ValueError(f"Unknown method: {method}")

    # Warmup
    for _ in range(num_warmup):
        output = inference_func(dummy_input, dirtiness_map=dmap, only_backbone=True)
    
    cache_args = {"anchor_features": output[1]} if method != "vanilla" else {}

    # FLOPs
    model.counting()
    model.clear_counts()
    inference_func(dummy_input, dirtiness_map=dmap, only_backbone=True, **cache_args)
    counts = model.total_counts()
    model.clear_counts()

    # --- Real Measurement ---
    torch.cuda.synchronize()
    
    # 1. Start Power Recording (Background)
    p_est.start()
    start_time = time.time()

    # 2. Run Workload (Main Thread)
    for tidx in range(num_repeats):
        with torch.cuda.nvtx.range(f"{method}_{patch_keep_rate}_t{tidx}"):
            inference_func(dummy_input, dirtiness_map=dmap, only_backbone=True, **cache_args)

            torch.cuda.synchronize()
    
    end_time = time.time()
    
    # 3. Stop Power Recording
    results = p_est.stop()
    
    # --- Calc ---
    total_time_sec = end_time - start_time
    latency_sec = total_time_sec / num_repeats

    # Results (mJ) -> Joule Per Inference
    dyn_board_j = (results["board_mj"] / 1000.0) / num_repeats
    dyn_proc_j  = (results["proc_mj"] / 1000.0) / num_repeats
    dyn_soc_j   = (results["soc_mj"] / 1000.0) / num_repeats

    cache_size = 0
    if method != "vanilla":
        cache = output[1]
        for key, value in cache.items():
            if isinstance(value, torch.Tensor):
                cache_size += value.element_size() * value.numel()

    return latency_sec, dyn_board_j, dyn_proc_j, dyn_soc_j, cache_size, counts


# --- 3. 메인 ---
@torch.no_grad()
def main():
    print("Initializing Power Estimator...")
    p_est = PowerEstimator(idle_load_duration=5, sampling_rate=30) 

    models_dict = {
        "ViT-base": MaskedRCNN_ViT_B_FPN_Contexted,
        # "Swin-base": CascadeMaskRCNN_Swin_B_Contexted,
    }

    # Configuration
    # keep_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    keep_rates = [0.1, 0.25]
    
    # methods = ["ours", "eventful", "maskvd", "stgt", "vanilla"]
    # methods = ["ours", "eventful"]
    methods = ["stgt"]
    # methods = ["vanilla"]
    
    input_sizes = [1024]
    
    print("\nStarting Measurement...")
    print("InputSize,Model,Method,KeepRate,Latency(s),DynE_Board(J),DynP_Board(W),DynE_Proc(J),DynP_Proc(W),DynE_SOC(J),DynP_SOC(W),FLOPs,Cache(MB)")

    for input_size in input_sizes:
        for mname, model_class in models_dict.items():
            try:
                # 모델 로드 전 메모리 정리
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                
                print(f"[Info] Loading {mname}...")
                model = model_class("cuda")
                model.eval()
            except Exception as e:
                print(f"[Error] Failed to load model {mname}: {e}")
                continue

            for method in methods:
                for keep_rate in keep_rates:
                    try:
                        lat, dyn_board, dyn_proc, dyn_soc, cache, num_count = measure_latency_memory_energy(
                            model, keep_rate, method, input_size, p_est
                        )
                        flops = sum(num_count.values())
                        cache_mb = cache / (1024 * 1024)
                        
                        # Zero Latency Protection
                        p_board = dyn_board/lat if lat > 0 else 0
                        p_proc = dyn_proc/lat if lat > 0 else 0
                        p_soc = dyn_soc/lat if lat > 0 else 0

                        print(f"{input_size},{mname},{method},{keep_rate},{lat:.4f},"
                              f"{dyn_board:.4f},{p_board:.4f},"
                              f"{dyn_proc:.4f},{p_proc:.4f},"
                              f"{dyn_soc:.4f},{p_soc:.4f},"
                              f"{flops},{cache_mb:.2f}")
                        sys.stdout.flush()
                        
                    except Exception as e:
                        print(f"Error in {method} with {keep_rate}: {e}")
                        import traceback
                        traceback.print_exc()
                    
                    # 쿨링 타임
                    time.sleep(5)

            del model

if __name__ == "__main__":
    main()