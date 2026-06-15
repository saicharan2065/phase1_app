import torch
import time
import sys

def test_vram(target_gb):
    print(f"Testing PyTorch VRAM allocation: {target_gb} GB")
    if not torch.cuda.is_available():
        print("CUDA/ROCm not available in this env.")
        return
        
    tensors = []
    try:
        start = time.time()
        for i in range(target_gb):
            # Allocate 1 GB
            t = torch.zeros((16384, 16384), device="cuda", dtype=torch.float32)
            # Force write to ensure it's not lazy allocated
            t[0, 0] = 1.0
            tensors.append(t)
            print(f"Allocated {i+1} GB", end="\r")
        
        end = time.time()
        print(f"\nSuccess! Allocated {target_gb} GB in {end - start:.2f} seconds.")
    except Exception as e:
        print(f"\nFailed during allocation: {e}")

if __name__ == "__main__":
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    test_vram(target)
