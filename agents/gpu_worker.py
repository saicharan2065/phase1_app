import sys
import time

def main():
    if len(sys.argv) < 2:
        return
    target_gb = int(sys.argv[1])
    
    try:
        import torch
        if torch.cuda.is_available():
            tensors = []
            # Allocate instantaneous VRAM using torch.zeros to bypass hardware TDR timeouts
            for _ in range(target_gb):
                # (16384 * 16384 * 4 bytes) = 1 GB
                tensors.append(torch.zeros((16384, 16384), device="cuda", dtype=torch.float32))
            
            # Create two random tensors for the heavy math loop
            compute_a = torch.randn((16384, 16384), device="cuda", dtype=torch.float32)
            compute_b = torch.randn((16384, 16384), device="cuda", dtype=torch.float32)
            
            # Enter infinite loop to peg GPU Compute at 100%
            while True:
                _ = torch.matmul(compute_a, compute_b)
    except Exception as e:
        print(f"MI300X Hardware Burn-in failed: {e}")

if __name__ == "__main__":
    main()
