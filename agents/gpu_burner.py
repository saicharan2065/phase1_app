import threading
import time

class GPUBurner:
    def __init__(self):
        self.is_running = False
        self.tensors = []
        self._thread = None
        
    def start_burn(self, target_gb):
        """Allocates target_gb of VRAM and runs matrix multiplications to 100% the GPU."""
        self.is_running = True
        self.tensors = []
        self._thread = threading.Thread(target=self._burn_loop, args=(target_gb,))
        self._thread.start()
        
    def stop_burn(self):
        """Stops the GPU computation and instantly frees the allocated VRAM."""
        self.is_running = False
        if self._thread:
            self._thread.join()
        self.tensors = []
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
            
    def _burn_loop(self, target_gb):
        try:
            import torch
            if torch.cuda.is_available():
                # Allocate exactly 1 GB of float32 tensors per loop
                for _ in range(target_gb):
                    if not self.is_running:
                        return
                    # (16384 * 16384 * 4 bytes) = 1,073,741,824 bytes = 1 GB
                    self.tensors.append(torch.randn((16384, 16384), device="cuda", dtype=torch.float32))
                
                # Keep the GPU Compute Units at 100% utilization
                while self.is_running:
                    if len(self.tensors) >= 2:
                        _ = torch.matmul(self.tensors[0], self.tensors[1])
                    else:
                        time.sleep(0.1)
        except Exception as e:
            print(f"MI300X Hardware Burn-in failed: {e}")
