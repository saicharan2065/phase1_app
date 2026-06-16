import subprocess
import os
import sys

class GPUBurner:
    def __init__(self):
        self.is_running = False
        self.process = None
        
    def start_burn(self, target_gb):
        """Allocates target_gb of VRAM via an isolated subprocess to prevent main server segfaults."""
        self.is_running = True
        worker_path = os.path.join(os.path.dirname(__file__), "gpu_worker.py")
        
        # Launch totally isolated Python process to protect Gradio from PyTorch OOMs/TDR crashes
        self.process = subprocess.Popen([sys.executable, worker_path, str(target_gb)])
        
    def stop_burn(self):
        """Stops the isolated GPU computation process and instantly frees the allocated VRAM."""
        self.is_running = False
        if self.process:
            if sys.platform == 'win32':
                subprocess.run(['taskkill', '/F', '/T', '/PID', str(self.process.pid)], capture_output=True)
            else:
                self.process.terminate()
                self.process.wait()
            self.process = None
