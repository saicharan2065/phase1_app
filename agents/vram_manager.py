import threading

class VRAMManager:
    """
    Centralized Singleton Registry for managing MI300X VRAM allocation.
    Ensures that a massive LLM is only mounted once and shared across all Inference engines
    to prevent PCIe bandwidth congestion and OOM crashes.
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(VRAMManager, cls).__new__(cls)
                cls._instance._initialize()
            return cls._instance
            
    def _initialize(self):
        self.models = {}
        self.tokenizers = {}
        self.mount_lock = threading.Lock()
        self.abort_requested = False
        
    def get_or_load_model(self, model_id, model_type="causal", use_4bit=False, force_cpu=False):
        """
        Returns the requested model if it is already cached in RAM/VRAM. 
        Otherwise, mounts it using Accelerate CPU offloading or strictly CPU.
        """
        with self.mount_lock:
            if model_id in self.models and self.models[model_id] is not None:
                return self.models[model_id], self.tokenizers.get(model_id)
                
            import torch
            import torch.nn as nn
            import psutil
            
            # Dynamically calculate currently FREE VRAM to prevent Accelerate over-allocation OOMs
            if torch.cuda.is_available() and not force_cpu:
                free_bytes, total_bytes = torch.cuda.mem_get_info(0)
                # Leave a 5GB safety buffer for context window overhead
                safe_free_gib = max(0, int(free_bytes / (1024**3)) - 5)
            else:
                safe_free_gib = 0
                
            ram_total_gb = int(psutil.virtual_memory().total / (1024**3))
            max_memory = {
                0: f"{safe_free_gib}GiB", # Only tell Accelerate what is ACTUALLY free right now
                "cpu": f"{ram_total_gb - 20}GiB" # Leave 20GB for OS
            }
            
            target_device_map = "cpu" if force_cpu else "auto"
            
            # MI300X ROCm PyTorch Compatibility Monkeypatch for BitsAndBytes
            if not hasattr(nn.Module, "set_submodule"):
                def set_submodule(self, target: str, module: nn.Module) -> None:
                    atoms: list[str] = target.split(".")
                    name = atoms.pop(-1)
                    mod = self
                    for item in atoms:
                        if not hasattr(mod, item):
                            raise AttributeError(f"Module has no attribute `{item}`")
                        mod = getattr(mod, item)
                        if not isinstance(mod, nn.Module):
                            raise AttributeError("`{}` is not an nn.Module".format(item))
                    setattr(mod, name, module)
                nn.Module.set_submodule = set_submodule
                
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from transformers import AutoProcessor, LlavaForConditionalGeneration
            
            if model_type == "vision":
                self.tokenizers[model_id] = AutoProcessor.from_pretrained(model_id)
                self.models[model_id] = LlavaForConditionalGeneration.from_pretrained(
                    model_id,
                    device_map=target_device_map,
                    max_memory=None if force_cpu else max_memory,
                    torch_dtype=torch.float16,
                    use_safetensors=True
                )
            else:
                self.tokenizers[model_id] = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
                self.models[model_id] = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map=target_device_map,
                    max_memory=None if force_cpu else max_memory,
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    use_safetensors=True
                )
            
            if getattr(self, "abort_requested", False):
                self.models.pop(model_id, None)
                self.tokenizers.pop(model_id, None)
                import torch, gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.abort_requested = False
                raise RuntimeError("Model mount aborted by user.")
                
            return self.models[model_id], self.tokenizers[model_id]
            
    def purge_vram(self):
        acquired = self.mount_lock.acquire(blocking=False)
        if not acquired:
            self.abort_requested = True
            return False
            
        try:
            import torch
            import gc
            
            # Delete all cached models
            for model_id in list(self.models.keys()):
                del self.models[model_id]
            for model_id in list(self.tokenizers.keys()):
                del self.tokenizers[model_id]
                
            self.models.clear()
            self.tokenizers.clear()
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            self.abort_requested = False
            return True
        finally:
            self.mount_lock.release()

# Global Singleton Instance
vram_manager = VRAMManager()
