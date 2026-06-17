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
        self.active_model_id = None
        self.model = None
        self.tokenizer = None
        self.mount_lock = threading.Lock()
        
    def get_or_load_model(self, model_id, use_4bit=True):
        """
        Returns the active model if it is already in VRAM. 
        Otherwise, purges VRAM and mounts the new model.
        """
        with self.mount_lock:
            if self.active_model_id == model_id and self.model is not None:
                return self.model, self.tokenizer
                
            # Clean up old model if a different one is loaded to prevent OOM
            if self.model is not None:
                import torch
                import gc
                del self.model
                del self.tokenizer
                self.model = None
                self.tokenizer = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            import torch
            import torch.nn as nn
            
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
                
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            
            if use_4bit:
                bnb_config = BitsAndBytesConfig(load_in_4bit=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    quantization_config=bnb_config,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    use_safetensors=True
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    use_safetensors=True
                )
                
            self.active_model_id = model_id
            return self.model, self.tokenizer

# Global Singleton Instance
vram_manager = VRAMManager()
