import time
import pandas as pd
import concurrent.futures
from threading import Lock
from threading import Lock

class BulkSARGenerator:
    def __init__(self):
        self._lock = Lock()
        self.is_running = False
        self.processed_count = 0
        self.total_count = 0
        self.results = []
        self.model_loaded = False
        self.model_loaded = False
        
    def _initialize_vram_engine(self):
        """Loads actual LLM into MI300X VRAM using PyTorch and Transformers."""
        self.status_message = "Loading real LLM into MI300X VRAM..."
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            from agents.vram_manager import vram_manager
            
            from tabs.model_management import get_active_model_state
            active_model = get_active_model_state()
            model_id = active_model if active_model and active_model != "None Selected" else "Qwen/Qwen1.5-0.5B"
            
            self.model, self.tokenizer = vram_manager.get_or_load_model(model_id, use_4bit=True)
            self.model_loaded = True
            self.status_message = f"Successfully mounted {model_id} onto MI300X."
        except ImportError as e:
            self.model_loaded = False
            raise RuntimeError(f"CRITICAL ERROR: Transformers/PyTorch libraries are missing from the environment. Real pipeline cannot execute! Details: {str(e)}")
        except Exception as e:
            self.model_loaded = False
            raise RuntimeError(f"MI300X VRAM MOUNT FAILURE: {str(e)}")
            
    def _process_batch(self, suspects):
        """Real GPU batch inference on a chunk of suspects using Hugging Face."""
        batch_results = []
        if self.model_loaded:
            try:
                import torch
                # Create real prompts
                prompts = [f"Write a Suspicious Activity Report for individual ID: {s}. Reason: High velocity transfers." for s in suspects]
                
                # Tokenize and push to MI300X VRAM
                inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=50)
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                
                # Real LLM Generation (Matrix Math on GPU)
                with torch.no_grad():
                    outputs = self.model.generate(**inputs, max_new_tokens=30, do_sample=True, temperature=0.7)
                    
                decoded_reports = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                
                for i, report in enumerate(decoded_reports):
                    batch_results.append({"Suspect ID": suspects[i], "SAR Report": report.replace(prompts[i], "").strip(), "Risk Level": "HIGH"})
            except Exception as e:
                raise RuntimeError(f"MI300X CUDA EXECUTION ERROR: {str(e)}")
        else:
            raise RuntimeError("Cannot process batch. MI300X VRAM model is not mounted.")
            
        with self._lock:
            self.processed_count += len(suspects)
            self.results.extend(batch_results)
            
    def run_bulk_inference(self, suspect_ids, batch_size=32, skip_gpu=False, sync_barrier=None):
        self.is_running = True
        try:
            self.total_count = len(suspect_ids)
            self.processed_count = 0
            self.results = []
            
            if not self.model_loaded:
                self._initialize_vram_engine()
                
            if sync_barrier:
                self.status_message = "WAITING FOR OTHER ENGINES TO MOUNT..."
                sync_barrier.wait()
                
            # Chunk the dataset into batches
            chunks = [suspect_ids[i:i + batch_size] for i in range(0, len(suspect_ids), batch_size)]
            
            # We process sequentially if using real model to prevent OOM
            for chunk in chunks:
                if not self.is_running: break
                self._process_batch(chunk)
                
            return self.results
        except Exception as e:
            if sync_barrier:
                try: sync_barrier.abort()
                except Exception: pass
            self.status_message = f"CRASH: {str(e)}"
            return []
        finally:
            self.is_running = False
        
    def stop(self):
        self.is_running = False
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
