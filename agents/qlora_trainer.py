import time
import concurrent.futures
from threading import Lock
from threading import Lock

class QLoRATrainer:
    def __init__(self):
        self._lock = Lock()
        self.is_training = False
        self.progress_percent = 0
        self.current_epoch = 0
        self.total_epochs = 3
        self.status_message = "IDLE"
        self.status_message = "IDLE"
        
    def _simulate_qlora_training(self, dataset_id, model_id, skip_gpu, sync_barrier=None):
        self.is_training = True
        self.progress_percent = 0
        self.current_epoch = 1
        
        # 1. MOUNTING
        self.status_message = f"MOUNTING: Allocating VRAM and streaming {model_id} from SSD..."
        
        has_libraries = False
        try:
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
                
            from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
            from peft import LoraConfig, get_peft_model
            from trl import SFTTrainer
            from datasets import Dataset
            has_libraries = True
        except ImportError as e:
            self.status_message = f"CRASH: Missing PyTorch libraries (peft/trl). Please install them. {str(e)}"
            self.is_training = False
            return
            
        if has_libraries:
            try:
                # Real pipeline execution
                self.status_message = "FREEZING base parameters & attaching real LoRA adapters..."
                
                actual_model = model_id
                
                with self._lock:
                    # Disable BitsAndBytes 4-bit to bypass ROCm mismatch
                    model = AutoModelForCausalLM.from_pretrained(
                        actual_model,
                        device_map="auto",
                        torch_dtype=torch.float16,
                        trust_remote_code=True,
                        use_safetensors=True
                    )
                    tokenizer = AutoTokenizer.from_pretrained(actual_model, trust_remote_code=True)
                    tokenizer.pad_token = tokenizer.eos_token
                
                lora_config = LoraConfig(
                    r=8, 
                    lora_alpha=16, 
                    target_modules=["q_proj", "v_proj"], 
                    lora_dropout=0.05, 
                    bias="none", 
                    task_type="CAUSAL_LM"
                )
                model = get_peft_model(model, lora_config)
                
                from data.dataset_manager import DatasetManager
                import pandas as pd
                dm = DatasetManager()
                
                self.status_message = f"LOADING REAL DATA: Pulling records from {dataset_id}..."
                if not dataset_id or dataset_id == "No valid dataset selected." or dataset_id == "No Datasets Cached":
                    raise RuntimeError("No valid target_dataset provided.")
                    
                ds = dm._load_dataset_records_sync(dataset_id, "5000") # Limit to 5000 for realistic QLoRA run
                if isinstance(ds, pd.DataFrame) and "Error" in ds.columns:
                    raise RuntimeError(f"Dataset load failed: {ds.iloc[0]['Error']}")
                
                # Convert real dataset into training texts
                training_texts = []
                if isinstance(ds, pd.DataFrame):
                    records = ds.to_dict('records')
                else:
                    records = [ds[i] for i in range(len(ds))]
                    
                for r in records:
                    if "text" in r and isinstance(r["text"], str):
                        training_texts.append(r["text"])
                    else:
                        row_str = " | ".join([f"{k}: {v}" for k, v in r.items()])
                        training_texts.append(f"Financial Record:\n{row_str}")
                        
                if not training_texts:
                    raise RuntimeError("Failed to extract training text from dataset.")
                    
                # Create actual training data
                real_data = Dataset.from_dict({"text": training_texts})
                
                args = TrainingArguments(
                    output_dir=f"storage/adapters/{actual_model.replace('/', '_')}_checkpoints",
                    num_train_epochs=self.total_epochs,
                    per_device_train_batch_size=1,
                    save_steps=10,
                    logging_steps=1,
                    learning_rate=2e-4,
                    fp16=True
                )
                
                trainer = SFTTrainer(
                    model=model,
                    train_dataset=real_data,
                    dataset_text_field="text",
                    max_seq_length=128,
                    args=args,
                )
                
                if sync_barrier:
                    self.status_message = "WAITING FOR OTHER ENGINES TO MOUNT..."
                    sync_barrier.wait()
                    
                self.status_message = "TRAINING: Executing real PyTorch backward pass on MI300X..."
                
                # We can't easily hook into Trainer for live progress bars in gradio without a custom callback
                # So we just run it and update progress manually
                for epoch in range(1, self.total_epochs + 1):
                    if not self.is_training: break
                    self.current_epoch = epoch
                    self.progress_percent = int((epoch / self.total_epochs) * 100)
                    # Train for 1 epoch at a time just for progress bar updates
                    args.num_train_epochs = 1
                    trainer.train()
                
                self.status_message = "SAVING: Writing real PEFT Adapter file..."
                save_path = f"storage/adapters/{actual_model.replace('/', '_')}_qlora"
                trainer.model.save_pretrained(save_path)
                
            except Exception as e:
                if sync_barrier:
                    try: sync_barrier.abort()
                    except Exception: pass
                self.status_message = f"CRASH: QLoRA Training Failed: {str(e)}"
                self.is_training = False
                
                # Prevent VRAM Leak: Delete local variables to free PyTorch objects held by the exception traceback
                import traceback
                traceback.clear_frames(e.__traceback__)
                if 'model' in locals(): del model
                if 'trainer' in locals(): del trainer
                import gc; gc.collect()
                return
                
        # 5. DEMOUNTING
        self.status_message = "DEMOUNTING: Purging VRAM. Freeing memory..."
        time.sleep(1)
        
        self.status_message = "COMPLETE: Neural Rewiring Finished."
        self.is_training = False
        
    def start_training(self, dataset_id, model_id, skip_gpu=False, sync_barrier=None):
        if self.is_training:
            return "Training is already in progress!"
            
        import threading
        threading.Thread(target=self._simulate_qlora_training, args=(dataset_id, model_id, skip_gpu, sync_barrier), daemon=True).start()
            
        return "Neural Rewiring Initialized in Background."
        
    def stop(self):
        self.is_training = False
        self.status_message = "ABORTED"
