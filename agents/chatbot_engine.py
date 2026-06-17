import time
import threading

class ChatbotEngine:
    def __init__(self):
        self.is_loading = False
        
    def generate_response(self, message, history, active_model_id, username="GUEST"):
        """
        Stream back response for gradio ChatInterface
        """
        if not active_model_id or active_model_id == "None Selected":
            yield "Please select an Active Model in the Model Management tab first!"
            return

        from agents.vram_manager import vram_manager
        
        # Clean the model ID to remove the appended size from the UI (e.g. ' (24.87 GB)')
        clean_model_id = active_model_id.split(" (")[0] if active_model_id and " (" in active_model_id else active_model_id
        
        # Prevent Vision Models in Text Chatbot
        if "llava" in clean_model_id.lower() or "vision" in clean_model_id.lower():
            yield f"ERROR: '{clean_model_id}' is a Vision-Language Model. The Text Chatbot only supports Causal Language Models (like Qwen or Deepseek)."
            return

        # Check if we need to load or switch models
        if clean_model_id not in vram_manager.models:
            yield f"Loading {clean_model_id} into System RAM (CPU)..."
            try:
                self.is_loading = True
                # Disable 4-bit quantization and force to CPU
                _MODEL, _TOKENIZER = vram_manager.get_or_load_model(clean_model_id, use_4bit=False, force_cpu=True)
                self.is_loading = False
            except Exception as e:
                self.is_loading = False
                yield f"CRITICAL ERROR: Failed to mount LLM {active_model_id} into MI300X VRAM. Details: {str(e)}"
                return
        
        _MODEL = vram_manager.models[clean_model_id]
        _TOKENIZER = vram_manager.tokenizers[clean_model_id]
        
        # If we successfully loaded the real model
        try:
            import torch
            # Basic prompt formatting (very naive, depends on the model's actual chat template)
            prompt = ""
            for user_msg, bot_msg in history:
                prompt += f"User: {user_msg}\nAssistant: {bot_msg}\n"
            prompt += f"User: {message}\nAssistant:"
            
            # Send inputs to the same device as the model
            model_device = next(_MODEL.parameters()).device
            inputs = _TOKENIZER(prompt, return_tensors="pt").to(model_device)
            
            # We would typically use TextIteratorStreamer here, but for simplicity we generate and return
            # To actually stream token-by-token with HF is complex, we'll yield chunks of the final output
            from transformers import TextIteratorStreamer
            streamer = TextIteratorStreamer(_TOKENIZER, skip_prompt=True, skip_special_tokens=True)
            
            generation_kwargs = dict(
                inputs, 
                streamer=streamer, 
                max_new_tokens=512, 
                temperature=0.7,
                do_sample=True
            )
            
            thread = threading.Thread(target=_MODEL.generate, kwargs=generation_kwargs)
            thread.start()
            
            generated_text = ""
            for new_text in streamer:
                generated_text += new_text
                yield generated_text
                
        except Exception as e:
            yield f"CRITICAL INFERENCE ERROR: {str(e)}"
