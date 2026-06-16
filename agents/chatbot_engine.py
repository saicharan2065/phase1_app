import time
import threading

# We track the globally loaded model to avoid re-loading 70B parameters on every chat message
_LOADED_MODEL_ID = None
_MODEL = None
_TOKENIZER = None

class ChatbotEngine:
    def __init__(self):
        self.is_loading = False
        
    def _load_model(self, model_id):
        global _LOADED_MODEL_ID, _MODEL, _TOKENIZER
        if _LOADED_MODEL_ID == model_id:
            return True, "Model already loaded."
            
        self.is_loading = True
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # Simulated 192GB VRAM optimization: device_map="auto" and fp16
            _TOKENIZER = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            _MODEL = AutoModelForCausalLM.from_pretrained(
                model_id, 
                device_map="auto", 
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            
            _LOADED_MODEL_ID = model_id
            self.is_loading = False
            return True, f"Successfully loaded {model_id} into MI300X VRAM."
        except Exception as e:
            self.is_loading = False
            return False, str(e)
            
    def generate_response(self, message, history, active_model_id, username="GUEST"):
        """
        Stream back response for gradio ChatInterface
        """
        if not active_model_id or active_model_id == "None Selected":
            yield "Please select an Active Model in the Model Management tab first!"
            return

        global _LOADED_MODEL_ID, _MODEL, _TOKENIZER
        
        # Check if we need to load or switch models
        if _LOADED_MODEL_ID != active_model_id:
            yield f"Loading {active_model_id} into 192GB VRAM... This may take a minute for 70B parameters..."
            success, msg = self._load_model(active_model_id)
            if not success:
                yield f"CRITICAL ERROR: Failed to mount LLM {active_model_id} into MI300X VRAM. Details: {msg}"
                return
                
        # If we successfully loaded the real model
        try:
            import torch
            # Basic prompt formatting (very naive, depends on the model's actual chat template)
            prompt = ""
            for user_msg, bot_msg in history:
                prompt += f"User: {user_msg}\nAssistant: {bot_msg}\n"
            prompt += f"User: {message}\nAssistant:"
            
            inputs = _TOKENIZER(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
            
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
