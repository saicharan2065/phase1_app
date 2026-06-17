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
        
        clean_model_id = active_model_id.split(" (")[0] if active_model_id and " (" in active_model_id else active_model_id
        
        # Prevent Vision Models in Text Chatbot
        if "llava" in clean_model_id.lower() or "vision" in clean_model_id.lower():
            yield f"ERROR: '{clean_model_id}' is a Vision-Language Model. The Text Chatbot only supports Causal Language Models."
            return

        # Check if we need to load the model
        if clean_model_id not in vram_manager.models:
            yield f"Loading {clean_model_id} into System RAM (CPU)... Please wait."
            try:
                self.is_loading = True
                _MODEL, _TOKENIZER = vram_manager.get_or_load_model(clean_model_id, use_4bit=False, force_cpu=True)
                self.is_loading = False
            except Exception as e:
                self.is_loading = False
                yield f"CRITICAL ERROR: Failed to load {clean_model_id}. Details: {str(e)}"
                return
        
        _MODEL = vram_manager.models[clean_model_id]
        _TOKENIZER = vram_manager.tokenizers[clean_model_id]
        
        try:
            import torch
            
            # Use Qwen's native chat template for proper formatting
            messages = []
            messages.append({"role": "system", "content": "You are a helpful financial compliance AI assistant. Keep answers concise and relevant."})
            if history:
                for user_msg, bot_msg in history:
                    if user_msg:
                        messages.append({"role": "user", "content": user_msg})
                    if bot_msg:
                        messages.append({"role": "assistant", "content": bot_msg})
            messages.append({"role": "user", "content": message})
            
            # Try using the tokenizer's built-in chat template
            try:
                prompt = _TOKENIZER.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                # Fallback to simple format if no chat template
                prompt = f"User: {message}\nAssistant:"
            
            # Send inputs to the same device as the model
            model_device = next(_MODEL.parameters()).device
            inputs = _TOKENIZER(prompt, return_tensors="pt", truncation=True, max_length=512).to(model_device)
            
            from transformers import TextIteratorStreamer
            streamer = TextIteratorStreamer(_TOKENIZER, skip_prompt=True, skip_special_tokens=True)
            
            generation_kwargs = dict(
                inputs, 
                streamer=streamer, 
                max_new_tokens=150,  # Keep responses short and focused
                temperature=0.7,
                do_sample=True,
                repetition_penalty=1.2,  # Prevent repetitive nonsense
                eos_token_id=_TOKENIZER.eos_token_id,
                pad_token_id=_TOKENIZER.eos_token_id,
            )
            
            thread = threading.Thread(target=_MODEL.generate, kwargs=generation_kwargs)
            thread.start()
            
            generated_text = ""
            for new_text in streamer:
                generated_text += new_text
                yield generated_text
                
        except Exception as e:
            yield f"CRITICAL INFERENCE ERROR: {str(e)}"
