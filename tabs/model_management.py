import gradio as gr
import psutil
import shutil
import os
from pathlib import Path

# Mock global state for models
PROTECTED_MODELS = ["meta-llama/Llama-2-7b-chat-hf", "mistralai/Mistral-7B-v0.1", "google/gemma-7b"]
USER_MODELS = ["gpt2", "facebook/opt-125m"]
ACTIVE_MODEL = "gpt2"

def clear_hf_cache():
    cache_dir = os.path.join(str(Path.home()), ".cache", "huggingface", "hub")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir, ignore_errors=True)
        return "System Cache Cleared. All default huggingface models uninstalled. They will re-download when used."
    return "Cache directory not found. No models are currently installed."

def get_system_stats():
    # RAM Usage
    ram = psutil.virtual_memory()
    ram_percent = ram.percent
    ram_used_gb = ram.used / (1024**3)
    ram_total_gb = ram.total / (1024**3)
    
    # Disk Usage
    disk = shutil.disk_usage("/")
    disk_percent = (disk.used / disk.total) * 100
    disk_used_gb = disk.used / (1024**3)
    disk_total_gb = disk.total / (1024**3)
    
    # VRAM Usage (Attempting to use nvidia-smi via torch if available)
    vram_info = "N/A (No GPU detected)"
    try:
        import torch
        if torch.cuda.is_available():
            vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            vram_reserved = torch.cuda.memory_reserved(0) / (1024**3)
            vram_allocated = torch.cuda.memory_allocated(0) / (1024**3)
            vram_info = f"{vram_allocated:.1f} GB allocated / {vram_total:.1f} GB total"
    except ImportError:
        pass
        
    stats = f"""
### Resource Usage
* **RAM Usage**: {ram_percent}% ({ram_used_gb:.1f} GB / {ram_total_gb:.1f} GB)
* **Disk Usage**: {disk_percent:.1f}% ({disk_used_gb:.1f} GB / {disk_total_gb:.1f} GB)
* **VRAM Usage**: {vram_info}
"""
    return stats

def install_model(hf_id):
    if not hf_id:
        return "Please provide a Hugging Face Model ID.", gr.update()
    if hf_id in USER_MODELS or hf_id in PROTECTED_MODELS:
        return f"Model '{hf_id}' is already installed.", gr.update()
    
    USER_MODELS.append(hf_id)
    return f"Successfully installed user model: {hf_id}", gr.update(choices=USER_MODELS)

def delete_model(model_id):
    global ACTIVE_MODEL
    if not model_id:
        return "No model selected.", gr.update()
    if model_id in USER_MODELS:
        USER_MODELS.remove(model_id)
        if ACTIVE_MODEL == model_id:
            ACTIVE_MODEL = None
        return f"Successfully deleted user model: {model_id}", gr.update(choices=USER_MODELS)
    return f"Cannot delete model '{model_id}'. It might be a protected system model or does not exist.", gr.update()

def activate_model(model_id):
    global ACTIVE_MODEL
    if not model_id:
        return "No model selected."
    if model_id in PROTECTED_MODELS or model_id in USER_MODELS:
        ACTIVE_MODEL = model_id
        return f"Successfully activated model: {model_id}"
    return f"Model '{model_id}' not found."

def create_model_management_tab():
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Protected System Models")
            sys_models = gr.Dropdown(choices=PROTECTED_MODELS, label="System Models", interactive=False)
            
            with gr.Row():
                sys_activate_btn = gr.Button("Activate System Model", variant="secondary")
                sys_clear_btn = gr.Button("Uninstall Default Models", variant="stop")
            
            gr.Markdown("### User Models")
            usr_models = gr.Dropdown(choices=USER_MODELS, label="Your Models", interactive=True)
            
            with gr.Row():
                usr_activate_btn = gr.Button("Activate User Model", variant="primary")
                usr_delete_btn = gr.Button("Delete User Model", variant="stop")
            
            action_out = gr.Textbox(label="Action Status", interactive=False)
            
            sys_activate_btn.click(fn=activate_model, inputs=sys_models, outputs=action_out)
            sys_clear_btn.click(fn=clear_hf_cache, outputs=action_out)
            usr_activate_btn.click(fn=activate_model, inputs=usr_models, outputs=action_out)
            usr_delete_btn.click(fn=delete_model, inputs=usr_models, outputs=[action_out, usr_models])
            
        with gr.Column():
            gr.Markdown("### Install Model from Hugging Face")
            new_model_id = gr.Textbox(label="Hugging Face Model ID (e.g., t5-small)")
            install_btn = gr.Button("Install Model")
            install_btn.click(fn=install_model, inputs=new_model_id, outputs=[action_out, usr_models])
            
            gr.Markdown("---")
            stats_out = gr.Markdown(get_system_stats())
            refresh_btn = gr.Button("Refresh Resource Stats")
            refresh_btn.click(fn=get_system_stats, outputs=stats_out)
