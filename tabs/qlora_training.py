import gradio as gr
import time
from agents.qlora_trainer import QLoRATrainer
from tabs.model_management import get_cached_hf_models, refresh_cached_models

trainer = QLoRATrainer()

def auto_detect_dataset():
    import os
    from huggingface_hub import scan_cache_dir
    try:
        cache = scan_cache_dir()
        latest_ds = None
        latest_time = 0
        for repo in cache.repos:
            if getattr(repo, "repo_type", "model") == "dataset":
                if repo.last_modified > latest_time:
                    latest_time = repo.last_modified
                    latest_ds = repo.repo_id
        if latest_ds:
            return latest_ds
        return "No datasets found."
    except Exception:
        return "No datasets found."

def run_training_ui(model_id, dataset_key, dataset_file):
    dataset_id = auto_detect_dataset()
    if dataset_key and dataset_key in getattr(sys.modules.get('data.dataset_manager'), 'GLOBAL_WORKSPACE_DATA', {}):
        pass # They selected a valid dataset
    elif dataset_file is not None:
        pass # They uploaded one
    elif not dataset_id or "No datasets" in dataset_id:
        yield "Error: Please select a workspace dataset, upload a file, or download one in the Dataset Marketplace."
        return
        
    if not model_id or "No models" in model_id:
        yield "Error: Please install a model in Model Management first."
        return
        
    # Start background job
    trainer.start_training("Local_Data_Source" if (dataset_key or dataset_file) else dataset_id, model_id)
    
    # Stream UI updates
    while trainer.is_training:
        time.sleep(0.5)
        bar = "█" * int(trainer.progress_percent / 2) + "░" * (50 - int(trainer.progress_percent / 2))
        display = f"{trainer.status_message}\n\n[{bar}] {trainer.progress_percent}%"
        yield display
        
    yield f"{trainer.status_message}\n\n[██████████████████████████████████████████████████] 100%"

def create_qlora_tab():
    gr.Markdown("### QLoRA Neural Rewiring Studio")
    gr.Markdown("Active Fine-Tuning Environment. Train your own custom financial crime detection intelligence. The system automatically detects your latest dataset and handles the entire 50GB VRAM lifecycle (Mounting -> Freezing -> Training -> Demounting).")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("#### Step 1: Data Target")
            auto_ds = gr.Textbox(label="Auto-Detected HF Target Dataset", value=auto_detect_dataset, interactive=False)
            refresh_ds_btn = gr.Button("↻ Rescan HF Cache", size="sm")
            
            with gr.Row():
                from data.dataset_manager import GLOBAL_WORKSPACE_DATA
                ds_dropdown = gr.Dropdown(choices=list(GLOBAL_WORKSPACE_DATA.keys()), label="Or Workspace Dataset", scale=4)
                refresh_btn = gr.Button("↻", size="sm", scale=1)
            ds_upload = gr.File(label="Or Upload Direct File")
            
            gr.Markdown("#### Step 2: Select Base Brain")
            model_dropdown = gr.Dropdown(choices=get_cached_hf_models(), label="Select Base AI Model")
            refresh_models_btn = gr.Button("↻ Refresh Models", size="sm")
            
            gr.Markdown("#### Step 3: Execute")
            start_btn = gr.Button("🚀 Start Neural Rewiring", variant="primary")
            
            refresh_btn.click(fn=lambda: gr.update(choices=list(GLOBAL_WORKSPACE_DATA.keys())), outputs=ds_dropdown)
            
        with gr.Column(scale=2):
            gr.Markdown("#### Live Training Telemetry")
            telemetry_out = gr.Textbox(label="VRAM Lifecycle Status", lines=10, interactive=False)
            
    refresh_ds_btn.click(fn=auto_detect_dataset, outputs=auto_ds)
    refresh_models_btn.click(fn=refresh_cached_models, outputs=model_dropdown)
    
    start_btn.click(fn=run_training_ui, inputs=[model_dropdown, ds_dropdown, ds_upload], outputs=telemetry_out)
