import gradio as gr
import time
from agents.qlora_trainer import QLoRATrainer
from tabs.model_management import get_cached_hf_models, refresh_cached_models

trainer = QLoRATrainer()

def run_training_ui(model_id, dataset_key, dataset_file, username):
    import sys
    import pandas as pd
    from data.dataset_manager import get_user_workspace
    if dataset_key and dataset_key in get_user_workspace(username):
        dataset_name = dataset_key
    elif dataset_file is not None:
        import os
        dataset_name = os.path.basename(dataset_file.name)
        
        # Sync into USER_WORKSPACE_DATA
        try:
            if dataset_name.endswith(".csv"):
                df = pd.read_csv(dataset_file.name)
            else:
                df = pd.read_json(dataset_file.name)
            get_user_workspace(username)[dataset_name] = df
        except:
            pass
            
    else:
        yield "Error: Please select a workspace dataset or upload a file."
        return
        
    if not model_id or "No models" in model_id:
        yield "Error: Please install a model in Model Management first."
        return
        
    # Start background job
    trainer.start_training(dataset_name, model_id)
    
    # Stream UI updates
    while trainer.is_training:
        time.sleep(0.5)
        bar = "█" * int(trainer.progress_percent / 2) + "░" * (50 - int(trainer.progress_percent / 2))
        display = f"{trainer.status_message}\n\n[{bar}] {trainer.progress_percent}%"
        yield display
        
    yield f"{trainer.status_message}\n\n[██████████████████████████████████████████████████] 100%"

def create_qlora_tab(session_user):
    gr.Markdown("### QLoRA Neural Rewiring Studio")
    gr.Markdown("Active Fine-Tuning Environment. Train your own custom financial crime detection intelligence. The system automatically detects your latest dataset and handles the entire 50GB VRAM lifecycle (Mounting -> Freezing -> Training -> Demounting).")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("#### Step 1: Data Target")
            with gr.Row():
                from data.dataset_manager import get_user_workspace
                ds_dropdown = gr.Dropdown(choices=[], label="Select Workspace Dataset", scale=4)
                refresh_btn = gr.Button("↻", size="sm", scale=1)
            ds_upload = gr.File(label="Or Upload Direct File")
            
            gr.Markdown("#### Step 2: Select Base Brain")
            model_dropdown = gr.Dropdown(choices=get_cached_hf_models(), label="Select Base AI Model")
            refresh_models_btn = gr.Button("↻ Refresh Models", size="sm")
            
            gr.Markdown("#### Step 3: Execute")
            start_btn = gr.Button("🚀 Start Neural Rewiring", variant="primary")
            
            refresh_btn.click(fn=lambda u: gr.update(choices=list(get_user_workspace(u).keys())), inputs=session_user, outputs=ds_dropdown)
            
        with gr.Column(scale=2):
            gr.Markdown("#### Live Training Telemetry")
            telemetry_out = gr.Textbox(label="VRAM Lifecycle Status", lines=10, interactive=False)
            
    refresh_models_btn.click(fn=refresh_cached_models, outputs=model_dropdown)
    
    start_btn.click(fn=run_training_ui, inputs=[model_dropdown, ds_dropdown, ds_upload, session_user], outputs=telemetry_out)
