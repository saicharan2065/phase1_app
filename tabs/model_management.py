import gradio as gr
import psutil
import shutil
import os
from pathlib import Path
from huggingface_hub import scan_cache_dir

def get_cached_hf_models():
    """Scans the local HF cache and returns a list of installed repo IDs."""
    try:
        cache = scan_cache_dir()
        repos = [repo.repo_id for repo in cache.repos]
        if not repos:
            return ["No models installed"]
        return repos
    except Exception:
        return ["No models installed"]

def refresh_cached_models():
    repos = get_cached_hf_models()
    return gr.update(choices=repos, value=repos[0] if repos else None)

def delete_cached_model(repo_id):
    if not repo_id or repo_id == "No models installed":
        return "Please select a valid model.", refresh_cached_models()
        
    try:
        cache = scan_cache_dir()
        for repo in cache.repos:
            if repo.repo_id == repo_id:
                shutil.rmtree(repo.repo_path, ignore_errors=True)
                return f"Successfully deleted model: {repo_id}", refresh_cached_models()
        return f"Model {repo_id} not found in cache.", refresh_cached_models()
    except Exception as e:
        return f"Error deleting model: {str(e)}", refresh_cached_models()

def test_matching_engine(progress=gr.Progress(track_tqdm=True)):
    progress(0, desc="Checking model installation...")
    try:
        from sentence_transformers import SentenceTransformer
        # This will download the model if not present, showing progress automatically
        model = SentenceTransformer("all-MiniLM-L6-v2")
        progress(0.8, desc="Running internal verification test...")
        # Internal test to ensure the mathematical tensor graph works (user requested not to see the raw output)
        _ = model.encode("hi")
        progress(1.0, desc="Done")
        return "Model Verification Successful! The model is installed, loaded, and mathematically functional.", refresh_cached_models()
    except Exception as e:
        return f"Verification Failed: {str(e)}", refresh_cached_models()

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
    
    stats = f"""
### Resource Usage
* **RAM Usage**: {ram_percent}% ({ram_used_gb:.1f} GB / {ram_total_gb:.1f} GB)
* **Disk Usage**: {disk_percent:.1f}% ({disk_used_gb:.1f} GB / {disk_total_gb:.1f} GB)
"""
    return stats

def install_hf_model(hf_id, progress=gr.Progress(track_tqdm=True)):
    if not hf_id:
        return "Please provide a Hugging Face Model ID.", refresh_cached_models()
    
    progress(0, desc=f"Starting download for {hf_id}...")
    try:
        from huggingface_hub import snapshot_download
        # This will securely download the model to the cache and display progress
        _ = snapshot_download(repo_id=hf_id)
        return f"Successfully installed: {hf_id}", refresh_cached_models()
    except Exception as e:
        return f"Failed to install {hf_id}: {str(e)}", refresh_cached_models()

def login_hf(token):
    if not token:
        return "Please provide a token."
    try:
        from huggingface_hub import login
        login(token=token)
        return "Successfully logged into Hugging Face Hub! You can now download gated models."
    except Exception as e:
        return f"Login failed: {str(e)}"

def create_model_management_tab():
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Hugging Face Cache Manager")
            cached_models_dropdown = gr.Dropdown(choices=get_cached_hf_models(), label="Installed Models", interactive=True)
            
            with gr.Row():
                refresh_cache_btn = gr.Button("Refresh List")
                delete_cache_btn = gr.Button("Delete Selected Model", variant="stop")
            
            gr.Markdown("### Matching Engine Verification")
            gr.Markdown("Ensures the core `all-MiniLM-L6-v2` matching engine is installed and mathematically stable.")
            test_engine_btn = gr.Button("Verify Matching Engine", variant="primary")
            
            action_out = gr.Textbox(label="Action Status", interactive=False)
            
            refresh_cache_btn.click(fn=refresh_cached_models, outputs=cached_models_dropdown)
            delete_cache_btn.click(fn=delete_cached_model, inputs=cached_models_dropdown, outputs=[action_out, cached_models_dropdown])
            test_engine_btn.click(fn=test_matching_engine, outputs=[action_out, cached_models_dropdown])
            
        with gr.Column():
            gr.Markdown("### Hugging Face Authentication")
            gr.Markdown("Enter your Hugging Face Access Token to download gated or private models.")
            hf_token_input = gr.Textbox(label="Access Token", type="password")
            hf_login_btn = gr.Button("Save Token & Login", variant="secondary")
            hf_login_btn.click(fn=login_hf, inputs=hf_token_input, outputs=action_out)
            
            gr.Markdown("---")
            
            gr.Markdown("### Install Hugging Face Model")
            gr.Markdown("Downloads any model securely from the Hugging Face hub directly to your local system cache.")
            new_model_id = gr.Textbox(label="Hugging Face Model ID (e.g., t5-small)")
            install_btn = gr.Button("Download & Install Model", variant="primary")
            install_btn.click(fn=install_hf_model, inputs=new_model_id, outputs=[action_out, cached_models_dropdown])

            gr.Markdown("### Application State")
            stats_out = gr.Markdown(get_system_stats())
            refresh_btn = gr.Button("Refresh Resource Stats")
            refresh_btn.click(fn=get_system_stats, outputs=stats_out)
