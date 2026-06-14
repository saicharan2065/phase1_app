import gradio as gr
import psutil
import shutil
import os
from pathlib import Path
from huggingface_hub import scan_cache_dir

GLOBAL_ACTIVE_MODEL = "None Selected"

def get_active_model_state():
    return GLOBAL_ACTIVE_MODEL

def set_active_model(repo_id):
    global GLOBAL_ACTIVE_MODEL
    if not repo_id or repo_id == "No models installed":
        return "Please select a valid model."
    clean_repo_id = repo_id.split(" (")[0] if " (" in repo_id else repo_id
    GLOBAL_ACTIVE_MODEL = clean_repo_id
    return f"Successfully set active model to: {clean_repo_id}"

def get_cached_hf_models():
    """Scans the local HF cache and returns a list of installed repo IDs with sizes in GB."""
    try:
        cache = scan_cache_dir()
        repos = []
        for repo in cache.repos:
            if getattr(repo, "repo_type", "model") != "model":
                continue
            size_gb = repo.size_on_disk / (1024**3)
            repos.append(f"{repo.repo_id} ({size_gb:.2f} GB)")
            
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
        return "Please select a valid model."
        
    try:
        cache = scan_cache_dir()
        
        # Extract the pure repo_id if it has the " (X.XX GB)" suffix
        clean_repo_id = repo_id.split(" (")[0] if " (" in repo_id else repo_id
        
        for repo in cache.repos:
            if repo.repo_id == clean_repo_id:
                shutil.rmtree(repo.repo_path, ignore_errors=True)
                return f"Successfully deleted model: {clean_repo_id}"
        return f"Model {clean_repo_id} not found in cache."
    except Exception as e:
        return f"Error deleting model: {str(e)}"

def test_matching_engine(progress=gr.Progress(track_tqdm=True)):
    progress(0, desc="Checking model installation...")
    try:
        from sentence_transformers import SentenceTransformer
        # This will download the model if not present, showing progress automatically
        model = SentenceTransformer("all-MiniLM-L6-v2")
        progress(0.8, desc="Running internal verification test...")
        # Internal test to ensure the mathematical tensor graph works
        _ = model.encode("hi")
        progress(1.0, desc="Done")
        return "Model Verification Successful! The model is installed, loaded, and mathematically functional."
    except Exception as e:
        return f"Verification Failed: {str(e)}"

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

def install_hf_model(hf_id, progress=gr.Progress()):
    if not hf_id:
        return "Please provide a Hugging Face Model ID."
    
    progress(0, desc=f"Fetching metadata for {hf_id}...")
    try:
        from huggingface_hub import HfApi, snapshot_download
        import concurrent.futures
        import os
        import time
        from pathlib import Path
        
        api = HfApi()
        info = api.model_info(repo_id=hf_id, files_metadata=True)
        
        files_to_download = [f for f in info.siblings]
        total_bytes = sum((f.size or 0) for f in files_to_download)
        total_gb = total_bytes / (1024**3)
        if total_bytes <= 0: total_bytes = 1
        
        safe_repo_name = hf_id.replace("/", "--")
        cache_dir = os.path.join(str(Path.home()), ".cache", "huggingface", "hub", f"models--{safe_repo_name}")
        
        # Get baseline size (if partial download exists)
        baseline_bytes = 0
        if os.path.exists(cache_dir):
            for dirpath, _, filenames in os.walk(cache_dir):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    if not os.path.islink(fp):
                        baseline_bytes += os.path.getsize(fp)
                        
        last_bytes = baseline_bytes
        last_time = time.time()
        speed_str = "0.00 MB/s"
        
        # Run standard snapshot_download in background so we can track disk bytes live
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(snapshot_download, repo_id=hf_id)
            
            while not future.done():
                current_bytes = 0
                if os.path.exists(cache_dir):
                    for dirpath, _, filenames in os.walk(cache_dir):
                        for f in filenames:
                            fp = os.path.join(dirpath, f)
                            if not os.path.islink(fp):  # Avoid double-counting HF symlinks
                                current_bytes += os.path.getsize(fp)
                                
                display_bytes = min(current_bytes, total_bytes)
                current_gb = display_bytes / (1024**3)
                
                current_time = time.time()
                time_diff = current_time - last_time
                
                # Update speed every 1 second
                if time_diff >= 1.0:
                    bytes_diff = current_bytes - last_bytes
                    speed_mb_s = (bytes_diff / (1024**2)) / time_diff
                    if speed_mb_s < 0: speed_mb_s = 0
                    speed_str = f"{speed_mb_s:.2f} MB/s"
                    
                    last_bytes = current_bytes
                    last_time = current_time
                
                progress(display_bytes / total_bytes, desc=f"Downloading... {current_gb:.2f} / {total_gb:.2f} GB | Speed: {speed_str}")
                time.sleep(0.5)
                
            future.result() # Raise any exceptions from the download thread
            
        progress(1.0, desc=f"Download Complete: {total_gb:.2f} / {total_gb:.2f} GB")
        return f"Successfully installed: {hf_id}"
    except Exception as e:
        return f"Failed to install {hf_id}: {str(e)}"

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
        action_out = gr.Textbox(label="Global Action Status", interactive=False)
        
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Hugging Face Cache Manager")
            cached_models_dropdown = gr.Dropdown(choices=get_cached_hf_models(), label="Installed Models", interactive=True)
            
            with gr.Row():
                refresh_cache_btn = gr.Button("Refresh List")
                set_active_btn = gr.Button("Set as Active", variant="primary")
                delete_cache_btn = gr.Button("Delete Selected", variant="stop")
            
            gr.Markdown("### Matching Engine Verification")
            gr.Markdown("Ensures the core `all-MiniLM-L6-v2` matching engine is installed and mathematically stable.")
            test_engine_btn = gr.Button("Verify Matching Engine", variant="primary")
            
        with gr.Column():
            gr.Markdown("### Hugging Face Authentication")
            gr.Markdown("Enter your Hugging Face Access Token to download gated or private models.")
            hf_token_input = gr.Textbox(label="Access Token", type="password")
            hf_login_btn = gr.Button("Save Token & Login", variant="secondary")
            
            gr.Markdown("---")
            
            gr.Markdown("### Install Hugging Face Model")
            gr.Markdown("Downloads any model securely from the Hugging Face hub directly to your local system cache.")
            new_model_id = gr.Textbox(label="Hugging Face Model ID (e.g., t5-small)")
            install_btn = gr.Button("Download & Install Model", variant="primary")

            gr.Markdown("### Application State")
            stats_out = gr.Markdown(get_system_stats())
            refresh_btn = gr.Button("Refresh Resource Stats")
            
    refresh_cache_btn.click(fn=refresh_cached_models, outputs=cached_models_dropdown)
    
    set_active_btn.click(fn=set_active_model, inputs=cached_models_dropdown, outputs=action_out)
    
    delete_cache_btn.click(fn=delete_cached_model, inputs=cached_models_dropdown, outputs=action_out).then(
        fn=refresh_cached_models, outputs=cached_models_dropdown
    )
    
    test_engine_btn.click(fn=test_matching_engine, outputs=action_out).then(
        fn=refresh_cached_models, outputs=cached_models_dropdown
    )
    
    hf_login_btn.click(fn=login_hf, inputs=hf_token_input, outputs=action_out)
    
    install_btn.click(fn=install_hf_model, inputs=new_model_id, outputs=action_out).then(
        fn=refresh_cached_models, outputs=cached_models_dropdown
    )
    
    refresh_btn.click(fn=get_system_stats, outputs=stats_out)
