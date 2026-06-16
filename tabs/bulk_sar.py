import gradio as gr
import pandas as pd
import time
import threading
from agents.bulk_sar_generator import BulkSARGenerator
from data.dataset_manager import DatasetManager

generator = BulkSARGenerator()
dm = DatasetManager()

def get_cached_hf_datasets():
    import os
    from huggingface_hub import scan_cache_dir
    try:
        cache = scan_cache_dir()
        repos = []
        for repo in cache.repos:
            if getattr(repo, "repo_type", "model") == "dataset":
                repos.append(repo.repo_id)
        if not repos:
            return ["No datasets found. Download in Dataset Marketplace first."]
        return repos
    except Exception:
        return ["No datasets found."]

def load_ds_preview(dataset_id):
    if not dataset_id or "No datasets" in dataset_id:
        return pd.DataFrame()
    df = dm.load_dataset_records(dataset_id, "100")
    return df

def run_vram_inference(dataset_id):
    if not dataset_id or "No datasets" in dataset_id:
        yield "Error: Select a valid dataset.", pd.DataFrame()
        return
        
    df = dm.load_dataset_records(dataset_id, "100") # Cap at 100 for fast demo
    if df.empty or "Error" in df.columns:
        yield "Error loading dataset.", pd.DataFrame()
        return
        
    suspect_ids = [f"SUSPECT_{i}" for i in range(len(df))]
    
    yield "Initializing 4-Bit VRAM Allocation... (Loading 131GB weights)", pd.DataFrame()
    
    t = threading.Thread(target=generator.run_bulk_inference, args=(suspect_ids, 32))
    t.start()
    
    while True:
        time.sleep(0.5)
        progress = f"VRAM Batch Processing: {generator.processed_count} / {len(suspect_ids)} completed."
        current_df = pd.DataFrame(generator.results)
        yield progress, current_df
        
        if not generator.is_running and generator.processed_count >= len(suspect_ids):
            break
            
    t.join()
    final_df = pd.DataFrame(generator.results)
    yield "Bulk VRAM Inference Complete!", final_df

def create_bulk_sar_tab(session_user=None):
    gr.Markdown("### Advanced VRAM Inference Engine")
    gr.Markdown("Select a real dataset downloaded from the **Dataset Marketplace**. This engine uses 4-Bit GPU Batching to analyze thousands of Suspects concurrently without crashing the webpage.")
    
    with gr.Row():
        with gr.Column(scale=1):
            ds_dropdown = gr.Dropdown(choices=get_cached_hf_datasets(), label="Select Cached Hugging Face Dataset")
            refresh_btn = gr.Button("Refresh Datasets", size="sm")
            run_btn = gr.Button("Launch VRAM Batch Inference", variant="primary")
            status_out = gr.Textbox(label="VRAM Status", interactive=False)
            
        with gr.Column(scale=2):
            preview_table = gr.Dataframe(label="Dataset Preview (First 100 rows)", max_height=200)
            
    refresh_btn.click(fn=lambda: gr.update(choices=get_cached_hf_datasets()), outputs=ds_dropdown)
    ds_dropdown.change(fn=load_ds_preview, inputs=ds_dropdown, outputs=preview_table)
    
    gr.Markdown("### Mass Batch Inference Results")
    results_table = gr.Dataframe(label="Generated Suspicious Activity Reports (SARs)")
    
    run_btn.click(fn=run_vram_inference, inputs=ds_dropdown, outputs=[status_out, results_table])
