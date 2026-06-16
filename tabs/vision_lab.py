import gradio as gr
import time
import pandas as pd
import threading
from agents.vision_forensics import VisionForensicsEngine

vision_engine = VisionForensicsEngine()

def run_vision_lab(dataset_key, file, username):
    if vision_engine.is_running:
        yield "Already scanning...", pd.DataFrame()
        return
        
    if file is not None:
        import os
        from data.dataset_manager import get_user_workspace
        filename = os.path.basename(file.name)
        try:
            if filename.endswith(".csv"):
                df = pd.read_csv(file.name)
            else:
                df = pd.read_json(file.name)
            get_user_workspace(username)[filename] = df
        except:
            pass # fallback to engine processing if invalid
            
    yield "Mounting Multi-Modal Vision Engine to MI300X VRAM...", pd.DataFrame()
    
    t = threading.Thread(target=vision_engine.run_mass_forensics)
    t.start()
    
    while True:
        time.sleep(0.5)
        bar_len = int((vision_engine.processed_count / vision_engine.total_documents) * 40)
        bar = "█" * bar_len + "░" * (40 - bar_len)
        progress = f"{vision_engine.status_message}\n[{bar}] {vision_engine.processed_count} / {vision_engine.total_documents} Scanned"
        
        current_df = pd.DataFrame(vision_engine.findings)
        yield progress, current_df
        
        if not vision_engine.is_running and vision_engine.processed_count >= vision_engine.total_documents:
            break
            
    t.join()
    final_df = pd.DataFrame(vision_engine.findings)
    yield f"COMPLETE: Scanned {vision_engine.total_documents} documents across MI300X HBM3 Memory.", final_df

def create_vision_lab_tab(session_user):
    gr.Markdown("### 👁️ MI300X Document Forensics Lab")
    gr.Markdown("Utilize your **192 GB of VRAM** to load massively parallel Vision-Language Models (VLMs). Because of the MI300X's memory bandwidth, we can batch-scan **10,000** High-Resolution Passports and KYC Documents simultaneously to detect AI-generated deep-fakes and pixel manipulations.")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("#### Document Batch Loader")
            
            with gr.Row():
                from data.dataset_manager import get_user_workspace
                ds_dropdown = gr.Dropdown(choices=[], label="Select Workspace Dataset", scale=4)
                refresh_btn = gr.Button("↻", size="sm", scale=1)
                
            ds_upload = gr.File(label="Or Upload Direct File")
            
            start_btn = gr.Button("🚀 Launch 10k Vision Batch Scan", variant="primary")
            
            refresh_btn.click(fn=lambda u: gr.update(choices=list(get_user_workspace(u).keys())), inputs=session_user, outputs=ds_dropdown)
            status_out = gr.Textbox(label="VLM Telemetry", lines=4, interactive=False)
            
        with gr.Column(scale=2):
            results_table = gr.Dataframe(label="Deep-Fake Detection Results", max_height=300)
            
    start_btn.click(fn=run_vision_lab, inputs=[ds_dropdown, ds_upload, session_user], outputs=[status_out, results_table])
