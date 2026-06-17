import gradio as gr
import time
import pandas as pd
import threading
from agents.vision_forensics import VisionForensicsEngine

vision_engine = VisionForensicsEngine()

def run_vision_lab(dataset_key, file, active_vlm, username):
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
            
    # Extract clean model ID
    clean_vlm = active_vlm.split(" (")[0] if active_vlm and " (" in active_vlm else active_vlm
    if clean_vlm == "None Selected" or clean_vlm == "No models installed" or not clean_vlm:
        clean_vlm = "llava-hf/llava-1.5-13b-hf"
            
    yield f"Mounting Multi-Modal Vision Engine ({clean_vlm}) to MI300X VRAM...", pd.DataFrame()
    
    t = threading.Thread(target=vision_engine.run_mass_forensics, kwargs={"model_id": clean_vlm})
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
    
    from tabs.model_management import get_cached_hf_models, delete_cached_model, install_hf_model
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("#### VLM Engine Management")
            with gr.Group():
                vlm_dropdown = gr.Dropdown(choices=get_cached_hf_models(), label="Active VLM Model", value="llava-hf/llava-1.5-13b-hf")
                vlm_refresh_btn = gr.Button("↻ Refresh Models", size="sm")
                
                with gr.Row():
                    vlm_delete_btn = gr.Button("🗑️ Delete Selected VLM")
                    vlm_delete_status = gr.Textbox(label="Status", interactive=False)
                    
                gr.Markdown("---")
                vlm_install_box = gr.Textbox(label="Download New VLM (Hugging Face ID)")
                vlm_install_btn = gr.Button("↓ Download & Install", variant="secondary")
                vlm_install_status = gr.Textbox(label="Install Status", interactive=False)
                
            gr.Markdown("#### Document Batch Loader")
            
            with gr.Row():
                from data.dataset_manager import get_user_workspace
                ds_dropdown = gr.Dropdown(choices=[], label="Select Workspace Dataset", scale=4)
                refresh_btn = gr.Button("↻", size="sm", scale=1)
                
            ds_upload = gr.File(label="Or Upload Direct File")
            
            start_btn = gr.Button("🚀 Launch 10k Vision Batch Scan", variant="primary")
            
            refresh_btn.click(fn=lambda u: gr.update(choices=list(get_user_workspace(u).keys())), inputs=session_user, outputs=ds_dropdown)
            status_out = gr.Textbox(label="VLM Telemetry", lines=4, interactive=False)
            
            # Wires for VLM Management
            vlm_refresh_btn.click(fn=lambda: gr.update(choices=get_cached_hf_models()), outputs=vlm_dropdown)
            vlm_delete_btn.click(fn=delete_cached_model, inputs=vlm_dropdown, outputs=vlm_delete_status).then(
                fn=lambda: gr.update(choices=get_cached_hf_models()), outputs=vlm_dropdown
            )
            vlm_install_btn.click(fn=install_hf_model, inputs=vlm_install_box, outputs=vlm_install_status).then(
                fn=lambda: gr.update(choices=get_cached_hf_models()), outputs=vlm_dropdown
            )
            
        with gr.Column(scale=2):
            results_table = gr.Dataframe(label="Deep-Fake Detection Results", max_height=300)
            
    start_btn.click(fn=run_vision_lab, inputs=[ds_dropdown, ds_upload, vlm_dropdown, session_user], outputs=[status_out, results_table])
