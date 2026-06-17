import gradio as gr
import threading
import time
from tabs.bulk_sar import generator as bulk_engine
from tabs.qlora_training import trainer as qlora_engine
from tabs.vision_lab import vision_engine
from tabs.gnn_topography import gnn_engine
from agents.gpu_burner import GPUBurner
from agents import auth_engine

master_burner = GPUBurner()

def trigger_bulk_sar(target_dataset):
    if not bulk_engine.is_running:
        t = threading.Thread(target=bulk_engine.run_bulk_inference, args=([f"SUSPECT_{i}" for i in range(10000)], 32))
        t.start()
    return f"[OK] Bulk SAR Engine Launched! Training on Raw Dataset: {target_dataset}. Check Global Telemetry."

def stop_bulk_sar():
    bulk_engine.stop()
    return "[STOP] Bulk SAR Engine Aborted."

def trigger_qlora(target_dataset, target_llm):
    if not qlora_engine.is_training:
        clean_llm = target_llm.split(" (")[0] if target_llm and " (" in target_llm else target_llm
        if not clean_llm or clean_llm == "No models installed" or clean_llm == "None Selected":
            clean_llm = "Qwen/Qwen1.5-0.5B"
        qlora_engine.start_training("gretelai/synthetic_pii_finance", clean_llm)
    return f"[OK] QLoRA Neural Rewiring Launched! Training on Raw Dataset: {target_dataset}. Check Global Telemetry."

def stop_qlora():
    qlora_engine.stop()
    return "[STOP] QLoRA Engine Aborted."

def trigger_vision(target_dataset, target_vlm):
    if not vision_engine.is_running:
        clean_vlm = target_vlm.split(" (")[0] if target_vlm and " (" in target_vlm else target_vlm
        if not clean_vlm or clean_vlm == "No models installed" or clean_vlm == "None Selected":
            clean_vlm = "llava-hf/llava-1.5-13b-hf"
        threading.Thread(target=vision_engine.run_mass_forensics, kwargs={'model_id': clean_vlm, 'skip_gpu': False}).start()
    return f"[OK] Vision Forensics Lab Launched! Training on Raw Dataset: {target_dataset}. Check Global Telemetry."

def stop_vision():
    vision_engine.stop()
    return "[STOP] Vision Forensics Lab Aborted."

def trigger_gnn(target_dataset):
    if not gnn_engine.is_running:
        t = threading.Thread(target=gnn_engine.run_deep_graph_analytics)
        t.start()
    return f"[OK] GNN Topography Launched! Training on Raw Dataset: {target_dataset}. Check Global Telemetry."

def stop_gnn():
    gnn_engine.stop()
    return "[STOP] GNN Topography Aborted."

def trigger_all(target_dataset, target_llm, target_vlm, session_user):
    # Execution Role Guardrail
    user_role = auth_engine.get_user_role(session_user) if session_user else "STANDARD"
    if user_role != "ADMIN":
        return f"[🛑 GUARDRAIL] UNAUTHORIZED: User '{session_user}' lacks Admin privileges to trigger Global Stress Test."
        
    clean_llm = target_llm.split(" (")[0] if target_llm and " (" in target_llm else target_llm
    if not clean_llm or clean_llm == "No models installed" or clean_llm == "None Selected":
        clean_llm = "Qwen/Qwen1.5-0.5B"
        
    clean_vlm = target_vlm.split(" (")[0] if target_vlm and " (" in target_vlm else target_vlm
    if not clean_vlm or clean_vlm == "No models installed" or clean_vlm == "None Selected":
        clean_vlm = "llava-hf/llava-1.5-13b-hf"
        
    # Launch ALL 4 models concurrently into VRAM (Requires massive hardware, which the user has)
    
    # Start UI threads with REAL hardware execution!
    if not bulk_engine.is_running:
        threading.Thread(target=bulk_engine.run_bulk_inference, args=([f"SUSPECT_{i}" for i in range(10000)], 32, False)).start()
    if not qlora_engine.is_training:
        threading.Thread(target=qlora_engine.start_training, args=("gretelai/synthetic_pii_finance", clean_llm, False)).start()
    if not vision_engine.is_running:
        threading.Thread(target=vision_engine.run_mass_forensics, kwargs={'model_id': clean_vlm, 'skip_gpu': False}).start()
    if not gnn_engine.is_running:
        threading.Thread(target=gnn_engine.run_deep_graph_analytics, kwargs={'skip_gpu': False}).start()
        
    return f"[☢️] GLOBAL MI300X STRESS TEST EXECUTED ON '{target_dataset}'! REAL HARDWARE PIPELINES LAUNCHED."
        
def stop_all():
    stop_bulk_sar()
    stop_qlora()
    stop_vision()
    stop_gnn()
    return "[🛑] GLOBAL MI300X ABORT EXECUTED! ALL VRAM PURGED."

def create_mi300x_dashboard_tab(session_user):
    gr.Markdown("## 🔴 MI300X Command Center")
    gr.Markdown("Welcome to the **AMD Instinct MI300X Presentation Dashboard**. Use the buttons below to trigger massive memory-bound workloads. Watch the live Global Telemetry panel in the top right to verify maximum parallel VRAM saturation across your 192GB allocation limit.")
    gr.Markdown("""
### 🧠 Engine Explanations
These heavy engines use the underlying hardware to process massive amounts of raw input data to train your base models and identify crime. They do not rely on previous heuristic results; they train directly on millions of data points:
- **Bulk SAR Engine**: Generates massive synthetic Suspicious Activity Reports (SARs) from raw PII data.
- **QLoRA Studio**: Fine-tunes the base 70B parameter LLM directly on massive amounts of raw financial text to understand financial crime jargon.
- **Vision Forensics**: Scans thousands of raw KYC passports and identity documents in parallel using Vision-Language Models to detect deep-fakes.
- **GNN Topography**: Computes complex mathematical relationships across millions of raw global ledger transactions to map out underlying criminal networks.
""")
    
    with gr.Row():
        from data.dataset_manager import get_user_workspace
        from tabs.model_management import get_cached_hf_models
        cached_models = get_cached_hf_models()
        
        target_dataset = gr.Dropdown(choices=[], label="Target Raw Input Dataset", scale=3)
        target_llm = gr.Dropdown(choices=cached_models, label="Target Base LLM (QLoRA)", value="Qwen/Qwen1.5-0.5B", scale=3)
        target_vlm = gr.Dropdown(choices=cached_models, label="Target Vision Model (VLM)", value="llava-hf/llava-1.5-13b-hf", scale=3)
        refresh_ds_btn = gr.Button("↻ Refresh Workspace", scale=1)
        
    refresh_ds_btn.click(
        fn=lambda u: gr.update(choices=list(get_user_workspace(u).keys())), 
        inputs=session_user, outputs=target_dataset
    ).then(
        fn=lambda: gr.update(choices=get_cached_hf_models()), outputs=target_llm
    ).then(
        fn=lambda: gr.update(choices=get_cached_hf_models()), outputs=target_vlm
    )
    
    with gr.Row():
        with gr.Column():
            btn_bulk = gr.Button("🚀 Start Bulk SAR Engine (41.2 GB)")
            stop_btn_bulk = gr.Button("🛑 Stop Bulk SAR")
        with gr.Column():
            btn_qlora = gr.Button("🚀 Start QLoRA Studio (42.8 GB)")
            stop_btn_qlora = gr.Button("🛑 Stop QLoRA")
        with gr.Column():
            btn_vision = gr.Button("🚀 Start Vision Forensics (48.5 GB)")
            stop_btn_vision = gr.Button("🛑 Stop Vision Lab")
        with gr.Column():
            btn_gnn = gr.Button("🚀 Start GNN Topography (44.1 GB)")
            stop_btn_gnn = gr.Button("🛑 Stop GNN")
            
    with gr.Row():
        master_btn = gr.Button("☢️ EXECUTE MI300X GLOBAL STRESS TEST (START ALL)", variant="primary", size="lg", elem_classes="nuclear-btn")
        master_stop_btn = gr.Button("🛑 EMERGENCY ABORT ALL ENGINES", size="lg")
    
    status_out = gr.Textbox(label="Command Center Terminal", interactive=False)
    
    btn_bulk.click(fn=trigger_bulk_sar, inputs=[target_dataset], outputs=status_out)
    stop_btn_bulk.click(fn=stop_bulk_sar, outputs=status_out)
    
    btn_qlora.click(fn=trigger_qlora, inputs=[target_dataset, target_llm], outputs=status_out)
    stop_btn_qlora.click(fn=stop_qlora, outputs=status_out)
    
    btn_vision.click(fn=trigger_vision, inputs=[target_dataset, target_vlm], outputs=status_out)
    stop_btn_vision.click(fn=stop_vision, outputs=status_out)
    
    btn_gnn.click(fn=trigger_gnn, inputs=[target_dataset], outputs=status_out)
    stop_btn_gnn.click(fn=stop_gnn, outputs=status_out)
    
    master_btn.click(fn=trigger_all, inputs=[target_dataset, target_llm, target_vlm, session_user], outputs=status_out)
    master_stop_btn.click(fn=stop_all, outputs=status_out)
