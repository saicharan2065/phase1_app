import gradio as gr
import threading
from tabs.bulk_sar import generator as bulk_engine
from tabs.qlora_training import trainer as qlora_engine
from tabs.vision_lab import vision_engine
from tabs.gnn_topography import gnn_engine

def trigger_bulk_sar():
    if not bulk_engine.is_running:
        t = threading.Thread(target=bulk_engine.run_bulk_inference, args=([f"SUSPECT_{i}" for i in range(10000)], 32))
        t.start()
    return "[OK] Bulk SAR Engine Launched! Check Global Telemetry."

def trigger_qlora():
    if not qlora_engine.is_training:
        qlora_engine.start_training("gretelai/synthetic_pii_finance", "DeepSeek-70B")
    return "[OK] QLoRA Neural Rewiring Launched! Check Global Telemetry."

def trigger_vision():
    if not vision_engine.is_running:
        t = threading.Thread(target=vision_engine.run_mass_forensics)
        t.start()
    return "[OK] Vision Forensics Lab Launched! Check Global Telemetry."

def trigger_gnn():
    if not gnn_engine.is_running:
        t = threading.Thread(target=gnn_engine.run_deep_graph_analytics)
        t.start()
    return "[OK] GNN Topography Launched! Check Global Telemetry."

def trigger_all():
    trigger_bulk_sar()
    trigger_qlora()
    trigger_vision()
    trigger_gnn()
    return "[☢️] GLOBAL MI300X STRESS TEST EXECUTED! ALL 4 ENGINES RUNNING."

def create_mi300x_dashboard_tab():
    gr.Markdown("## 🔴 MI300X Command Center")
    gr.Markdown("Welcome to the **AMD Instinct MI300X Presentation Dashboard**. Use the buttons below to trigger massive memory-bound workloads. Watch the live Global Telemetry panel in the top right to verify maximum parallel VRAM saturation across your 192GB allocation limit.")
    
    with gr.Row():
        btn_bulk = gr.Button("🚀 Start Bulk SAR Engine (41.2 GB)")
        btn_qlora = gr.Button("🚀 Start QLoRA Studio (42.8 GB)")
        btn_vision = gr.Button("🚀 Start Vision Forensics (48.5 GB)")
        btn_gnn = gr.Button("🚀 Start GNN Topography (44.1 GB)")
        
    master_btn = gr.Button("☢️ EXECUTE MI300X GLOBAL STRESS TEST (START ALL)", variant="primary", size="lg", elem_classes="nuclear-btn")
    
    status_out = gr.Textbox(label="Command Center Terminal", interactive=False)
    
    btn_bulk.click(fn=trigger_bulk_sar, outputs=status_out)
    btn_qlora.click(fn=trigger_qlora, outputs=status_out)
    btn_vision.click(fn=trigger_vision, outputs=status_out)
    btn_gnn.click(fn=trigger_gnn, outputs=status_out)
    master_btn.click(fn=trigger_all, outputs=status_out)
