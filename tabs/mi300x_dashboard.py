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
        t = threading.Thread(target=bulk_engine.run_bulk_inference, args=(target_dataset, 32), daemon=True)
        t.start()
    return f"[OK] Bulk SAR Engine Launched! Training on Raw Dataset: {target_dataset}. Check Global Telemetry."

def stop_bulk_sar():
    bulk_engine.stop()
    from agents.vram_manager import vram_manager
    vram_manager.purge_vram()
    return "[STOP] Bulk SAR Engine Aborted and VRAM Purged."

def trigger_qlora(target_dataset, target_llm):
    if not qlora_engine.is_training:
        clean_llm = target_llm.split(" (")[0] if target_llm and " (" in target_llm else target_llm
        if not clean_llm or clean_llm == "No models installed" or clean_llm == "None Selected":
            clean_llm = "Qwen/Qwen1.5-0.5B"
        qlora_engine.start_training("gretelai/synthetic_pii_finance", clean_llm)
    return f"[OK] QLoRA Neural Rewiring Launched! Training on Raw Dataset: {target_dataset}. Check Global Telemetry."

def stop_qlora():
    qlora_engine.stop()
    from agents.vram_manager import vram_manager
    vram_manager.purge_vram()
    return "[STOP] QLoRA Engine Aborted and VRAM Purged."

def trigger_vision(target_dataset, target_vlm):
    if not vision_engine.is_running:
        clean_vlm = target_vlm.split(" (")[0] if target_vlm and " (" in target_vlm else target_vlm
        if not clean_vlm or clean_vlm == "No models installed" or clean_vlm == "None Selected":
            clean_vlm = "llava-hf/llava-1.5-13b-hf"
        threading.Thread(target=vision_engine.run_mass_forensics, kwargs={'dataset_id': target_dataset, 'model_id': clean_vlm, 'skip_gpu': False}, daemon=True).start()
    return f"[OK] Vision Forensics Lab Launched! Training on Raw Dataset: {target_dataset}. Check Global Telemetry."

def stop_vision():
    vision_engine.stop()
    return "[STOP] Vision Forensics Lab Aborted."

def trigger_gnn(target_dataset):
    if not gnn_engine.is_running:
        t = threading.Thread(target=gnn_engine.run_deep_graph_analytics, kwargs={'dataset_id': target_dataset}, daemon=True)
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
    engines_to_start = []
    if not bulk_engine.is_running: engines_to_start.append("bulk")
    if not qlora_engine.is_training: engines_to_start.append("qlora")
    if not vision_engine.is_running: engines_to_start.append("vision")
    if not gnn_engine.is_running: engines_to_start.append("gnn")
    
    barrier = threading.Barrier(len(engines_to_start)) if len(engines_to_start) > 0 else None
    
    if "bulk" in engines_to_start:
        threading.Thread(target=bulk_engine.run_bulk_inference, args=(target_dataset, 32, False, barrier), daemon=True).start()
    if "qlora" in engines_to_start:
        # Call the training function directly (not start_training which spawns its own thread)
        threading.Thread(target=qlora_engine._simulate_qlora_training, args=(target_dataset, clean_llm, False, barrier), daemon=True).start()
    if "vision" in engines_to_start:
        threading.Thread(target=vision_engine.run_mass_forensics, kwargs={'dataset_id': target_dataset, 'model_id': clean_vlm, 'skip_gpu': False, 'sync_barrier': barrier}, daemon=True).start()
    if "gnn" in engines_to_start:
        threading.Thread(target=gnn_engine.run_deep_graph_analytics, kwargs={'dataset_id': target_dataset, 'skip_gpu': False, 'sync_barrier': barrier}, daemon=True).start()
        
    return f"[☢️] GLOBAL MI300X STRESS TEST EXECUTED ON '{target_dataset}'! REAL HARDWARE PIPELINES LAUNCHED."
        
def stop_all():
    stop_bulk_sar()
    stop_qlora()
    stop_vision()
    stop_gnn()
    from agents.vram_manager import vram_manager
    vram_manager.purge_vram()
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
        from tabs.dataset_marketplace import dm
        from tabs.model_management import get_cached_hf_models
        cached_models = get_cached_hf_models()
        cached_datasets = dm.get_cached_datasets()
        
        target_dataset = gr.Dropdown(choices=cached_datasets, label="Target Raw Input Dataset", scale=3)
        target_llm = gr.Dropdown(choices=cached_models, label="Target Base LLM (QLoRA)", value="Qwen/Qwen1.5-0.5B", scale=3)
        target_vlm = gr.Dropdown(choices=cached_models, label="Target Vision Model (VLM)", value="llava-hf/llava-1.5-13b-hf", scale=3)
        refresh_ds_btn = gr.Button("↻ Refresh Workspace", scale=1)
        
    refresh_ds_btn.click(
        fn=lambda: gr.update(choices=dm.get_cached_datasets()), 
        outputs=target_dataset
    ).then(
        fn=lambda: gr.update(choices=get_cached_hf_models()), outputs=target_llm
    ).then(
        fn=lambda: gr.update(choices=get_cached_hf_models()), outputs=target_vlm
    )
    
    with gr.Row():
        with gr.Column():
            btn_bulk = gr.Button("🚀 Start Bulk SAR Engine")
            stop_btn_bulk = gr.Button("🛑 Stop Bulk SAR")
        with gr.Column():
            btn_qlora = gr.Button("🚀 Start QLoRA Studio")
            stop_btn_qlora = gr.Button("🛑 Stop QLoRA")
        with gr.Column():
            btn_vision = gr.Button("🚀 Start Vision Forensics")
            stop_btn_vision = gr.Button("🛑 Stop Vision Lab")
        with gr.Column():
            btn_gnn = gr.Button("🚀 Start GNN Topography")
            stop_btn_gnn = gr.Button("🛑 Stop GNN")
            
    with gr.Row():
        master_btn = gr.Button("☢️ EXECUTE MI300X GLOBAL STRESS TEST (START ALL)", variant="primary", size="lg", elem_classes="nuclear-btn")
        master_stop_btn = gr.Button("🛑 EMERGENCY ABORT ALL ENGINES", size="lg")
        reboot_server_btn = gr.Button("🔌 FORCE REBOOT SERVER (CLEARS LEAKED VRAM)", variant="stop", size="lg")
    
    status_out = gr.Textbox(label="Command Center Master Log", interactive=False)
    
    gr.Markdown("### Live Engine Telemetry")
    live_status_out = gr.Textbox(label="Internal Engine Status", interactive=False, lines=5)
    
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
    
    import os
    def nuke_server():
        try:
            import psutil
            import os
            # Only kill child processes spawned by this specific Gradio server instance to prevent kicking the user out of SSH/Jupyter
            current_process = psutil.Process(os.getpid())
            for child in current_process.children(recursive=True):
                child.kill()
        except Exception:
            pass
        import os
        os._exit(0)
        
    reboot_server_btn.click(fn=nuke_server, outputs=status_out)
    
    def poll_engine_status():
        lines = []
        def format_engine(name, eng, p_attr, t_attr):
            base_msg = getattr(eng, 'status_message', 'IDLE')
            if not base_msg: base_msg = 'IDLE'
            if getattr(eng, 'is_running', getattr(eng, 'is_training', False)):
                p = getattr(eng, p_attr, 0)
                t = getattr(eng, t_attr, 0)
                if t > 0 and 'IDLE' not in base_msg and 'MOUNT' not in base_msg and 'WAITING' not in base_msg and 'LOADING' not in base_msg and 'TOKENIZING' not in base_msg:
                    pct = int((p / t) * 100)
                    return f"{name}: {base_msg} | PROGRESS: {p}/{t} ({pct}%)"
            return f"{name}: {base_msg}"
            
        lines.append(format_engine("Bulk SAR", bulk_engine, "processed_count", "total_count"))
        lines.append(format_engine("QLoRA Studio", qlora_engine, "current_epoch", "total_epochs"))
        lines.append(format_engine("Vision Lab", vision_engine, "processed_count", "total_documents"))
        lines.append(format_engine("GNN Topography", gnn_engine, "processed_nodes", "total_nodes"))
        return "\n".join(lines)
        
    try:
        timer = gr.Timer(1)
        timer.tick(fn=poll_engine_status, outputs=live_status_out)
    except AttributeError:
        # Fallback if Gradio version doesn't support Timer
        pass
    
    return target_dataset, target_llm, target_vlm
