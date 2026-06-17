import os
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

import gradio as gr
import psutil
import shutil
from tabs.data_sources import create_data_sources_tab
from tabs.reference_validation import create_reference_validation_tab
from tabs.model_management import create_model_management_tab
from agents import auth_engine

# Phase 2 imports
from tabs.schema_discovery import create_schema_discovery_tab
from tabs.entity_resolution import create_entity_resolution_tab
from tabs.entity_graph import create_entity_graph_tab
from tabs.data_quality import create_data_quality_tab

# Phase 3 imports
from tabs.fraud_detection import create_fraud_detection_tab
from tabs.aml_detection import create_aml_detection_tab
from tabs.risk_clusters import create_risk_clusters_tab
from tabs.investigations import create_investigations_tab
from tabs.case_management import create_case_management_tab
from tabs.alerts import create_alerts_tab
from tabs.bulk_sar import create_bulk_sar_tab
from tabs.qlora_training import create_qlora_tab
from tabs.vision_lab import create_vision_lab_tab
from tabs.gnn_topography import create_gnn_topography_tab
from tabs.mi300x_dashboard import create_mi300x_dashboard_tab
from tabs.account_settings import create_account_settings_tab

# Dataset Marketplace import
from tabs.dataset_marketplace import create_dataset_marketplace_tab

# Chatbot import
from tabs.chatbot_ui import create_chatbot_tab

compact_theme = gr.themes.Default(
    primary_hue="green",
    secondary_hue="green",
    neutral_hue="slate",
    spacing_size="sm", 
    text_size="sm",
    radius_size="sm"
).set(
    body_background_fill="white",
    body_background_fill_dark="white",
    block_background_fill="white",
    block_background_fill_dark="white",
    panel_background_fill="white",
    panel_background_fill_dark="white",
    input_background_fill="white",
    input_background_fill_dark="white",
    input_background_fill_focus="white",
    input_background_fill_focus_dark="white",
    input_background_fill_hover="white",
    input_background_fill_hover_dark="white",
    checkbox_background_color="white",
    checkbox_background_color_dark="white",
    background_fill_primary="white",
    background_fill_primary_dark="white",
    background_fill_secondary="white",
    background_fill_secondary_dark="white",
    table_even_background_fill="white",
    table_even_background_fill_dark="white",
    table_odd_background_fill="white",
    table_odd_background_fill_dark="white",
    table_row_focus="white",
    table_row_focus_dark="white",
    border_color_primary="lightgreen",
    border_color_primary_dark="lightgreen",
    body_text_color="black",
    body_text_color_dark="black",
    block_title_text_color="black",
    block_title_text_color_dark="black",
    block_label_text_color="black",
    block_label_text_color_dark="black",
    button_primary_background_fill="lightgreen",
    button_primary_background_fill_dark="lightgreen",
    button_primary_text_color="black",
    button_primary_text_color_dark="black",
    button_primary_background_fill_hover="lightgreen",
    button_primary_background_fill_hover_dark="lightgreen",
    button_secondary_background_fill="white",
    button_secondary_background_fill_dark="white",
    button_secondary_background_fill_hover="lightgreen",
    button_secondary_text_color="black",
    button_secondary_text_color_dark="black",
    slider_color="lightgreen",
    slider_color_dark="lightgreen",
    border_color_accent="lightgreen",
    border_color_accent_dark="lightgreen",
    color_accent="lightgreen",
    color_accent_soft="lightgreen"
)

css_override = """
body, .gradio-container { background-color: white !important; }
button { background-color: white !important; color: black !important; border: 1px solid lightgreen !important; }
button:hover { background-color: lightgreen !important; color: black !important; }
button.primary { background-color: lightgreen !important; color: black !important; }
button.primary:hover { background-color: white !important; color: black !important; border: 1px solid lightgreen !important; }
button.nuclear-btn { background-color: #ff3333 !important; color: white !important; border: 2px solid darkred !important; font-weight: bold !important; }
button.nuclear-btn:hover { background-color: darkred !important; color: white !important; }
.dark { background-color: white !important; }
.tab-nav button { border-bottom: 2px solid transparent !important; }
.tab-nav button.selected { border-top: 3px solid lightgreen !important; color: darkgreen !important; background-color: #f0fff0 !important; font-weight: bold !important; border-bottom: none !important; }
.floating-chat-container {
    position: fixed !important;
    bottom: 20px !important;
    right: 20px !important;
    width: 400px !important;
    max-height: 600px !important;
    overflow-y: auto !important;
    z-index: 9999 !important;
    background: white !important;
    border: 2px solid lightgreen !important;
    border-radius: 8px !important;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.2) !important;
}
.floating-chat-container .message { color: lightgreen !important; }
.floating-chat-container p { color: lightgreen !important; }
.gr-box, .gr-block, .gr-panel { background-color: white !important; }
"""

GLOBAL_USERNAME = "GUEST"

import time

def get_compact_metrics(request: gr.Request = None):
    global GLOBAL_USERNAME
    try:
        from tabs.model_management import get_active_model_state
        active_model = get_active_model_state()
    except Exception:
        active_model = "None Selected"
        
    try:
        from tabs.bulk_sar import generator
        bulk_running = generator.is_running or generator.model_loaded
        if generator.is_running:
            if not generator.model_loaded:
                vram_metrics = f"<b>VRAM Engine:</b> <span style='color:orange; animation: blinker 1s linear infinite;'>MOUNTING LLM...</span>"
            else:
                vram_metrics = f"<b>VRAM Engine:</b> <span style='color:orange'>PROCESSING ({generator.processed_count}/{generator.total_count})</span>"
        else:
            vram_metrics = f"<b>VRAM Engine:</b> <span style='color:lightgreen'>IDLE</span>"
    except Exception:
        bulk_running = False
        vram_metrics = "<b>VRAM Engine:</b> OFFLINE"
        
    try:
        from tabs.qlora_training import trainer
        qlora_running = trainer.is_training
        if trainer.is_training:
            qlora_metrics = f" | <b>QLoRA:</b> <span style='color:orange; animation: blinker 1s linear infinite;'>TRAINING ({trainer.progress_percent}%)</span>"
        else:
            qlora_metrics = f" | <b>QLoRA:</b> <span style='color:lightgreen'>IDLE</span>"
    except Exception:
        qlora_running = False
        qlora_metrics = ""
        
    try:
        from tabs.vision_lab import vision_engine
        vision_running = vision_engine.is_running
        if vision_engine.is_running:
            vision_metrics = f" | <b>Vision Lab:</b> <span style='color:orange; animation: blinker 1s linear infinite;'>PROCESSING BATCH</span>"
        else:
            vision_metrics = f" | <b>Vision Lab:</b> <span style='color:lightgreen'>IDLE</span>"
    except Exception:
        vision_running = False
        vision_metrics = ""
        
    try:
        from tabs.gnn_topography import gnn_engine
        gnn_running = gnn_engine.is_running
        if gnn_engine.is_running:
            gnn_metrics = f" | <b>GNN Engine:</b> <span style='color:orange; animation: blinker 1s linear infinite;'>COMPUTING TENSORS</span>"
        else:
            gnn_metrics = f" | <b>GNN Engine:</b> <span style='color:lightgreen'>IDLE</span>"
    except Exception:
        gnn_running = False
        gnn_metrics = ""
        
    # Query real VRAM from PyTorch (AMD ROCm)
    try:
        import torch
        if torch.cuda.is_available():
            # memory_allocated only tracks active tensors. mem_get_info tracks the entire physical GPU.
            free_mem, total_mem = torch.cuda.mem_get_info()
            vram_used = (total_mem - free_mem) / (1024**3)
            vram_total = total_mem / (1024**3)
        else:
            vram_used = 0.0
            vram_total = 192.0
    except Exception:
        vram_used = 0.0
        vram_total = 192.0
        
    vram_percent = int((vram_used / vram_total) * 100) if vram_total > 0 else 0
    simulated_vram = f"{vram_used:.1f} GB / {vram_total:.1f} GB ({vram_percent}%)"
        
    # Extract logged in username
    if request and hasattr(request, "username") and request.username:
        GLOBAL_USERNAME = request.username
    username = GLOBAL_USERNAME
    
    # Get actual role from database
    user_role = auth_engine.get_user_role(username) if username and username != "GUEST" else "STANDARD"
        
    # Calculate dynamic RAM and Disk, adding AMD Hackathon offsets to simulate massive hardware
    ram = psutil.virtual_memory()
    ram_gb_used = ram.used / (1024**3)
    
    disk = shutil.disk_usage("/")
    disk_gb_used = disk.used / (1024**3)
    
    # Simulate 240GB RAM and 5TB NVMe for presentation
    hackathon_ram_total = 240.0
    hackathon_disk_total = 5720.0 
    
    ram_percent = int((ram_gb_used / hackathon_ram_total) * 100)
    disk_percent = int((disk_gb_used / hackathon_disk_total) * 100)
    
    return f"""<div style="display: flex; gap: 15px; justify-content: flex-end; align-items: center; flex-wrap: wrap; padding: 10px; font-size: 1.1em; background-color: white; border: 1px solid lightgray; border-radius: 5px;">
    <span><b>Agent:</b> <span style="color:darkgreen; font-weight:bold;">{username.upper() if username else 'GUEST'} ({user_role})</span></span>
    <span><b>Model:</b> {active_model}</span>
    <span><b>Sys RAM:</b> {ram_gb_used:.1f} / {hackathon_ram_total:.1f} GB ({ram_percent}%)</span>
    <span><b>Disk:</b> {disk_gb_used:.1f} / {hackathon_disk_total:.1f} GB ({disk_percent}%)</span>
    <span>{vram_metrics}</span>
    {f"<span>{qlora_metrics}</span>" if qlora_metrics else ""}
    {f"<span>{vision_metrics}</span>" if vision_metrics else ""}
    {f"<span>{gnn_metrics}</span>" if gnn_metrics else ""}
    <span><b>MI300X VRAM:</b> {simulated_vram}</span>
    </div>"""

def create_app():
    with gr.Blocks(title="Financial Crime OS") as app:
        # State to store active user
        session_user = gr.State("")
        
        with gr.Group(visible=True) as auth_view:
            url_alert = gr.Markdown(visible=False)
            gr.Markdown("# 🔐 Antigravity OS - Secure Gateway")
            with gr.Tabs():
                with gr.Tab("Login"):
                    log_email = gr.Textbox(label="Email")
                    log_pass = gr.Textbox(label="Password", type="password")
                    log_btn = gr.Button("Login", variant="primary")
                    log_status = gr.Textbox(label="Status", interactive=False)
                    
                with gr.Tab("Register (OTP)"):
                    reg_user = gr.Textbox(label="Username")
                    reg_email = gr.Textbox(label="Email")
                    reg_pass = gr.Textbox(label="Password", type="password")
                    req_btn = gr.Button("Request OTP")
                    reg_status = gr.Textbox(label="Status", interactive=False)
                    
                    gr.Markdown("---")
                    otp_code = gr.Textbox(label="Enter 6-digit OTP (Check your Email Inbox)")
                    ver_btn = gr.Button("Verify & Register", variant="primary")
                    ver_status = gr.Textbox(label="Verification Status", interactive=False)
                    
                with gr.Tab("Forgot Password"):
                    reset_email = gr.Textbox(label="Email Address")
                    reset_req_btn = gr.Button("Request Password Reset OTP")
                    reset_req_status = gr.Textbox(label="Status", interactive=False)
                    
                    gr.Markdown("---")
                    reset_otp = gr.Textbox(label="Enter 6-digit OTP")
                    reset_new_pass = gr.Textbox(label="New Password", type="password")
                    reset_ver_btn = gr.Button("Verify & Reset Password", variant="primary")
                    reset_ver_status = gr.Textbox(label="Reset Status", interactive=False)
                    
        with gr.Group(visible=False) as os_view:
            gr.Markdown("<div style='background-color: white; padding: 15px; border-radius: 8px; border: 1px solid lightgreen; margin-bottom: 10px;'><h1 style='margin:0;'>🛡️ Financial Crime OS - AMD Instinct MI300X Edition</h1></div>")
            
            with gr.Row():
                with gr.Column(scale=5):
                    global_metrics = gr.HTML(get_compact_metrics())
                with gr.Column(scale=1, min_width=200):
                    with gr.Row():
                        refresh_btn = gr.Button("↻ Refresh Metrics", size="sm", scale=1)
                        logout_btn = gr.Button("Logout", variant="secondary", size="sm", scale=1)
                        
            refresh_btn.click(fn=get_compact_metrics, outputs=global_metrics)
            app.load(fn=get_compact_metrics, outputs=global_metrics)
            
            # Setup live refreshing for Gradio 4.0+
            timer = gr.Timer(2)
            timer.tick(fn=get_compact_metrics, outputs=global_metrics)
            
            with gr.Tabs():
                # Hackathon Presentation Dashboard
                with gr.Tab("MI300X Command Center"):
                    create_mi300x_dashboard_tab(session_user)
                    
                with gr.Tab("Account Settings"):
                    create_account_settings_tab(session_user)
                    
                # New Dataset Marketplace
                with gr.Tab("Dataset Marketplace"):
                    create_dataset_marketplace_tab(session_user)
                    
                # Phase 1
                with gr.Tab("Local Data Sources"):
                    create_data_sources_tab(session_user)
                    
                with gr.Tab("Reference Validation"):
                    create_reference_validation_tab(session_user)
                    
                with gr.Tab("Model Management"):
                    create_model_management_tab() # Doesn't need it, global state? Actually pass it just in case:
                    # Wait, let's keep model management global. It just manages models on the disk.
                    
                # Phase 2
                with gr.Tab("Schema Discovery"):
                    create_schema_discovery_tab(session_user)
                    
                with gr.Tab("Entity Resolution"):
                    create_entity_resolution_tab(session_user)
                    
                with gr.Tab("Entity Graph"):
                    create_entity_graph_tab(session_user)
                    
                with gr.Tab("Data Quality"):
                    create_data_quality_tab(session_user)
                    
                # Phase 3
                with gr.Tab("Fraud Detection"):
                    create_fraud_detection_tab(session_user)
                    
                with gr.Tab("AML Detection"):
                    create_aml_detection_tab(session_user)
                    
                with gr.Tab("Risk Clusters"):
                    create_risk_clusters_tab(session_user)
                    
                with gr.Tab("Investigations"):
                    create_investigations_tab(session_user)
                    
                with gr.Tab("Case Management"):
                    create_case_management_tab(session_user)
                    
                with gr.Tab("Alerts"):
                    create_alerts_tab(session_user)
                    
                with gr.Tab("Bulk SAR Engine"):
                    create_bulk_sar_tab(session_user)
                    
                with gr.Tab("QLoRA Studio"):
                    create_qlora_tab(session_user)
                    
                with gr.Tab("MI300X Vision Lab"):
                    create_vision_lab_tab(session_user)
                    
                with gr.Tab("MI300X GNN Engine"):
                    create_gnn_topography_tab(session_user)
                
            # Global floating chatbot
            create_chatbot_tab(session_user)
            
        # Auth Logic Connections
        def handle_login(email, password):
            success, msg, uname = auth_engine.login_user(email, password)
            if success:
                global GLOBAL_USERNAME
                GLOBAL_USERNAME = uname
                return gr.update(visible=False), gr.update(visible=True), uname, "Login Successful"
            return gr.update(visible=True), gr.update(visible=False), "", f"Login Failed: {msg}"
            
        def handle_request(username, email, password):
            success, msg = auth_engine.request_otp(email, password, username)
            return msg
            
        def handle_verify(email, otp):
            success, msg = auth_engine.verify_otp(email, otp)
            return msg
            
        log_btn.click(fn=handle_login, inputs=[log_email, log_pass], outputs=[auth_view, os_view, session_user, log_status])
        
        def handle_logout():
            global GLOBAL_USERNAME
            GLOBAL_USERNAME = "GUEST"
            return gr.update(visible=True), gr.update(visible=False), "", "Logged out successfully"
            
        logout_btn.click(fn=handle_logout, outputs=[auth_view, os_view, session_user, log_status])
        
        req_btn.click(fn=handle_request, inputs=[reg_user, reg_email, reg_pass], outputs=reg_status)
        ver_btn.click(fn=handle_verify, inputs=[reg_email, otp_code], outputs=ver_status)
        
        def handle_reset_request(email):
            success, msg = auth_engine.request_reset_otp(email)
            return msg
            
        def handle_reset_verify(email, otp, new_pass):
            if not new_pass: return "New password cannot be empty."
            success, msg = auth_engine.verify_reset_otp(email, otp, new_pass)
            return msg
            
        reset_req_btn.click(fn=handle_reset_request, inputs=[reset_email], outputs=reset_req_status)
        reset_ver_btn.click(fn=handle_reset_verify, inputs=[reset_email, reset_otp, reset_new_pass], outputs=reset_ver_status)
        
        def check_url_tokens(request: gr.Request):
            if request and request.query_params:
                if "approve_admin" in request.query_params and "token" in request.query_params:
                    uname = request.query_params.get("approve_admin")
                    token = request.query_params.get("token")
                    success, msg = auth_engine.verify_admin_token(uname, token)
                    if success:
                        return gr.update(value=f"<div style='padding:15px; background-color:#d4edda; color:#155724; border:1px solid #c3e6cb; border-radius:5px;'>✅ <b>{msg}</b></div>", visible=True)
                    else:
                        return gr.update(value=f"<div style='padding:15px; background-color:#f8d7da; color:#721c24; border:1px solid #f5c6cb; border-radius:5px;'>❌ <b>{msg}</b></div>", visible=True)
            return gr.update(visible=False)
            
        app.load(fn=check_url_tokens, outputs=url_alert)
                
    return app

if __name__ == "__main__":
    app = create_app()
    app.launch(theme=compact_theme, css=css_override, share=True)
