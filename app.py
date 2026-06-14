import gradio as gr
import psutil
import shutil
from tabs.data_sources import create_data_sources_tab
from tabs.reference_validation import create_reference_validation_tab
from tabs.model_management import create_model_management_tab

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

# Dataset Marketplace import
from tabs.dataset_marketplace import create_dataset_marketplace_tab

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
button { background-color: white !important; color: black !important; border: 1px solid lightgreen !important; }
button:hover { background-color: lightgreen !important; color: black !important; }
button.primary { background-color: lightgreen !important; color: black !important; }
button.primary:hover { background-color: white !important; color: black !important; border: 1px solid lightgreen !important; }
.gradio-container { background-color: white !important; }
.dark { background-color: white !important; }
"""

GLOBAL_USERNAME = "GUEST"

def get_compact_metrics(request: gr.Request = None):
    global GLOBAL_USERNAME
    try:
        from tabs.model_management import get_active_model_state
        active_model = get_active_model_state()
    except Exception:
        active_model = "None Selected"
        
    try:
        from tabs.bulk_sar import generator
        if generator.is_running:
            vram_metrics = f"<b>VRAM Engine:</b> <span style='color:orange'>PROCESSING ({generator.processed_count}/{generator.total_count})</span>"
        else:
            vram_metrics = f"<b>VRAM Engine:</b> <span style='color:lightgreen'>IDLE</span>"
            
        simulated_vram = "41.2 GB / 48.0 GB (85%)" if generator.is_running or generator.model_loaded else "0.0 GB / 48.0 GB (0%)"
    except Exception:
        vram_metrics = "<b>VRAM Engine:</b> OFFLINE"
        simulated_vram = "0.0 GB"
        
    try:
        from tabs.qlora_training import trainer
        if trainer.is_training:
            qlora_metrics = f" | <b>QLoRA:</b> <span style='color:orange; animation: blinker 1s linear infinite;'>TRAINING ({trainer.progress_percent}%)</span>"
        else:
            qlora_metrics = f" | <b>QLoRA:</b> <span style='color:lightgreen'>IDLE</span>"
    except Exception:
        qlora_metrics = ""
        
    # Extract logged in username
    if request and hasattr(request, "username") and request.username:
        GLOBAL_USERNAME = request.username
    username = GLOBAL_USERNAME
        
    ram = psutil.virtual_memory()
    ram_gb_used = ram.used / (1024**3)
    ram_gb_total = ram.total / (1024**3)
    
    disk = shutil.disk_usage("/")
    disk_gb_used = disk.used / (1024**3)
    disk_gb_total = disk.total / (1024**3)
    
    return f"""<div style="text-align: right; padding-top: 10px; font-size: 0.9em; line-height: 1.4;">
    <b>Agent:</b> <span style="color:lightgreen; font-weight:bold;">{username.upper()}</span> | <b>Active Model:</b> {active_model}<br>
    <b>System RAM:</b> {ram_gb_used:.1f} GB / {ram_gb_total:.1f} GB ({ram.percent}%) | <b>Disk:</b> {disk_gb_used:.1f} GB / {disk_gb_total:.1f} GB<br>
    {vram_metrics}{qlora_metrics} | <b>VRAM Usage:</b> {simulated_vram}
    </div>"""

def create_app():
    with gr.Blocks(title="Financial Crime OS", css=css_override) as app:
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("# Financial Crime Operating System")
            with gr.Column(scale=1):
                global_metrics = gr.HTML(get_compact_metrics())
                refresh_btn = gr.Button("↻ Refresh Metrics", size="sm")
                refresh_btn.click(fn=get_compact_metrics, outputs=global_metrics)
                app.load(fn=get_compact_metrics, outputs=global_metrics)
                
                # Setup live refreshing for Gradio 4.0+
                timer = gr.Timer(2)
                timer.tick(fn=get_compact_metrics, outputs=global_metrics)
        
        with gr.Tabs():
            # New Dataset Marketplace
            with gr.Tab("Dataset Marketplace"):
                create_dataset_marketplace_tab()
                
            # Phase 1
            with gr.Tab("Local Data Sources"):
                create_data_sources_tab()
                
            with gr.Tab("Reference Validation"):
                create_reference_validation_tab()
                
            with gr.Tab("Model Management"):
                create_model_management_tab()
                
            # Phase 2
            with gr.Tab("Schema Discovery"):
                create_schema_discovery_tab()
                
            with gr.Tab("Entity Resolution"):
                create_entity_resolution_tab()
                
            with gr.Tab("Entity Graph"):
                create_entity_graph_tab()
                
            with gr.Tab("Data Quality"):
                create_data_quality_tab()
                
            # Phase 3
            with gr.Tab("Fraud Detection"):
                create_fraud_detection_tab()
                
            with gr.Tab("AML Detection"):
                create_aml_detection_tab()
                
            with gr.Tab("Risk Clusters"):
                create_risk_clusters_tab()
                
            with gr.Tab("Investigations"):
                create_investigations_tab()
                
            with gr.Tab("Case Management"):
                create_case_management_tab()
                
            with gr.Tab("Alerts"):
                create_alerts_tab()
                
            with gr.Tab("Reports"):
                create_bulk_sar_tab()
                
            with gr.Tab("QLoRA Studio"):
                create_qlora_tab()
                
    return app

def auth_check(username, password):
    valid_users = {
        "admin": "admin123",
        "agent1": "agent123",
        "agent2": "agent123"
    }
    return username in valid_users and valid_users[username] == password

if __name__ == "__main__":
    app = create_app()
    app.launch(theme=compact_theme, auth=auth_check, auth_message="Financial Crime OS - Please log in (admin/admin123 or agent1/agent123)")
