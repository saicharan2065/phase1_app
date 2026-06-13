import gradio as gr
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
from tabs.reports import create_reports_tab

# Dataset Marketplace import
from tabs.dataset_marketplace import create_dataset_marketplace_tab

compact_theme = gr.themes.Default(
    primary_hue="emerald",
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
    border_color_primary="lightgreen",
    border_color_primary_dark="lightgreen",
    body_text_color="black",
    body_text_color_dark="black",
    block_title_text_color="black",
    block_title_text_color_dark="black",
    block_label_text_color="black",
    block_label_text_color_dark="black",
    button_primary_background_fill="*primary_500",
    button_primary_background_fill_dark="*primary_600",
    button_primary_text_color="white",
    button_primary_text_color_dark="white",
    button_secondary_text_color="black",
    button_secondary_text_color_dark="black"
)

def create_app():
    with gr.Blocks(title="Financial Crime OS") as app:
        gr.Markdown("# Financial Crime Operating System")
        
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
                create_reports_tab()
                
    return app

if __name__ == "__main__":
    app = create_app()
    app.launch(theme=compact_theme)
