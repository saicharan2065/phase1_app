from utils.dummy_generator import generate_dummy_data
import gradio as gr
import pandas as pd
from validation.data_quality import DataQualityAnalyzer
from data.dataset_manager import GLOBAL_WORKSPACE_DATA

def run_dq_ui(dataset_key):
    if not dataset_key or dataset_key not in GLOBAL_WORKSPACE_DATA:
        return "Please select a valid dataset from the workspace.", 0.0, pd.DataFrame(), None
        
    try:
        df = GLOBAL_WORKSPACE_DATA[dataset_key]
        analyzer = DataQualityAnalyzer()
        overall_score, col_quality_df, fig = analyzer.analyze(df)
        
        return "Analysis Complete", overall_score, col_quality_df, fig
    except Exception as e:
        return f"Error: {str(e)}", 0.0, pd.DataFrame(), None

def create_data_quality_tab():
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Select Workspace Dataset")
            with gr.Row():
                ds_dropdown = gr.Dropdown(choices=list(GLOBAL_WORKSPACE_DATA.keys()), label="Dataset", scale=4)
                refresh_btn = gr.Button("↻ Refresh", size="sm", scale=1)
            analyze_btn = gr.Button("Run Quality Analysis", variant="primary")
            
            refresh_btn.click(fn=lambda: gr.update(choices=list(GLOBAL_WORKSPACE_DATA.keys())), outputs=ds_dropdown)
            
        with gr.Column():
            gr.Markdown("### Summary")
            status_out = gr.Textbox(label="Status", interactive=False)
            overall_score = gr.Number(label="Overall Data Quality Score (%)", interactive=False)
            
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Per-Column Quality")
            quality_table = gr.Dataframe(label="Column Metrics")
        with gr.Column(scale=1):
            gr.Markdown("### Quality Visualization")
            quality_plot = gr.Plot(label="Column Quality Chart")
            
    analyze_btn.click(
        fn=run_dq_ui,
        inputs=ds_dropdown,
        outputs=[status_out, overall_score, quality_table, quality_plot]
    )
