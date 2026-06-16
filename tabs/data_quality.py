from utils.dummy_generator import generate_dummy_data
import gradio as gr
import pandas as pd
from validation.data_quality import DataQualityAnalyzer
from data.dataset_manager import get_user_workspace

def run_dq_ui(dataset_key, file, username):
    if file is not None:
        try:
            if file.name.endswith('.csv'): df = pd.read_csv(file.name)
            elif file.name.endswith('.xlsx'): df = pd.read_excel(file.name)
            elif file.name.endswith('.json'): df = pd.read_json(file.name)
            else: return "Unsupported file type.", 0.0, pd.DataFrame(), None
        except Exception as e:
            return f"File error: {e}", 0.0, pd.DataFrame(), None
    elif dataset_key and dataset_key in get_user_workspace(username):
        df = get_user_workspace(username)[dataset_key]
    else:
        return "Please select a valid dataset from the workspace, or upload a file.", 0.0, pd.DataFrame(), None
        
    try:
        analyzer = DataQualityAnalyzer()
        overall_score, col_quality_df, fig = analyzer.analyze(df)
        
        return "Analysis Complete", overall_score, col_quality_df, fig
    except Exception as e:
        return f"Error: {str(e)}", 0.0, pd.DataFrame(), None

def create_data_quality_tab(session_user):
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Select Workspace Dataset")
            with gr.Row():
                ds_dropdown = gr.Dropdown(choices=[], label="Dataset", scale=4)
                refresh_btn = gr.Button("↻ Refresh", size="sm", scale=1)
                
            gr.Markdown("### Or Upload Direct File")
            ds_upload = gr.File(label="Dataset Fallback")
            
            analyze_btn = gr.Button("Run Quality Analysis", variant="primary")
            
            refresh_btn.click(fn=lambda u: gr.update(choices=list(get_user_workspace(u).keys())), inputs=session_user, outputs=ds_dropdown)
            
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
        inputs=[ds_dropdown, ds_upload, session_user],
        outputs=[status_out, overall_score, quality_table, quality_plot]
    )
