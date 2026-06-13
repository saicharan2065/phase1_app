import gradio as gr
import pandas as pd
from validation.data_quality import DataQualityAnalyzer

def analyze_data_quality(file):
    if file is None:
        return "Please upload a dataset.", 0.0, pd.DataFrame(), None
        
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file.name)
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file.name)
        elif file.name.endswith('.json'):
            df = pd.read_json(file.name)
        else:
            return "Unsupported file type.", 0.0, pd.DataFrame(), None
            
        analyzer = DataQualityAnalyzer()
        overall_score, col_quality_df, fig = analyzer.analyze(df)
        
        return "Analysis Complete", overall_score, col_quality_df, fig
    except Exception as e:
        return f"Error: {str(e)}", 0.0, pd.DataFrame(), None

def create_data_quality_tab():
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Upload Dataset")
            ds_upload = gr.File(label="Dataset")
            analyze_btn = gr.Button("Analyze Data Quality", variant="primary")
            
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
        fn=analyze_data_quality,
        inputs=ds_upload,
        outputs=[status_out, overall_score, quality_table, quality_plot]
    )
