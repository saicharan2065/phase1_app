import gradio as gr
import pandas as pd
from validation.matching_engine import ReferenceDataMatchingEngine

from data.dataset_manager import GLOBAL_WORKSPACE_DATA

def validate_datasets(source_key, ref_key, source_file, ref_file):
    def load_data(key, file):
        if file is not None:
            try:
                if file.name.endswith('.csv'): return pd.read_csv(file.name)
                elif file.name.endswith('.xlsx'): return pd.read_excel(file.name)
                elif file.name.endswith('.json'): return pd.read_json(file.name)
                return None
            except:
                return None
        elif key and key in GLOBAL_WORKSPACE_DATA:
            return GLOBAL_WORKSPACE_DATA[key]
        return None

    source_df = load_data(source_key, source_file)
    ref_df = load_data(ref_key, ref_file)
    
    if source_df is None or ref_df is None:
        return "Please select valid source and reference datasets from the workspace, or upload files.", 0.0, pd.DataFrame()
        
    try:
        
        engine = ReferenceDataMatchingEngine()
        score, mismatches = engine.match_records(source_df, ref_df)
        
        msg = f"Validation completed between {source_key} and {ref_key}"
        return msg, score, mismatches
    except Exception as e:
        return f"Error: {str(e)}", 0.0, pd.DataFrame([{"Error": str(e)}])

def create_reference_validation_tab():
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Source Dataset")
            with gr.Row():
                source_ds = gr.Dropdown(choices=list(GLOBAL_WORKSPACE_DATA.keys()), label="Source Workspace Dataset", scale=4)
                src_refresh = gr.Button("↻", size="sm", scale=1)
            gr.Markdown("#### Or Upload")
            source_upload = gr.File(label="Upload Source")
            
        with gr.Column():
            gr.Markdown("### Reference Dataset")
            with gr.Row():
                ref_ds = gr.Dropdown(choices=list(GLOBAL_WORKSPACE_DATA.keys()), label="Reference Workspace Dataset", scale=4)
                ref_refresh = gr.Button("↻", size="sm", scale=1)
            gr.Markdown("#### Or Upload")
            ref_upload = gr.File(label="Upload Reference")
            
    val_btn = gr.Button("Run Reference Matching Engine", variant="primary")
    
    src_refresh.click(fn=lambda: gr.update(choices=list(GLOBAL_WORKSPACE_DATA.keys())), outputs=source_ds)
    ref_refresh.click(fn=lambda: gr.update(choices=list(GLOBAL_WORKSPACE_DATA.keys())), outputs=ref_ds)
    
    with gr.Row():
        val_msg = gr.Textbox(label="Validation Status", interactive=False)
        match_score = gr.Number(label="Overall Match Score (%)", interactive=False)
        
    gr.Markdown("### Record Matches and Mismatches")
    mismatch_table = gr.Dataframe(label="Matching Output")
    
    val_btn.click(
        fn=validate_datasets,
        inputs=[source_ds, ref_ds, source_upload, ref_upload],
        outputs=[val_msg, match_score, mismatch_table]
    )
