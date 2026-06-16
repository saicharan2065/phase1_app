import gradio as gr
import pandas as pd
from validation.matching_engine import ReferenceDataMatchingEngine

from data.dataset_manager import GLOBAL_WORKSPACE_DATA

def validate_datasets(source_key, ref_key):
    if not source_key or not ref_key or source_key not in GLOBAL_WORKSPACE_DATA or ref_key not in GLOBAL_WORKSPACE_DATA:
        return "Please select valid source and reference datasets from the workspace.", 0.0, pd.DataFrame()
        
    try:
        source_df = GLOBAL_WORKSPACE_DATA[source_key]
        ref_df = GLOBAL_WORKSPACE_DATA[ref_key]
        
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
            
        with gr.Column():
            gr.Markdown("### Reference Dataset")
            with gr.Row():
                ref_ds = gr.Dropdown(choices=list(GLOBAL_WORKSPACE_DATA.keys()), label="Reference Workspace Dataset", scale=4)
                ref_refresh = gr.Button("↻", size="sm", scale=1)
            
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
        inputs=[source_ds, ref_ds],
        outputs=[val_msg, match_score, mismatch_table]
    )
