import gradio as gr
import pandas as pd
from validation.matching_engine import ReferenceDataMatchingEngine

def validate_datasets(source_file, ref_file):
    if source_file is None or ref_file is None:
        return "Please upload both source and reference datasets.", 0.0, pd.DataFrame()
        
    try:
        def load_df(f):
            if f.name.endswith('.csv'): return pd.read_csv(f.name)
            elif f.name.endswith('.xlsx'): return pd.read_excel(f.name)
            elif f.name.endswith('.json'): return pd.read_json(f.name)
            return pd.DataFrame()
            
        source_df = load_df(source_file)
        ref_df = load_df(ref_file)
        
        engine = ReferenceDataMatchingEngine()
        score, mismatches = engine.match_records(source_df, ref_df)
        
        msg = f"Validation completed between {source_file.name.split('/')[-1]} and {ref_file.name.split('/')[-1]}"
        return msg, score, mismatches
    except Exception as e:
        return f"Error: {str(e)}", 0.0, pd.DataFrame([{"Error": str(e)}])

def create_reference_validation_tab():
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Source Dataset")
            source_ds = gr.File(label="Upload Source Dataset")
            
        with gr.Column():
            gr.Markdown("### Reference Dataset")
            ref_ds = gr.File(label="Upload Reference Dataset")
            
    val_btn = gr.Button("Run Reference Matching Engine", variant="primary")
    
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
