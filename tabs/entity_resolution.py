from utils.dummy_generator import generate_dummy_data
import gradio as gr
import pandas as pd
from agents.entity_resolution_agent import EntityResolutionAgent

def resolve_entities_ui(file):
    if file is None:
        return "Please upload a dataset.", pd.DataFrame()
        
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file.name)
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file.name)
        elif file.name.endswith('.json'):
            df = pd.read_json(file.name)
        else:
            return "Unsupported file type.", pd.DataFrame()
            
        agent = EntityResolutionAgent()
        results = agent.resolve_entities(df)
        
        if not results:
            return "No duplicates or related entities found.", pd.DataFrame()
            
        res_df = pd.DataFrame(results)
        return "Entity Resolution Complete", res_df
    except Exception as e:
        return f"Error: {str(e)}", pd.DataFrame()

def create_entity_resolution_tab():
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Upload Dataset")
            ds_upload = gr.File(label="Dataset")
            dummy_btn = gr.Button('Generate Dummy Data', size='sm')
            dummy_btn.click(fn=lambda: generate_dummy_data('entity_resolution'), outputs=ds_upload)
            resolve_btn = gr.Button("Run Entity Resolution", variant="primary")
            
        with gr.Column():
            gr.Markdown("### Resolution Results")
            status_out = gr.Textbox(label="Status", interactive=False)
            
    res_table = gr.Dataframe(label="Related Entities / Duplicates")
            
    resolve_btn.click(fn=resolve_entities_ui, inputs=ds_upload, outputs=[status_out, res_table])
