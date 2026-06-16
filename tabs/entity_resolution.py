from utils.dummy_generator import generate_dummy_data
import gradio as gr
import pandas as pd
from agents.entity_resolution_agent import EntityResolutionAgent

from data.dataset_manager import GLOBAL_WORKSPACE_DATA

def resolve_entities_ui(dataset_key, file):
    if file is not None:
        try:
            if file.name.endswith('.csv'): df = pd.read_csv(file.name)
            elif file.name.endswith('.xlsx'): df = pd.read_excel(file.name)
            elif file.name.endswith('.json'): df = pd.read_json(file.name)
            else: return "Unsupported file type.", pd.DataFrame()
        except Exception as e:
            return f"File error: {e}", pd.DataFrame()
    elif dataset_key and dataset_key in GLOBAL_WORKSPACE_DATA:
        df = GLOBAL_WORKSPACE_DATA[dataset_key]
    else:
        return "Please select a valid dataset from the workspace, or upload a file.", pd.DataFrame()
        
    try:
        agent = EntityResolutionAgent()
        results = agent.resolve_entities(df)
        
        if not results:
            return "No duplicates or related entities found.", pd.DataFrame()
            
        res_df = pd.DataFrame(results)
        
        # Prevent browser crash by truncating massive dataframes
        if len(res_df) > 100:
            preview_df = res_df.head(100)
            return f"Entity Resolution Complete (Showing first 100 of {len(res_df)} matches)", preview_df
            
        return "Entity Resolution Complete", res_df
    except Exception as e:
        return f"Error: {str(e)}", pd.DataFrame()

def create_entity_resolution_tab():
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Select Workspace Dataset")
            with gr.Row():
                ds_dropdown = gr.Dropdown(choices=list(GLOBAL_WORKSPACE_DATA.keys()), label="Dataset", scale=4)
                refresh_btn = gr.Button("↻ Refresh", size="sm", scale=1)
                
            gr.Markdown("### Or Upload Direct File")
            ds_upload = gr.File(label="Dataset Fallback")
            
            resolve_btn = gr.Button("Run Entity Resolution", variant="primary")
            
            refresh_btn.click(fn=lambda: gr.update(choices=list(GLOBAL_WORKSPACE_DATA.keys())), outputs=ds_dropdown)
            
        with gr.Column():
            gr.Markdown("### Resolution Results")
            status_out = gr.Textbox(label="Status", interactive=False)
            
    res_table = gr.Dataframe(label="Related Entities / Duplicates")
            
    resolve_btn.click(fn=resolve_entities_ui, inputs=[ds_dropdown, ds_upload], outputs=[status_out, res_table])
