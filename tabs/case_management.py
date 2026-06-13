from utils.dummy_generator import generate_dummy_data
import gradio as gr
from cases.case_manager import CaseManager
import pandas as pd

def refresh_cases():
    cm = CaseManager()
    return cm.get_all_cases_df()

def create_case(entity_id, reason):
    cm = CaseManager()
    cm.open_case(entity_id, reason)
    return refresh_cases()

def update_case(case_id, new_status):
    cm = CaseManager()
    cm.update_status(case_id, new_status)
    return refresh_cases()

def create_case_management_tab():
    gr.Markdown("### Case Management")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("#### Open New Case")
            ent_id = gr.Textbox(label="Entity ID")
            reason = gr.Textbox(label="Reason")
            open_btn = gr.Button("Open Case")
            
        with gr.Column():
            gr.Markdown("#### Update Case Status")
            case_id = gr.Textbox(label="Case ID")
            status = gr.Dropdown(["OPEN", "IN_PROGRESS", "REVIEW", "CLOSED"], label="Status")
            update_btn = gr.Button("Update Status")
            
    cases_df = gr.Dataframe(label="All Cases", value=refresh_cases)
    
    open_btn.click(fn=create_case, inputs=[ent_id, reason], outputs=cases_df)
    update_btn.click(fn=update_case, inputs=[case_id, status], outputs=cases_df)
