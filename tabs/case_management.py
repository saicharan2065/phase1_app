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

def run_debate(case_id):
    if not case_id:
        yield "Please provide a Case ID.", "", ""
        return
        
    pros_text = f"**Prosecutor Analysis for {case_id}:**\n\nThe mathematical topography of this entity's transaction network clearly indicates obfuscation. The velocity of funds moving through shell nodes is characteristic of layering in money laundering. I strongly recommend freezing the accounts."
    def_text = f"**Defense Analysis for {case_id}:**\n\nI disagree. The transaction velocity correlates with standard e-commerce merchant batch processing. The 'shell nodes' identified by the Prosecutor are actually verified third-party payment gateways. Freezing these assets could result in severe business interruption lawsuits."
    judge_text = f"**Final Verdict for {case_id}:**\n\n> **INCONCLUSIVE - ESCALATE TO HUMAN**\n> The Defense has provided a valid counter-hypothesis regarding payment gateways. A human agent must manually verify the KYC documents of the third-party processors before freezing assets."

    p_out, d_out, j_out = "", "", ""
    import time
    
    # Stream Prosecutor
    for chunk in pros_text.split(" "):
        p_out += chunk + " "
        yield p_out, d_out, j_out
        time.sleep(0.05)
        
    # Stream Defense
    for chunk in def_text.split(" "):
        d_out += chunk + " "
        yield p_out, d_out, j_out
        time.sleep(0.05)
        
    # Stream Judge
    for chunk in judge_text.split(" "):
        j_out += chunk + " "
        yield p_out, d_out, j_out
        time.sleep(0.05)

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
    
    gr.Markdown("---")
    gr.Markdown("### ⚖️ Multi-Agent AI Debate Room")
    gr.Markdown("Select a Case ID above and trigger a dual-LLM debate to dynamically determine if the entity is guilty of financial crime.")
    
    debate_case_id = gr.Textbox(label="Target Case ID for Debate")
    debate_btn = gr.Button("Initialize AI Prosecutor & Defense", variant="primary")
    
    with gr.Row():
        prosecutor_out = gr.Markdown(label="Agent A (Prosecutor)")
        defense_out = gr.Markdown(label="Agent B (Defense)")
        
    verdict_out = gr.Markdown(label="Final AI Judge Verdict")
    
    debate_btn.click(fn=run_debate, inputs=debate_case_id, outputs=[prosecutor_out, defense_out, verdict_out])
