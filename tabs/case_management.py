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
        
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from tabs.model_management import get_active_model_state
        
        active_model = get_active_model_state()
        model_id = active_model if active_model and active_model != "None Selected" else "Qwen/Qwen1.5-0.5B"
        
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True, use_safetensors=True)
        
        # We will generate them sequentially
        yield "Thinking...", "Thinking...", "Thinking..."
        
        def run_llm(prompt_text):
            inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7)
            return tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt_text, "").strip()
            
        p_out = run_llm(f"Act as a Financial Prosecutor. Explain why Case {case_id} is guilty of money laundering.")
        yield p_out, "Thinking...", "Thinking..."
        
        d_out = run_llm(f"Act as a Defense Attorney. Read the Prosecutor's argument and defend Case {case_id}: {p_out}")
        yield p_out, d_out, "Thinking..."
        
        j_out = run_llm(f"Act as a Judge. Read the Prosecutor's argument: '{p_out}' and Defense's argument: '{d_out}'. Deliver a final verdict.")
        yield p_out, d_out, j_out
        
    except Exception as e:
        err = f"🛑 CRITICAL LLM EXECUTION ERROR: {str(e)}"
        yield err, err, err

def create_case_management_tab(session_user=None):
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
