import gradio as gr
from alerts.alert_engine import AlertEngine

def refresh_alerts():
    ae = AlertEngine()
    return ae.get_all_alerts_df()

def generate_mock_alert(ent_id, level):
    ae = AlertEngine()
    ae.generate_alert(ent_id, "Manual Trigger", level, "Manually generated alert from UI")
    return refresh_alerts()

def create_alerts_tab():
    gr.Markdown("### Alert Engine")
    
    with gr.Row():
        ent_id = gr.Textbox(label="Entity ID")
        level = gr.Dropdown(["INFO", "LOW", "MEDIUM", "HIGH", "CRITICAL"], label="Level")
        trigger_btn = gr.Button("Trigger Manual Alert")
        
    alerts_df = gr.Dataframe(label="Alerts Database", value=refresh_alerts)
    
    trigger_btn.click(fn=generate_mock_alert, inputs=[ent_id, level], outputs=alerts_df)
