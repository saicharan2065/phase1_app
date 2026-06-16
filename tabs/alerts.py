from utils.dummy_generator import generate_dummy_data
import gradio as gr
from alerts.alert_engine import AlertEngine

def refresh_alerts():
    ae = AlertEngine()
    return ae.get_all_alerts_df()

def generate_mock_alert(ent_id, level, target_email):
    ae = AlertEngine()
    ae.generate_alert(ent_id, "Manual Trigger", level, "Manually generated alert from UI", target_email=target_email)
    return refresh_alerts()

def save_smtp(email):
    ae = AlertEngine()
    msg = ae.save_smtp_config(email, "", "")
    return msg

def test_smtp():
    ae = AlertEngine()
    success, msg = ae.test_smtp_connection()
    return msg

def load_smtp_ui():
    ae = AlertEngine()
    config = ae.load_smtp_config()
    if config:
        return config.get("email", "")
    return ""

def create_alerts_tab(session_user=None):
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### Alert Dispatcher")
            gr.Markdown("Manually trigger alerts here. If an alert is HIGH or CRITICAL, it will automatically dispatch an email to the target user.")
            with gr.Row():
                ent_id = gr.Textbox(label="Entity ID")
                level = gr.Dropdown(["INFO", "LOW", "MEDIUM", "HIGH", "CRITICAL"], label="Level", value="CRITICAL")
                target_email = gr.Textbox(label="Target User Emails (Comma-separated list)")
                trigger_btn = gr.Button("Dispatch Alert to User", variant="primary")
            
            alerts_df = gr.Dataframe(label="Alerts Database", value=refresh_alerts)
        with gr.Column(scale=1):
            gr.Markdown("### Agent Configuration")
            gr.Markdown("Alert dispatch logic has been centralized. SMTP profiles are now handled by the secure authentication gateway.")
            
    trigger_btn.click(fn=generate_mock_alert, inputs=[ent_id, level, target_email], outputs=alerts_df)
