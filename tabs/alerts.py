from utils.dummy_generator import generate_dummy_data
import gradio as gr
from alerts.alert_engine import AlertEngine

def refresh_alerts():
    ae = AlertEngine()
    return ae.get_all_alerts_df()

def generate_mock_alert(ent_id, level):
    ae = AlertEngine()
    ae.generate_alert(ent_id, "Manual Trigger", level, "Manually generated alert from UI")
    return refresh_alerts()

def save_smtp(server, port, email, password, recipient):
    ae = AlertEngine()
    msg = ae.save_smtp_config(server, port, email, password, recipient)
    return msg

def test_smtp():
    ae = AlertEngine()
    success, msg = ae.test_smtp_connection()
    return msg

def load_smtp_ui():
    ae = AlertEngine()
    config = ae.load_smtp_config()
    if config:
        return config.get("server", ""), config.get("port", "587"), config.get("email", ""), config.get("password", ""), config.get("recipient", "")
    return "", "587", "", "", ""

def create_alerts_tab():
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### Alert Engine")
            gr.Markdown("Manually trigger alerts here. If an alert is HIGH or CRITICAL, it will automatically dispatch an email.")
            with gr.Row():
                ent_id = gr.Textbox(label="Entity ID")
                level = gr.Dropdown(["INFO", "LOW", "MEDIUM", "HIGH", "CRITICAL"], label="Level", value="CRITICAL")
                trigger_btn = gr.Button("Trigger Manual Alert", variant="primary")
            
            alerts_df = gr.Dataframe(label="Alerts Database", value=refresh_alerts)
            trigger_btn.click(fn=generate_mock_alert, inputs=[ent_id, level], outputs=alerts_df)
            
        with gr.Column(scale=1):
            gr.Markdown("### SMTP Mail Configuration")
            gr.Markdown("Configure your mail server to dispatch system alerts.")
            
            smtp_server = gr.Textbox(label="SMTP Server", placeholder="smtp.gmail.com")
            smtp_port = gr.Textbox(label="Port", placeholder="587 or 465")
            smtp_email = gr.Textbox(label="Sender Email (e.g., your_email@gmail.com)")
            smtp_pass = gr.Textbox(label="App Password", type="password")
            smtp_target = gr.Textbox(label="Recipient Email (Where to send alerts)")
            
            with gr.Row():
                save_btn = gr.Button("Save Configuration")
                test_btn = gr.Button("Test Connection", variant="secondary")
                
            smtp_status = gr.Textbox(label="SMTP Status", interactive=False)
            
            # Load initial config on boot
            # In Gradio, we can use a dummy trigger or just initialize the components with default values 
            # by fetching them once during construction.
            sv, pt, em, pw, tg = load_smtp_ui()
            smtp_server.value = sv
            smtp_port.value = pt
            smtp_email.value = em
            smtp_pass.value = pw
            smtp_target.value = tg
            
            save_btn.click(fn=save_smtp, inputs=[smtp_server, smtp_port, smtp_email, smtp_pass, smtp_target], outputs=smtp_status)
            test_btn.click(fn=test_smtp, outputs=smtp_status)
