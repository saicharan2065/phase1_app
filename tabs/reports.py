from utils.dummy_generator import generate_dummy_data
import gradio as gr
from reports.report_generator import ReportGenerator

def generate_pdf(title):
    rg = ReportGenerator()
    lines = ["This is an automated Phase 3 Investigation Report.", "Further context would be added here based on actual case details."]
    filepath = rg.generate_pdf_report("test_report", title, lines)
    return f"Report generated at: {filepath}"

def create_reports_tab():
    gr.Markdown("### Reporting Engine")
    title = gr.Textbox(label="Report Title")
    gen_btn = gr.Button("Generate PDF Report")
    out = gr.Textbox(label="Output")
    
    gen_btn.click(fn=generate_pdf, inputs=title, outputs=out)
