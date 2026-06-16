import gradio as gr
import pandas as pd
import random

from data.dataset_manager import get_user_workspace

def check_dataframe_guardrails(df, filename="dataset"):
    # Universal Domain Guardrail (block pure operational/biometric data, allow if mixed with financial context)
    cols = [str(c).lower() for c in df.columns]
    
    # Pure operational / biological / physical keywords across sectors
    operational_keywords = [
        "heart_rate", "blood_pressure", "cholesterol", "biometric", "bmi", "oxygen_level", # Medical
        "machine_temperature", "motor_rpm", "concrete_drying", "factory_humidity", # Manufacturing
        "shirt_size", "fabric_type", "customer_review", # Retail
        "wind_speed", "turbine_voltage", "grid_frequency", "weather", "temperature", # Energy/Environment
        "paint_color", "roof_type", "square_footage" # Real Estate pure physical
    ]
    
    # Universal Financial/Transactional keywords
    financial_keywords = ["claim", "amount", "billing", "insurance", "cost", "price", "payment", "transaction", "invoice", "vendor", "contract", "escrow", "refund", "wire", "transfer", "buyer", "balance"]
    
    has_operational = any(any(kw in c for kw in operational_keywords) for c in cols)
    has_financial = any(any(kw in c for kw in financial_keywords) for c in cols)
    
    if has_operational and not has_financial:
        raise ValueError(f"Compliance Error: Pure Operational/Physical data detected in '{filename}'. The Financial Crime OS strictly rejects non-financial data across all sectors to prevent data poisoning.")
        
    return True


def load_hf_dataset(dataset_name, username):
    if not dataset_name:
        return "Please enter a dataset name."
    try:
        from datasets import load_dataset
        ds = load_dataset(dataset_name, split="train")
        df = ds.to_pandas()
        
        # Check DataFrame guardrails
        check_dataframe_guardrails(df, dataset_name)
        
        get_user_workspace(username)[dataset_name] = df
        return f"Successfully loaded Hugging Face Dataset: {dataset_name} ({len(df)} rows)"
    except Exception as e:
        return f"Failed to load dataset: {e}"

def upload_file(file, username):
    if file is None:
        return "No file uploaded."
    try:
        import os
        
        # File Size Guardrail (500MB)
        file_size_mb = os.path.getsize(file.name) / (1024 * 1024)
        if file_size_mb > 500:
            return f"Guardrail Alert: File size ({file_size_mb:.1f} MB) exceeds the 500MB limit to prevent memory exhaustion."
            
        filename = os.path.basename(file.name)
        if filename.endswith(".csv"):
            df = pd.read_csv(file.name)
        elif filename.endswith(".xlsx") or filename.endswith(".xls"):
            df = pd.read_excel(file.name)
        elif filename.endswith(".json"):
            df = pd.read_json(file.name)
        else:
            return "Unsupported file format."
            
        # Check DataFrame guardrails
        check_dataframe_guardrails(df, filename)
        
        get_user_workspace(username)[filename] = df
        return f"Successfully uploaded and registered: {filename} ({len(df)} rows)"
    except Exception as e:
        return f"Failed to process file: {e}"

def import_url(url, username):
    if not url:
        return "Please enter a valid URL."
    try:
        df = pd.read_csv(url)
        filename = url.split("/")[-1] or "imported_data.csv"
        
        # Check DataFrame guardrails
        check_dataframe_guardrails(df, filename)
        
        get_user_workspace(username)[filename] = df
        return f"Successfully imported data from URL: {filename} ({len(df)} rows)"
    except Exception as e:
        return f"Failed to import URL: {e}"

def create_data_sources_tab(session_user):
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Load Hugging Face Dataset")
            hf_ds_name = gr.Textbox(label="Dataset Name (e.g., imdb)")
            hf_load_btn = gr.Button("Load Dataset")
            hf_out = gr.Textbox(label="Output", interactive=False)
            hf_load_btn.click(fn=load_hf_dataset, inputs=[hf_ds_name, session_user], outputs=hf_out)
            
        with gr.Column():
            gr.Markdown("### Upload CSV/JSON/XLSX")
            file_upload = gr.File(label="Upload File", file_types=[".csv", ".json", ".xlsx"])
            file_out = gr.Textbox(label="Output", interactive=False)
            file_upload.upload(fn=upload_file, inputs=[file_upload, session_user], outputs=file_out)
            
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Import URL")
            gr.Markdown("Download CSV data directly from a public URL.")
            url_input = gr.Textbox(label="Data URL (.csv)")
            url_btn = gr.Button("Import")
            url_out = gr.Textbox(label="Output", interactive=False)
            url_btn.click(fn=import_url, inputs=[url_input, session_user], outputs=url_out)
            
        with gr.Column():
            gr.Markdown("### Global Workspace Registry")
            gr.Markdown("Any data loaded on this page is instantly available globally in the Dataset Marketplace under the **Local Workspace Data** category.")
