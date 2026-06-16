import gradio as gr
import pandas as pd
import random

from data.dataset_manager import GLOBAL_WORKSPACE_DATA

def load_hf_dataset(dataset_name):
    if not dataset_name:
        return "Please enter a dataset name."
    try:
        from datasets import load_dataset
        ds = load_dataset(dataset_name, split="train")
        df = ds.to_pandas()
        GLOBAL_WORKSPACE_DATA[dataset_name] = df
        return f"Successfully loaded Hugging Face Dataset: {dataset_name} ({len(df)} rows)"
    except Exception as e:
        return f"Failed to load dataset: {e}"

def upload_file(file):
    if file is None:
        return "No file uploaded."
    try:
        import os
        filename = os.path.basename(file.name)
        if filename.endswith(".csv"):
            df = pd.read_csv(file.name)
        elif filename.endswith(".xlsx") or filename.endswith(".xls"):
            df = pd.read_excel(file.name)
        elif filename.endswith(".json"):
            df = pd.read_json(file.name)
        else:
            return "Unsupported file format."
        
        GLOBAL_WORKSPACE_DATA[filename] = df
        return f"Successfully uploaded and registered: {filename} ({len(df)} rows)"
    except Exception as e:
        return f"Failed to process file: {e}"

def import_url(url):
    if not url:
        return "Please enter a valid URL."
    try:
        df = pd.read_csv(url)
        filename = url.split("/")[-1] or "imported_data.csv"
        GLOBAL_WORKSPACE_DATA[filename] = df
        return f"Successfully imported data from URL: {filename} ({len(df)} rows)"
    except Exception as e:
        return f"Failed to import URL: {e}"

def create_data_sources_tab():
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Load Hugging Face Dataset")
            hf_ds_name = gr.Textbox(label="Dataset Name (e.g., imdb)")
            hf_load_btn = gr.Button("Load Dataset")
            hf_out = gr.Textbox(label="Output", interactive=False)
            hf_load_btn.click(fn=load_hf_dataset, inputs=hf_ds_name, outputs=hf_out)
            
        with gr.Column():
            gr.Markdown("### Upload CSV/JSON/XLSX")
            file_upload = gr.File(label="Upload File", file_types=[".csv", ".json", ".xlsx"])
            file_out = gr.Textbox(label="Output", interactive=False)
            file_upload.upload(fn=upload_file, inputs=file_upload, outputs=file_out)
            
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Import URL")
            gr.Markdown("Download CSV data directly from a public URL.")
            url_input = gr.Textbox(label="Data URL (.csv)")
            url_btn = gr.Button("Import")
            url_out = gr.Textbox(label="Output", interactive=False)
            url_btn.click(fn=import_url, inputs=url_input, outputs=url_out)
            
        with gr.Column():
            gr.Markdown("### Global Workspace Registry")
            gr.Markdown("Any data loaded on this page is instantly available globally in the Dataset Marketplace under the **Local Workspace Data** category.")
