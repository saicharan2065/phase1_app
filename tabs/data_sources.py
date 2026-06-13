import gradio as gr
import pandas as pd
import random

def load_hf_dataset(dataset_name):
    if not dataset_name:
        return "Please enter a dataset name."
    return f"Successfully loaded Hugging Face Dataset: {dataset_name} (Mocked)"

def upload_file(file):
    if file is None:
        return "No file uploaded."
    return f"Successfully uploaded file: {file.name}"

def import_url(url):
    if not url:
        return "Please enter a valid URL."
    return f"Successfully imported data from URL: {url} (Mocked)"

def generate_synthetic_data(num_records):
    # Mock synthetic data generation
    data = {
        "id": range(1, int(num_records) + 1),
        "name": [f"User_{i}" for i in range(1, int(num_records) + 1)],
        "value": [round(random.uniform(10.0, 100.0), 2) for _ in range(int(num_records))],
        "status": [random.choice(["Active", "Inactive", "Pending"]) for _ in range(int(num_records))]
    }
    df = pd.DataFrame(data)
    return df

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
            url_input = gr.Textbox(label="Data URL")
            url_btn = gr.Button("Import")
            url_out = gr.Textbox(label="Output", interactive=False)
            url_btn.click(fn=import_url, inputs=url_input, outputs=url_out)
            
        with gr.Column():
            gr.Markdown("### Generate Synthetic Data")
            num_records = gr.Number(value=100, label="Number of Records", precision=0)
            synth_btn = gr.Button("Generate")
            synth_out = gr.Dataframe(label="Synthetic Data Preview")
            synth_btn.click(fn=generate_synthetic_data, inputs=num_records, outputs=synth_out)
