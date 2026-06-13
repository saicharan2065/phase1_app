import gradio as gr
import pandas as pd
from data.dataset_manager import DatasetManager
from validation.matching_engine import ReferenceDataMatchingEngine

dm = DatasetManager()

def update_dataset_choices(category):
    if category == "Custom Hugging Face Dataset":
        return gr.update(choices=[], visible=False), gr.update(visible=True)
    
    datasets = dm.fetch_datasets_by_category(category)
    return gr.update(choices=datasets, value=datasets[0] if datasets else None, visible=True), gr.update(visible=False)

def update_estimates(limit_str):
    mem, time = dm.estimate_memory_and_time(limit_str)
    return f"**Est. Mem:** {mem} | **Time:** {time}"

def load_dataset_ui(category, ds_dropdown, custom_ds, limit_str):
    dataset_id = custom_ds if category == "Custom Hugging Face Dataset" else ds_dropdown
    if not dataset_id:
        return pd.DataFrame(), "Please select or enter a dataset ID."
        
    df = dm.load_dataset_records(dataset_id, limit_str)
    if not df.empty and "Error" not in df.columns:
        msg = f"Loaded {len(df)} rows from {dataset_id}"
    else:
        msg = df.iloc[0]["Error"] if "Error" in df.columns else "Dataset is empty."
        
    return df, msg

def get_preview_info(df):
    if df is None or df.empty:
        return "No data loaded."
        
    info = f"**Rows:** {len(df)} | **Columns:** {len(df.columns)} | **Missing:** {df.isna().sum().sum()}"
    return info

def refresh_cache_info():
    info = dm.get_cache_info()
    return f"**Cache Size:** {info['size_mb']:.2f} MB\n\n**Location:** {info['location']}"

def clear_cache_ui():
    msg = dm.clear_cache()
    return refresh_cache_info()

def run_comparison(source_df, ref_df):
    if source_df is None or source_df.empty or ref_df is None or ref_df.empty:
        return "Load Source and Reference datasets first.", pd.DataFrame()
        
    engine = ReferenceDataMatchingEngine()
    score, mismatches = engine.match_records(source_df, ref_df)
    return f"Validation Score: {score}%", mismatches

def create_dataset_marketplace_tab():
    categories = list(dm.category_map.keys()) + ["Custom Hugging Face Dataset"]
    limits = ["100", "1000", "10000", "100000", "1000000", "ALL"]
    
    with gr.Row():
        # Source Panel
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("#### SOURCE DATASET")
                s_cat = gr.Dropdown(choices=categories, label="Category")
                s_ds = gr.Dropdown(choices=[], label="Available Datasets")
                s_custom = gr.Textbox(label="Custom Dataset ID", visible=False)
                
                with gr.Row():
                    s_limit = gr.Dropdown(choices=limits, value="1000", label="Records")
                    s_est = gr.Markdown(update_estimates("1000"))
                
                s_load_btn = gr.Button("Load Source", variant="primary")
                s_status = gr.Textbox(label="Status", interactive=False)
                s_info = gr.Markdown("")
                
            s_df_state = gr.State()
            
            s_cat.change(update_dataset_choices, inputs=s_cat, outputs=[s_ds, s_custom])
            s_limit.change(update_estimates, inputs=s_limit, outputs=s_est)
            
            s_load_btn.click(
                load_dataset_ui, 
                inputs=[s_cat, s_ds, s_custom, s_limit], 
                outputs=[s_df_state, s_status]
            ).then(
                get_preview_info,
                inputs=s_df_state,
                outputs=s_info
            )

        # Reference Panel
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("#### REFERENCE DATASET")
                r_cat = gr.Dropdown(choices=categories, label="Category")
                r_ds = gr.Dropdown(choices=[], label="Available Datasets")
                r_custom = gr.Textbox(label="Custom Dataset ID", visible=False)
                
                with gr.Row():
                    r_limit = gr.Dropdown(choices=limits, value="1000", label="Records")
                    r_est = gr.Markdown(update_estimates("1000"))
                
                r_load_btn = gr.Button("Load Reference", variant="primary")
                r_status = gr.Textbox(label="Status", interactive=False)
                r_info = gr.Markdown("")
                
            r_df_state = gr.State()
            
            r_cat.change(update_dataset_choices, inputs=r_cat, outputs=[r_ds, r_custom])
            r_limit.change(update_estimates, inputs=r_limit, outputs=r_est)
            
            r_load_btn.click(
                load_dataset_ui, 
                inputs=[r_cat, r_ds, r_custom, r_limit], 
                outputs=[r_df_state, r_status]
            ).then(
                get_preview_info,
                inputs=r_df_state,
                outputs=r_info
            )

    # Actions and Previews
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("#### COMPARISON ACTIONS")
                with gr.Row():
                    compare_btn = gr.Button("Compare", variant="primary")
                    swap_btn = gr.Button("Swap")
                    clear_btn = gr.Button("Clear")
                
                comp_status = gr.Textbox(label="Comparison Status", interactive=False)
                
            with gr.Group():
                gr.Markdown("#### CACHE MANAGEMENT")
                cache_info = gr.Markdown(refresh_cache_info())
                clear_cache_btn = gr.Button("Clear Cache")
                clear_cache_btn.click(clear_cache_ui, outputs=cache_info)

        with gr.Column(scale=3):
            gr.Markdown("#### DATA PREVIEWS")
            with gr.Tabs():
                with gr.Tab("Source Preview"):
                    s_preview_table = gr.Dataframe(max_height=300)
                with gr.Tab("Reference Preview"):
                    r_preview_table = gr.Dataframe(max_height=300)
                with gr.Tab("Validation Results"):
                    comp_table = gr.Dataframe(max_height=300)
                    
            # Wire previews
            s_load_btn.click(
                lambda df: df.head(100) if df is not None and not df.empty else pd.DataFrame(),
                inputs=s_df_state, outputs=s_preview_table
            )
            r_load_btn.click(
                lambda df: df.head(100) if df is not None and not df.empty else pd.DataFrame(),
                inputs=r_df_state, outputs=r_preview_table
            )
            
            compare_btn.click(
                run_comparison,
                inputs=[s_df_state, r_df_state],
                outputs=[comp_status, comp_table]
            )
            
            # Simple swap logic
            def swap_datasets(s_df, r_df):
                return r_df, s_df
                
            swap_btn.click(
                swap_datasets,
                inputs=[s_df_state, r_df_state],
                outputs=[s_df_state, r_df_state]
            ).then(
                lambda df: df.head(100) if df is not None and not df.empty else pd.DataFrame(),
                inputs=s_df_state, outputs=s_preview_table
            ).then(
                lambda df: df.head(100) if df is not None and not df.empty else pd.DataFrame(),
                inputs=r_df_state, outputs=r_preview_table
            )
            
            # Simple clear logic
            def clear_datasets():
                return pd.DataFrame(), pd.DataFrame(), "", ""
                
            clear_btn.click(
                clear_datasets,
                outputs=[s_df_state, r_df_state, s_status, r_status]
            ).then(
                lambda: pd.DataFrame(), outputs=s_preview_table
            ).then(
                lambda: pd.DataFrame(), outputs=r_preview_table
            )
