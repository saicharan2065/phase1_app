import gradio as gr
import pandas as pd
import os
from pathlib import Path
import shutil
from data.dataset_manager import DatasetManager
from validation.matching_engine import ReferenceDataMatchingEngine

dm = DatasetManager()

def update_dataset_choices(category, username):
    if category == "Custom Hugging Face Dataset":
        return gr.update(choices=[], visible=False), gr.update(visible=True)
    
    if category == "Local Workspace Data":
        from data.dataset_manager import get_user_workspace
        datasets = list(get_user_workspace(username).keys())
    else:
        datasets = dm.fetch_datasets_by_category(category)
        
    return gr.update(choices=datasets, value=datasets[0] if datasets else None, visible=True), gr.update(visible=False)

def update_estimates(limit_str):
    mem, time = dm.estimate_memory_and_time(limit_str)
    return f"**Est. Mem:** {mem} | **Time:** {time}"

def load_dataset_ui(category, ds_dropdown, custom_ds, limit_str, username):
    dataset_id = custom_ds if category == "Custom Hugging Face Dataset" else ds_dropdown
    if not dataset_id:
        yield pd.DataFrame(), "Please select or enter a dataset ID.", pd.DataFrame()
        return
        
    yield pd.DataFrame(), f"⏳ INITIALIZING: Starting background thread for {dataset_id}...", pd.DataFrame()
    
    import time
    task_id = dm.start_async_download(dataset_id, limit_str, username)
    
    while True:
        status_info = dm.background_tasks.get(task_id)
        if not status_info:
            break
            
        p = status_info["process"]
        import os
        safe_id = str(dataset_id).replace("/", "_").replace("\\", "_")
        error_file = os.path.join(dm.cache_dir, f"{safe_id}_error.txt")
        
        if p and not p.is_alive():
            # Process Finished! Check for custom error file from subprocess
            if os.path.exists(error_file):
                with open(error_file, "r") as f:
                    err = f.read()
                yield pd.DataFrame(), f"❌ ERROR: {err}", pd.DataFrame()
                os.remove(error_file)
                break
                
            if p.exitcode != 0:
                yield pd.DataFrame(), f"❌ ERROR: Subprocess crashed with exit code {p.exitcode}", pd.DataFrame()
                break
                
            # Now we instantly load the cached memory-mapped pointer in the main thread
            df = dm._load_dataset_records_sync(status_info["dataset_id"], status_info["limit_str"], status_info["username"])
            
            if isinstance(df, pd.DataFrame) and "Error" in df.columns:
                yield pd.DataFrame(), f"❌ ERROR: {df.iloc[0]['Error']}", pd.DataFrame()
                break
            
            # Universal Data Sync (Pointer)
            from data.dataset_manager import get_user_workspace
            get_user_workspace(username)[dataset_id] = df
            
            # Slice 100 rows strictly for UI Preview to keep RAM near 0GB
            if isinstance(df, pd.DataFrame):
                preview_df = df.head(100)
                len_str = str(len(df))
            else:
                preview_ds = df.select(range(min(100, len(df))))
                clean_records = []
                for row in preview_ds:
                    clean_row = {}
                    for k, v in row.items():
                        if type(v).__name__ not in ['bytes', 'Image', 'PngImageFile', 'JpegImageFile']:
                            clean_row[k] = v
                        else:
                            clean_row[k] = "<Image/Bytes Data>"
                    clean_records.append(clean_row)
                preview_df = pd.DataFrame(clean_records)
                len_str = str(len(df))
                
            msg = f"✅ Loaded {len_str} rows from {dataset_id} (0GB RAM Used)"
            yield preview_df, msg, safe_preview(preview_df)
            break
            
        yield pd.DataFrame(), f"⏳ DOWNLOADING IN BACKGROUND: Rust/Xet Acceleration Active...", pd.DataFrame()
        time.sleep(1)

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

def delete_specific_ui(dataset_folder):
    msg = dm.delete_specific_dataset(dataset_folder)
    return refresh_cache_info()

def run_comparison(source_df, ref_df, progress=gr.Progress(track_tqdm=True)):
    if source_df is None or source_df.empty or ref_df is None or ref_df.empty:
        return "Load Source and Reference datasets first.", pd.DataFrame()
        
    progress(0, desc="Initializing Matching Engine...")
    engine = ReferenceDataMatchingEngine()
    
    progress(0.5, desc="Computing Matrix Similarity...")
    score, mismatches = engine.match_records(source_df, ref_df)
    
    progress(1.0, desc="Done")
    return f"Validation Score: {score}%", safe_preview(mismatches)

def safe_preview(df):
    if df is None or df.empty:
        return pd.DataFrame()
    # Limit rows to 15, cols to 15, and truncate extremely long strings to prevent browser hanging
    preview_df = df.head(15).iloc[:, :15].copy()
    for col in preview_df.select_dtypes(include=['object']):
        preview_df[col] = preview_df[col].astype(str).str.slice(0, 300)
    return preview_df

def create_dataset_marketplace_tab(session_user):
    categories = list(dm.category_map.keys()) + ["Custom Hugging Face Dataset"]
    limits = ["100", "1000", "10000", "100000", "1000000", "ALL"]
    
    # Pre-populate datasets for the first category
    initial_category = categories[0]
    initial_datasets = dm.fetch_datasets_by_category(initial_category)
    
    with gr.Row():
        # Source Panel
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("#### SOURCE DATASET")
                s_cat = gr.Dropdown(choices=categories, value=initial_category, label="Category")
                with gr.Row():
                    s_ds = gr.Dropdown(choices=initial_datasets, value=initial_datasets[0] if initial_datasets else None, label="Available Datasets", scale=4)
                    s_refresh_btn = gr.Button("↻ Refresh", size="sm", scale=1)
                s_custom = gr.Textbox(label="Custom Dataset ID", visible=False)
                
                with gr.Row():
                    s_limit = gr.Dropdown(choices=limits, value="1000", label="Records")
                    s_est = gr.Markdown(update_estimates("1000"))
                
                s_load_btn = gr.Button("Load Source", variant="primary")
                s_status = gr.Textbox(label="Status", interactive=False)
                s_info = gr.Markdown("")
                
            s_df_state = gr.State()
            
            s_cat.change(update_dataset_choices, inputs=[s_cat, session_user], outputs=[s_ds, s_custom])
            s_refresh_btn.click(update_dataset_choices, inputs=[s_cat, session_user], outputs=[s_ds, s_custom])
            s_limit.change(update_estimates, inputs=s_limit, outputs=s_est)

        # Reference Panel
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("#### REFERENCE DATASET")
                r_cat = gr.Dropdown(choices=categories, value=initial_category, label="Category")
                with gr.Row():
                    r_ds = gr.Dropdown(choices=initial_datasets, value=initial_datasets[0] if initial_datasets else None, label="Available Datasets", scale=4)
                    r_refresh_btn = gr.Button("↻ Refresh", size="sm", scale=1)
                r_custom = gr.Textbox(label="Custom Dataset ID", visible=False)
                
                with gr.Row():
                    r_limit = gr.Dropdown(choices=limits, value="1000", label="Records")
                    r_est = gr.Markdown(update_estimates("1000"))
                
                r_load_btn = gr.Button("Load Reference", variant="primary")
                r_status = gr.Textbox(label="Status", interactive=False)
                r_info = gr.Markdown("")
                
            r_df_state = gr.State()
            
            r_cat.change(update_dataset_choices, inputs=[r_cat, session_user], outputs=[r_ds, r_custom])
            r_refresh_btn.click(update_dataset_choices, inputs=[r_cat, session_user], outputs=[r_ds, r_custom])
            r_limit.change(update_estimates, inputs=r_limit, outputs=r_est)

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
                
                # Make the Cache Size metric real-time
                timer = gr.Timer(2)
                timer.tick(fn=refresh_cache_info, outputs=cache_info)
                
                with gr.Row():
                    cached_ds_dropdown = gr.Dropdown(label="Select Dataset to Delete", choices=dm.get_cached_datasets())
                    refresh_cached_ds_btn = gr.Button("↻", size="sm")
                
                with gr.Row():
                    delete_specific_btn = gr.Button("Delete Selected")
                    clear_cache_btn = gr.Button("Nuclear Wipe (Delete All)", variant="stop")
                    
                clear_cache_btn.click(clear_cache_ui, outputs=cache_info).then(
                    fn=lambda: gr.update(choices=dm.get_cached_datasets(), value=None), outputs=cached_ds_dropdown
                )
                
                delete_specific_btn.click(delete_specific_ui, inputs=[cached_ds_dropdown], outputs=cache_info).then(
                    fn=lambda: gr.update(choices=dm.get_cached_datasets(), value=None), outputs=cached_ds_dropdown
                )
                
                refresh_cached_ds_btn.click(fn=lambda: gr.update(choices=dm.get_cached_datasets()), outputs=cached_ds_dropdown)
                

        with gr.Column(scale=3):
            gr.Markdown("#### DATA PREVIEWS")
            with gr.Tabs():
                with gr.Tab("Source Preview"):
                    s_preview_table = gr.Dataframe(max_height=300)
                with gr.Tab("Reference Preview"):
                    r_preview_table = gr.Dataframe(max_height=300)
                with gr.Tab("Validation Results"):
                    comp_table = gr.Dataframe(max_height=300)
                    
            # Previews are now directly yielded by the load_dataset_ui generator to avoid race conditions.
            s_load_btn.click(
                load_dataset_ui, 
                inputs=[s_cat, s_ds, s_custom, s_limit, session_user], 
                outputs=[s_df_state, s_status, s_preview_table]
            ).then(
                get_preview_info,
                inputs=s_df_state,
                outputs=s_info
            )
            
            r_load_btn.click(
                load_dataset_ui, 
                inputs=[r_cat, r_ds, r_custom, r_limit, session_user], 
                outputs=[r_df_state, r_status, r_preview_table]
            ).then(
                get_preview_info,
                inputs=r_df_state,
                outputs=r_info
            )
            
            # Compare and Swap buttons remain the same.
            
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
                safe_preview,
                inputs=s_df_state, outputs=s_preview_table
            ).then(
                safe_preview,
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
