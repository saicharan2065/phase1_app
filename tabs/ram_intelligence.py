import gradio as gr
from agents.entity_memory import entity_memory_index
from agents.scenario_simulator import scenario_simulator
from agents.ram_graph_engine import ram_graph_engine
import threading
import pandas as pd
import psutil

def create_ram_intelligence_tab():
    with gr.TabItem("2TB RAM Intelligence", id="ram_intel"):
        gr.Markdown("## 2TB RAM Intelligence: Extreme Scale Analytics")
        gr.Markdown("This dashboard bypasses Graphics Card limits by utilizing your massive 2000 GB System RAM. It downloads real Hugging Face datasets and performs extreme-scale analysis on millions of rows instantly. **No simulations, no fake data.**")
        
        from tabs.dataset_marketplace import dm
        available_datasets = dm.get_cached_datasets()
        
        with gr.Row():
            target_dataset = gr.Dropdown(
                choices=available_datasets,
                value=available_datasets[0] if available_datasets and available_datasets[0] != "No Datasets Cached" else "No Datasets Cached",
                label="Target Dataset (Real Hugging Face Data)",
                allow_custom_value=True,
                scale=4
            )
            refresh_ds_btn = gr.Button("↻ Refresh Datasets", scale=1)
            
        def refresh_datasets():
            ds = dm.get_cached_datasets()
            return gr.update(choices=ds)
            
        refresh_ds_btn.click(fn=refresh_datasets, outputs=target_dataset)

        with gr.Tabs():
            # Section 1: RAM-Based Vector Search
            with gr.TabItem("Entity Vector Memory"):
                gr.Markdown("### 1. Semantic Search Engine")
                gr.Markdown("Click 'Build Memory' to convert every row in the real dataset into searchable text. Then, type a human sentence (like 'rapid account movement') and it will instantly find the closest matching real rows from the dataset.")
                
                with gr.Row():
                    build_mem_btn = gr.Button("🧠 Build Vector Memory (Real Data)", variant="primary")
                    clear_mem_btn = gr.Button("🗑️ Flush Memory", variant="stop")
                    
                mem_status = gr.Textbox(label="Vector DB Status", interactive=False)
                mem_progress = gr.HTML()
                
                with gr.Row():
                    search_query = gr.Textbox(label="Semantic Query (e.g., 'fraudulent account with massive amounts')", scale=4)
                    search_btn = gr.Button("🔍 Search Entities", scale=1)
                    
                search_results = gr.Dataframe(label="Top Matching Entities", interactive=False)
                
                def get_mem_status():
                    pct = getattr(entity_memory_index, 'progress_percent', 0)
                    html = f"<div style='width: 100%; background-color: #ddd;'><div style='width: {pct}%; height: 10px; background-color: #4CAF50;'></div></div>" if pct > 0 else ""
                    return entity_memory_index.status, html
                    
                def trigger_build_mem(ds):
                    import threading
                    t = threading.Thread(target=entity_memory_index.build_index, args=(ds, 50000))
                    t.start()
                    return "Initializing memory build sequence in background..."
                    
                build_mem_btn.click(fn=trigger_build_mem, inputs=[target_dataset], outputs=mem_status)
                clear_mem_btn.click(fn=entity_memory_index.clear, outputs=mem_status)
                search_btn.click(fn=entity_memory_index.search, inputs=[search_query], outputs=search_results)
                
                try:
                    timer_mem = gr.Timer(2)
                    timer_mem.tick(fn=get_mem_status, outputs=[mem_status, mem_progress])
                except AttributeError:
                    pass

            # Section 2: Real-Time Scenario Simulator
            with gr.TabItem("Live Stream Processor"):
                gr.Markdown("### 2. Live Data Streamer")
                gr.Markdown("Click 'Start Live Stream' to literally stream the actual rows from the dataset into memory one-by-one, mimicking a live banking network. It flags high-risk transactions in real-time.")
                
                with gr.Row():
                    scenario_type = gr.Dropdown(["Structuring", "Layering", "Mule Network", "Shell Company Web"], value="Layering", label="Detection Scenario")
                    tps_slider = gr.Slider(minimum=1, maximum=1000, value=50, label="Transactions Per Second (TPS)")
                    duration_slider = gr.Slider(minimum=10, maximum=600, value=60, label="Stream Duration (Seconds)")
                    
                with gr.Row():
                    start_stream_btn = gr.Button("⚡ Start Live Stream", variant="primary")
                    stop_stream_btn = gr.Button("🛑 Stop Stream", variant="stop")
                    
                stream_status = gr.Textbox(label="Engine Status", interactive=False)
                stream_progress = gr.HTML()
                
                with gr.Row():
                    processed_box = gr.Textbox(label="Entities Processed", interactive=False)
                    alerts_box = gr.Textbox(label="Neural Alerts Triggered", interactive=False)
                    risk_box = gr.Textbox(label="Avg Live Risk", interactive=False)
                    ram_box = gr.Textbox(label="RAM Footprint", interactive=False)
                    
                live_table = gr.Dataframe(label="High Risk Stream Log", interactive=False)
                
                def get_stream_metrics():
                    df = pd.DataFrame(scenario_simulator.live_entities)
                    ram_str = f"{psutil.Process().memory_info().rss / (1024*1024*1024):.2f} GB"
                    pct = getattr(scenario_simulator, 'progress_percent', 0)
                    html = f"<div style='width: 100%; background-color: #ddd;'><div style='width: {pct}%; height: 10px; background-color: #FF5722;'></div></div>" if pct > 0 else ""
                    return (
                        scenario_simulator.status,
                        html,
                        str(scenario_simulator.total_processed),
                        str(scenario_simulator.total_alerts),
                        f"{scenario_simulator.current_risk_avg:.1f}%",
                        ram_str,
                        df
                    )
                
                start_stream_btn.click(
                    fn=scenario_simulator.start_stream, 
                    inputs=[target_dataset, scenario_type, duration_slider, tps_slider], 
                    outputs=stream_status
                )
                stop_stream_btn.click(fn=scenario_simulator.stop_stream, outputs=stream_status)
                
                try:
                    timer_stream = gr.Timer(1)
                    timer_stream.tick(
                        fn=get_stream_metrics, 
                        outputs=[stream_status, stream_progress, processed_box, alerts_box, risk_box, ram_box, live_table]
                    )
                except AttributeError:
                    pass

            # Section 3: Massive Graph Investigation
            with gr.TabItem("NetworkX Graph Topography"):
                gr.Markdown("### 3. Massive Network Graph")
                gr.Markdown("Click 'Construct Topology Map' to analyze the real sender/receiver columns in the dataset. It maps every single transaction as a giant web in your System RAM to find the hidden 'hubs' moving the most money.")
                
                with gr.Row():
                    build_graph_btn = gr.Button("🕸️ Construct Topology Map (Real Data)", variant="primary")
                    analyze_graph_btn = gr.Button("🔬 Run PageRank Analysis")
                    clear_graph_btn = gr.Button("🗑️ Flush Topology", variant="stop")
                    
                graph_status = gr.Textbox(label="Graph Engine Status", interactive=False)
                graph_progress = gr.HTML()
                
                with gr.Row():
                    node_box = gr.Textbox(label="Total Nodes", interactive=False)
                    edge_box = gr.Textbox(label="Total Edges", interactive=False)
                    graph_ram_box = gr.Textbox(label="RAM Footprint", interactive=False)
                    
                graph_findings = gr.Dataframe(label="Highest Centrality Hubs", interactive=False)
                
                def trigger_build_graph(ds):
                    t = threading.Thread(target=ram_graph_engine.build_graph, args=(ds, 200000))
                    t.start()
                    return "Initializing massive graph mapping..."
                    
                def get_graph_metrics():
                    ram_str = f"{ram_graph_engine.ram_footprint:.2f} GB"
                    
                    pct = getattr(ram_graph_engine, 'progress_percent', 0)
                    html = f"<div style='width: 100%; background-color: #ddd;'><div style='width: {pct}%; height: 10px; background-color: #2196F3;'></div></div>" if pct > 0 else ""
                    
                    return ram_graph_engine.status, html, f"{ram_graph_engine.node_count:,}", f"{ram_graph_engine.edge_count:,}", ram_str
                    
                build_graph_btn.click(fn=trigger_build_graph, inputs=[target_dataset], outputs=graph_status)
                analyze_graph_btn.click(fn=ram_graph_engine.analyze_graph, outputs=graph_findings)
                clear_graph_btn.click(fn=ram_graph_engine.clear, outputs=graph_status)
                
                try:
                    timer_graph = gr.Timer(2)
                    timer_graph.tick(fn=get_graph_metrics, outputs=[graph_status, graph_progress, node_box, edge_box, graph_ram_box])
                except AttributeError:
                    pass

            # Section 4: Neural Fine-Tuning
            with gr.TabItem("Neural Fine-Tuning"):
                gr.Markdown("### 4. Deep Learning Fine-Tuning Engine")
                gr.Markdown("Click 'Start CPU-Bound Fine-Tuning' to train a Language Model directly on your System RAM using the real data. This bypasses the Graphics Card entirely to avoid hardware memory deadlocks.")
                
                from tabs.model_management import get_cached_hf_models
                available_models = get_cached_hf_models()
                
                with gr.Row():
                    model_to_tune = gr.Dropdown(
                        choices=available_models, 
                        value=available_models[0] if available_models and available_models[0] != "No Models Cached" else "No Models Cached", 
                        label="Target Model to Mount",
                        allow_custom_value=True
                    )
                    epochs = gr.Slider(1, 10, value=3, step=1, label="Epochs")
                    
                with gr.Row():
                    start_tune_btn = gr.Button("🧠 Start CPU-Bound Fine-Tuning", variant="primary")
                    stop_tune_btn = gr.Button("🛑 Abort Training", variant="stop")
                    
                tune_status = gr.Textbox(label="Training Status", interactive=False)
                progress_bar = gr.HTML()
                
                def trigger_tune(ds, model, ep):
                    from tabs.qlora_training import trainer
                    return trainer.start_training(ds, model, ep)
                    
                def abort_tune():
                    from tabs.qlora_training import trainer
                    return trainer.abort_training()
                    
                def get_tune_status():
                    from tabs.qlora_training import trainer
                    pct = trainer.progress_percent
                    html = f"<div style='width: 100%; background-color: #ddd;'><div style='width: {pct}%; height: 20px; background-color: lightgreen;'></div></div>"
                    return trainer.status_message, html
                    
                start_tune_btn.click(fn=trigger_tune, inputs=[target_dataset, model_to_tune, epochs], outputs=tune_status)
                stop_tune_btn.click(fn=abort_tune, outputs=tune_status)
                
                try:
                    timer_tune = gr.Timer(2)
                    timer_tune.tick(fn=get_tune_status, outputs=[tune_status, progress_bar])
                except AttributeError:
                    pass
                    
        def refresh_ui_elements():
            ds = dm.get_cached_datasets()
            from tabs.model_management import get_cached_hf_models
            mods = get_cached_hf_models()
            return gr.update(choices=ds), gr.update(choices=mods)
            
        refresh_ds_btn.click(fn=refresh_ui_elements, outputs=[target_dataset, model_to_tune])
