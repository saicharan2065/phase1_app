import gradio as gr
from agents.entity_memory import entity_memory_index
from agents.scenario_simulator import scenario_simulator
from agents.ram_graph_engine import ram_graph_engine
import threading

def create_ram_intelligence_tab():
    with gr.TabItem("2TB RAM Intelligence", id="ram_intel"):
        gr.Markdown("## 2TB RAM System Intelligence Suite")
        gr.Markdown("Executing massive multi-modal workloads entirely within System RAM using real Hugging Face datasets. Bypassing limited VRAM constraints to enable near-infinite scale.")
        
        with gr.Row():
            target_dataset = gr.Dropdown(
                choices=[
                    "AiresPucrs/laundry-financial-transactions",
                    "AiresPucrs/credit-card-fraud",
                    "AiresPucrs/synthetic-bank-transactions",
                    "AiresPucrs/aml-laundering",
                    "No Datasets Cached"
                ],
                value="AiresPucrs/laundry-financial-transactions",
                label="Target Dataset (Real Hugging Face Data)",
                allow_custom_value=True
            )
            
        with gr.Tabs():
            # Section 1: RAM-Based Vector Search
            with gr.TabItem("Entity Vector Memory"):
                gr.Markdown("### In-Memory Semantic Vector Index")
                gr.Markdown("Constructs massive TF-IDF vector embeddings of raw dataset rows and stores them persistently in System RAM for instantaneous semantic entity matching.")
                
                with gr.Row():
                    build_mem_btn = gr.Button("🧠 Build Vector Memory (Real Data)", variant="primary")
                    clear_mem_btn = gr.Button("🗑️ Flush Memory", variant="stop")
                    
                mem_status = gr.Textbox(label="Vector DB Status", interactive=False)
                
                with gr.Row():
                    search_query = gr.Textbox(label="Semantic Query (e.g., 'fraudulent account with massive amounts')", scale=4)
                    search_btn = gr.Button("🔍 Search Entities", scale=1)
                    
                search_results = gr.Dataframe(label="Top Matching Entities", interactive=False)
                
                def trigger_build_mem(ds):
                    t = threading.Thread(target=entity_memory_index.build_index, args=(ds, 50000))
                    t.start()
                    return "Initializing memory build sequence..."
                    
                def get_mem_status():
                    return entity_memory_index.status
                    
                build_mem_btn.click(fn=trigger_build_mem, inputs=[target_dataset], outputs=mem_status)
                clear_mem_btn.click(fn=entity_memory_index.clear, outputs=mem_status)
                search_btn.click(fn=entity_memory_index.search, inputs=[search_query], outputs=search_results)
                
                try:
                    timer_mem = gr.Timer(2)
                    timer_mem.tick(fn=get_mem_status, outputs=mem_status)
                except AttributeError:
                    pass

            # Section 2: Real-Time Scenario Simulator
            with gr.TabItem("Live Stream Processor"):
                gr.Markdown("### Real-Time Financial Stream Simulator")
                gr.Markdown("Sequentially streams actual records from the dataset at high velocity to simulate live banking networks.")
                
                with gr.Row():
                    scenario_type = gr.Dropdown(["Structuring", "Layering", "Mule Network", "Shell Company Web"], value="Layering", label="Detection Scenario")
                    tps_slider = gr.Slider(minimum=1, maximum=1000, value=50, label="Transactions Per Second (TPS)")
                    duration_slider = gr.Slider(minimum=10, maximum=600, value=60, label="Stream Duration (Seconds)")
                    
                with gr.Row():
                    start_stream_btn = gr.Button("⚡ Start Live Stream", variant="primary")
                    stop_stream_btn = gr.Button("🛑 Stop Stream", variant="stop")
                    
                stream_status = gr.Textbox(label="Stream Engine Status", interactive=False)
                
                with gr.Row():
                    processed_box = gr.Textbox(label="Total Processed", interactive=False)
                    alerts_box = gr.Textbox(label="Total Alerts", interactive=False)
                    risk_box = gr.Textbox(label="Average Risk Score", interactive=False)
                    ram_box = gr.Textbox(label="RAM Footprint", interactive=False)
                    
                live_table = gr.Dataframe(label="Live High-Risk Entities", interactive=False)
                
                start_stream_btn.click(
                    fn=scenario_simulator.start_stream, 
                    inputs=[target_dataset, scenario_type, duration_slider, tps_slider], 
                    outputs=stream_status
                )
                stop_stream_btn.click(fn=scenario_simulator.stop_stream, outputs=stream_status)
                
                try:
                    timer_stream = gr.Timer(1)
                    timer_stream.tick(
                        fn=scenario_simulator.get_metrics, 
                        outputs=[stream_status, processed_box, alerts_box, risk_box, ram_box, live_table]
                    )
                except AttributeError:
                    pass

            # Section 3: Massive Graph Investigation
            with gr.TabItem("NetworkX Graph Topography"):
                gr.Markdown("### Massive Graph Investigation Engine")
                gr.Markdown("Maps relationships between entities purely in memory to compute complex shortest paths and network centralities without hardware limitations.")
                
                with gr.Row():
                    build_graph_btn = gr.Button("🕸️ Construct Topology Map (Real Data)", variant="primary")
                    analyze_graph_btn = gr.Button("🔬 Run PageRank Analysis")
                    clear_graph_btn = gr.Button("🗑️ Flush Topology", variant="stop")
                    
                graph_status = gr.Textbox(label="Graph Engine Status", interactive=False)
                
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
                    return ram_graph_engine.status, f"{ram_graph_engine.node_count:,}", f"{ram_graph_engine.edge_count:,}", ram_str
                    
                build_graph_btn.click(fn=trigger_build_graph, inputs=[target_dataset], outputs=graph_status)
                analyze_graph_btn.click(fn=ram_graph_engine.analyze_graph, outputs=graph_findings)
                clear_graph_btn.click(fn=ram_graph_engine.clear, outputs=graph_status)
                
                try:
                    timer_graph = gr.Timer(2)
                    timer_graph.tick(fn=get_graph_metrics, outputs=[graph_status, node_box, edge_box, graph_ram_box])
                except AttributeError:
                    pass
