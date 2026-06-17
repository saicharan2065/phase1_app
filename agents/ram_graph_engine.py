import pandas as pd
import networkx as nx
import threading
import psutil
import time

class MassiveRAMGraphEngine:
    def __init__(self):
        self.G = nx.DiGraph()
        self.is_built = False
        self.build_lock = threading.Lock()
        self.status = "IDLE"
        self.node_count = 0
        self.edge_count = 0
        self.ram_footprint = 0.0
        self.findings = []
        
    def build_graph(self, dataset_id, max_records=200000):
        with self.build_lock:
            self.status = f"PULLING REAL DATA: Fetching {max_records} records from {dataset_id}..."
            try:
                from data.dataset_manager import DatasetManager
                dm = DatasetManager()
                
                if not dataset_id or dataset_id == "No valid dataset selected." or dataset_id == "No Datasets Cached":
                    raise RuntimeError("No valid target_dataset provided.")
                    
                ds = dm._load_dataset_records_sync(dataset_id, str(max_records))
                if isinstance(ds, pd.DataFrame) and "Error" in ds.columns:
                    raise RuntimeError(f"Dataset load failed: {ds.iloc[0]['Error']}")
                
                self.status = "CONSTRUCTING: Mapping real entities into NetworkX RAM graph..."
                
                if isinstance(ds, pd.DataFrame):
                    df = ds.copy()
                else:
                    df = pd.DataFrame([ds[i] for i in range(len(ds))])
                
                # Attempt to find sender/receiver columns dynamically for a real edge map
                sender_cols = ['nameOrig', 'Sender', 'Originator', 'Account', 'Source']
                receiver_cols = ['nameDest', 'Receiver', 'Beneficiary', 'Destination', 'Target']
                
                s_col = next((c for c in sender_cols if c in df.columns), None)
                r_col = next((c for c in receiver_cols if c in df.columns), None)
                
                self.G.clear()
                
                if s_col and r_col:
                    # Pure real data mapping
                    edges = list(zip(df[s_col], df[r_col]))
                    self.G.add_edges_from(edges)
                else:
                    # Fallback mapping if standard columns don't exist: connect index to first string column
                    str_cols = df.select_dtypes(include=['object', 'string']).columns
                    if len(str_cols) > 0:
                        edges = list(zip(df.index.astype(str), df[str_cols[0]]))
                        self.G.add_edges_from(edges)
                    else:
                        raise RuntimeError("Dataset does not have mappable entity relationships.")
                        
                self.node_count = self.G.number_of_nodes()
                self.edge_count = self.G.number_of_edges()
                self.ram_footprint = psutil.Process().memory_info().rss / (1024**3)
                
                self.is_built = True
                self.status = f"READY: Mapped {self.node_count:,} real nodes and {self.edge_count:,} edges into {self.ram_footprint:.2f} GB RAM."
            except Exception as e:
                self.status = f"CRASH: {str(e)}"
                self.is_built = False
                
    def analyze_graph(self):
        if not self.is_built:
            return pd.DataFrame({"Error": ["Graph not built. Please construct graph first."]})
            
        with self.build_lock:
            self.status = "ANALYZING: Executing high-RAM PageRank across all nodes..."
            try:
                start = time.time()
                # Run PageRank on the real graph to find massive hubs (mule network centers)
                pr = nx.pagerank(self.G, alpha=0.85, max_iter=20)
                
                sorted_nodes = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:15]
                
                self.findings = []
                for node, score in sorted_nodes:
                    degree = self.G.degree[node] if node in self.G else 0
                    self.findings.append({
                        "Central Entity ID": str(node)[:30],
                        "PageRank Score": f"{score:.6f}",
                        "Direct Connections": degree,
                        "Risk Topology": "High Centrality Hub"
                    })
                    
                elapsed = time.time() - start
                self.ram_footprint = psutil.Process().memory_info().rss / (1024**3)
                self.status = f"COMPLETED: Analysis finished in {elapsed:.1f}s. Total RAM footprint: {self.ram_footprint:.2f} GB."
                
                return pd.DataFrame(self.findings)
            except Exception as e:
                self.status = f"ANALYSIS CRASH: {str(e)}"
                return pd.DataFrame({"Error": [str(e)]})
                
    def clear(self):
        with self.build_lock:
            self.G.clear()
            self.is_built = False
            self.status = "CLEARED: Memory released."
            self.node_count = 0
            self.edge_count = 0
            self.ram_footprint = 0.0

ram_graph_engine = MassiveRAMGraphEngine()
