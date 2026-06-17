import time
import concurrent.futures
from threading import Lock
from threading import Lock

class GNNEngine:
    def __init__(self):
        self._lock = Lock()
        self.is_running = False
        self.total_nodes = 500000000 # 500 Million
        self.processed_nodes = 0
        self.status_message = "IDLE"
        self.findings = []
        self.status_message = "IDLE"
        self.findings = []
        
    def _compute_graph_tensors(self, records):
        if not self.is_running:
            return
            
        try:
            import torch
            import torch.nn.functional as F
            from torch_geometric.nn import GCNConv
            
            class SimpleGCN(torch.nn.Module):
                def __init__(self, in_channels, hidden_channels, out_channels):
                    super().__init__()
                    self.conv1 = GCNConv(in_channels, hidden_channels)
                    self.conv2 = GCNConv(hidden_channels, out_channels)

                def forward(self, x, edge_index):
                    x = self.conv1(x, edge_index)
                    x = F.relu(x)
                    x = F.dropout(x, p=0.5, training=self.training)
                    x = self.conv2(x, edge_index)
                    return x
                    
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = SimpleGCN(in_channels=16, hidden_channels=32, out_channels=2).to(device)
            
            num_nodes = len(records)
            if num_nodes == 0: return
            
            # Simple feature hash based on row content
            x_data = []
            for r in records:
                val = hash(str(r)) % 10000 / 10000.0
                x_data.append([val] * 16)
            x = torch.tensor(x_data, dtype=torch.float).to(device)
            
            # Create a basic sequential edge index to simulate a transaction chain
            sources = list(range(num_nodes - 1))
            targets = list(range(1, num_nodes))
            edge_index = torch.tensor([sources, targets], dtype=torch.long).to(device)
            
            # Forward pass (Real hardware burn)
            with torch.no_grad():
                out = model(x, edge_index)
                
            # If standard deviation of embeddings is anomalous, flag it
            if out.std().item() > 0.01:
                with self._lock:
                    self.findings.append({
                        "Sub-Graph": str(records[0])[:50] + "...",
                        "Topology": "Anomalous Chain (Detected via GCN Embedding)",
                        "Node Count": num_nodes
                    })
                    
            with self._lock:
                self.processed_nodes += num_nodes
                
        except ImportError as e:
            raise RuntimeError(f"CRITICAL ERROR: torch_geometric libraries missing. {str(e)}")
        except Exception as e:
            raise RuntimeError(f"MI300X CUDA EXECUTION ERROR: {str(e)}")
            
    def run_deep_graph_analytics(self, dataset_id=None, skip_gpu=False, sync_barrier=None):
        self.is_running = True
        self.processed_nodes = 0
        self.findings = []
        
        try:
            from data.dataset_manager import DatasetManager
            import pandas as pd
            dm = DatasetManager()
            
            if not dataset_id or dataset_id == "No valid dataset selected." or dataset_id == "No Datasets Cached":
                raise RuntimeError("No valid target_dataset provided.")
                
            self.status_message = f"LOADING REAL DATA: Pulling records from {dataset_id}..."
            ds = dm._load_dataset_records_sync(dataset_id, "50000") # Pull up to 50k nodes
            if isinstance(ds, pd.DataFrame) and "Error" in ds.columns:
                raise RuntimeError(f"Dataset load failed: {ds.iloc[0]['Error']}")
                
            if isinstance(ds, pd.DataFrame):
                records = ds.to_dict('records')
            else:
                records = [ds[i] for i in range(len(ds))]
                
            self.total_nodes = len(records)
            
            # Phase 1: Construct Adjacency Matrix in RAM
            self.status_message = f"RAM ALLOCATION: Constructing {self.total_nodes} Node Graph in System RAM..."
            
            if sync_barrier:
                self.status_message = "WAITING FOR OTHER ENGINES TO MOUNT..."
                sync_barrier.wait()
                
            # Phase 2: Compute in VRAM
            self.status_message = "VRAM COMPUTE: Running Graph Convolutional Network on MI300X..."
            
            batch_size = 5000
            chunks = [records[i:i + batch_size] for i in range(0, len(records), batch_size)]
            
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                list(executor.map(self._compute_graph_tensors, chunks))
                
            self.status_message = "COMPLETE: Global Network Analyzed."
        except Exception as e:
            if sync_barrier:
                try: sync_barrier.abort()
                except Exception: pass
            self.status_message = f"CRASH: {str(e)}"
        finally:
            self.is_running = False
            
        return self.findings
        
    def stop(self):
        self.is_running = False
        self.status_message = "ABORTED"
