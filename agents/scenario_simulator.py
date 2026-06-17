import time
import pandas as pd
import threading
import psutil

class RealTimeScenarioStreamer:
    def __init__(self):
        self.is_running = False
        self.stream_lock = threading.Lock()
        self.status = "IDLE"
        self.total_processed = 0
        self.total_alerts = 0
        self.current_risk_avg = 0.0
        self.live_entities = []
        self._thread = None
        self.dataset_records = []
        
    def _fetch_real_data(self, dataset_id, max_records="50000"):
        self.status = f"PULLING REAL DATA: Fetching {max_records} records from {dataset_id}..."
        try:
            from data.dataset_manager import DatasetManager
            dm = DatasetManager()
            
            if not dataset_id or dataset_id == "No valid dataset selected." or dataset_id == "No Datasets Cached":
                raise RuntimeError("No valid target_dataset provided.")
                
            ds = dm._load_dataset_records_sync(dataset_id, str(max_records))
            if isinstance(ds, pd.DataFrame) and "Error" in ds.columns:
                raise RuntimeError(f"Dataset load failed: {ds.iloc[0]['Error']}")
            
            if isinstance(ds, pd.DataFrame):
                self.dataset_records = ds.to_dict('records')
            else:
                self.dataset_records = [ds[i] for i in range(len(ds))]
                
            return True
        except Exception as e:
            self.status = f"CRASH: {str(e)}"
            return False

    def _stream_loop(self, tps, duration):
        start_time = time.time()
        record_idx = 0
        max_idx = len(self.dataset_records)
        
        while self.is_running and (time.time() - start_time) < duration:
            if record_idx >= max_idx:
                record_idx = 0 # Loop back to start to simulate continuous stream
                
            # Process 'tps' (transactions per second) records
            batch = self.dataset_records[record_idx : record_idx + tps]
            record_idx += tps
            
            with self.stream_lock:
                self.total_processed += len(batch)
                
                # Simple deterministic rule to flag real records based on keywords or amount
                for r in batch:
                    r_str = str(r).lower()
                    risk = 0
                    if 'amount' in r:
                        try:
                            amt = float(r['amount'])
                            if amt > 9000: risk += 50
                            if amt > 90000: risk += 40
                        except: pass
                        
                    if 'fraud' in r_str or 'scam' in r_str or 'high_risk' in r_str:
                        risk += 80
                        
                    if risk > 60:
                        self.total_alerts += 1
                        entity_id = r.get('nameDest', r.get('Account', r.get('id', 'Unknown')))
                        
                        # Keep top 10 live entities
                        self.live_entities.insert(0, {
                            "Real Entity ID": str(entity_id)[:20],
                            "Risk": risk,
                            "Raw Data Sneak Peek": str(r)[:100] + "..."
                        })
                        if len(self.live_entities) > 10:
                            self.live_entities.pop()
                            
                # Update avg risk roughly
                self.current_risk_avg = min(100.0, (self.total_alerts / max(1, self.total_processed)) * 100 * 5)
            
            time.sleep(1.0) # 1 second tick
            
        self.is_running = False
        self.status = "COMPLETED"

    def start_stream(self, dataset_id, scenario_type, duration_sec, tps):
        if self.is_running:
            return "Already running."
            
        self.total_processed = 0
        self.total_alerts = 0
        self.current_risk_avg = 0.0
        self.live_entities = []
        
        if not self._fetch_real_data(dataset_id):
            return self.status
            
        self.is_running = True
        self.status = f"STREAMING: {scenario_type} simulation using actual {dataset_id} records."
        
        self._thread = threading.Thread(target=self._stream_loop, args=(int(tps), int(duration_sec)))
        self._thread.start()
        return self.status
        
    def stop_stream(self):
        self.is_running = False
        self.status = "STOPPED"
        return self.status

    def get_metrics(self):
        with self.stream_lock:
            ram_mb = psutil.Process().memory_info().rss / (1024 * 1024)
            return (
                self.status,
                self.total_processed,
                self.total_alerts,
                f"{self.current_risk_avg:.1f}/100",
                f"{ram_mb:.1f} MB",
                pd.DataFrame(self.live_entities) if self.live_entities else pd.DataFrame({"Notice": ["Waiting for high-risk flags..."]})
            )

scenario_simulator = RealTimeScenarioStreamer()
