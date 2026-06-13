import json
import os
import uuid
from datetime import datetime

class AlertEngine:
    def __init__(self, storage_dir="storage/alerts"):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)

    def generate_alert(self, entity_id, source, level, description):
        alert_id = f"ALT-{str(uuid.uuid4())[:8].upper()}"
        alert_data = {
            "Alert ID": alert_id,
            "Entity ID": entity_id,
            "Source": source, # e.g., 'Fraud Detection', 'AML Detection'
            "Level": level,   # INFO, LOW, MEDIUM, HIGH, CRITICAL
            "Description": description,
            "Timestamp": datetime.now().isoformat()
        }
        
        filepath = os.path.join(self.storage_dir, f"{alert_id}.json")
        with open(filepath, "w") as f:
            json.dump(alert_data, f, indent=4)
            
        return alert_data

    def get_all_alerts_df(self):
        import pandas as pd
        import glob
        alerts = []
        for filepath in glob.glob(os.path.join(self.storage_dir, "*.json")):
            with open(filepath, "r") as f:
                alerts.append(json.load(f))
                
        if not alerts:
            return pd.DataFrame(columns=["Alert ID", "Entity ID", "Source", "Level", "Description", "Timestamp"])
        return pd.DataFrame(alerts).sort_values("Timestamp", ascending=False)
