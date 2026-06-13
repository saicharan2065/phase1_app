from cases.case_storage import CaseStorage
import pandas as pd
import os

class CaseReporter:
    def __init__(self):
        self.storage = CaseStorage()

    def export_case(self, case_id, export_format="json"):
        case = self.storage.get_case(case_id)
        if not case:
            return None
            
        export_dir = "storage/reports"
        os.makedirs(export_dir, exist_ok=True)
        
        if export_format == "json":
            filepath = os.path.join(export_dir, f"{case_id}_export.json")
            import json
            with open(filepath, "w") as f:
                json.dump(case, f, indent=4)
            return filepath
            
        elif export_format == "csv":
            filepath = os.path.join(export_dir, f"{case_id}_export.csv")
            df = pd.DataFrame([case])
            df.to_csv(filepath, index=False)
            return filepath
            
        return None
