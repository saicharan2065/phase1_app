import json
import os
import glob
from datetime import datetime

class CaseStorage:
    def __init__(self, storage_dir="storage/cases"):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)

    def save_case(self, case_data):
        case_id = case_data.get("Case ID")
        if not case_id:
            return False
            
        case_data["Last Updated"] = datetime.now().isoformat()
        filepath = os.path.join(self.storage_dir, f"{case_id}.json")
        with open(filepath, "w") as f:
            json.dump(case_data, f, indent=4)
        return True

    def get_case(self, case_id):
        filepath = os.path.join(self.storage_dir, f"{case_id}.json")
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                return json.load(f)
        return None

    def get_all_cases(self):
        cases = []
        for filepath in glob.glob(os.path.join(self.storage_dir, "*.json")):
            with open(filepath, "r") as f:
                cases.append(json.load(f))
        return cases
