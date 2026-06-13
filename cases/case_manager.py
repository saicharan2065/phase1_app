import uuid
from datetime import datetime
from cases.case_storage import CaseStorage

class CaseManager:
    def __init__(self):
        self.storage = CaseStorage()

    def open_case(self, entity_id, reason, initial_priority="MEDIUM"):
        case_id = f"CASE-{str(uuid.uuid4())[:8].upper()}"
        case_data = {
            "Case ID": case_id,
            "Entity ID": entity_id,
            "Status": "OPEN",
            "Priority": initial_priority,
            "Reason": reason,
            "Assignee": "Unassigned",
            "Created At": datetime.now().isoformat(),
            "Notes": []
        }
        self.storage.save_case(case_data)
        return case_id

    def update_status(self, case_id, new_status):
        case = self.storage.get_case(case_id)
        if case:
            case["Status"] = new_status
            self.storage.save_case(case)
            return True
        return False

    def assign_case(self, case_id, assignee):
        case = self.storage.get_case(case_id)
        if case:
            case["Assignee"] = assignee
            self.storage.save_case(case)
            return True
        return False

    def add_note(self, case_id, note):
        case = self.storage.get_case(case_id)
        if case:
            case["Notes"].append({"Timestamp": datetime.now().isoformat(), "Note": note})
            self.storage.save_case(case)
            return True
        return False

    def get_all_cases_df(self):
        import pandas as pd
        cases = self.storage.get_all_cases()
        if not cases:
            return pd.DataFrame(columns=["Case ID", "Entity ID", "Status", "Priority", "Assignee", "Created At"])
        return pd.DataFrame(cases)
