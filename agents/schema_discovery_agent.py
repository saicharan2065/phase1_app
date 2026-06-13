import json

class SchemaDiscoveryAgent:
    def __init__(self):
        self.standard_entities = {
            "PERSON_NAME": ["customer_name", "full_name", "name", "first_name", "last_name"],
            "PHONE": ["mobile", "phone_number", "phone", "cell", "contact"],
            "ADDRESS": ["residence", "address", "location", "street"],
            "INCOME": ["salary", "annual_income", "income", "wage"],
            "TRANSACTION_AMOUNT": ["txn_amt", "amount", "transaction_amount"],
            "MERCHANT_ID": ["merchant_id", "merchant"]
        }

    def discover_schema(self, columns):
        schema_mapping = {entity: [] for entity in self.standard_entities}
        
        for col in columns:
            col_lower = col.lower().strip()
            matched = False
            # Exact match check
            for entity, keywords in self.standard_entities.items():
                if col_lower in keywords:
                    schema_mapping[entity].append(col)
                    matched = True
                    break
            
            # Simple substring matching
            if not matched:
                for entity, keywords in self.standard_entities.items():
                    if any(kw in col_lower for kw in keywords):
                        schema_mapping[entity].append(col)
                        break
                        
        # Filter out empty mappings
        result = {k: v for k, v in schema_mapping.items() if v}
        return result
