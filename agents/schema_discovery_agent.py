import json

class SchemaDiscoveryAgent:
    def __init__(self):
        self.standard_entities = {
            "PERSON_NAME": ["customer_name", "full_name", "name", "first_name", "last_name", "emp_fname", "lname", "fname", "surname"],
            "PHONE": ["mobile", "phone_number", "phone", "cell", "contact"],
            "ADDRESS": ["residence", "address", "location", "street", "zip_code", "city", "state"],
            "INCOME": ["salary", "annual_income", "income", "wage"],
            "TRANSACTION_AMOUNT": ["txn_amt", "amount", "transaction_amount", "balance", "account_balance", "curr_bal"],
            "MERCHANT_ID": ["merchant_id", "merchant", "vendor"],
            "DATE_OF_BIRTH": ["dob", "date_of_birth", "birth_date", "bday"],
            "SOCIAL_SECURITY": ["ssn", "social_security", "national_id", "tax_id"],
            "CREDIT_CARD": ["cc_num", "credit_card", "card_number", "pan"],
            "IP_ADDRESS": ["ip_addr", "ip_address", "login_ip"],
            "EMAIL_ADDRESS": ["email", "contact_email", "email_address"]
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
