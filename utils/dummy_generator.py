import os
import tempfile
import pandas as pd
import random
from datetime import datetime, timedelta

def generate_dummy_data(domain="generic", num_records=15):
    """
    Generates a dummy Pandas DataFrame based on the specified domain context.
    Saves it to a temporary CSV file and returns the file path.
    """
    records = []
    
    # Ensure minimum
    num_records = max(1, int(num_records))
    
    if domain == "fraud":
        for i in range(num_records):
            records.append({
                "transaction_id": f"TXN-{random.randint(10000, 99999)}",
                "account_id": f"ACC-{random.randint(100, 999)}",
                "amount": round(random.uniform(10.0, 15000.0), 2),
                "timestamp": (datetime.now() - timedelta(hours=random.randint(1, 72))).strftime("%Y-%m-%d %H:%M:%S"),
                "merchant": random.choice(["AMAZON", "WALMART", "CRYPTO_EXCHANGE", "LOCAL_CAFE", "OFFSHORE_CORP"]),
                "is_flagged": random.choice([True, False, False, False])
            })
            
    elif domain == "aml":
        # Generate some structuring and velocity patterns
        groups = max(1, num_records // 11)
        for i in range(groups):
            acct = f"ACC-{random.randint(100, 105)}"
            # Normal txns
            records.append({"account_id": acct, "amount": random.uniform(100, 5000), "type": "WIRE"})
            # Structuring (multiple just below 10k)
            records.append({"account_id": acct, "amount": random.uniform(9000, 9999), "type": "CASH"})
            records.append({"account_id": acct, "amount": random.uniform(9000, 9999), "type": "CASH"})
            # Velocity
            for _ in range(8):
                records.append({"account_id": acct, "amount": random.uniform(100, 500), "type": "P2P"})
                
    elif domain == "fraud_rings":
        # Generate shared PII to trigger connected graph components
        rings = max(1, num_records // 3)
        for i in range(rings):
            shared_phone = f"555-RING-{i}"
            shared_address = f"123 Fraud St Apt {i}"
            for j in range(3): # 3 customers sharing same info
                records.append({
                    "customer_id": f"CUST_{i}_{j}",
                    "name": f"Fake User {i}-{j}",
                    "phone": shared_phone,
                    "address": shared_address,
                    "device": f"DEVICE_MAC_{i}"
                })
            
    elif domain in ["entity_resolution", "entity_graph", "case_management"]:
        for i in range(num_records):
            records.append({
                "entity_id": f"E-{random.randint(100, 999)}",
                "name": random.choice(["John Doe", "J. Doe", "Jonathan Doe", "Acme Corp", "Acme Corporation"]),
                "dob": random.choice(["1980-01-01", "1980-01-02", "Unknown", "1975-05-15"]),
                "address": random.choice(["123 Main St", "123 Main Street, Apt 4", "PO BOX 99", "Unknown"]),
                "phone": random.choice(["555-0100", "555-0101", "+1-555-0100", "None"])
            })
            
    elif domain in ["schema_discovery", "data_quality"]:
        for i in range(num_records):
            records.append({
                "id": i,
                "first_name": random.choice(["Alice", "Bob", None, "Charlie"]),
                "last_name": random.choice(["Smith", "Jones", "", "Brown"]),
                "email": random.choice(["test@example.com", "invalid-email", None, "admin@corp.com"]),
                "balance": random.choice([100.50, -50.0, "ERROR", 0.0, 5000.0])
            })
            
    else:
        # Default generic data
        for i in range(num_records):
            records.append({
                "id": i,
                "value": random.random(),
                "status": random.choice(["Active", "Inactive", "Pending"])
            })
            
    df = pd.DataFrame(records)
    
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, f"dummy_data_{domain}.csv")
    df.to_csv(file_path, index=False)
    
    return file_path
