import json
import os
import uuid
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

class AlertEngine:
    def __init__(self, storage_dir="storage/alerts"):
        self.storage_dir = storage_dir
        self.smtp_config_path = "storage/smtp_config.json"
        os.makedirs(self.storage_dir, exist_ok=True)
        os.makedirs("storage", exist_ok=True)

    def save_smtp_config(self, server, port, email, password, recipient):
        config = {
            "server": server,
            "port": port,
            "email": email,
            "password": password,
            "recipient": recipient
        }
        with open(self.smtp_config_path, "w") as f:
            json.dump(config, f, indent=4)
        return "SMTP Configuration Saved Successfully!"

    def load_smtp_config(self):
        if os.path.exists(self.smtp_config_path):
            with open(self.smtp_config_path, "r") as f:
                return json.load(f)
        return None

    def test_smtp_connection(self):
        config = self.load_smtp_config()
        if not config:
            return False, "No SMTP Configuration found. Please save first."
        try:
            msg = MIMEMultipart()
            msg["From"] = config["email"]
            msg["To"] = config["recipient"]
            msg["Subject"] = "Financial Crime OS - Test Email"
            msg.attach(MIMEText("This is a test email to verify your SMTP configuration works.", "plain"))
            
            with smtplib.SMTP(config["server"], int(config["port"])) as server:
                server.starttls()
                server.login(config["email"], config["password"])
                server.send_message(msg)
            return True, "Test email sent successfully!"
        except Exception as e:
            return False, f"SMTP Error: {str(e)}"

    def _dispatch_email(self, alert_data):
        config = self.load_smtp_config()
        if not config:
            return
        try:
            msg = MIMEMultipart()
            msg["From"] = config["email"]
            msg["To"] = config["recipient"]
            msg["Subject"] = f"[{alert_data['Level']}] Financial Crime Alert: {alert_data['Alert ID']}"
            
            body = f"""
            FINANCIAL CRIME ALERT TRIGGERED
            
            Alert ID: {alert_data['Alert ID']}
            Entity ID: {alert_data['Entity ID']}
            Source: {alert_data['Source']}
            Level: {alert_data['Level']}
            
            Description:
            {alert_data['Description']}
            
            Timestamp: {alert_data['Timestamp']}
            """
            msg.attach(MIMEText(body, "plain"))
            
            with smtplib.SMTP(config["server"], int(config["port"])) as server:
                server.starttls()
                server.login(config["email"], config["password"])
                server.send_message(msg)
        except Exception as e:
            print(f"Failed to send email alert: {str(e)}")

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
            
        # Dispatch email if HIGH or CRITICAL
        if level in ["HIGH", "CRITICAL"]:
            self._dispatch_email(alert_data)
            
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
