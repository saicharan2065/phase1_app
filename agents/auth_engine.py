import json
import os
import random
import time

DB_PATH = "users.json"
PENDING_OTPS = {}  # email -> {otp, expiry, password}

def _load_db():
    if not os.path.exists(DB_PATH):
        return {"users": {"admin": "admin123", "agent1": "agent123"}}
    with open(DB_PATH, "r") as f:
        try:
            return json.load(f)
        except:
            return {"users": {}}

def _save_db(db):
    with open(DB_PATH, "w") as f:
        json.dump(db, f, indent=4)

def login_user(email, password):
    db = _load_db()
    users = db.get("users", {})
    if email in users and users[email] == password:
        return True, "Login successful"
    return False, "Invalid email or password"

def request_otp(email, password):
    if not email or not password:
        return False, "Email and password are required."
        
    db = _load_db()
    if email in db.get("users", {}):
        return False, "User already registered. Please login."
        
    otp_code = str(random.randint(100000, 999999))
    
    # Store OTP with a 5 minute expiry
    PENDING_OTPS[email] = {
        "otp": otp_code,
        "password": password,
        "expiry": time.time() + 300
    }
    
    # Simulate sending email by printing to terminal
    print("\n" + "="*50)
    print("🔐 SECURITY NOTIFICATION - MOCK EMAIL SERVER 🔐")
    print(f"To: {email}")
    print(f"Subject: Your Antigravity OS Verification Code")
    print(f"OTP Code: {otp_code}")
    print("="*50 + "\n")
    
    return True, f"OTP sent to {email}. Please check your terminal console."

def verify_otp(email, otp_code):
    if email not in PENDING_OTPS:
        return False, "No pending registration found for this email."
        
    record = PENDING_OTPS[email]
    if time.time() > record["expiry"]:
        del PENDING_OTPS[email]
        return False, "OTP has expired. Please request a new one."
        
    if record["otp"] != otp_code:
        return False, "Invalid OTP code."
        
    # Success, register user
    db = _load_db()
    db.setdefault("users", {})[email] = record["password"]
    _save_db(db)
    
    del PENDING_OTPS[email]
    return True, "Registration successful. You can now login."
