import json
import os
import random
import time
import re
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

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
    if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
        return False, "Invalid email address format.", ""
    db = _load_db()
    users = db.get("users", {})
    if email in users:
        # Check if new struct dict or old string pwd
        record = users[email]
        pwd = record.get("password") if isinstance(record, dict) else record
        if pwd == password:
            uname = record.get("username", email.split('@')[0]) if isinstance(record, dict) else email.split('@')[0]
            return True, "Login successful", uname
    return False, "Invalid email or password", ""

def request_otp(email, password, username):
    if not email or not password or not username:
        return False, "Email, password, and username are required."
        
    if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
        return False, "Invalid email address format."
        
    db = _load_db()
    if email in db.get("users", {}):
        return False, "User already registered. Please login."
        
    otp_code = str(random.randint(100000, 999999))
    
    # Store OTP with a 5 minute expiry
    PENDING_OTPS[email] = {
        "otp": otp_code,
        "password": password,
        "username": username,
        "expiry": time.time() + 300
    }
    
    # Send ACTUAL email
    try:
        msg = MIMEMultipart()
        msg["From"] = "pocof72065@gmail.com"
        msg["To"] = email
        msg["Subject"] = "Your Antigravity OS Verification Code"
        
        body = f"Welcome to Financial Crime OS.\n\nYour OTP Verification Code is: {otp_code}\n\nThis code will expire in 5 minutes."
        msg.attach(MIMEText(body, "plain"))
        
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login("pocof72065@gmail.com", "mwep jrif xapl somt")
            server.send_message(msg)
            
        print(f"\n🔐 SECURITY NOTIFICATION: Actual OTP email sent to {email} via Gmail SMTP. 🔐\n")
        return True, f"OTP sent securely to {email}. Please check your inbox."
    except Exception as e:
        print(f"SMTP Error: {e}")
        return False, f"Failed to send email: {e}"

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
    db.setdefault("users", {})[email] = {
        "password": record["password"],
        "username": record.get("username", email.split('@')[0])
    }
    _save_db(db)
    
    del PENDING_OTPS[email]
    return True, "Registration successful. You can now login."

# --- ROLE MANAGEMENT ---

def get_user_role(username):
    if not username: return "STANDARD"
    # Hardcode 'admin' username to always be ADMIN for safety
    if username.lower() == "admin": return "ADMIN"
    
    db = _load_db()
    for email, record in db.get("users", {}).items():
        if isinstance(record, dict) and record.get("username") == username:
            return record.get("role", "STANDARD")
    return "STANDARD"

def request_admin_privilege(username):
    db = _load_db()
    for email, record in db.get("users", {}).items():
        if isinstance(record, dict) and record.get("username") == username:
            if record.get("role") == "ADMIN":
                return "You are already an Admin."
            record["pending_admin"] = True
            _save_db(db)
            return "Admin privileges requested. Waiting for approval."
    return "User not found."

def get_pending_admin_requests():
    db = _load_db()
    pending = []
    for email, record in db.get("users", {}).items():
        if isinstance(record, dict) and record.get("pending_admin") == True:
            pending.append(record.get("username", email))
    return pending if pending else ["No pending requests"]

def approve_admin_request(admin_user, target_user):
    if get_user_role(admin_user) != "ADMIN":
        return f"Error: User {admin_user} is not authorized to approve admins."
        
    db = _load_db()
    for email, record in db.get("users", {}).items():
        if isinstance(record, dict) and record.get("username") == target_user:
            record["role"] = "ADMIN"
            record["pending_admin"] = False
            _save_db(db)
            return f"Successfully granted Admin privileges to {target_user}."
    return "Target user not found."
