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
        record = users[email]
        pwd = record.get("password") if isinstance(record, dict) else record
        if pwd == password:
            if isinstance(record, dict) and record.get("status") == "PENDING":
                return False, "Account pending Admin approval.", ""
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
        "username": record.get("username", email.split('@')[0]),
        "status": "PENDING"
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
        if isinstance(record, str): continue
        record_uname = record.get("username", email.split('@')[0] if '@' in email else email)
        if record_uname == username:
            return record.get("role", "STANDARD")
    return "STANDARD"

def request_admin_privilege(username, base_url="http://127.0.0.1:7860"):
    if not username or username == "GUEST":
        return "You must be logged in to request privileges."
        
    db = _load_db()
    for email, record in db.get("users", {}).items():
        # Auto-migrate legacy string records
        if isinstance(record, str):
            record = {"password": record, "username": email.split('@')[0] if '@' in email else email, "status": "APPROVED"}
            db["users"][email] = record
            
        record_uname = record.get("username", email.split('@')[0] if '@' in email else email)
        
        if record_uname == username:
            if record.get("role") == "ADMIN" or username.lower() == "admin":
                return "You are already an Admin."
                
            # Generate secure token
            import secrets
            token = secrets.token_hex(16)
            record["admin_token"] = token
            _save_db(db)
            
            # Send Email Link to requesting user
            try:
                msg = MIMEMultipart()
                msg["From"] = "pocof72065@gmail.com"
                msg["To"] = email
                msg["Subject"] = "Admin Privilege Approval Link"
                
                # Dynamic URL
                link = f"{base_url}/?approve_admin={username}&token={token}"
                
                body = f"Hello {username},\n\nYou requested Admin privileges on the Financial Crime OS.\n\nClick the following link to authorize the upgrade:\n{link}\n\nDo not share this link."
                msg.attach(MIMEText(body, "plain"))
                
                with smtplib.SMTP("smtp.gmail.com", 587) as server:
                    server.starttls()
                    server.login("pocof72065@gmail.com", "mwep jrif xapl somt")
                    server.send_message(msg)
                    
                print(f"\n🔐 SECURITY: Admin approval link sent to {email} via SMTP.\n")
                return f"Secure approval link sent to {email}. Click the link to upgrade."
            except Exception as e:
                return f"SMTP Error sending approval link: {str(e)}"
                
    return "User not found. (Are you logged in?)"

def verify_admin_token(username, token):
    if not username or not token: return False, "Missing parameters."
    db = _load_db()
    for email, record in db.get("users", {}).items():
        if isinstance(record, str): continue
        record_uname = record.get("username", email.split('@')[0] if '@' in email else email)
        if record_uname == username:
            if record.get("admin_token") == token:
                record["role"] = "ADMIN"
                record["admin_token"] = None # Consume token
                _save_db(db)
                return True, f"Success! {username} has been upgraded to Admin."
            return False, "Invalid or expired token."
    return False, "User not found."

def get_pending_users():
    db = _load_db()
    pending = []
    for email, record in db.get("users", {}).items():
        if isinstance(record, dict) and record.get("status") == "PENDING":
            record_uname = record.get("username", email.split('@')[0] if '@' in email else email)
            pending.append(record_uname)
    return pending if pending else ["No pending registrations"]

def approve_user_account(admin_user, target_user):
    if get_user_role(admin_user) != "ADMIN":
        return f"Error: User {admin_user} is not authorized to approve accounts."
        
    db = _load_db()
    for email, record in db.get("users", {}).items():
        if isinstance(record, str): continue
        record_uname = record.get("username", email.split('@')[0] if '@' in email else email)
        if record_uname == target_user:
            record["status"] = "APPROVED"
            _save_db(db)
            return f"Successfully activated account for {target_user}."
    return "Target user not found."
