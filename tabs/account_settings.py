import gradio as gr
from agents import auth_engine

def get_role_display(username):
    role = auth_engine.get_user_role(username)
    return f"**Current Role:** {role}"

def handle_request(username, request: gr.Request):
    # Dynamically extract the base URL from the incoming web request
    base_url = str(request.headers.get("origin", "http://127.0.0.1:7860")) if request else "http://127.0.0.1:7860"
    return auth_engine.request_admin_privilege(username, base_url)

def get_requests():
    choices = auth_engine.get_pending_users()
    return gr.update(choices=choices)

def handle_approve(admin_user, target_user):
    if not target_user or "No pending" in target_user:
        return "Please select a valid user to approve."
    return auth_engine.approve_user_account(admin_user, target_user)

def create_account_settings_tab(session_user):
    gr.Markdown("### ⚙️ Account & Security Settings")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("#### Role Management")
            role_display = gr.Markdown(value="**Current Role:** UNKNOWN")
            request_btn = gr.Button("Request Admin Privileges", variant="secondary")
            request_status = gr.Textbox(label="Request Status", interactive=False)
            
        with gr.Column(scale=2):
            gr.Markdown("#### Admin Control Panel: User Activations")
            gr.Markdown("Only current Administrators can activate newly registered user accounts.")
            with gr.Row():
                pending_dropdown = gr.Dropdown(choices=["No pending registrations"], label="Pending Account Activations", scale=4)
                refresh_btn = gr.Button("↻ Refresh", size="sm", scale=1)
                
            approve_btn = gr.Button("Activate Account", variant="primary")
            approve_status = gr.Textbox(label="Action Status", interactive=False)
            
    # Load initial role when tab is rendered
    session_user.change(fn=get_role_display, inputs=session_user, outputs=role_display)
    
    request_btn.click(fn=handle_request, inputs=session_user, outputs=request_status)
    refresh_btn.click(fn=get_requests, outputs=pending_dropdown)
    approve_btn.click(fn=handle_approve, inputs=[session_user, pending_dropdown], outputs=approve_status).then(
        fn=get_requests, outputs=pending_dropdown
    )
