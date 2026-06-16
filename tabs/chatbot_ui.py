import gradio as gr
from tabs.model_management import get_active_model_state
from agents.chatbot_engine import ChatbotEngine

engine = ChatbotEngine()

def chat_interface_fn(message, history, username):
    active_model = get_active_model_state()
    
    # We yield from the generator
    for output in engine.generate_response(message, history, active_model, username):
        yield output

def create_chatbot_tab(session_user):
    with gr.Accordion("💬 AI Assistant", open=False, elem_classes="floating-chat-container"):
        with gr.Row():
            active_model_display = gr.Textbox(label="Loaded Model", value=get_active_model_state, interactive=False, scale=4)
            refresh_btn = gr.Button("↻", size="sm", scale=1)
            
        refresh_btn.click(fn=get_active_model_state, outputs=active_model_display)
        
        chatbot = gr.ChatInterface(
            fn=chat_interface_fn,
            additional_inputs=[session_user],
            description="Antigravity OS Contextual Chatbot."
        )
