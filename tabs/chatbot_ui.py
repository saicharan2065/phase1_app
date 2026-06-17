import gradio as gr
from tabs.model_management import get_active_model_state
from agents.chatbot_engine import ChatbotEngine

engine = ChatbotEngine()

def chat_interface_fn(message, history, session_user, chatbot_model):
    if not chatbot_model:
        yield "Please select a model from the dropdown."
        return
        
    # We yield from the generator
    for output in engine.generate_response(message, history, chatbot_model, session_user):
        yield output

def create_chatbot_tab(session_user):
    from tabs.model_management import get_cached_hf_models
    with gr.Accordion("🤖 AI Assistant", open=False, elem_classes="floating-chat-container"):
        with gr.Row():
            cached = get_cached_hf_models()
            default_model = "Qwen/Qwen1.5-0.5B" if "Qwen/Qwen1.5-0.5B" in cached else (cached[0] if cached else None)
            chatbot_model_drop = gr.Dropdown(choices=cached, label="Chatbot LLM", value=default_model, scale=4, interactive=True)
            refresh_btn = gr.Button("↻", size="sm", scale=1)
            
        refresh_btn.click(fn=lambda: gr.update(choices=get_cached_hf_models()), outputs=chatbot_model_drop)
        
        chatbot = gr.ChatInterface(
            fn=chat_interface_fn,
            additional_inputs=[session_user, chatbot_model_drop],
            description="Antigravity OS Contextual Chatbot."
        )
