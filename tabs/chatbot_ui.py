import gradio as gr
from agents.chatbot_engine import ChatbotEngine

engine = ChatbotEngine()

def chat_interface_fn(message, history, session_user):
    # We yield from the generator
    for output in engine.generate_response(message, history, "Qwen/Qwen1.5-0.5B", session_user):
        yield output

def create_chatbot_tab(session_user):
    with gr.Accordion("🤖 AI Assistant (Qwen 0.5B)", open=False, elem_classes="floating-chat-container"):
        chatbot = gr.ChatInterface(
            fn=chat_interface_fn,
            additional_inputs=[session_user],
            description="Antigravity OS Chatbot — Powered by Qwen 0.5B on System RAM."
        )
