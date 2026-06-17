import gradio as gr
from agents.chatbot_engine import ChatbotEngine

engine = ChatbotEngine()

def chat_interface_fn(message, history, session_user):
    # We yield from the generator
    for output in engine.generate_response(message, history, "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", session_user):
        yield output

def create_chatbot_tab(session_user):
    with gr.Accordion("🤖 AI Assistant (DeepSeek-R1 32B)", open=False, elem_classes="floating-chat-container"):
        chatbot = gr.ChatInterface(
            fn=chat_interface_fn,
            additional_inputs=[session_user],
            description="Antigravity OS Chatbot — Powered by DeepSeek-R1 32B on System RAM."
        )
