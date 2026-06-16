import gradio as gr
from tabs.model_management import get_active_model_state
from agents.chatbot_engine import ChatbotEngine

engine = ChatbotEngine()

def chat_interface_fn(message, history):
    active_model = get_active_model_state()
    
    # We yield from the generator
    for output in engine.generate_response(message, history, active_model):
        yield output

def create_chatbot_tab():
    gr.Markdown("### MI300X Local AI Assistant")
    gr.Markdown("Talk directly to the AI model loaded into your MI300X VRAM. Ensure you have selected an Active Model in the **Model Management** tab.")
    
    with gr.Row():
        active_model_display = gr.Textbox(label="Currently Loaded Model", value=get_active_model_state, interactive=False)
        refresh_btn = gr.Button("↻ Refresh Model State", size="sm")
        
    refresh_btn.click(fn=get_active_model_state, outputs=active_model_display)
    
    chatbot = gr.ChatInterface(
        fn=chat_interface_fn,
        title="Antigravity OS Assistant",
        description="Ask me anything about your datasets or financial crime analysis."
    )
