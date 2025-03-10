import os
import gradio as gr
import requests
import json
import time
from datetime import datetime


def query_local_llm(prompt, history=None):
    messages = []
    if history:
        for human, ai in history:
            messages.append({"role": "user", "content": human})
            messages.append({"role": "assistant", "content": ai})
    
    messages.append({"role": "user", "content": prompt})
    
    try:
        from llama_cpp import Llama
        
        llm = Llama.from_pretrained(
            repo_id="Lakshith-403/optima_qwen-3b-instruct-ai-research-finetune",
            filename="q4_model_final.gguf",
        )
        
        formatted_prompt = ""
        for msg in messages:
            role_prefix = "User: " if msg["role"] == "user" else "Assistant: "
            formatted_prompt += role_prefix + msg["content"] + "\n"
        
        response = llm(
            formatted_prompt,
            max_tokens=1000,
            temperature=0.3,
        )
        
        return response["choices"][0]["text"]
    except Exception as e:
        return f"Error querying LLM: {str(e)}"


def save_chat_history(chat_history, filename=None):
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chat_history_{timestamp}.txt"
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write("# AI Research Assistant Chat History\n\n")
        for i, (human, ai) in enumerate(chat_history, 1):
            f.write(f"## Exchange {i}\n\n")
            f.write(f"**User:** {human}\n\n")
            f.write(f"**Assistant:** {ai}\n\n")
            f.write("---\n\n")
    
    return filename


def initialize_app():
    with gr.Blocks(title="AI Research Assistant", theme=gr.themes.Soft()) as demo:
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("# AI Research Assistant")
                gr.Markdown("Ask questions about artificial intelligence research.")
            
            with gr.Column(scale=1):
                gr.Markdown("### Model Settings")
                temperature = gr.Slider(minimum=0.1, maximum=1.0, value=0.3, step=0.1, label="Temperature")
        
        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(height=500, elem_id="chatbot")
                with gr.Row():
                    msg = gr.Textbox(
                        label="Ask a question",
                        placeholder="What are the recent advances in transformer models?",
                        scale=8
                    )
                    submit_btn = gr.Button("Send", variant="primary", scale=1)
                
                with gr.Row():
                    clear_btn = gr.Button("Clear Chat")
                    save_btn = gr.Button("Save Chat History")
            
            with gr.Column(scale=1):
                gr.Markdown("### System Status")
                status = gr.Textbox(label="Status", value="Ready", interactive=False)
                response_time = gr.Textbox(label="Response Time", value="0.0s", interactive=False)
                
        
        saved_file = gr.Textbox(label="Saved File", visible=False)
        
        def respond(message, chat_history, temp):
            if not message.strip():
                return "", chat_history, "Please enter a question", "0.0s"
            
            status_msg = "Processing query..."
            yield "", chat_history, status_msg, "0.0s"
            
            start_time = time.time()
            try:
                bot_message = query_local_llm(message, chat_history)
                chat_history.append((message, bot_message))
                elapsed = time.time() - start_time
                status_msg = "Ready"
                time_msg = f"{elapsed:.2f}s"
            except Exception as e:
                elapsed = time.time() - start_time
                status_msg = f"Error: {str(e)}"
                time_msg = f"{elapsed:.2f}s"
            
            yield "", chat_history, status_msg, time_msg
        
        def save_history(chat_history):
            if not chat_history:
                return "No chat history to save"
            filename = save_chat_history(chat_history)
            return filename
        
        msg.submit(respond, [msg, chatbot, temperature], [msg, chatbot, status, response_time])
        submit_btn.click(respond, [msg, chatbot, temperature], [msg, chatbot, status, response_time])
        clear_btn.click(lambda: ([], "Ready", "0.0s"), None, [chatbot, status, response_time])
        save_btn.click(save_history, [chatbot], [saved_file])
        
        saved_file.change(lambda x: gr.Info(f"Chat history saved to {x}"), [saved_file], None)
    
    return demo

if __name__ == "__main__":
    print("Initializing AI Research Assistant...")
    demo = initialize_app()
    demo.launch(share=False) 