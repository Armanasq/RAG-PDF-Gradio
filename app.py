import gradio as gr
from llama_index.core import StorageContext, load_index_from_storage, VectorStoreIndex, SimpleDirectoryReader, ChatPromptTemplate
from llama_index.llms.huggingface import HuggingFaceInferenceAPI
from dotenv import load_dotenv
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
import os
import base64
import subprocess
from langdetect import detect

args = "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"
# Load environment variables
command = f"huggingface-cli login --token {args}"

subprocess.run(command, shell=True, check=True)
print("Logged in successfully!")

# Configure the Llama index settings to use GPU (if available)
Settings.llm = HuggingFaceInferenceAPI(
    model_name="mistralai/Mistral-7B-Instruct-v0.3",
    tokenizer_name="mistralai/Mistral-7B-Instruct-v0.3",
    context_window=5000,
    max_new_tokens=1024,
    generate_kwargs={"temperature": 0.1},
    device=0  # Use GPU 0, set to -1 for CPU
)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    device=0  # Use GPU 0, set to -1 for CPU
)

# Define the directory for persistent storage and data
PERSIST_DIR = "./db"
DATA_DIR = "data"

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PERSIST_DIR, exist_ok=True)

def displayPDF(file):
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    return pdf_display

def data_ingestion():
    documents = SimpleDirectoryReader(DATA_DIR).load_data()
    storage_context = StorageContext.from_defaults()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)

def handle_query(query, lang):
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
    chat_text_qa_msgs = [
    (
        "user",
        """You are a Q&A assistant named FarzanBot. Your main goal is to provide answers as accurately as possible, based on the instructions and context you have been given. If a question does not match the provided context or is outside the scope of the document, kindly advise the user to ask questions within the context of the document.
        Context:
        {context_str}
        Question:
        {query_str}
        """
    )
    ]
    text_qa_template = ChatPromptTemplate.from_messages(chat_text_qa_msgs)
    
    query_engine = index.as_query_engine(text_qa_template=text_qa_template)
    answer = query_engine.query(query)
    
    if hasattr(answer, 'response'):
        return answer.response, lang
    elif isinstance(answer, dict) and 'response' in answer:
        return answer['response'], lang
    else:
        return "Sorry, I couldn't find an answer.", lang

def process_file(uploaded_file):
    if uploaded_file:
        filepath = "data/saved_pdf.pdf"
        with open(filepath, "wb") as f:
            f.write(uploaded_file)
        data_ingestion()
        return displayPDF(filepath)
    return "No file uploaded."

def chat(file, query, history):
    if file:
        pdf_display = process_file(file)
    else:
        pdf_display = "No PDF uploaded."
    
    if query:
        lang = detect(query)
        response, lang = handle_query(query, lang)
        history.append(("user", query))
        history.append(("assistant", response))
    else:
        response = "Please enter your query."

    return pdf_display, history, lang

# Gradio UI
with gr.Blocks(css=".gradio-container {background-color: #f0f0f0;} .input-container {margin-bottom: 20px;}") as demo:
    gr.Markdown("# (PDF) Information and Inference🗞️")
    gr.Markdown("Retrieval-Augmented Generation")
    gr.Markdown("Start chat ...🚀")
    
    with gr.Row(equal_height=True):
        with gr.Column():
            file_input = gr.File(label="Upload your PDF Files", type="binary")
            file_submit_button = gr.Button("Upload PDF", variant="primary")
        
        with gr.Column():
            text_input = gr.Textbox(label="Ask me anything about the content of the PDF:")
            submit_button = gr.Button("Submit Query", variant="secondary")

    pdf_output = gr.HTML(label="PDF Display")
    chatbox = gr.Chatbot(label="Chat History")

    file_submit_button.click(process_file, inputs=[file_input], outputs=[pdf_output])
    submit_button.click(chat, inputs=[file_input, text_input, chatbox], outputs=[pdf_output, chatbox])

demo.launch()
