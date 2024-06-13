# PDF Information and Inference 🗞️

## Overview
This project provides a Gradio interface to upload a PDF, extract its content, and interact with it using natural language queries. The model answers questions based on the content of the uploaded PDF, using a combination of HuggingFace and Llama Index.

## Features
- Upload PDF files.
- Extract and display PDF content.
- Ask questions about the PDF content.
- Get answers from the content using a Q&A assistant.

## Setup

### Prerequisites
- Python 3.8+
- Install the necessary libraries using `pip`:
    ```bash
    pip install gradio llama-index huggingface_hub python-dotenv langdetect
    ```

### Environment Setup
1. **HuggingFace Token**: Replace `args` with your HuggingFace token:
    ```python
    args = "your_huggingface_token"
    ```
2. **Directory Setup**: Ensure the directories for data storage exist:
    ```python
    PERSIST_DIR = "./db"
    DATA_DIR = "data"
    ```

## Running the Application

1. **Login to HuggingFace**:
    ```python
    command = f"huggingface-cli login --token {args}"
    subprocess.run(command, shell=True, check=True)
    ```

2. **Configure Llama Index**:
    - Set up the language model and embedding model to use GPU if available:
    ```python
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
    ```

3. **Launch the Gradio Interface**:
    ```python
    demo.launch()
    ```

## Usage
1. **Upload PDF**:
    - Click "Upload your PDF Files" and select your PDF.
    - Click "Upload PDF" to process and display the PDF content.

2. **Ask Questions**:
    - Enter your query in the textbox labeled "Ask me anything about the content of the PDF".
    - Click "Submit Query" to get answers based on the PDF content.

## Code Breakdown
### Main Functions

- **`displayPDF(file)`**:
    Converts the uploaded PDF into a base64-encoded string for displaying within an iframe.

- **`data_ingestion()`**:
    Loads documents from the data directory and creates a vector store index for persistent storage.

- **`handle_query(query, lang)`**:
    Loads the stored index, creates a query engine, and processes the query to generate an answer.

- **`process_file(uploaded_file)`**:
    Handles file upload, saves the file to the data directory, and initiates data ingestion.

- **`chat(file, query, history)`**:
    Manages the chat interface, processes the uploaded file, detects query language, and appends responses to the chat history.

### Gradio Interface
The Gradio interface is built using the `gr.Blocks` module, with separate columns for file upload and text input. The responses are displayed in a chatbox.

```python
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
```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributions
Feel free to fork the project, make improvements, and submit pull requests. Your contributions are welcome!

## Support
For any issues or questions, please open an issue on the GitHub repository or contact the maintainer.

---

Happy coding! 🚀
