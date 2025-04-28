# Tokenizer CLI

A simple command-line tool to tokenize text using Hugging Face models.

## Setup

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **(Optional) Set Hugging Face Token:** For models that require authentication (e.g., gated models like Llama), set the `HF_TOKEN` environment variable:
    ```bash
    export HF_TOKEN='your_hugging_face_token'
    ```
    Replace `'your_hugging_face_token'` with your actual token.

## Usage

Run the script from your terminal:

```bash
python main.py
```

The tool will prompt you to select a tokenizer model (either from a predefined list or by entering a custom Hugging Face repository name) and then ask for the text you want to tokenize.

Example:
```
🤗 Welcome to the Hugging Face Tokenizer CLI! 🤗
[?] Select a model or choose 'Enter custom model name': 
 > meta-llama/Llama-3.1-70B-Instruct
   google/gemma-2-9b-it
   mistralai/Mistral-7B-Instruct-v0.3
   bert-base-uncased
   gpt2
   Enter custom model name


⏳ Loading tokenizer for 'meta-llama/Llama-3.1-70B-Instruct'...
tokenizer_config.json: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 55.4k/55.4k [00:00<00:00, 9.34MB/s]
tokenizer.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 9.09M/9.09M [00:00<00:00, 16.2MB/s]
special_tokens_map.json: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 296/296 [00:00<00:00, 1.16MB/s]
✅ Tokenizer loaded successfully.

--------------------
[?] Enter the text you want to tokenize: Hello World!

💬 Input Text:
'Hello World!'

✨ Tokenization Results ✨
-------------------------
🔹 Token IDs:  [128000, 9906, 4435, 0]
🔹 Tokens:     ['<|begin_of_text|>', 'Hello', 'ĠWorld', '!']
-------------------------
[?] Tokenize another string with the same model? (Y/n):
```