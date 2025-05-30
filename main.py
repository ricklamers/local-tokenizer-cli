import inquirer
from transformers import AutoTokenizer, PreTrainedTokenizerFast, PreTrainedTokenizer
import sys
import os
import re

# --- Configuration ---
COMMON_MODELS = [
    "meta-llama/Llama-3.1-70B-Instruct",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
    "meta-llama/Llama-3.2-1B",
    "google/gemma-2-9b-it",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "bert-base-uncased",
    "gpt2",
    "Enter custom model name",
]

# Actions for the main loop
ACTIONS = {
    "tokenize": "Tokenize text",
    "decode": "Decode token IDs",
    "change": "Change model",
    "exit": "Exit",
}

# --- Helper Functions ---

def get_tokenizer(model_name: str) -> PreTrainedTokenizer | PreTrainedTokenizerFast | None:
    """Loads a tokenizer from Hugging Face, handling potential errors."""
    try:
        # Suppress excessive logging from transformers unless it's an error
        previous_log_level = os.environ.get("TRANSFORMERS_VERBOSITY", None)
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"

        print(f"\n⏳ Loading tokenizer for '{model_name}'...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("✅ Tokenizer loaded successfully.")

        # Restore previous log level if it existed
        if previous_log_level is not None:
            os.environ["TRANSFORMERS_VERBOSITY"] = previous_log_level
        else:
            del os.environ["TRANSFORMERS_VERBOSITY"]

        return tokenizer
    except OSError:
        print(f"❌ Error: Could not find tokenizer for model '{model_name}'. Please check the repository name.")
        return None
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        # Restore previous log level in case of other errors
        if previous_log_level is not None:
            os.environ["TRANSFORMERS_VERBOSITY"] = previous_log_level
        else:
            if "TRANSFORMERS_VERBOSITY" in os.environ:
                 del os.environ["TRANSFORMERS_VERBOSITY"]
        return None

def select_model() -> str | None:
    """Prompts the user to select or enter a model name."""
    questions = [
        inquirer.List(
            'model',
            message="Select a model or choose 'Enter custom model name'",
            choices=COMMON_MODELS,
            carousel=True,
        ),
    ]
    answers = inquirer.prompt(questions)
    if not answers: # Handle ctrl-c or unexpected exit
        return None

    selected_option = answers['model']

    if selected_option == "Enter custom model name":
        custom_model_question = [
            inquirer.Text('custom_model', message="Enter the Hugging Face model repository name (e.g., 'bert-base-uncased')")
        ]
        custom_answers = inquirer.prompt(custom_model_question)
        if not custom_answers:
             return None
        return custom_answers['custom_model'].strip()
    else:
        return selected_option

def select_action() -> str | None:
    """Prompts the user to select an action."""
    questions = [
        inquirer.List(
            'action',
            message="What would you like to do?",
            choices=list(ACTIONS.values()),
            carousel=False, # Keep it simple
        ),
    ]
    answers = inquirer.prompt(questions)
    if not answers:
        return None # Handle ctrl-c

    # Find the key corresponding to the selected value
    for key, value in ACTIONS.items():
        if value == answers['action']:
            return key
    return None # Should not happen

def get_input_string() -> str | None:
    """Prompts the user to enter the string to tokenize using an editor for multi-line support."""
    questions = [
        inquirer.Editor('text', message="Enter the text you want to tokenize (this will open your default editor)")
    ]
    answers = inquirer.prompt(questions)
    if not answers: # Handle ctrl-c or unexpected exit
        return None
    return answers['text']

def get_token_ids_input() -> list[int] | None:
    """Prompts the user for comma-separated token IDs and parses them."""
    try:
        print("Enter comma-separated token IDs (e.g., 101, 2054, 102): ")
        id_string = input().strip()
    except EOFError: # Handle Ctrl+D
        print("\nAction cancelled.")
        return None
    except KeyboardInterrupt: # Handle Ctrl+C
        print("\nAction cancelled.")
        return None

    if not id_string:
        print("❌ Error: No input provided.")
        return None

    # Split by comma, allowing for optional whitespace around commas
    raw_ids = [item.strip() for item in re.split(r'\s*,\s*', id_string)]

    token_ids = []
    try:
        for raw_id in raw_ids:
            if raw_id: # Avoid empty strings if there are trailing/leading commas
                token_ids.append(int(raw_id))
    except ValueError:
        print("❌ Error: Invalid input. Please enter only comma-separated numbers.")
        return None

    if not token_ids:
        print("❌ Error: No valid token IDs entered.")
        return None

    return token_ids

# --- Processing Functions ---

def process_text_tokenization(tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast):
    """Handles getting text input and showing tokenization results."""
    input_string = get_input_string()
    if input_string is None: # Handle ctrl-c during text input
        print("\nAction cancelled.")
        return

    if not input_string.strip():
        print("💬 Please enter some text to tokenize.")
        return

    print(f"\n💬 Input Text:\n'{input_string}'")

    try:
        # Encode the string
        encoded_output = tokenizer.encode(input_string)
        # Decode back to tokens
        tokens = tokenizer.convert_ids_to_tokens(encoded_output)

        print("\n✨ Tokenization Results ✨")
        print("-" * 25)
        print(f"🔹 Token IDs:  {encoded_output}")
        print(f"🔹 Tokens:     {tokens}")
        print(f"🔹 Token Count: {len(encoded_output)}")
        print("-" * 25)

    except Exception as e:
        print(f"❌ Error during tokenization: {e}")


def process_id_decoding(tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast):
    """Handles getting token ID input and showing decoded tokens."""
    token_ids = get_token_ids_input()

    if token_ids is None:
        # Error message printed in get_token_ids_input or action cancelled
        return

    print(f"\n🔢 Input Token IDs:\n{token_ids}")

    try:
        # Decode the IDs
        # Note: skip_special_tokens=True might be desirable depending on use case,
        # but let's keep them for now to be explicit.
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        # Optional: Convert tokens back to a string
        decoded_string = tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)


        print("\n✨ Decoding Results ✨")
        print("-" * 25)
        print(f"🔹 Tokens:         {tokens}")
        print(f"🔹 Decoded String: '{decoded_string}'")
        print("-" * 25)

    except Exception as e:
        print(f"❌ Error during decoding: {e}")


# --- Main Execution ---

def main():
    """Main function to run the tokenizer CLI."""
    print("🤗 Welcome to the Hugging Face Tokenizer CLI! 🤗")

    while True: # Outer loop for model selection
        # 1. Select Model
        model_name = select_model()
        if not model_name:
            print("\n👋 Goodbye!")
            sys.exit(0) # User exited model selection
        if not model_name.strip():
            print("\n❌ Error: Model name cannot be empty.")
            continue # Ask for model again

        # 2. Load Tokenizer
        tokenizer = get_tokenizer(model_name)
        if tokenizer is None:
            # Error message handled in get_tokenizer
            continue # Ask for model again

        # 3. Action Loop (Tokenize/Decode/Change/Exit)
        while True:
            print("\n" + "="*30)
            action = select_action()
            print("="*30)


            if action == "tokenize": # Compare directly with the key
                 process_text_tokenization(tokenizer)
            elif action == "decode": # Compare directly with the key
                 process_id_decoding(tokenizer)
            elif action == "change": # Compare directly with the key
                 print("\n🔄 Changing model...")
                 break # Break inner loop to re-select model
            elif action == "exit": # Compare directly with the key
                 print("\n👋 Goodbye!")
                 sys.exit(0) # Exit completely
            elif action is None: # Handle ctrl-c during action selection
                 print("\n👋 Goodbye!")
                 sys.exit(0)


if __name__ == "__main__":
    main()
