import inquirer
from transformers import AutoTokenizer, PreTrainedTokenizerFast, PreTrainedTokenizer
import sys
import os

# --- Configuration ---
COMMON_MODELS = [
    "meta-llama/Llama-3.1-70B-Instruct",
    "google/gemma-2-9b-it",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "bert-base-uncased",
    "gpt2",
    "Enter custom model name",
]

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

def get_input_string() -> str | None:
    """Prompts the user to enter the string to tokenize."""
    questions = [
        inquirer.Text('text', message="Enter the text you want to tokenize")
    ]
    answers = inquirer.prompt(questions)
    if not answers: # Handle ctrl-c or unexpected exit
        return None
    return answers['text']

# --- Main Execution ---

def main():
    """Main function to run the tokenizer CLI."""
    print("🤗 Welcome to the Hugging Face Tokenizer CLI! 🤗")

    # 1. Select Model
    model_name = select_model()
    if not model_name:
        print("\nExiting.")
        sys.exit(0)
    if not model_name.strip():
        print("\n❌ Error: Model name cannot be empty.")
        sys.exit(1)

    # 2. Load Tokenizer
    tokenizer = get_tokenizer(model_name)
    if tokenizer is None:
        sys.exit(1)

    # 3. Get Input and Tokenize Loop
    while True:
        print("\n" + "-"*20)
        input_string = get_input_string()
        if input_string is None: # Handle ctrl-c during text input
            print("\nExiting.")
            break

        if not input_string.strip():
            print("💬 Please enter some text to tokenize.")
            continue

        print(f"\n💬 Input Text:\n'{input_string}'")

        try:
            # Encode the string
            encoded_output = tokenizer.encode(input_string)

            # Decode back to tokens (some tokenizers might require specific methods)
            tokens = tokenizer.convert_ids_to_tokens(encoded_output)

            print("\n✨ Tokenization Results ✨")
            print("-" * 25)
            print(f"🔹 Token IDs:  {encoded_output}")
            print(f"🔹 Tokens:     {tokens}")
            print("-" * 25)

        except Exception as e:
            print(f"❌ Error during tokenization: {e}")

        # Ask to continue
        continue_questions = [
            inquirer.Confirm('continue', message="Tokenize another string with the same model?", default=True),
        ]
        continue_answers = inquirer.prompt(continue_questions)
        if not continue_answers or not continue_answers['continue']:
            print("\n👋 Goodbye!")
            break


if __name__ == "__main__":
    main()
