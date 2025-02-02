import json
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from baukit import Trace, TraceDict
from tqdm import tqdm  # Progress bar for better tracking


def load_data(file_path):
    """Load JSONL file containing classification results and separate jailbreak vs non-jailbreak prompts."""
    with file_path.open("r", encoding="utf-8") as file:
        data_list = [json.loads(line) for line in file]

    # Categorize data
    data_jailbreak = [data for data in data_list if data["valid"]]
    data_non_jailbreak = [data for data in data_list if not data["valid"]]

    print(f"Non-jailbreak samples: {len(data_non_jailbreak)}")
    print(f"Jailbreak samples: {len(data_jailbreak)}")

    return data_jailbreak, data_non_jailbreak


def setup_model(model_name, device):
    """Load the transformer model and tokenizer, moving the model to the specified device."""
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Model loaded: {model_name}")
    return model, tokenizer


def extract_activations(model, tokenizer, chat_prompt, device):
    """Extract activation vectors for a given chat prompt."""
    activation_vectors = []

    with torch.no_grad():  # Avoid storing computation graph
        with TraceDict(model, layers=hook_layers, retain_input=True, retain_output=True) as rep:
            for i, layer in enumerate(model.model.layers):
                # Format chat prompt
                prompt_text = tokenizer.apply_chat_template(chat_prompt, tokenize=False, add_generation_prompt=True)

                # Tokenize and move input to device
                inputs = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt").to(device)

                with Trace(layer) as cache:
                    _ = model(input_ids=inputs)  # Run model forward pass
                    activation_vector = cache.output[0].detach().cpu().numpy()[:, -1:, :]

                activation_vectors.append(activation_vector)

                # Free up memory
                del inputs, activation_vector
                torch.cuda.empty_cache()

    return np.array(activation_vectors)


def process_prompts(data_list, description, output_file):
    """Extract activations for a list of prompts and save the results."""
    activations = []

    for prompt in tqdm(data_list, desc=description, unit="prompt"):
        chat_prompt = prompt['input_text']
        if len(chat_prompt[0]['content']) < 10000:  # Prevent excessively long inputs
            activations.append(extract_activations(model, tokenizer, chat_prompt, device))

    np.save(output_file, np.array(activations))
    print(f"Saved activations to {output_file}")


if __name__ == "__main__":
    # Define paths and model
    file_path = Path("./prompts/classification_results_final_new_intervention_more_rejection.jsonl")
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    device = "cuda:0"

    print(f"Using device: {device}")

    # Load data
    data_jailbreak, data_non_jailbreak = load_data(file_path)

    # Load model and tokenizer
    model, tokenizer = setup_model(model_name, device)

    # Define hook layers
    hook_layers = [f"model.layers.{l}.mlp" for l in range(len(model.model.layers))]

    # Process and save activations
    process_prompts(data_jailbreak, "Processing Jailbreaking Prompts", "./activations/prompts_jailbreaking_after_intervention.npy")
    process_prompts(data_non_jailbreak, "Processing Non-Jailbreaking Prompts", "./activations/no_prompts_jailbreaking_after_intervention.npy")