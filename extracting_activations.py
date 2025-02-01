import json
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from baukit import Trace, TraceDict
from tqdm import tqdm  # Import tqdm for progress bar


def apply_activation_modification(steering_vec, k):
    """Modify the model activations by applying a mask to the k largest activations."""
    def hook(output):
        steering_vec_array = steering_vec.detach().cpu().numpy()[0][0]

        # Get indices of the k largest absolute activations
        top_k_indices = np.argsort(np.abs(steering_vec_array))[-k:]

        # Create mask and apply it
        mask = np.zeros_like(steering_vec_array)
        mask[top_k_indices] = 1
        steering_vec_masked = steering_vec_array * mask

        # Convert back to tensor and apply to output
        return (output[0] + torch.tensor(steering_vec_masked, device=output[0].device),) + output[1:]
    
    return hook

# Load JSONL data
file_path = Path("./prompts/classification_results_final_new_intervention_more_rejection.jsonl")
with file_path.open("r", encoding="utf-8") as file:
    data_list = [json.loads(line) for line in file]

# Categorize data
data_jailbreak = [data for data in data_list if data["valid"]]
data_non_jailbreak = [data for data in data_list if not data["valid"]]

print(f"Non-jailbreak samples: {len(data_non_jailbreak)}")
print(f"Jailbreak samples: {len(data_jailbreak)}")

# Device setup
device = "cuda:0"
print(f"Using device: {device}")


# Load model and tokenizer
model_name = "meta-llama/Llama-3.2-3B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(f"Model loaded: {model_name}")

# Define hook layers
hook_layers = [f"model.layers.{l}.mlp" for l in range(len(model.model.layers))]


def extract_activations():
    activation_vectors = []
    with TraceDict(model, layers=hook_layers, retain_input=True, retain_output=True) as rep:
        for i, layer in enumerate(model.model.layers):
            inputs = tokenizer("happy", return_tensors="pt").to(device)

            with Trace(layer) as cache:
                _ = model(**inputs)
                activation_vector = cache.output[0].detach().cpu().numpy()[:, -1:, :]
            
            activation_vectors.append(activation_vector)

    return np.array(activation_vectors)




def extract_activations_encode_all(chat_prompt):
    """Extract activations for a given chat prompt and clean GPU memory after execution."""
    activation_vectors = []

    with torch.no_grad():  # Prevent unnecessary computation graph storage
        with TraceDict(model, layers=hook_layers, retain_input=True, retain_output=True) as rep:
            for i, layer in enumerate(model.model.layers):
                # Format chat prompt correctly
                prompt_text = tokenizer.apply_chat_template(chat_prompt, tokenize=False, add_generation_prompt=True)

                # Encode and move to device
                inputs = tokenizer.encode(prompt_text, add_special_tokens=False,return_tensors="pt").to(device)

                with Trace(layer) as cache:
                    _ = model(input_ids=inputs)  # Pass inputs correctly
                    activation_vector = cache.output[0].detach().cpu().numpy()[:, -1:, :]
                
                activation_vectors.append(activation_vector)

                del inputs, activation_vector
                torch.cuda.empty_cache()  

    return np.array(activation_vectors)


prompts_jailbreaking = []
processed = 0


for jailbreaking_prompt in tqdm(data_jailbreak, desc="Processing Jailbreaking Prompts", unit="prompt"):
    
    chat_prompt = jailbreaking_prompt['input_text']
    if len(chat_prompt[0]['content']) < 10000:
        activation_vectors = extract_activations_encode_all(chat_prompt)
        prompts_jailbreaking.append(activation_vectors)

no_prompts_jailbreaking = []
for non_jailbreaking_prompt in tqdm(data_non_jailbreak, desc="Processing Non-Jailbreaking Prompts", unit="prompt"):
    chat_prompt = non_jailbreaking_prompt['input_text']
    print(len(chat_prompt))
    if len(chat_prompt[0]['content']) < 10000:
        activation_vectors = extract_activations_encode_all(chat_prompt)
        no_prompts_jailbreaking.append(activation_vectors)

# Save the activation vectors
np.save("./activations/prompts_jailbreaking_after_intervention.npy", np.array(prompts_jailbreaking))
np.save("./activations/no_prompts_jailbreaking_after_intervention.npy", np.array(no_prompts_jailbreaking))


