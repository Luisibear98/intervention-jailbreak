import json
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from llm_guard.output_scanners import NoRefusal
from llm_guard.output_scanners.no_refusal import MatchType
from baukit import Trace


# === CONFIGURATION ===
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
LAYER_IDX = 17  # Layer to modify
COEFF = 1.3  # Scaling factor for intervention
DEVICE = "cuda"

x
def load_activations():
    """Load and reshape activation vectors."""
    prompts_jailbreaking = np.load("./activations/prompts_jailbreaking.npy", allow_pickle=True)
    no_prompts_jailbreaking = np.load("./activations/no_prompts_jailbreaking.npy", allow_pickle=True)

    prompts_jailbreaking = prompts_jailbreaking.reshape(len(prompts_jailbreaking), 28, 3072)
    no_prompts_jailbreaking = no_prompts_jailbreaking.reshape(len(no_prompts_jailbreaking), 28, 3072)

    return prompts_jailbreaking, no_prompts_jailbreaking


def compute_activation_difference(prompts_jailbreaking, no_prompts_jailbreaking, layer_idx):
    """Compute the difference in activations between jailbreaking and non-jailbreaking prompts."""
    layer_data_jailbreak = prompts_jailbreaking[:, layer_idx, :]
    layer_data_non_jailbreak = no_prompts_jailbreaking[:, layer_idx, :]

    mean_activation_jailbreak = np.mean(layer_data_jailbreak, axis=0)
    mean_activation_non_jailbreak = np.mean(layer_data_non_jailbreak, axis=0)

    activation_difference = mean_activation_jailbreak - mean_activation_non_jailbreak
    return torch.tensor(activation_difference, dtype=torch.float32, device=DEVICE)


def modify_activations(steering_vec, k):
    """Applies a mask to the k largest activations in the vector."""
    def hook(output):
        steering_vec_array = steering_vec.cpu().numpy()
        top_k_indices = np.argsort(np.abs(steering_vec_array))[-k:]

        mask = np.zeros_like(steering_vec_array)
        mask[top_k_indices] = 1

        masked_steering_vec = torch.tensor(steering_vec_array * mask, device=DEVICE)

        return (output[0] + masked_steering_vec,) + output[1:]

    return hook


def process_prompt(prompt, model, tokenizer, activation_difference, num_neurons, scanner, output_file):
    """Processes a single prompt by applying activation intervention and classifying its output."""
    input_text = prompt['input_text']
    input_ids = tokenizer.apply_chat_template(input_text, return_tensors="pt").to(DEVICE)

    target_module = model.model.layers[LAYER_IDX]
    with Trace(target_module, edit_output=modify_activations(COEFF * activation_difference, num_neurons)):
        outputs = model.generate(input_ids=input_ids, max_new_tokens=2048)
        response = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # Extract assistant response
    start_token = "<|start_header_id|>assistant<|end_header_id|>"
    end_token = "<|eot_id|>"
    start_idx = response.find(start_token)

    if start_idx != -1:
        start_idx += len(start_token)
        end_idx = response.rfind(end_token)
        assistant_response = response[start_idx:end_idx].strip() if end_idx != -1 else response[start_idx:].strip()
    else:
        assistant_response = ""

    # Validate response
    sanitized_output, is_valid, risk_score = scanner.scan(input_text[1]["content"], assistant_response)

    result_entry = {
        "input_text": input_text,
        "assistant_response": assistant_response,
        "valid": is_valid,
        "risk_score": risk_score,
    }

    # Save results
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(result_entry, ensure_ascii=False) + "\n")


def main():
    """Main execution function."""
    # Load activation data
    prompts_jailbreaking, no_prompts_jailbreaking = load_activations()

    # Compute activation differences
    activation_difference = compute_activation_difference(prompts_jailbreaking, no_prompts_jailbreaking, LAYER_IDX)
    num_neurons = len(activation_difference)

    print(f"Activation Difference Shape: {activation_difference.shape}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")

    # Load prompts
    input_file = Path("./prompts/classification_results_final_new_ne.jsonl")
    output_file = Path("./prompts/classification_results_final_new_intervention_more_acceptance.jsonl")

    with input_file.open("r", encoding="utf-8") as file:
        data_list = [json.loads(line) for line in file]

    # Initialize scanner
    scanner = NoRefusal(threshold=0.5, match_type=MatchType.FULL)

    # Process prompts
    for prompt in data_list:
        process_prompt(prompt, model, tokenizer, activation_difference, num_neurons, scanner, output_file)

    # Free GPU memory
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()