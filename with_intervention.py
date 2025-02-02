import random
import json
import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from llm_guard import scan_output
from llm_guard.output_scanners import NoRefusal
from llm_guard.output_scanners.no_refusal import MatchType
import numpy as np
from baukit import Trace, TraceDict
from pathlib import Path


prompts_jailbreaking = np.load("./activations/prompts_jailbreaking.npy", allow_pickle=True)
no_prompts_jailbreaking = np.load("./activations/no_prompts_jailbreaking.npy", allow_pickle=True)
prompts_jailbreaking = np.array(prompts_jailbreaking).reshape(len(prompts_jailbreaking), 28, 3072)
no_prompts_jailbreaking = np.array(no_prompts_jailbreaking).reshape(len(no_prompts_jailbreaking), 28, 3072)

layer_idx = 17

layer_data_jailbreak = prompts_jailbreaking[:, layer_idx, :] 
layer_data_non_jailbreak = no_prompts_jailbreaking[:, layer_idx, :]  # (samples, features)


# Compute the mean activation across samples for both conditions
mean_activation_jailbreak = np.mean(layer_data_jailbreak, axis=0)  # Mean across samples (axis=0)
mean_activation_non_jailbreak = np.mean(layer_data_non_jailbreak, axis=0)

activation_difference = mean_activation_jailbreak - mean_activation_non_jailbreak

activation_difference_normalized = activation_difference # activation_difference / np.linalg.norm(activation_difference)
activation_difference_normalized = torch.tensor(activation_difference_normalized, dtype=torch.float32)

print("Activation Difference Shape:", activation_difference.shape)
print("Normalized Activation Difference:", activation_difference_normalized)


scanner = NoRefusal(threshold=0.5, match_type=MatchType.FULL)


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct",
    device_map="auto",
)

output_file = "prompts/classification_results_final_new_intervention_more_acceptance.jsonl"


coeff = 0.8


def act_add(steering_vec, k):
    def hook(output):
        steering_vec_array = np.array(steering_vec.cpu())
        
        top_k_indices = np.argsort(np.abs(steering_vec_array))[-k:]

        mask = np.zeros_like(steering_vec_array)
        mask[top_k_indices] = 1
        
        steering_vec_masked = steering_vec_array * mask
        
        steering_vec_masked = torch.tensor(steering_vec_masked).to('cuda')
        
        return (output[0] + steering_vec_masked,) + output[1:]
    
    return hook


file_path = Path("./prompts/classification_results_final_new_ne.jsonl")
with file_path.open("r", encoding="utf-8") as file:
    data_list = [json.loads(line) for line in file]


module = model.model.layers[17]
top_neurons_to_affect = len(activation_difference_normalized)
coeff = 1.3
for prompt in data_list:  
    with Trace(module,edit_output=act_add(coeff*activation_difference_normalized,top_neurons_to_affect)) as _:
        input_text = prompt['input_text']
        input_ids = tokenizer.apply_chat_template(input_text, return_tensors="pt").to("cuda")
        outputs = model.generate(input_ids=input_ids.to('cuda'), max_new_tokens=2048)
        response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        start_token = "<|start_header_id|>assistant<|end_header_id|>"
        end_token = "<|eot_id|>"
        start_idx = response.find(start_token)

        if start_idx != -1:
            start_idx += len(start_token)
            end_idx = response.rfind(end_token)
            assistant_response = response[start_idx:end_idx].strip() if end_idx != -1 else response[start_idx:].strip()
        else:
            assistant_response = ""

        print(assistant_response)
        sanitized_output, is_valid, risk_score = scanner.scan(input_text[1]["content"], assistant_response)


        result_entry = {
            "input_text": input_text,
            "assistant_response": assistant_response,
            "valid":is_valid,
            "risck_score":risk_score,
        }
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result_entry, ensure_ascii=False) + "\n")

