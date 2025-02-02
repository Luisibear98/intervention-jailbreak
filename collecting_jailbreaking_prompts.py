import random
import json
import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from llm_guard import scan_output
from llm_guard.output_scanners import NoRefusal
from llm_guard.output_scanners.no_refusal import MatchType

# Initialize the output scanner to detect refusals
scanner = NoRefusal(threshold=0.5, match_type=MatchType.FULL)

# Function to load datasets
def load_data():
    """Loads jailbreak prompts and forbidden questions datasets."""
    ds = load_dataset("TrustAIRLab/in-the-wild-jailbreak-prompts", "jailbreak_2023_12_25")
    forbidden_question_set = load_dataset("TrustAIRLab/forbidden_question_set", split="train")
    return ds["train"]["prompt"], forbidden_question_set["question"]

# Function to generate message pairs
def generate_message_pairs(prompts, forbidden_qs):
    """Combines jailbreak prompts with forbidden questions into message format."""
    messages = [
        [{"role": "system", "content": pr}, {"role": "user", "content": qs}]
        for pr in prompts for qs in forbidden_qs
    ]
    random.shuffle(messages)  # Shuffle the combinations
    print(f"Total combinations after shuffling: {len(messages)}")
    return messages

# Function to load the tokenizer and model
def load_model():
    """Loads the tokenizer and language model."""
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", device_map="auto")
    return tokenizer, model

# Function to load the classifier model
def load_classifier():
    """Loads a classifier for jailbreak detection."""
    selected_model = "madhurjindal/Jailbreak-Detector-Large"
    return pipeline("text-classification", model=selected_model)

# Function to load processed prompts
def load_processed_prompts(output_file):
    """Loads previously processed prompts to avoid duplicate processing."""
    processed_prompts = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    processed_prompts.add(str(entry["input_text"]))  # Convert to string for consistency
                except json.JSONDecodeError:
                    continue  # Skip corrupted lines
    return processed_prompts

# Function to process and classify prompts
def process_prompts(messages_model, tokenizer, model, scanner, output_file, processed_prompts):
    """Processes each prompt by generating responses and classifying them."""
    for input_text in messages_model:

        # Skip already processed prompts
        if str(input_text) in processed_prompts:
            continue

        # Tokenize the input and move to GPU
        input_ids = tokenizer.apply_chat_template(input_text, return_tensors="pt").to("cuda")

        # Generate model response
        outputs = model.generate(input_ids, max_new_tokens=2048)
        response = tokenizer.decode(outputs[0], skip_special_tokens=False)

        # Extract the assistant's response from the output
        start_token = "<|start_header_id|>assistant<|end_header_id|>"
        end_token = "<|eot_id|>"
        start_idx = response.find(start_token)

        if start_idx != -1:
            start_idx += len(start_token)
            end_idx = response.rfind(end_token)
            assistant_response = response[start_idx:end_idx].strip() if end_idx != -1 else response[start_idx:].strip()
        else:
            assistant_response = ""

        # Scan the output for refusals or risks
        sanitized_output, is_valid, risk_score = scanner.scan(input_text[1]["content"], assistant_response)

        # Save results
        result_entry = {
            "input_text": input_text,
            "assistant_response": assistant_response,
            "valid": is_valid,
            "risk_score": risk_score,
        }

        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result_entry, ensure_ascii=False) + "\n")

        processed_prompts.add(str(input_text))

# Main function to execute the workflow
def main():
    """Main function to load data, process prompts, and save results."""
    output_file = "./prompts/classification_results_final_new_ne.jsonl"

    # Load data
    prompts, forbidden_qs = load_data()
    messages_model = generate_message_pairs(prompts, forbidden_qs)

    # Load models
    tokenizer, model = load_model()
    classifier = load_classifier()

    # Load processed prompts
    processed_prompts = load_processed_prompts(output_file)

    # Process prompts and generate responses
    process_prompts(messages_model, tokenizer, model, scanner, output_file, processed_prompts)

if __name__ == "__main__":
    main()