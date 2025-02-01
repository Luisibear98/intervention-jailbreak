import random
import json
import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from llm_guard import scan_output
from llm_guard.output_scanners import NoRefusal
from llm_guard.output_scanners.no_refusal import MatchType

scanner = NoRefusal(threshold=0.5, match_type=MatchType.FULL)

# Cargar datasets
ds = load_dataset("TrustAIRLab/in-the-wild-jailbreak-prompts", "jailbreak_2023_12_25")
forbidden_question_set = load_dataset("TrustAIRLab/forbidden_question_set", split="train")

# Extraer prompts y preguntas prohibidas
prompts = ds["train"]["prompt"]
forbidden_qs = forbidden_question_set["question"]

# Combinar prompts y preguntas prohibidas en formato de mensajes
messages_model = [
    [
        {"role": "system", "content": pr},
        {"role": "user", "content": qs},
    ]
    for pr in prompts for qs in forbidden_qs
]

# Barajar la lista
random.shuffle(messages_model)
print(f"Total combinaciones tras mezclar: {len(messages_model)}")

# Cargar tokenizador y modelo
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct",
    device_map="auto",
)

# Configurar clasificador
selected_model = "madhurjindal/Jailbreak-Detector-Large"
classifier = pipeline("text-classification", model=selected_model)

# Archivo de salida
output_file = "./prompts/classification_results_final_new_ne.jsonl"

# Cargar prompts ya procesados
processed_prompts = set()
if os.path.exists(output_file):
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
                processed_prompts.add(str(entry["input_text"]))  # Convertir a string para evitar errores
            except json.JSONDecodeError:
                continue  # Saltar líneas corruptas

# Loop de inferencia
for input_text in messages_model:  # Limitar a los primeros 100

    # Verificar si ya se ha procesado
    if str(input_text) in processed_prompts:
        continue

    # Convertir entrada en tensor
    input_ids = tokenizer.apply_chat_template(input_text, return_tensors="pt").to("cuda")
    
    # Generar respuesta
    outputs = model.generate(input_ids, max_new_tokens=2048)
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # Extraer respuesta del asistente
    start_token = "<|start_header_id|>assistant<|end_header_id|>"
    end_token = "<|eot_id|>"
    start_idx = response.find(start_token)

    if start_idx != -1:
        start_idx += len(start_token)
        end_idx = response.rfind(end_token)
        assistant_response = response[start_idx:end_idx].strip() if end_idx != -1 else response[start_idx:].strip()
    else:
        assistant_response = ""

    
    sanitized_output, is_valid, risk_score = scanner.scan( input_text[1]["content"], assistant_response)

    result_entry = {
        "input_text": input_text,
        "assistant_response": assistant_response,
        "valid":is_valid,
        "risck_score":risk_score,
    }
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(result_entry, ensure_ascii=False) + "\n")

    processed_prompts.add(str(input_text))