from vidyut.prakriya import *
from vidyut.lipi import  *
import openai
import os
import json
import re
import pandas as pd
from datetime import datetime
import pandas as pd
from datasets import Dataset, DatasetDict

# Load data 
filename = "sanskrit_morphology_dataset_20250523_133240.json"
with open(filename, 'r', encoding='utf-8') as f:
   dataset = json.load(f)

# Create HuggingFace dataset
from datasets import Dataset

hf_dataset = Dataset.from_list(dataset)
hf_dataset.save_to_disk("sanskrit_morphology_hf_dataset")

# After the dataset creation loop, format for Prime-RL
# Create HuggingFace dataset in Prime-RL format
def format_for_prime_rl(dataset):
    """Format dataset for Prime-RL training"""
    formatted_data = []
    
    for entry in dataset:
        # Prime-RL expects a specific format for RL training
        formatted_entry = {
            "prompt": entry["prompt"],
            "question": json.dumps({
                "dhatu_iast": entry["dhatu_iast"],
                "gana_iast": entry["gana_iast"],
                "prayoga_iast": entry["prayoga_iast"],
                "lakara_iast": entry["lakara_iast"],
                "purusha_iast": entry["purusha_iast"],
                "vacana_iast": entry["vacana_iast"]
            }),
            "answer": entry["conjugated_verb_iast"],
            "model_answer": entry.get("model_answer", ""),
            "is_correct": entry.get("is_correct", False),
            # Additional metadata for verification
            "metadata": {
                "dhatu_slp1": entry["dhatu_slp1"],
                "conjugated_verb_slp1": entry["conjugated_verb_slp1"],
                "derivation_steps": entry.get("derivation_steps", [])
            }
        }
        formatted_data.append(formatted_entry)
    
    return formatted_data

# Format and save
formatted_dataset = format_for_prime_rl(dataset)
hf_dataset = Dataset.from_list(formatted_dataset)

# Split into train/validation
train_test_split = hf_dataset.train_test_split(test_size=0.1)
dataset_dict = DatasetDict({
    'train': train_test_split['train'],
    'validation': train_test_split['test']
})

# Save in Prime-RL compatible format
dataset_dict.save_to_disk("sanskrit_morphology_prime_rl")
dataset_dict.push_to_hub("vkethana-sanskrit-morphology-prime-rl")
