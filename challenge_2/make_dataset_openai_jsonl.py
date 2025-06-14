from vidyut.prakriya import *
from vidyut.lipi import  *
import os
import json
import re
import pandas as pd
from datetime import datetime

# You have download Vidyut data beforehand
# I include a copy of it in the repo for simplicity 
# (but this is not good practice)
morphological_data_path = "vidyut-0.4.0/prakriya/"

v = Vyakarana(log_steps=True)

def get_human_readable_dhatu(dhatu):
    human_readable_dhatu = v.derive(dhatu)
    assert human_readable_dhatu
    human_readable_dhatu = human_readable_dhatu[0].text
    return transliterate(human_readable_dhatu, Scheme.Slp1, Scheme.Iast)

def generate_jsonl_dataset():
    data = Data(morphological_data_path)
    dhatu_list = [e.dhatu for e in data.load_dhatu_entries()]

    desired_dhatus = [
        "bhāṣ",
        "gam",
        "bhū",
        "dṛś",
        "śru", # Warning: The library says that this is a 1st class dhaatu, not a 5th class (?). Nevertheless, it conjugates the verb correctly.
        "car",
        "han",
        "vad",
        "vac",
        "kṛ"
    ]

    dhatus = []
    for i in dhatu_list:
        hrd = get_human_readable_dhatu(i)
        if hrd in desired_dhatus:
            desired_dhatus.remove(hrd)  # don't want to duplicate
            dhatus.append(i)

    print("Obtained dhatu list successfully")

    # System message for the developer role
    system_message = """You are an expert in Sanskrit grammar. You will conjugate Sanskrit verb roots according to Paninian rules. I will give you a Sanskrit dhātu (verb root) along with morphological markers also given in terms of their Sanskrit names. You must conjugate the verb correctly.
Output the conjugated verb form in JSON format: { "conjugated_verb": "your_answer_here" }
Note: Use IAST transliteration (ā, ī, ū, ṛ, ṝ, ḷ, ṃ, ḥ, ñ, ṅ, ṭ, ḍ, ṇ, ś, ṣ). Be careful to not confuse "h" and "ḥ"! They aren't interchangeable.
Please don't include back ticks (```) in your response or any other form of Markdown formatting. Just give me raw JSON output which I will then parse using Python. Thanks. Now here's the input. Read it, then output your answer as JSON following the specifications above:"""

    jsonl_lines = []
    prayoga = Prayoga.Kartari
    
    for dhatu in dhatus:
        '''
        For reference:
        Lat = simple present
        Lan = imperfect
        Lot = imperative
        Lit = reduplicating past tense (this one might be hard for model)
        Lin = optative
        '''
        for lakara in [Lakara.Lat, Lakara.Lit, Lakara.VidhiLin, Lakara.Lot, Lakara.Lan]:
            for purusha in Purusha.choices():
                for vacana in Vacana.choices():
                    prakriyas = v.derive(Pada.Tinanta(
                        dhatu=dhatu,
                        prayoga=prayoga,
                        lakara=lakara,
                        purusha=purusha,
                        vacana=vacana,
                    ))

                    if prakriyas:  # Make sure we have results
                        ground_truth = prakriyas[0]
                        translit = lambda x: transliterate(str(x), Scheme.Slp1, Scheme.Iast)

                        # Create the user input content
                        #lakara_clean = str(v.derive(lakara))
                        lakara_clean = str(lakara).replace('~','')
                        #print("____",lakara,v.derive(lakara))
                        user_input = f'''{{
    "dhātu": "{get_human_readable_dhatu(dhatu)}",
    "gaṇa": "{translit(dhatu.gana)}",
    "prayoga": "{translit(prayoga)}",
    "lakara": "{translit(lakara_clean)}",
    "purusha": "{translit(purusha)}",
    "vacana": "{translit(vacana)}"
}}'''

                        # Extract derivation history
                        derivation_history = []
                        for i, step in enumerate(ground_truth.history):
                            #new_result = [v.derive(elem) for elem in step.result]
                            new_result = [elem for elem in step.result]
                            derivation_history.append({
                                "code": step.code,
                                "text": transliterate(' + '.join(new_result), Scheme.Slp1, Scheme.Iast)
                            })

                        # Create the JSONL entry
                        jsonl_entry = {
                            "messages": [
                                {
                                    "role": "developer",
                                    "content": system_message
                                },
                                {
                                    "role": "user",
                                    "content": user_input
                                }
                            ],
                            "dhatu": get_human_readable_dhatu(dhatu),
                            "gana": translit(dhatu.gana),
                            "prayoga": translit(prayoga),
                            "lakara": translit(lakara_clean),
                            "purusha": translit(purusha),
                            "vacana": translit(vacana),
                            "expected_answer": translit(ground_truth.text),
                            "derivation_history": derivation_history
                        }
                        
                        jsonl_lines.append(jsonl_entry)
                        
                        # Optional: print progress
                        print(f"Generated entry for {get_human_readable_dhatu(dhatu)} - {translit(lakara_clean)} - {translit(purusha)} - {translit(vacana)}")

    return jsonl_lines

def write_jsonl_file(data, filename):
    """Write data to JSONL file (one JSON object per line)"""
    with open(filename, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

def split_dataset(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """Split dataset into train, validation, and test sets"""
    import random
    random.seed(42)  # For reproducibility
    
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)
    
    total_size = len(shuffled_data)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    
    train_data = shuffled_data[:train_size]
    val_data = shuffled_data[train_size:train_size + val_size]
    test_data = shuffled_data[train_size + val_size:]
    
    return train_data, val_data, test_data

if __name__ == "__main__":
    # Check if morphological_data_path exists
    if not os.path.exists(morphological_data_path):
        print(f"Path {morphological_data_path} does not exist. Please download the vidyut data first.")
        exit(1)

    # Generate dataset
    print("Generating JSONL dataset...")
    dataset = generate_jsonl_dataset()
    
    print(f"Generated {len(dataset)} training examples")
    
    # Split the dataset
    train_data, val_data, test_data = split_dataset(dataset)
    
    # Write to separate files
    write_jsonl_file(train_data, "sanskrit_morphology_train.jsonl")
    write_jsonl_file(val_data, "sanskrit_morphology_val.jsonl")
    write_jsonl_file(test_data, "sanskrit_morphology_test.jsonl")
    
    print(f"Dataset split and saved:")
    print(f"  Training set: {len(train_data)} examples -> sanskrit_morphology_train.jsonl")
    print(f"  Validation set: {len(val_data)} examples -> sanskrit_morphology_val.jsonl")
    print(f"  Test set: {len(test_data)} examples -> sanskrit_morphology_test.jsonl")
    
    # Also create a single combined file if needed
    write_jsonl_file(dataset, "sanskrit_morphology_complete.jsonl")
    print(f"  Complete dataset: {len(dataset)} examples -> sanskrit_morphology_complete.jsonl")
    
    # Print a sample entry for verification
    if dataset:
        print("\nSample entry:")
        print(json.dumps(dataset[0], indent=2, ensure_ascii=False))
