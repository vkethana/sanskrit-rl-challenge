import os
import random
import json
from openai import OpenAI
import vidyut
from vidyut.kosha import Kosha
from vidyut.prakriya import (
    Vyakarana,
    Data,
    Dhatu,
    Gana,
    Pada,
    Prayoga,
    Purusha,
    Vacana,
    Lakara
)

# --- Setup and Sanity Check ---
path = "vidyut-0.4.0"
if not os.path.isdir(path):
    print(f"Downloading Vidyut data to {path}...")
    vidyut.download_data(path)

# Load the dictionary (Kosha) for dhatus
kosha = Kosha(os.path.join(path, "kosha"))
print("Sample entries for 'gacCati':")
for entry in kosha.get("gacCati"):
    print(entry)

# Initialize the Vyakarana engine for generating forms
vyakarana = Vyakarana(log_steps=False)

# Load dhatus from the Data class
data = Data(path)
dhatus = [e.dhatu for e in data.load_dhatu_entries()]
print(f"Loaded {len(dhatus)} dhatus from Dhatupatha")

# --- Problem Generator ---
def generate_problem(dhatu, morphological_data):
    """
    Given a dhatu and morphological_data dict, create a JSON-in/JSON-out prompt
    for the model to conjugate the verb.
    """
    json_input = json.dumps({"dhAtu": dhatu, **morphological_data}, ensure_ascii=False)
    
    prompt = f"""
You are given a Sanskrit dhAtu (verb root) and a set of morphological features in Leipzig Glossing Format.
Please conjugate the verb according to these features and respond *only* with a JSON object as shown in the example.

Example Input:
```json
{{
  "dhAtu": "gam",
  "person": 3,
  "number": "SG",
  "tense": "PRES",
  "mood": "IND",
  "voice": "ACT"
}}
```

Your Input:
```json
{json_input}
```

Example Output:
```json
{{
  "conjugated_verb": "gacCati"
}}
```
"""
    
    return prompt.strip()

# --- Gold-Standard Generator (Verifier) ---
def get_gold_form(dhatu_obj, morph):
    """
    Use the Vidyut-Prakriya Vyakarana to generate the correct conjugated form.
    """
    try:
        # Map morphological features to vidyut.prakriya enums
        person_map = {1: Purusha.Uttama, 2: Purusha.Madhyama, 3: Purusha.Prathama}
        number_map = {"SG": Vacana.Eka, "DU": Vacana.Dvi, "PL": Vacana.Bahu}
        tense_map = {
            "PRES": Lakara.Lat,
            "IMP": Lakara.Lot, 
            "AOR": Lakara.Lun,
            "PERF": Lakara.Lit
        }
        voice_map = {"ACT": Prayoga.Kartari, "MID": Prayoga.Karmani}
        
        # Generate the form using the dhatu object directly (it already has the correct gana)
        prakriyas = vyakarana.derive(Pada.Tinanta(
            dhatu=dhatu_obj,
            prayoga=voice_map[morph['voice']],
            lakara=tense_map[morph['tense']],
            purusha=person_map[morph['person']],
            vacana=number_map[morph['number']],
        ))
        
        # Return the first result if available
        return prakriyas[0].text if prakriyas else None
        
    except Exception as e:
        print(f"Error generating gold form: {e}")
        return None

# --- Random Sampler for Morphological Features ---
def sample_morph():
    return {
        "person": random.choice([1, 2, 3]),
        "number": random.choice(["SG", "DU", "PL"]),
        "tense": random.choice(["PRES", "IMP"]),  # Simplified to supported tenses
        "mood": random.choice(["IND"]),  # Simplified for now
        "voice": random.choice(["ACT"]),  # Simplified for now
    }

# --- Main Loop ---
if __name__ == "__main__":
    # 1. Randomly pick a dhAtu from the loaded dhatus
    dhatu_obj = random.choice(dhatus)
    dhatu_text = dhatu_obj.aupadeshika  # Get the text form for the prompt
    morph = sample_morph()
    
    # 2. Generate the problem prompt
    prompt = generate_problem(dhatu_text, morph)
    print("\n--- Prompt for Model ---")
    print(prompt)
    
    # 3. Ask the model (ChatGPT API)
    api_key = os.getenv("OPENAI_API_KEY")
    
    if api_key:
        try:
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}]
            )
            model_output = response.choices[0].message.content.strip()
            print("\n--- Model Output ---")
            print(model_output)
        except Exception as e:
            print(f"\n--- Error calling OpenAI API ---")
            print(f"Error: {e}")
            model_output = '{"conjugated_verb": "ERROR"}'
    else:
        print("\n--- No OpenAI API Key Found ---")
        print("Set OPENAI_API_KEY environment variable to test with the model")
        model_output = '{"conjugated_verb": "NO_API_KEY"}'
    
    # 4. Compute gold answer via Vidyut
    gold = get_gold_form(dhatu_obj, morph)
    print("\n--- Gold Answer from Vidyut ---")
    print(json.dumps({"conjugated_verb": gold}, ensure_ascii=False))
    
    # 5. Compare and report
    try:
        model_ans = json.loads(model_output)
        correct = (model_ans.get("conjugated_verb") == gold)
    except json.JSONDecodeError:
        correct = False
    
    print(f"\nDhatu: {dhatu_text} (gana: {dhatu_obj.gana})")
    print(f"Features: {morph}")
    print("Result:", "Correct!" if correct else "Incorrect.")
