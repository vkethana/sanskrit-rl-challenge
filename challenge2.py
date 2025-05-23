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

# Make sure API key is set
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("Warning: OPENAI_API_KEY not set. Set it with: export OPENAI_API_KEY='your-key-here'")
    assert False

client = openai.OpenAI(api_key=api_key)

def query_chatgpt(prompt):
    """
    Query ChatGPT API for the Sanskrit verb conjugation.
    
    Args:
        prompt (str): The prompt asking for verb conjugation
        
    Returns:
        str: The conjugated verb form extracted from ChatGPT's response
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",  # or "gpt-3.5-turbo" for faster/cheaper
            messages=[
                {"role": "system", "content": "You are an expert in Sanskrit grammar, especially Paninian rules and verb conjugation."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Low temperature for consistent outputs
            max_tokens=150
        )
        
        answer_text = response.choices[0].message.content
        answer_text.replace('```json', '')
        answer_text.replace('```', '')
        
        # Try to extract the conjugated verb from JSON response
        try:
            # Look for JSON in the response
            json_match = re.search(r'\{[^}]+\}', answer_text)
            if json_match:
                answer_json = json.loads(json_match.group())
                return answer_json.get("conjugated_verb", "")
        except json.JSONDecodeError:
            pass
        
        # If JSON parsing fails, try to extract the answer directly
        # Look for text in quotes after "conjugated_verb"
        match = re.search(r'"conjugated_verb"\s*:\s*"([^"]+)"', answer_text)
        if match:
            return match.group(1)
        
        # Last resort: return the full response
        return answer_text.strip()
            
    except Exception as e:
        print(f"Error calling ChatGPT: {e}")
        return None


path = "vidyut-0.4.0/prakriya/"
data = Data(path)
dhatus = [e.dhatu for e in data.load_dhatu_entries()]

v = Vyakarana(log_steps=False)

dataset = []
prayoga = Prayoga.Kartari
lakara=Lakara.Lat
#for dhatu in dhatus:
for dhatu in dhatus[0:1]:
        #for lakara in Lakara.choices():
        for purusha in Purusha.choices():
            for vacana in Vacana.choices():
                prakriyas = v.derive(Pada.Tinanta(
                    dhatu=dhatu,
                    prayoga=prayoga,
                    lakara=lakara,
                    purusha=purusha,
                    vacana=vacana,
                ))
                ground_truth = prakriyas[0]
                translit = lambda x: transliterate(str(x), Scheme.Slp1, Scheme.Iast)

                pre_prompt = '''
                You are an expert in Sanskrit grammar. You will conjugate Sanskrit verb roots according to Paninian rules. I will give you a Sanskrit dhātu (verb root) along with morphological markers also given in terms of their Sanskrit names. You must conjugate the verb correctly.
                Input:
                '''

                lakara_clean = str(lakara).replace("~","")
                prompt = f'''
                {{
                    "dhātu": "{translit(dhatu.aupadeshika)}",
                    "gaṇa": "{translit(dhatu.gana)}",
                    "prayoga": "{translit(prayoga)}",
                    "lakara": "{translit(lakara_clean)}",
                    "purusha": "{translit(purusha)}",
                    "vacana": "{translit(vacana)}"
                }}'''
                # TODO: ADD MEANING AS A FIELD ABOVE ^^^

                post_prompt = '''
                Output the conjugated verb form in JSON format:
                {{
                    "conjugated_verb": "your_answer_here"
                }}
                Note: Use IAST transliteration (ā, ī, ū, ṛ, ṝ, ḷ, ṃ, ḥ, ñ, ṅ, ṭ, ḍ, ṇ, ś, ṣ)
                Please don't include back ticks (```) in your response or any other form of markdown formatting. Just give me your raw JSON output which I can then throw into my JSON parser. Thanks.
                '''
                print(prompt)
                # After generating the prompt in your loop:
                chatgpt_answer = query_chatgpt(pre_prompt+prompt+post_prompt)
                if chatgpt_answer:
                    ground_truth_iast = translit(ground_truth.text)
                    print(f"Ground truth: {ground_truth_iast}")
                    print(f"ChatGPT answer: {chatgpt_answer}")
                    # Simple comparison (normalize for variations)
                    is_correct = chatgpt_answer.lower().strip() == ground_truth_iast.lower().strip()
                    print(f"Correct: {'✓' if is_correct else '✗'}")

                    # Create dataset entry
                    dataset_entry = {
                        # Input fields
                        "dhatu_slp1": dhatu.aupadeshika,
                        "dhatu_iast": translit(dhatu.aupadeshika),
                        "gana": str(dhatu.gana),
                        "gana_iast": translit(dhatu.gana),
                        "prayoga": str(prayoga),
                        "prayoga_iast": translit(prayoga),
                        "lakara": str(lakara).replace("~",""),
                        "lakara_iast": translit(str(lakara).replace("~","")),
                        "purusha": str(purusha),
                        "purusha_iast": translit(purusha),
                        "vacana": str(vacana),
                        "vacana_iast": translit(vacana),
                        
                        # Output fields
                        "conjugated_verb_slp1": ground_truth.text,
                        "conjugated_verb_iast": ground_truth_iast,
                        
                        # Full prompt for training
                        "prompt": pre_prompt + prompt + post_prompt,
                        
                        # Model performance
                        "model_answer": chatgpt_answer,
                        "is_correct": is_correct,
                        
                        # Derivation steps (useful for debugging/analysis)
                        "derivation_steps": [
                            {
                                "code": step.code,
                                "result": ' + '.join(step.result)
                            } for step in prakriyas[0].history
                        ]
                    }
                    
                    dataset.append(dataset_entry)


# Save as JSON
with open(f"sanskrit_morphology_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

# Save as CSV for easy viewing
df = pd.DataFrame(dataset)
df.to_csv(f"sanskrit_morphology_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False)

print(f"\nDataset created with {len(dataset)} entries")
print(f"Accuracy: {sum(entry['is_correct'] for entry in dataset)}/{len(dataset)} = {sum(entry['is_correct'] for entry in dataset)/len(dataset)*100:.1f}%")

