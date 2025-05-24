''' IMPORTANT
Before running this script, please run `huggingface-cli login` in your terminal
in order to log into hugging face.
'''

from vidyut.prakriya import *
from vidyut.lipi import  *
import os
import json
import re
import pandas as pd
from datetime import datetime
import pandas as pd
from datasets import Dataset, DatasetDict

# You have download this data beforehand
morphological_data_path = "vidyut-0.4.0/prakriya/"
# I include a copy of it in the repo for simplicity (this is not good practice)

def generate_dataset_of_problems():
    data = Data(morphological_data_path)
    dhatus = [e.dhatu for e in data.load_dhatu_entries()]

    v = Vyakarana(log_steps=False)

    dataset = []
    prayoga = Prayoga.Kartari
    lakara=Lakara.Lat
    for dhatu in dhatus[0:20]:
            print(str(dhatu))
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

                        # Full prompt for training
                        "text": pre_prompt + prompt + post_prompt,

                        # Output fields
                        "answer_slp1": ground_truth.text,
                        "answer_iast": translit(ground_truth.text),
                        "completion": translit(ground_truth.text),
                    }
                    dataset.append(dataset_entry)
    return dataset

if __name__ == "__main__":

    # Check if morphological_data_path exists
    if not os.path.exists(morphological_data_path):
        print(f"Path {morphological_data_path} does not exist. Please download the vidyut data first.")
        exit(1)

    dataset = generate_dataset_of_problems()
    dataset_name = "sanskrit-morphology-rl"
    '''

    hf_dataset = Dataset.from_list(dataset)
    hf_dataset.save_to_disk(dataset_name)

    # This wont work unless ur signed into hugging face:
    try:
        hf_dataset.push_to_hub(dataset_name)
    except Exception as e:
        print(f"Error pushing to Hugging Face: {e}")
        print("Please make sure you are logged in to Hugging Face CLI.")
        print("You can log in using the command `huggingface-cli login`.")
    '''
