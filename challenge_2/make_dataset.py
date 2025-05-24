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

# You have download Vidyut data beforehand
# I include a copy of it in the repo for simplicity 
# (but this is not good practice)
morphological_data_path = "vidyut-0.4.0/prakriya/"

v = Vyakarana(log_steps=False)

def get_human_readable_dhatu(dhatu):
    human_readable_dhatu = v.derive(dhatu)
    assert human_readable_dhatu
    human_readable_dhatu = human_readable_dhatu[0].text
    return transliterate(human_readable_dhatu, Scheme.Slp1, Scheme.Iast)

def generate_dataset_of_problems():
    data = Data(morphological_data_path)
    dhatu_list = [e.dhatu for e in data.load_dhatu_entries()]

    desired_dhatus = [
        "bhāṣ",
        "gam",
        "bhū",
        "dṛś",
        "śru", # Warning: The library says that this is a 1st class dhaatu, not a 5th class (?). Nevertheless, it conjugates the verb correctly.
        "car",
        "han"
    ]

    dhatus = []
    for i in dhatu_list:
        hrd = get_human_readable_dhatu(i)
        if hrd in desired_dhatus:
            desired_dhatus.remove(hrd) # don't want to duplicate
            dhatus.append(i)

    print("Obtained dhatu list successfully")

    dataset = []
    prayoga = Prayoga.Kartari
    lakara=Lakara.Lat
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
                    ground_truth = prakriyas[0]
                    translit = lambda x: transliterate(str(x), Scheme.Slp1, Scheme.Iast)

                    pre_prompt = '''
                    You are an expert in Sanskrit grammar. You will conjugate Sanskrit verb roots according to Paninian rules. I will give you a Sanskrit dhātu (verb root) along with morphological markers also given in terms of their Sanskrit names. You must conjugate the verb correctly.
                    Input:
                    '''

                    lakara_clean = str(lakara).replace("~","")
                    prompt = f'''
                    {{
                        "dhātu": "{get_human_readable_dhatu(dhatu)}",
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
                    Note: Use IAST transliteration (ā, ī, ū, ṛ, ṝ, ḷ, ṃ, ḥ, ñ, ṅ, ṭ, ḍ, ṇ, ś, ṣ). Be careful to not confuse "h" and "ḥ"! They aren't interchangeable.
                    Please don't include back ticks (```) in your response or any other form of Markdown formatting. Just give me raw JSON output which I will then parse using Python. Thanks.
                    '''

                    print(prompt)
                    print(translit(ground_truth.text))
                    # Create dataset entry
                    dataset_entry = {
                        # Input fields
                        "dhatu_iast": get_human_readable_dhatu(dhatu),
                        "gana_iast": translit(dhatu.gana),
                        "prayoga_iast": translit(prayoga),
                        "lakara_iast": translit(str(lakara).replace("~","")),
                        "purusha_iast": translit(purusha),
                        "vacana_iast": translit(vacana),

                        # Full prompt for training
                        "text": pre_prompt + prompt + post_prompt,

                        # Output fields
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

    hf_dataset = Dataset.from_list(dataset)
    hf_dataset.save_to_disk(dataset_name)

    # This wont work unless ur signed into hugging face:
    try:
        hf_dataset.push_to_hub(dataset_name)
    except Exception as e:
        print(f"Error pushing to Hugging Face: {e}")
        print("Please make sure you are logged in to Hugging Face CLI.")
        print("You can log in using the command `huggingface-cli login`.")
