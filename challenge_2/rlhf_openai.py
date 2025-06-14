import os
import requests
import json

# Get the API key from environment
api_key = os.environ["OPENAI_API_KEY"]
headers = {"Authorization": f"Bearer {api_key}"}

grading_function = """
import json
from rapidfuzz import fuzz, utils

def grade(sample, item) -> float:
    try:
        # Extract the model's output text
        output_text = sample["output_text"].strip()
        
        # Parse the model's JSON response to get derivation steps
        try:
            model_response = json.loads(output_text)
            model_derivation = model_response.get("derivation_history", [])
        except json.JSONDecodeError:
            # If JSON parsing fails, return 0
            return 0.0
        
        # Get the expected derivation history from the item
        expected_derivation = item.get("derivation_history", [])
        
        # If no expected derivation history, return 0
        if not expected_derivation:
            return 0.0
        
        # If model provides no derivation steps, return 0
        if not model_derivation:
            return 0.0
        
        total_steps = len(expected_derivation)
        correct_streak = 0
        
        # Compare each step in sequence until we find a mismatch
        for i in range(min(len(model_derivation), len(expected_derivation))):
            expected_step = expected_derivation[i]
            model_step = model_derivation[i]
            
            # Extract code and text from expected step
            expected_code = str(expected_step.get("code", "")).strip()
            expected_text = str(expected_step.get("text", "")).strip()
            
            # Extract code and text from model step
            model_code = str(model_step.get("code", "")).strip()
            model_text = str(model_step.get("text", "")).strip()
            
            # Both code and text must match exactly (or very closely for text due to transliteration variations)
            code_match = (expected_code == model_code)
            text_similarity = fuzz.WRatio(model_text, expected_text, processor=utils.default_process) / 100.0
            text_match = (text_similarity >= 0.95)  # Allow slight transliteration differences
            
            # Step is correct only if both code and text match
            if code_match and text_match:
                correct_streak += 1
            else:
                # Once a step is wrong, break the streak
                break
        
        # Score is the fraction of correct consecutive steps from the beginning
        score = correct_streak / total_steps if total_steps > 0 else 0.0
        return score
        
    except Exception as e:
        return 0.0
"""

# Define the grader
grader = {
    "type": "python",
    "source": grading_function
}

# Validate the grader
print("Validating grader...")
payload = {"grader": grader}
response = requests.post(
    "https://api.openai.com/v1/fine_tuning/alpha/graders/validate",
    json=payload,
    headers=headers
)
print("Validate request_id:", response.headers.get("x-request-id", "N/A"))
print("Validate response:", response.text)

# Test the grader with Sanskrit morphology examples
print("\n" + "="*50)
print("Testing grader with Sanskrit derivation examples")
print("="*50)

# Test case 1: Perfect derivation match (all steps correct)
test_payload_1 = {
    "grader": grader,
    "item": {
        "derivation_history": [
            {"code": "1.3.1", "text": "bhū"},
            {"code": "3.2.123", "text": "bhū + lam̐ṭ"},
            {"code": "1.3.2", "text": "bhū + lam̐ṭ"},
            {"code": "1.3.3", "text": "bhū + lam̐ṭ"}
        ]
    },
    "model_sample": """{
        "conjugated_verb": "bhavati",
        "derivation_history": [
            {"code": "1.3.1", "text": "bhū"},
            {"code": "3.2.123", "text": "bhū + lam̐ṭ"},
            {"code": "1.3.2", "text": "bhū + lam̐ṭ"},
            {"code": "1.3.3", "text": "bhū + lam̐ṭ"}
        ]
    }"""
}

response = requests.post(
    "https://api.openai.com/v1/fine_tuning/alpha/graders/run",
    json=test_payload_1,
    headers=headers
)
print("Test 1 (Perfect derivation match - should score 1.0):")
print("Request_id:", response.headers.get("x-request-id", "N/A"))
print("Response:", response.text)

# Test case 2: Partial derivation match (first 2 steps correct, then wrong)
test_payload_2 = {
    "grader": grader,
    "item": {
        "derivation_history": [
            {"code": "1.3.1", "text": "bhū"},
            {"code": "3.2.123", "text": "bhū + lam̐ṭ"},
            {"code": "1.3.2", "text": "bhū + lam̐ṭ"},
            {"code": "1.3.3", "text": "bhū + lam̐ṭ"}
        ]
    },
    "model_sample": """{
        "conjugated_verb": "bhavati", 
        "derivation_history": [
            {"code": "1.3.1", "text": "bhū"},
            {"code": "3.2.123", "text": "bhū + lam̐ṭ"},
            {"code": "1.3.9", "text": "bhū + l"},
            {"code": "1.3.3", "text": "bhū + lam̐ṭ"}
        ]
    }"""
}

response = requests.post(
    "https://api.openai.com/v1/fine_tuning/alpha/graders/run",
    json=test_payload_2,
    headers=headers
)
print("\nTest 2 (Partial match - first 2 correct, should score 0.5):")
print("Request_id:", response.headers.get("x-request-id", "N/A"))
print("Response:", response.text)

# Test case 3: Wrong from the start
test_payload_3 = {
    "grader": grader,
    "item": {
        "derivation_history": [
            {"code": "1.3.1", "text": "bhū"},
            {"code": "3.2.123", "text": "bhū + lam̐ṭ"},
            {"code": "1.3.2", "text": "bhū + lam̐ṭ"},
            {"code": "1.3.3", "text": "bhū + lam̐ṭ"}
        ]
    },
    "model_sample": """{
        "conjugated_verb": "bhavati",
        "derivation_history": [
            {"code": "2.1.1", "text": "gam"},
            {"code": "3.2.123", "text": "bhū + lam̐ṭ"},
            {"code": "1.3.2", "text": "bhū + lam̐ṭ"},
            {"code": "1.3.3", "text": "bhū + lam̐ṭ"}
        ]
    }"""
}

response = requests.post(
    "https://api.openai.com/v1/fine_tuning/alpha/graders/run",
    json=test_payload_3,
    headers=headers
)
print("\nTest 3 (Wrong from start - should score 0.0):")
print("Request_id:", response.headers.get("x-request-id", "N/A"))
print("Response:", response.text)

# Test case 4: No derivation provided by model
test_payload_4 = {
    "grader": grader,
    "item": {
        "derivation_history": [
            {"code": "1.3.1", "text": "bhū"},
            {"code": "3.2.123", "text": "bhū + lam̐ṭ"}
        ]
    },
    "model_sample": '{"conjugated_verb": "bhavati"}'  # No derivation_history
}

response = requests.post(
    "https://api.openai.com/v1/fine_tuning/alpha/graders/run",
    json=test_payload_4,
    headers=headers
)
print("\nTest 4 (No derivation from model - should score 0.0):")
print("Request_id:", response.headers.get("x-request-id", "N/A"))
print("Response:", response.text)
