import os
import requests
import json

# Get the API key from environment
api_key = os.environ["OPENAI_API_KEY"]
headers = {"Authorization": f"Bearer {api_key}"}

grading_function = """
import json
import re
from rapidfuzz import fuzz, utils

def normalize_string(s):
    \"\"\"Normalize strings for comparison - lowercase, remove extra spaces\"\"\"
    if not s:
        return ""
    return str(s).lower().strip()

def extract_numbers(s):
    \"\"\"Extract all numbers from a string\"\"\"
    if not s:
        return []
    return re.findall(r'\\d+', str(s))

def grade(sample, item) -> float:
    try:
        # Extract the model's output text
        output_text = sample["output_text"].strip()
        
        # Parse the model's JSON response
        try:
            model_response = json.loads(output_text)
        except json.JSONDecodeError:
            # If JSON parsing fails, return 0
            return 0.0
        
        # Get expected answer from the item
        expected = item.get("expected_answer", {})
        
        # Extract fields from model response and expected answer
        model_author = normalize_string(model_response.get("author", ""))
        model_work = normalize_string(model_response.get("work", ""))
        model_book = normalize_string(model_response.get("book", ""))
        model_chapter = normalize_string(model_response.get("chapter", ""))
        model_verse = str(model_response.get("verse", "")).strip()
        model_confidence = float(model_response.get("confidence", 0.0))
        
        expected_author = normalize_string(expected.get("author", ""))
        expected_work = normalize_string(expected.get("work", ""))
        expected_book = normalize_string(expected.get("book", ""))
        expected_chapter = normalize_string(expected.get("chapter", ""))
        expected_verse = str(expected.get("verse", "")).strip()
        
        # Scoring weights for different levels of the hierarchy
        weights = {
            "author": 2.0,
            "work": 3.0,
            "book": 1.5,
            "chapter": 1.5,
            "verse": 4.0,
            "confidence": 0.5
        }
        
        total_possible_score = sum(weights.values())
        earned_score = 0.0
        
        # Score author match
        if expected_author and expected_author != "unknown":
            if model_author == expected_author:
                earned_score += weights["author"]
            elif model_author and fuzz.WRatio(model_author, expected_author, processor=utils.default_process) >= 80:
                earned_score += weights["author"] * 0.7
        
        # Score work match
        if expected_work and expected_work != "unknown":
            if model_work == expected_work:
                earned_score += weights["work"]
            elif model_work and fuzz.WRatio(model_work, expected_work, processor=utils.default_process) >= 80:
                earned_score += weights["work"] * 0.7
        
        # Score book match
        if expected_book and expected_book != "unknown":
            if model_book == expected_book:
                earned_score += weights["book"]
            elif model_book and fuzz.WRatio(model_book, expected_book, processor=utils.default_process) >= 80:
                earned_score += weights["book"] * 0.7
        
        # Score chapter match
        if expected_chapter and expected_chapter != "unknown":
            if model_chapter == expected_chapter:
                earned_score += weights["chapter"]
            elif model_chapter and fuzz.WRatio(model_chapter, expected_chapter, processor=utils.default_process) >= 80:
                earned_score += weights["chapter"] * 0.7
        
        # Score verse match
        if expected_verse and expected_verse != "0":
            expected_numbers = extract_numbers(expected_verse)
            model_numbers = extract_numbers(model_verse)
            
            if model_verse == expected_verse:
                earned_score += weights["verse"]
            elif expected_numbers and model_numbers:
                if any(num in expected_numbers for num in model_numbers):
                    earned_score += weights["verse"] * 0.8
                try:
                    expected_num = int(expected_numbers[0]) if expected_numbers else 0
                    model_num = int(model_numbers[0]) if model_numbers else 0
                    
                    if expected_num > 0:
                        diff = abs(expected_num - model_num)
                        if diff <= 1:
                            earned_score += weights["verse"] * 0.6
                        elif diff <= 5:
                            earned_score += weights["verse"] * 0.3
                except ValueError:
                    pass
        
        # Score confidence appropriateness
        difficulty = item.get("difficulty", "medium")
        
        if 0.0 <= model_confidence <= 1.0:
            if difficulty == "easy" and model_confidence >= 0.7:
                earned_score += weights["confidence"]
            elif difficulty == "medium" and 0.4 <= model_confidence <= 0.8:
                earned_score += weights["confidence"]
            elif difficulty == "hard" and model_confidence <= 0.6:
                earned_score += weights["confidence"]
            else:
                earned_score += weights["confidence"] * 0.5
        
        # Bonus for complete correct identification
        all_fields_correct = (
            (not expected_author or expected_author == "unknown" or model_author == expected_author) and
            (not expected_work or expected_work == "unknown" or model_work == expected_work) and
            (not expected_book or expected_book == "unknown" or model_book == expected_book) and
            (not expected_chapter or expected_chapter == "unknown" or model_chapter == expected_chapter) and
            (not expected_verse or expected_verse == "0" or model_verse == expected_verse)
        )
        
        if all_fields_correct:
            earned_score += 2.0
        
        # Normalize score to 0-1 range
        final_score = min(earned_score / (total_possible_score + 2.0), 1.0)
        
        return max(0.0, final_score)
        
    except Exception as e:
        return 0.0
"""

# Define the grader
grader = {
    "type": "python",
    "source": grading_function
}

# Validate the grader
print("Validating Sanskrit librarian grader...")
payload = {"grader": grader}
response = requests.post(
    "https://api.openai.com/v1/fine_tuning/alpha/graders/validate",
    json=payload,
    headers=headers
)
print("Validate request_id:", response.headers.get("x-request-id", "N/A"))
print("Validate response:", response.text)

# Test the grader with Sanskrit text identification examples
print("\n" + "="*60)
print("Testing grader with Sanskrit text identification examples")
print("="*60)

# Test case 1: Perfect identification (all fields correct)
test_payload_1 = {
    "grader": grader,
    "item": {
        "expected_answer": {
            "author": "abhinavagupta",
            "work": "tantraloka",
            "book": "1",
            "chapter": "prathamamahnika",
            "verse": "42",
            "confidence": 1.0
        },
        "difficulty": "medium"
    },
    "model_sample": """{
        "author": "abhinavagupta",
        "work": "tantraloka",
        "book": "1", 
        "chapter": "prathamamahnika",
        "verse": "42",
        "confidence": 0.75
    }"""
}

response = requests.post(
    "https://api.openai.com/v1/fine_tuning/alpha/graders/run",
    json=test_payload_1,
    headers=headers
)
print("Test 1 (Perfect identification - should score high):")
print("Request_id:", response.headers.get("x-request-id", "N/A"))
print("Response:", response.text)

# Test case 2: Partial identification (author and work correct, but wrong verse)
test_payload_2 = {
    "grader": grader,
    "item": {
        "expected_answer": {
            "author": "abhinavagupta",
            "work": "tantraloka",
            "book": "1",
            "chapter": "unknown",
            "verse": "42",
            "confidence": 1.0
        },
        "difficulty": "medium"
    },
    "model_sample": """{
        "author": "abhinavagupta",
        "work": "tantraloka",
        "book": "unknown",
        "chapter": "unknown", 
        "verse": "43",
        "confidence": 0.6
    }"""
}

response = requests.post(
    "https://api.openai.com/v1/fine_tuning/alpha/graders/run",
    json=test_payload_2,
    headers=headers
)
print("\nTest 2 (Partial match - author/work correct, verse close):")
print("Request_id:", response.headers.get("x-request-id", "N/A"))
print("Response:", response.text)

# Test case 3: Wrong identification (completely incorrect)
test_payload_3 = {
    "grader": grader,
    "item": {
        "expected_answer": {
            "author": "abhinavagupta", 
            "work": "tantraloka",
            "book": "1",
            "chapter": "unknown",
            "verse": "42",
            "confidence": 1.0
        },
        "difficulty": "easy"
    },
    "model_sample": """{
        "author": "kalidasa",
        "work": "meghaduta",
        "book": "unknown",
        "chapter": "unknown",
        "verse": "15",
        "confidence": 0.9
    }"""
}

response = requests.post(
    "https://api.openai.com/v1/fine_tuning/alpha/graders/run",
    json=test_payload_3,
    headers=headers
)
print("\nTest 3 (Wrong identification - should score low):")
print("Request_id:", response.headers.get("x-request-id", "N/A"))
print("Response:", response.text)

# Test case 4: Fuzzy matching (transliteration variants)
test_payload_4 = {
    "grader": grader,
    "item": {
        "expected_answer": {
            "author": "nagarjuna",
            "work": "ratnavali", 
            "book": "unknown",
            "chapter": "unknown",
            "verse": "25",
            "confidence": 1.0
        },
        "difficulty": "hard"
    },
    "model_sample": """{
        "author": "nāgārjuna",
        "work": "ratnāvalī",
        "book": "unknown",
        "chapter": "unknown",
        "verse": "24",
        "confidence": 0.4
    }"""
}

response = requests.post(
    "https://api.openai.com/v1/fine_tuning/alpha/graders/run",
    json=test_payload_4,
    headers=headers
)
print("\nTest 4 (Fuzzy matching - transliteration variants, close verse):")
print("Request_id:", response.headers.get("x-request-id", "N/A"))
print("Response:", response.text)

# Test case 5: Confidence scoring test (overconfident on hard problem)
test_payload_5 = {
    "grader": grader,
    "item": {
        "expected_answer": {
            "author": "unknown",
            "work": "unknown",
            "book": "unknown", 
            "chapter": "unknown",
            "verse": "0",
            "confidence": 1.0
        },
        "difficulty": "hard"
    },
    "model_sample": """{
        "author": "bhartrhari",
        "work": "vairagya",
        "book": "1",
        "chapter": "unknown",
        "verse": "50",
        "confidence": 0.95
    }"""
}

response = requests.post(
    "https://api.openai.com/v1/fine_tuning/alpha/graders/run",
    json=test_payload_5,
    headers=headers
)
print("\nTest 5 (Overconfident on hard/unknown problem):")
print("Request_id:", response.headers.get("x-request-id", "N/A"))
print("Response:", response.text)

# Test case 6: Malformed JSON response
test_payload_6 = {
    "grader": grader,
    "item": {
        "expected_answer": {
            "author": "abhinavagupta",
            "work": "tantraloka",
            "book": "1",
            "chapter": "unknown",
            "verse": "42",
            "confidence": 1.0
        },
        "difficulty": "medium"
    },
    "model_sample": """This is not valid JSON at all, just some text response."""
}

response = requests.post(
    "https://api.openai.com/v1/fine_tuning/alpha/graders/run",
    json=test_payload_6,
    headers=headers
)
print("\nTest 6 (Malformed JSON - should score 0.0):")
print("Request_id:", response.headers.get("x-request-id", "N/A"))
print("Response:", response.text)

print("\n" + "="*60)
print("Grader testing complete!")
print("="*60)
