import os
from openai import OpenAI
from openai.types.fine_tuning import ReinforcementMethod, ReinforcementHyperparameters
from pathlib import Path

# Initialize OpenAI client
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def upload_files(output_dir="sanskrit_dataset_output"):
    """Upload the JSONL files to OpenAI"""
    
    output_path = Path(output_dir)
    
    # Find the most recent files (assuming they have timestamps)
    train_files = list(output_path.glob("sanskrit_quote_id_train_*.jsonl"))
    val_files = list(output_path.glob("sanskrit_quote_id_val_*.jsonl"))
    
    if not train_files or not val_files:
        raise FileNotFoundError(f"No training/validation files found in {output_dir}")
    
    # Get the most recent files
    train_file = max(train_files, key=lambda x: x.stat().st_mtime)
    val_file = max(val_files, key=lambda x: x.stat().st_mtime)
    
    print(f"Using training file: {train_file}")
    print(f"Using validation file: {val_file}")
    
    # Upload training file
    print("Uploading training file...")
    with open(train_file, "rb") as f:
        training_file = client.files.create(
            file=f,
            purpose="fine-tune"
        )
    print(f"Training file uploaded: {training_file.id}")
    
    # Upload validation file
    print("Uploading validation file...")
    with open(val_file, "rb") as f:
        validation_file = client.files.create(
            file=f,
            purpose="fine-tune"
        )
    print(f"Validation file uploaded: {validation_file.id}")
    
    return training_file.id, validation_file.id

# Define the custom grader for Sanskrit text identification
from openai.types.graders import PythonGrader

sanskrit_librarian_grader = PythonGrader(
    name="Sanskrit Librarian Text Identification Grader",
    type="python",
    source="""
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
        # Higher weights for more specific identifications
        weights = {
            "author": 2.0,      # Important but broad
            "work": 3.0,        # More specific than author
            "book": 1.5,        # Structural but less critical
            "chapter": 1.5,     # Structural but less critical  
            "verse": 4.0,       # Most specific and valuable
            "confidence": 0.5   # Bonus for appropriate confidence
        }
        
        total_possible_score = sum(weights.values())
        earned_score = 0.0
        
        # Score author match
        if expected_author and expected_author != "unknown":
            if model_author == expected_author:
                earned_score += weights["author"]
            elif model_author and fuzz.WRatio(model_author, expected_author, processor=utils.default_process) >= 80:
                # Partial credit for close matches (transliteration variants)
                earned_score += weights["author"] * 0.7
        
        # Score work match
        if expected_work and expected_work != "unknown":
            if model_work == expected_work:
                earned_score += weights["work"]
            elif model_work and fuzz.WRatio(model_work, expected_work, processor=utils.default_process) >= 80:
                # Partial credit for close matches
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
        
        # Score verse match (special handling for numbers)
        if expected_verse and expected_verse != "0":
            expected_numbers = extract_numbers(expected_verse)
            model_numbers = extract_numbers(model_verse)
            
            if model_verse == expected_verse:
                # Exact match
                earned_score += weights["verse"]
            elif expected_numbers and model_numbers:
                # Check if any numbers match
                if any(num in expected_numbers for num in model_numbers):
                    earned_score += weights["verse"] * 0.8
                # Partial credit for being close numerically
                try:
                    expected_num = int(expected_numbers[0]) if expected_numbers else 0
                    model_num = int(model_numbers[0]) if model_numbers else 0
                    
                    if expected_num > 0:
                        diff = abs(expected_num - model_num)
                        if diff <= 1:  # Off by 1
                            earned_score += weights["verse"] * 0.6
                        elif diff <= 5:  # Off by up to 5
                            earned_score += weights["verse"] * 0.3
                except ValueError:
                    pass
        
        # Score confidence appropriateness
        # Reward reasonable confidence levels (not overconfident for hard problems)
        difficulty = item.get("difficulty", "medium")
        
        if 0.0 <= model_confidence <= 1.0:
            if difficulty == "easy" and model_confidence >= 0.7:
                earned_score += weights["confidence"]
            elif difficulty == "medium" and 0.4 <= model_confidence <= 0.8:
                earned_score += weights["confidence"]
            elif difficulty == "hard" and model_confidence <= 0.6:
                earned_score += weights["confidence"]
            else:
                # Partial credit for reasonable confidence
                earned_score += weights["confidence"] * 0.5
        
        # Bonus scoring for complete correct identification
        all_fields_correct = (
            (not expected_author or expected_author == "unknown" or model_author == expected_author) and
            (not expected_work or expected_work == "unknown" or model_work == expected_work) and
            (not expected_book or expected_book == "unknown" or model_book == expected_book) and
            (not expected_chapter or expected_chapter == "unknown" or model_chapter == expected_chapter) and
            (not expected_verse or expected_verse == "0" or model_verse == expected_verse)
        )
        
        if all_fields_correct:
            earned_score += 2.0  # Bonus for perfect identification
        
        # Normalize score to 0-1 range
        final_score = min(earned_score / (total_possible_score + 2.0), 1.0)  # +2.0 for bonus
        
        return max(0.0, final_score)
        
    except Exception as e:
        return 0.0
"""
)

def create_rl_job(training_file_id, validation_file_id):
    """Create the reinforcement learning fine-tuning job"""
    
    print("Creating RL fine-tuning job...")
    
    job = client.fine_tuning.jobs.create(
        training_file=training_file_id,
        validation_file=validation_file_id,
        model="gpt-4o-mini-2024-07-18",  # Using GPT-4o-mini as base model
        method={
            "type": "reinforcement",
            "reinforcement": ReinforcementMethod(
                grader=sanskrit_librarian_grader,
                hyperparameters=ReinforcementHyperparameters(
                    reasoning_effort="medium",  # Can be "low", "medium", or "high"
                    n_epochs=3,
                    # batch_size=8,  # Uncomment to set custom batch size
                    # learning_rate_multiplier=1.0,  # Uncomment to set custom learning rate
                )
            )
        },
        seed=42,
    )
    
    print(f"RL Job created successfully!")
    print(f"Job ID: {job.id}")
    print(f"Status: {job.status}")
    print(f"Model: {job.model}")
    
    return job

def monitor_job(job_id):
    """Monitor the progress of the fine-tuning job"""
    
    print(f"\nMonitoring job {job_id}...")
    print("You can also monitor this at: https://platform.openai.com/finetune")
    
    # Get current job status
    job = client.fine_tuning.jobs.retrieve(job_id)
    print(f"Current status: {job.status}")
    
    if job.status == "completed":
        print(f"✅ Job completed! Fine-tuned model: {job.fine_tuned_model}")
    elif job.status == "failed":
        print(f"❌ Job failed. Error: {job.error}")
    else:
        print(f"🔄 Job is {job.status}. Check back later.")
    
    return job

def test_grader_locally():
    """Test the grader logic locally before submitting"""
    
    # Sample test case
    sample = {
        "output_text": '''{"author": "abhinavagupta", "work": "tantraloka", "book": "1", "chapter": "unknown", "verse": "42", "confidence": 0.85}'''
    }
    
    item = {
        "expected_answer": {
            "author": "abhinavagupta",
            "work": "tantraloka", 
            "book": "1",
            "chapter": "unknown",
            "verse": "42",
            "confidence": 1.0
        },
        "difficulty": "medium"
    }
    
    # This would need the actual grader code extracted for local testing
    print("Sample grading test:")
    print(f"Sample output: {sample['output_text']}")
    print(f"Expected: {item['expected_answer']}")
    print("Note: Run this with the actual grader logic extracted for local testing")

if __name__ == "__main__":
    try:
        # Step 0: Test grader logic locally (optional)
        print("🧪 Testing grader logic...")
        test_grader_locally()
        print()
        
        '''
        # Step 1: Upload files
        training_file_id, validation_file_id = upload_files()
        
        # Step 2: Create RL job
        job = create_rl_job(training_file_id, validation_file_id)
        
        # Step 3: Monitor job (initial check)
        monitor_job(job.id)
        
        print(f"\n📋 Job Summary:")
        print(f"   Job ID: {job.id}")
        print(f"   Training File: {training_file_id}")
        print(f"   Validation File: {validation_file_id}")
        print(f"   Base Model: gpt-4o-mini-2024-07-18")
        print(f"   Grader: Custom Sanskrit librarian grader")
        
        print(f"\n🎯 Grading Strategy:")
        print(f"   - Author identification: 2.0 points")
        print(f"   - Work identification: 3.0 points") 
        print(f"   - Book identification: 1.5 points")
        print(f"   - Chapter identification: 1.5 points")
        print(f"   - Verse identification: 4.0 points")
        print(f"   - Confidence calibration: 0.5 points")
        print(f"   - Perfect identification bonus: 2.0 points")
        print(f"   - Difficulty-aware confidence scoring")
        print(f"   - Fuzzy matching for transliteration variants")
        
        print(f"\n🔗 Monitor your job at:")
        print(f"   https://platform.openai.com/finetune/{job.id}")
        
        print(f"\n📝 To check status later, run:")
        print(f"   python -c \"from openai import OpenAI; client = OpenAI(); print(client.fine_tuning.jobs.retrieve('{job.id}'))\"")
        '''
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have:")
        print("1. Set your OPENAI_API_KEY environment variable")
        print("2. Generated the JSONL files by running the dataset generator script")
        print("3. Have the 'sanskrit_dataset_output' directory with your training files")
        print("4. Have sufficient credits in your OpenAI account")
        print("5. Installed required packages: pip install rapidfuzz")
