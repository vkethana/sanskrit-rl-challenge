import os
from openai import OpenAI
from openai.types.fine_tuning import ReinforcementMethod, ReinforcementHyperparameters

# Initialize OpenAI client
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# First, upload your training and validation files
def upload_files():
    """Upload the JSONL files to OpenAI"""
    
    # Upload training file
    print("Uploading training file...")
    with open("sanskrit_morphology_train.jsonl", "rb") as f:
        training_file = client.files.create(
            file=f,
            purpose="fine-tune"
        )
    print(f"Training file uploaded: {training_file.id}")
    
    # Upload validation file
    print("Uploading validation file...")
    with open("sanskrit_morphology_val.jsonl", "rb") as f:
        validation_file = client.files.create(
            file=f,
            purpose="fine-tune"
        )
    print(f"Validation file uploaded: {validation_file.id}")
    
    return training_file.id, validation_file.id

# Define the custom grader for Sanskrit morphology
from openai.types.graders import PythonGrader

sanskrit_grader = PythonGrader(
    name="Sanskrit Morphology Derivation Grader",
    type="python",
    source="""
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
)

def create_rl_job(training_file_id, validation_file_id):
    """Create the reinforcement learning fine-tuning job"""
    
    print("Creating RL fine-tuning job...")
    
    job = client.fine_tuning.jobs.create(
        training_file=training_file_id,
        validation_file=validation_file_id,
        model="o4-mini-2025-04-16",  # Using GPT-4o as base model
        method={
            "type": "reinforcement",
            "reinforcement": ReinforcementMethod(
                grader=sanskrit_grader,
                hyperparameters=ReinforcementHyperparameters(
                    reasoning_effort="medium",  # Can be "low", "medium", or "high"
                    # You can also set other hyperparameters like:
                    n_epochs=3,
                    # batch_size=8,
                    # learning_rate_multiplier=1.0,
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

if __name__ == "__main__":
    try:
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
        print(f"   Base Model: gpt-4o-2024-08-06")
        print(f"   Grader: Custom Sanskrit morphology grader")
        
        print(f"\n🔗 Monitor your job at:")
        print(f"   https://platform.openai.com/finetune/{job.id}")
        
        print(f"\n📝 To check status later, run:")
        print(f"   python -c \"from openai import OpenAI; client = OpenAI(); print(client.fine_tuning.jobs.retrieve('{job.id}'))\"")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have:")
        print("1. Set your OPENAI_API_KEY environment variable")
        print("2. Generated the JSONL files by running the dataset generator script")
        print("3. Have sufficient credits in your OpenAI account")
