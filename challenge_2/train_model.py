# Run this in Colab

# Cell 1: Install Prime-RL in Colab
!apt-get update
!apt-get install -y git-lfs
!pip install uv

# Clone Prime-RL
!git clone https://github.com/PrimeIntellect-ai/prime-rl.git
%cd prime-rl

# Install dependencies using uv
!curl -LsSf https://astral.sh/uv/install.sh | sh
!source $HOME/.local/bin/env
!uv sync

# Cell 2: Install additional dependencies
!pip install vidyut datasets huggingface_hub

# Cell 3: Create custom verifier for Sanskrit
%%writefile src/zeroband/verifiers/sanskrit_morphology_verifier.py
from vidyut.prakriya import *
from vidyut.lipi import *
import json

class SanskritMorphologyVerifier:
    """Verifier for Sanskrit morphology following GENESYS format"""
    
    def __init__(self):
        self.data = Data("../vidyut-0.4.0/prakriya/")
        self.v = Vyakarana(log_steps=False)
        self.translit = lambda x: transliterate(str(x), Scheme.Slp1, Scheme.Iast)
    
    def verify(self, task, response):
        """
        Verify response against task using Vidyut.
        Returns (is_correct, metadata) tuple.
        """
        try:
            # Extract answer from model response
            import re
            match = re.search(r'"conjugated_verb"\s*:\s*"([^"]+)"', response)
            if match:
                model_answer = match.group(1)
            else:
                model_answer = response.strip().split()[-1]
            
            # Get correct answer
            correct_answer = task["answer_iast"]
            
            # Normalize and compare
            is_correct = model_answer.lower().strip() == correct_answer.lower().strip()
            
            return is_correct, {
                "model_answer": model_answer,
                "correct_answer": correct_answer
            }
            
        except Exception as e:
            return False, {"error": str(e)}

# Cell 4: Create config for Sanskrit training
%%writefile configs/training/sanskrit_morphology.toml
[model]
name = "microsoft/DialoGPT-small"  # Small model for Colab
model_type = "causal"

[data]
dataset = "vkethana/sanskrit-morphology-rl"
num_workers = 2
batch_size = 8  # Small batch for Colab

[training]
learning_rate = 1e-5
num_epochs = 2
gradient_accumulation_steps = 4
save_steps = 100

[inference]
temperature = 0.7
max_new_tokens = 50

[verifier]
type = "custom"
module = "verifiers.sanskrit_morphology_verifier"
class = "SanskritMorphologyVerifier"

# Cell 6: Run training (simplified for Colab)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# For Colab, we'll use a simplified training script
!python -m torch.distributed.run --nproc_per_node=1 \
    src/zeroband/train.py @ configs/training/sanskrit_morphology.toml \
    --output_dir=/content/drive/MyDrive/sanskrit_rl_model

# Cell 7: Alternative - Use GENESYS directly for generation/verification
!git clone https://github.com/PrimeIntellect-ai/genesys.git
%cd genesys
!uv sync --extra sglang

# Generate responses
!uv run python src/genesys/generate.py \
    --dataset vkethana/sanskrit-morphology-genesys \
    --model microsoft/DialoGPT-small \
    --output_dir /content/drive/MyDrive/sanskrit_responses

# Verify responses  
!uv run python src/genesys/verify.py \
    --file /content/drive/MyDrive/sanskrit_responses/out_*.jsonl \
    --verifier sanskrit_morphology
