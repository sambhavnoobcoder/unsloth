#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test conversion of vision models to GGUF format (fixed)
"""

import os
import sys
import torch
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from unsloth import FastModel
from huggingface_hub import login as hf_login

def main():
    """Test converting a vision model to GGUF format"""
    # Authenticate with Hugging Face (if HF_TOKEN is set in environment)
    token = os.getenv("HF_TOKEN")
    if token:
        print("Using HF_TOKEN from environment")
        hf_login(token=token)
    
    # Use an existing vision model
    model_name = "unsloth/Llama-3.1-11B-Vision-Instruct-bnb-4bit"
    print(f"Testing with vision model: {model_name}")
    
    # Create temporary directory for saving GGUF
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = os.path.join(temp_dir, "test_vision_model")
        
        # Load the model - using FastModel instead of FastVisionModel
        print("Loading model...")
        model, tokenizer = FastModel.from_pretrained(
            model_name=model_name,
            max_seq_length=2048,
            load_in_4bit=True,
            token=token,
        )
        
        # Save in GGUF format with q8_0 quantization
        print("Converting to GGUF...")
        gguf_path = model.save_pretrained_gguf(
            save_directory=output_dir,
            tokenizer=tokenizer,
            quantization_method="q8_0"
        )
        
        # Check results
        if os.path.exists(gguf_path) and str(gguf_path).endswith(".gguf"):
            print(f"SUCCESS! Created GGUF file at: {gguf_path}")
            return True
        else:
            print(f"FAILED! GGUF file not found or has wrong extension: {gguf_path}")
            return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
