#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test conversion of vision models to GGUF format (fixed)
"""

import os
import sys
import torch
import tempfile
import unittest
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unsloth
import unsloth.models.loader as _loader
# Monkey-patch compile_transformers to avoid None return unpacking error
# This ensures that from_pretrained doesn’t fail unpacking a None return
def _compile_transformers_stub(*args, **kwargs):
    return None, False
_loader.unsloth_compile_transformers = _compile_transformers_stub

# Use the public FastVisionModel for vision tasks
from unsloth import FastVisionModel
from huggingface_hub import login as hf_login

class TestVisionGGUF(unittest.TestCase):
    """Test saving vision models to GGUF format"""

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_vision_gguf_conversion(self):
        """Test converting a vision model to GGUF format"""
        # Authenticate with Hugging Face (if HF_TOKEN is set in environment)
        token = os.getenv("HF_TOKEN")
        if token:
            hf_login(token=token)

        # Use an existing vision model for testing (base, without explicit suffix)
        model_name = "unsloth/Llama-3.2-11B-Vision-Instruct"
        print(f"Testing with vision model: {model_name}")

        # Pick dtype: use bfloat16 if supported, else float16
        desired_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if desired_dtype == torch.float16:
            print("Device does not support bfloat16. Using float16 instead.")

        # Create temporary directory for saving GGUF
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "test_vision_model"

            # Load the vision model
            model, tokenizer = FastVisionModel.from_pretrained(
                model_name=model_name,
                max_seq_length=2048,
                dtype=desired_dtype,
                load_in_4bit=True,
                trust_remote_code=True,
                token=token,
            )

            # Save in GGUF format with q8_0 quantization
            gguf_path = model.save_pretrained_gguf(
                save_directory=str(output_dir),
                tokenizer=tokenizer,
                quantization_method="q8_0"
            )

            # Assertions: file exists and has .gguf extension
            self.assertTrue(os.path.exists(gguf_path), f"GGUF file not found at {gguf_path}")
            self.assertTrue(str(gguf_path).endswith(".gguf"), f"File does not have .gguf extension: {gguf_path}")

            print(f"Successfully created GGUF file at: {gguf_path}")

if __name__ == "__main__":
    unittest.main()
