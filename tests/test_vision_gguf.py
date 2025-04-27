#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test conversion of vision models to GGUF format
"""

import os
import sys
import torch
import tempfile
import unittest
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from unsloth import FastModel

class TestVisionGGUF(unittest.TestCase):
    """Test saving vision models to GGUF format"""

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_vision_gguf_conversion(self):
        """Test converting a vision model to GGUF format"""
        # Use a small vision model for testing
        model_name = "unsloth/Llama-3.2-1B-Vision-Instruct-bnb-4bit"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Load the model
            model, tokenizer = FastModel.from_pretrained(
                model_name=model_name,
                max_seq_length=2048,
                dtype=torch.bfloat16,
                load_in_4bit=True
            )
            
            # Save the model in GGUF format
            gguf_path = model.save_pretrained_gguf(
                save_directory=os.path.join(temp_dir, "test_vision_model"),
                tokenizer=tokenizer,
                quantization_method="q8_0"
            )
            
            # Check that the GGUF file was created
            self.assertTrue(os.path.exists(gguf_path))
            self.assertTrue(gguf_path.endswith(".gguf"))
            
            print(f"Successfully created GGUF file at: {gguf_path}")

if __name__ == "__main__":
    unittest.main() 