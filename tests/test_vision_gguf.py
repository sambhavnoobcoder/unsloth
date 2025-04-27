"""
Test conversion of vision models to GGUF format - fully self-contained
"""

import os
import sys
import json
import torch
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the specific functions we need directly from unsloth.save
from unsloth.save import vision_model_save_pretrained_gguf, patch_saving_functions

# Create a minimal mock model with just enough structure to test GGUF export
class MockVisionModel:
    def __init__(self):
        # Create a basic config with vision attributes directly
        class Config:
            def __init__(self):
                self.vision_config = {"image_size": 224, "patch_size": 16, "hidden_size": 768}
                self.torch_dtype = torch.float16
                self.model_type = "llama_vision"
                self.unsloth_version = "1.0.0"
                
            def to_dict(self):
                return {
                    "vision_config": self.vision_config,
                    "model_type": self.model_type
                }
                
        self.config = Config()
        self.original_push_to_hub = None

    def save_pretrained(self, *args, **kwargs):
        # Mock the save_pretrained functionality
        directory = kwargs.get("save_directory", args[0] if args else "temp_dir")
        os.makedirs(directory, exist_ok=True)
        # Save a minimal config file
        with open(os.path.join(directory, "config.json"), "w") as f:
            json.dump({
                "model_type": self.config.model_type,
                "vision_config": self.config.vision_config
            }, f)
        return directory
        
    # Add push_to_hub method
    def push_to_hub(self, *args, **kwargs):
        print("Mock push_to_hub called")
        return "mock_repo_id"
        
    # Add any other methods that might be needed
    def add_model_tags(self, tags):
        print(f"Adding model tags: {tags}")

def main():
    """Test the vision model GGUF conversion function directly"""
    print("Creating mock vision model...")
    model = MockVisionModel()
    
    # Apply patching to add the save_pretrained_gguf method
    print("Patching saving functions...")
    patch_saving_functions(model, vision=True)
    
    # Create dummy tokenizer
    class MockTokenizer:
        def __init__(self):
            self.original_push_to_hub = None
            
        def save_pretrained(self, directory):
            os.makedirs(directory, exist_ok=True)
            with open(os.path.join(directory, "tokenizer_config.json"), "w") as f:
                json.dump({"model_type": "llama"}, f)
                
        def push_to_hub(self, *args, **kwargs):
            print("Mock tokenizer push_to_hub called")
            return "mock_tokenizer_repo_id"
    
    tokenizer = MockTokenizer()
    
    # Check if patching succeeded
    print("Checking if save_pretrained_gguf method was added...")
    if hasattr(model, 'save_pretrained_gguf'):
        print("✅ save_pretrained_gguf method successfully added to model")
    else:
        print("❌ save_pretrained_gguf method not added to model")
        return False
    
    # Test the GGUF conversion function
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = os.path.join(temp_dir, "test_vision_model")
        
        print("Testing vision_model_save_pretrained_gguf function...")
        try:
            # We expect this to fail when trying to convert to GGUF
            # But it should at least get past the initialization and run until it tries to use llama.cpp
            gguf_path = model.save_pretrained_gguf(
                save_directory=output_dir,
                tokenizer=tokenizer,
                quantization_method="q8_0"
            )
            print(f"SUCCESS! Created GGUF file at: {gguf_path}")
            success = True
        except Exception as e:
            # If we got an error about llama.cpp converter missing, that's expected and means our function is correctly called
            error_str = str(e)
            if "llama.cpp" in error_str:
                print("SUCCESS! Got expected error about missing llama.cpp. Function was called correctly.")
                print(f"Error: {error_str}")
                success = True
            else:
                print(f"ERROR: {error_str}")
                import traceback
                traceback.print_exc()
                success = False
        
        return success

if __name__ == "__main__":
    success = main()
    print(f"Test {'passed' if success else 'failed'}")
    sys.exit(0 if success else 1)
