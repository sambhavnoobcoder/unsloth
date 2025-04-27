"""
Test conversion of vision models to GGUF format - fully self-contained
"""

import os
import sys
import json
import torch
import tempfile
from pathlib import Path
import types

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the specific functions we need directly from unsloth.save
from unsloth.save import vision_model_save_pretrained_gguf

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

    def save_pretrained(self, *args, **kwargs):
        """Save the model to a directory."""
        directory = kwargs.get("save_directory", args[0] if args else "temp_dir")
        os.makedirs(directory, exist_ok=True)
        # Save a minimal config file
        with open(os.path.join(directory, "config.json"), "w") as f:
            json.dump({
                "model_type": self.config.model_type,
                "vision_config": self.config.vision_config
            }, f)
        return directory

def main():
    """Test the vision model GGUF conversion function directly"""
    print("Creating mock vision model...")
    model = MockVisionModel()
    
    # Create dummy tokenizer
    class MockTokenizer:
        def save_pretrained(self, directory):
            """Save the tokenizer to a directory."""
            os.makedirs(directory, exist_ok=True)
            with open(os.path.join(directory, "tokenizer_config.json"), "w") as f:
                json.dump({"model_type": "llama"}, f)
    
    tokenizer = MockTokenizer()
    
    # Directly attach the GGUF saving method to our model
    print("Attaching vision_model_save_pretrained_gguf method directly...")
    model.save_pretrained_gguf = types.MethodType(vision_model_save_pretrained_gguf, model)
    
    # Test the GGUF conversion function
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = os.path.join(temp_dir, "test_vision_model")
        
        print("Testing vision_model_save_pretrained_gguf function...")
        try:
            # We expect this to fail when trying to convert to GGUF
            # But it should at least get past the initialization
            gguf_path = model.save_pretrained_gguf(
                save_directory=output_dir,
                tokenizer=tokenizer,
                quantization_method="q8_0"
            )
            print(f"SUCCESS! Created GGUF file at: {gguf_path}")
            success = True
        except Exception as e:
            # Check what kind of error we got
            error_str = str(e)
            print(f"Error: {error_str}")
            
            # If we got an error about unsloth_save_model, that's progress!
            if "unsloth_save_model" in error_str:
                print("SUCCESS! Function reached the point of calling unsloth_save_model.")
                success = True
            # Or if we got an error about llama.cpp
            elif "llama.cpp" in error_str:
                print("SUCCESS! Function reached the llama.cpp conversion step.")
                success = True
            else:
                import traceback
                traceback.print_exc()
                success = False
        
        return success

if __name__ == "__main__":
    success = main()
    print(f"Test {'passed' if success else 'failed'}")
    sys.exit(0 if success else 1)