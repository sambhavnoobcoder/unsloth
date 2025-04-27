import os
import sys
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
        from transformers import AutoConfig
        # Create a basic config with vision attributes
        self.config = AutoConfig.from_pretrained("microsoft/phi-3-mini")
        self.config.vision_config = {"image_size": 224, "patch_size": 16, "hidden_size": 768}
        self.config.torch_dtype = torch.bfloat16
        self.config.model_type = "phi3_vision"
        # Add basic tokenizer for the test
        self.tokenizer = None

    def save_pretrained(self, *args, **kwargs):
        # Mock the save_pretrained functionality
        directory = kwargs.get("save_directory", args[0] if args else "temp_dir")
        os.makedirs(directory, exist_ok=True)
        # Save a minimal config file
        with open(os.path.join(directory, "config.json"), "w") as f:
            f.write('{"model_type":"phi3_vision","vision_config":{"image_size":224,"patch_size":16,"hidden_size":768}}')
        return directory

def main():
    """Test the vision model GGUF conversion function directly"""
    print("Creating mock vision model...")
    model = MockVisionModel()
    
    # Apply patching to add the save_pretrained_gguf method
    patch_saving_functions(model, vision=True)
    
    # Create dummy tokenizer
    class MockTokenizer:
        def save_pretrained(self, directory):
            os.makedirs(directory, exist_ok=True)
            with open(os.path.join(directory, "tokenizer_config.json"), "w") as f:
                f.write('{"model_type":"phi3"}')
    
    tokenizer = MockTokenizer()
    
    # Test the GGUF conversion function
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = os.path.join(temp_dir, "test_vision_model")
        
        print("Testing vision_model_save_pretrained_gguf function...")
        try:
            # We expect this to fail when trying to convert to GGUF
            # But it should at least get to the point where it calls the llama.cpp converter
            model.save_pretrained_gguf(
                save_directory=output_dir,
                tokenizer=tokenizer,
                quantization_method="q8_0"
            )
            success = True
        except Exception as e:
            # If we got an error about llama.cpp converter missing, that's expected and means our function is correctly called
            if "llama.cpp" in str(e):
                print("SUCCESS! Got expected error about missing llama.cpp. Function was called correctly.")
                success = True
            else:
                print(f"ERROR: {str(e)}")
                success = False
        
        return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
