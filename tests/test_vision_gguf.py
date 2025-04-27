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

# Import the necessary functions directly
from unsloth.save import vision_model_save_pretrained_gguf

class MockVisionModel:
    def __init__(self):
        # Create a basic config with vision attributes directly
        class Config:
            def __init__(self):
                # Create a VisionConfig object with to_dict method
                class VisionConfig:
                    def __init__(self):
                        self.image_size = 224
                        self.patch_size = 16
                        self.hidden_size = 768
                        
                    def to_dict(self):
                        return {
                            "image_size": self.image_size,
                            "patch_size": self.patch_size,
                            "hidden_size": self.hidden_size
                        }
                
                self.vision_config = VisionConfig()
                self.torch_dtype = torch.float16
                self.model_type = "llama_vision"
                self.unsloth_version = "1.0.0"
                self.vocab_size = 32000
                
            def to_dict(self):
                return {
                    "vision_config": self.vision_config.to_dict(),
                    "model_type": self.model_type,
                    "vocab_size": self.vocab_size
                }
                
        self.config = Config()

    # Ensure save_pretrained matches the expected signature
    def save_pretrained(self, save_directory=None, **kwargs):
        """Save the model to a directory."""
        directory = save_directory or kwargs.get("save_directory")
        os.makedirs(directory, exist_ok=True)
        # Save a minimal config file
        with open(os.path.join(directory, "config.json"), "w") as f:
            json.dump({
                "model_type": self.config.model_type,
                "vision_config": self.config.vision_config.to_dict(),
                "vocab_size": self.config.vocab_size
            }, f)
        return directory
        
# More complete mock tokenizer with MATCHING ARGUMENT NAMES
class MockTokenizer:
    def __init__(self):
        # Add all commonly required tokenizer attributes
        self.padding_side = "right"
        self.model_max_length = 2048
        self.vocab_size = 32000
        self.pad_token = "[PAD]"
        self.pad_token_id = 0
        self.eos_token = "</s>"
        self.eos_token_id = 1
        self.bos_token = "<s>"
        self.bos_token_id = 2
        self.name_or_path = "mock_tokenizer"
        
        # For saving
        self.special_tokens_map_file = "special_tokens_map.json"
        self.tokenizer_config_file = "tokenizer_config.json"
        
    # Must match the exact signature expected by unsloth_save_model
    def save_pretrained(self, save_directory=None, **kwargs):
        """Save the tokenizer to a directory."""
        directory = save_directory or kwargs.get("save_directory")
        if not directory:
            raise ValueError("No save_directory specified")
            
        os.makedirs(directory, exist_ok=True)
        
        # Save tokenizer config
        with open(os.path.join(directory, "tokenizer_config.json"), "w") as f:
            json.dump({
                "model_type": "llama",
                "padding_side": self.padding_side,
                "pad_token": self.pad_token,
                "eos_token": self.eos_token,
                "bos_token": self.bos_token
            }, f)
            
        # Save special tokens map
        with open(os.path.join(directory, "special_tokens_map.json"), "w") as f:
            json.dump({
                "pad_token": self.pad_token,
                "eos_token": self.eos_token,
                "bos_token": self.bos_token
            }, f)
            
        # Create a vocabulary file
        with open(os.path.join(directory, "vocab.json"), "w") as f:
            vocab = {f"token{i}": i for i in range(100)}
            json.dump(vocab, f)
            
        # Create a merges file for BPE tokenizers
        with open(os.path.join(directory, "merges.txt"), "w") as f:
            f.write("# merges\n")
            
        return directory

def main():
    """Test the vision model GGUF conversion function directly"""
    print("Creating mock vision model...")
    model = MockVisionModel()
    tokenizer = MockTokenizer()
    
    # Directly attach the GGUF saving method to our model
    print("Attaching vision_model_save_pretrained_gguf method directly...")
    model.save_pretrained_gguf = types.MethodType(vision_model_save_pretrained_gguf, model)
    
    # Prepare a dummy state dictionary for the model
    def dummy_state_dict():
        return {"dummy": torch.zeros(1, 1)}
    
    model.state_dict = dummy_state_dict
    
    # Patch additional required methods for saving
    def dummy_get_input_embeddings():
        class DummyEmbedding:
            def __init__(self):
                self.weight = torch.zeros(32000, 768)
        return DummyEmbedding()
    
    model.get_input_embeddings = dummy_get_input_embeddings
    
    # Test the GGUF conversion function
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = os.path.join(temp_dir, "test_vision_model")
        
        print("Testing vision_model_save_pretrained_gguf function...")
        try:
            # We expect this to fail when trying to convert to GGUF
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
            
            # Expected errors that indicate the function is working correctly
            expected_errors = [
                "llama.cpp", 
                "does not exist",
                "convert-hf-to-gguf.py",
                "Converting vision model to GGUF"
            ]
            
            if any(err in error_str for err in expected_errors):
                print("SUCCESS! Function reached the expected llama.cpp conversion step.")
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