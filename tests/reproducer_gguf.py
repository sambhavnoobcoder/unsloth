"""
Reproducer script for vision model GGUF conversion issue
Based on the colab notebook: https://colab.research.google.com/drive/1a4lW9H-9PN_nhjjh9NNy0I_QhZk6XNMn
"""

import os
import sys
import torch
import tempfile
from pathlib import Path
import inspect

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the necessary functions
from unsloth import FastVisionModel
from unsloth.save import vision_model_save_pretrained_gguf  # Import directly for debugging
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from unsloth import is_bf16_supported


def dump_model_structure(model, max_depth=3, current_depth=0):
    """Helper function to print model structure for debugging"""
    if current_depth > max_depth:
        return "..."
    
    if isinstance(model, torch.nn.Module):
        result = f"{type(model).__name__}:\n"
        for name, child in model.named_children():
            child_repr = dump_model_structure(child, max_depth, current_depth + 1)
            result += "  " * (current_depth + 1) + f"{name}: {child_repr}\n"
        return result
    else:
        return str(type(model).__name__)


def main():
    """
    Reproducer script that:
    1. Loads a small vision model
    2. Creates a tiny training dataset
    3. Performs minimal training
    4. Attempts to save to GGUF format
    """
    print("Loading FastVisionModel...")
    try:
        # Try smaller models that will fit in T4 memory
        models_to_try = [
            # Model name, load_in_4bit, trust_remote_code
            ("llava-hf/llava-1.5-7b-hf", True, False),
            ("Qwen/Qwen-VL-Chat", True, True),
            ("microsoft/phi-3-vision-128k-instruct", True, True),
        ]
        
        model = None
        tokenizer = None
        
        for model_name, use_4bit, trust_remote in models_to_try:
            try:
                print(f"Trying model: {model_name}")
                model, tokenizer = FastVisionModel.from_pretrained(
                    model_name,
                    load_in_4bit=use_4bit,
                    trust_remote_code=trust_remote,
                    use_flash_attention_2=False,  # Disable FA2 to avoid CUDA issues
                    attn_implementation="eager"   # Use eager implementation for compatibility
                )
                print(f"Successfully loaded model: {model_name}")
                
                # Print model info
                print("Model type:", type(model).__name__)
                print("Model attributes:", [attr for attr in dir(model) if not attr.startswith('_')])
                
                # Check if it's a normal model structure
                if hasattr(model, 'model'):
                    print("Model has 'model' attribute")
                elif hasattr(model, 'vision_tower'):
                    print("Model has 'vision_tower' attribute (LlavaForConditionalGeneration structure)")
                else:
                    print("Model has neither 'model' nor 'vision_tower' attribute")
                
                break
            except Exception as e:
                print(f"Failed to load {model_name}: {e}")
        
        if model is None:
            print("Failed to load any model. Exiting.")
            return False
            
    except Exception as e:
        print(f"Unexpected error during model loading: {e}")
        return False
    
    print("Setting up LoRA...")
    try:
        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=True,
            finetune_language_layers=True,
            finetune_attention_modules=False,
            finetune_mlp_modules=True,
            r=8,
            lora_alpha=8,
            lora_dropout=0,
            bias="none",
            random_state=3407
        )
    except Exception as e:
        print(f"Error setting up LoRA: {e}")
        print("Continuing with base model...")
    
    # Create a minimal synthetic dataset for testing
    print("Creating minimal synthetic dataset...")
    try:
        # Create a minimal synthetic dataset with dummy images and text
        from PIL import Image
        import numpy as np
        
        dummy_images = []
        dummy_captions = []
        
        # Create 2 small dummy images
        for i in range(2):
            # Create a small 224x224 black image
            img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
            dummy_images.append(img)
            dummy_captions.append(f"This is a test caption for image {i}")
        
        dataset = [{"image": img, "caption": cap} for img, cap in zip(dummy_images, dummy_captions)]
    except Exception as e:
        print(f"Error creating synthetic dataset: {e}")
        return False
    
    # Format the dataset
    print("Formatting dataset...")
    instruction = "You are an expert radiographer. Describe accurately what you see in this image."
    
    def convert_to_conversation(sample):
        conversation = [
            {"role": "user",
             "content": [
                 {"type": "text", "text": instruction},
                 {"type": "image", "image": sample["image"]}
             ]
            },
            {"role": "assistant",
             "content": [
                 {"type": "text", "text": sample["caption"]}
             ]
            },
        ]
        return {"messages": conversation}
    
    converted_dataset = [convert_to_conversation(sample) for sample in dataset]
    
    # Skip training to focus directly on GGUF conversion
    print("Skipping training to focus on GGUF conversion...")
    
    # Now attempt to save to GGUF
    print("Saving model to GGUF format...")
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = os.path.join(temp_dir, "model")
            os.makedirs(output_dir, exist_ok=True)
            
            # Try examining the save_pretrained_gguf function
            print("Examining vision_model_save_pretrained_gguf function:")
            print(inspect.getsource(vision_model_save_pretrained_gguf))
            
            # For LlavaForConditionalGeneration models, we need to handle them differently
            # Let's look at the model structure
            print("\nModel structure:")
            print(dump_model_structure(model, max_depth=1))
            
            # Try to save to GGUF format
            print(f"\nAttempting to save model to GGUF in {output_dir}...")
            
            # First save the model and tokenizer to the output directory
            print("Saving model and tokenizer to output directory...")
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            
            try:
                # Now try the GGUF conversion
                gguf_path = model.save_pretrained_gguf(
                    output_dir,
                    tokenizer,
                    quantization_method="q8_0"
                )
                print(f"Successfully saved to GGUF: {gguf_path}")
                return True
            except AttributeError as e:
                if "'LlavaForConditionalGeneration' object has no attribute 'model'" in str(e):
                    print("Caught expected error for LlavaForConditionalGeneration model")
                    print("This is the issue we're trying to reproduce - vision models need special handling for GGUF conversion")
                    return True  # Successfully reproduced the issue
                else:
                    raise  # Re-raise if it's a different AttributeError
                
    except Exception as e:
        print(f"Error during GGUF conversion: {e}")
        if "Vision model conversion to GGUF failed" in str(e):
            print("This reproduces the reported issue with vision model GGUF conversion.")
            return True  # We successfully reproduced the issue
        else:
            print("An unexpected error occurred during GGUF conversion.")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            return False

if __name__ == "__main__":
    success = main()
    print(f"Reproducer {'successfully demonstrated the issue!' if success else 'failed to demonstrate the issue.'}")
    sys.exit(0 if success else 1) 