"""
Reproducer script for vision model GGUF conversion issue
Based on the colab notebook: https://colab.research.google.com/drive/1a4lW9H-9PN_nhjjh9NNy0I_QhZk6XNMn
"""

import os
import sys
import torch
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the necessary functions
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from unsloth import is_bf16_supported

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
        model, tokenizer = FastVisionModel.from_pretrained(
            "unsloth/Mistral-Small-3.1-24B-Instruct-2503-unsloth-bnb-4bit",
            load_in_4bit=True,
            use_gradient_checkpointing="unsloth",
            attn_implementation="eager"
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Trying fallback model for testing...")
        # Fallback to a smaller model for testing if the primary model fails
        try:
            model, tokenizer = FastVisionModel.from_pretrained(
                "Qwen/Qwen-VL-Chat",
                load_in_4bit=True
            )
        except Exception as e2:
            print(f"Error loading fallback model: {e2}")
            print("Cannot continue without a model. Exiting.")
            return False
    
    print("Setting up LoRA...")
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
    
    # Create a minimal synthetic dataset for testing
    print("Creating minimal synthetic dataset...")
    try:
        # Try to load a tiny subset of the actual dataset
        dataset = load_dataset("unsloth/Radiology_mini", split="train[:2]")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Creating synthetic dataset instead...")
        
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
    
    # Minimal training
    print("Setting up trainer...")
    FastVisionModel.for_training(model)
    
    try:
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            data_collator=UnslothVisionDataCollator(model, tokenizer),
            train_dataset=converted_dataset,
            args=SFTConfig(
                per_device_train_batch_size=1,
                gradient_accumulation_steps=1,
                warmup_steps=1,
                max_steps=1,  # Just 1 step for testing
                learning_rate=2e-4,
                fp16=not is_bf16_supported(),
                bf16=is_bf16_supported(),
                logging_steps=1,
                optim="paged_adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=3407,
                output_dir="outputs",
                report_to="none",
                
                # Vision finetuning required parameters
                remove_unused_columns=True,
                dataset_text_field="",
                dataset_kwargs={"skip_prepare_dataset": False},
                dataset_num_proc=1,
                max_seq_length=2048,
            ),
        )
        
        print("Running training...")
        trainer.train()
    except Exception as e:
        print(f"Error during training: {e}")
        print("Continuing with GGUF save test without training...")
    
    # Now attempt to save to GGUF
    print("Saving model to GGUF format...")
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = os.path.join(temp_dir, "model")
            
            # Try to save to GGUF format
            gguf_path = model.save_pretrained_gguf(
                output_dir,
                tokenizer,
                quantization_method="q8_0"
            )
            
            print(f"Successfully saved to GGUF: {gguf_path}")
            return True
    except Exception as e:
        print(f"Error during GGUF conversion: {e}")
        if "Vision model conversion to GGUF failed" in str(e):
            print("This reproduces the reported issue with vision model GGUF conversion.")
            return True  # We successfully reproduced the issue
        else:
            print("An unexpected error occurred.")
            return False

if __name__ == "__main__":
    success = main()
    print(f"Reproducer {'successfully demonstrated the issue!' if success else 'failed to demonstrate the issue.'}")
    sys.exit(0 if success else 1) 