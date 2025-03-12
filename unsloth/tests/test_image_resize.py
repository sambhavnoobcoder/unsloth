from unsloth import FastModel
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import gc

# Create a synthetic test image
def create_test_image(size=(800, 600)):
    # Create a colorful test image
    img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    # Add some colors and patterns
    for i in range(size[1]):
        for j in range(size[0]):
            img[i, j, 0] = (i * 256) // size[1]  # R channel
            img[i, j, 1] = (j * 256) // size[0]  # G channel
            img[i, j, 2] = ((i+j) * 128) // (size[0]+size[1])  # B channel
    return Image.fromarray(img)

# Create test image
image = create_test_image(size=(800, 600))
print(f"Original image size: {image.size}")

# Display the image
plt.figure(figsize=(10, 6))
plt.imshow(image)
plt.title(f"Test image: {image.size[0]}x{image.size[1]}")
plt.show()

# Test different image sizes
max_sizes_to_test = [None, 224, 448, (640, 320)]
memory_usages = []
tensor_shapes = []

# Try Qwen2-VL which uses a more standardized image processing approach
model_name = "unsloth/Qwen2-VL-7B-Instruct-unsloth-bnb-4bit"

for max_size in max_sizes_to_test:
    print(f"\nTesting with max_image_size={max_size}")
    
    # Track memory before loading
    torch.cuda.empty_cache()
    gc.collect()
    mem_before = torch.cuda.memory_allocated()
    
    # Load model with specified image size
    model, processor = FastModel.from_pretrained(
        model_name,
        max_image_size=max_size
    )
    
    # Prepare input prompt for Qwen2-VL - make sure image token is properly detected
    prompt = processor.image_token + "\nWhat's shown in this image?"
    
    # Process the image
    print(f"Using processor: {processor.__class__.__name__}")
    
    try:
        # For tracking how the image flows through the system, print debug info
        print(f"Image processor attributes: {[m for m in dir(processor) if 'image' in m and not m.startswith('_')]}")
        
        # Process input with image
        inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")
        
        # Print inputs structure
        for key in inputs.keys():
            if torch.is_tensor(inputs[key]):
                print(f"Input '{key}' shape: {tuple(inputs[key].shape)}")
            else:
                print(f"Input '{key}' type: {type(inputs[key])}")
        
        # Check for pixel_values
        if "pixel_values" in inputs:
            shape = tuple(inputs.pixel_values.shape)
            tensor_shapes.append(shape)
            print(f"Processed image tensor shape: {shape}")
        else:
            print("No pixel_values found in inputs")
            tensor_shapes.append("Unknown")
        
        # Check memory usage
        mem_after = torch.cuda.memory_allocated()
        memory_usage = (mem_after - mem_before) / 1024**2
        memory_usages.append(memory_usage)
        print(f"Memory usage: {memory_usage:.2f} MB")
        
        # Generate output - use a smaller max_new_tokens to avoid long generation
        output = model.generate(**inputs, max_new_tokens=20)
        print(processor.decode(output[0], skip_special_tokens=True))
    
    except Exception as e:
        print(f"Error processing: {e}")
        import traceback
        traceback.print_exc()
        tensor_shapes.append("Error")
        memory_usages.append(0)
    
    # Clean up to avoid memory accumulation
    del model
    del processor
    if 'inputs' in locals():
        del inputs
    if 'output' in locals():
        del output
    torch.cuda.empty_cache()
    gc.collect()

# Print summary
print("\n--- Summary ---")
for i, max_size in enumerate(max_sizes_to_test):
    print(f"Max size: {max_size}, Tensor shape: {tensor_shapes[i]}, Memory: {memory_usages[i]:.2f} MB")