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