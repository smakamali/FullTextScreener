# test the GPU

import torch
from transformers import pipeline

# Initialize pipeline with GPU
model_id = "mistral-large-latest"  # Example model
pipe = pipeline(
    "text-generation",
    model=model_id,
    device="cuda",  # Explicitly use GPU
    torch_dtype=torch.float16,  # Use half-precision for memory efficiency
    model_kwargs={"load_in_4bit": True}  # Enable 4-bit quantization
)

# Test GPU inference
output = pipe("Explain GPU acceleration in simple terms", max_new_tokens=100)
print(output[0]['generated_text'])