import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Set device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define matrices
a = torch.randn(3, 4).to(device)
b = torch.randn(4, 5).to(device)

# Perform matrix multiplication on GPU
c = torch.matmul(a, b)

# Transfer result back to CPU
c = c.cpu()

# Print result
print(c)