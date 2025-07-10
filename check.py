import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"PyTorch CUDA version: {torch.version.cuda}")

if torch.cuda.is_available():
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    # You can add more checks here, like memory usage:
    # print("Memory Usage:")
    # print(f"  Allocated: {round(torch.cuda.memory_allocated(0) / 1024**3, 1)} GB")
    # print(f"  Cached: {round(torch.cuda.memory_reserved(0) / 1024**3, 1)} GB")
else:
    print('No GPU detected by PyTorch. Operations will run on CPU.')