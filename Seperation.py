import demucs.api
import subprocess
import os

import torch

# 1. Check if CUDA (GPU support) is available
cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")

if cuda_available:
    # 2. Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {num_gpus}")

    # 3. Get the name of the current GPU (usually device 0 by default)
    if num_gpus > 0:
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU Name: {gpu_name}")

        # 4. Get the current CUDA device being used by PyTorch
        current_device_index = torch.cuda.current_device()
        print(f"Current CUDA Device Index: {current_device_index}")

        # 5. Check memory usage (useful during/after running a model)
        print("Memory Usage:")
        print(f"  Allocated: {round(torch.cuda.memory_allocated(0) / 1024**3, 1)} GB")
        print(f"  Cached: {round(torch.cuda.memory_reserved(0) / 1024**3, 1)} GB")
    else:
        print("CUDA is available but no GPUs detected.")
else:
    print("CUDA is not available. PyTorch will use CPU.")

# How your code determines the device for operations:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nYour operations will run on: {device}")

# Example of moving a tensor to the determined device
x = torch.randn(5, 5) # Created on CPU by default
print(f"Tensor x device (initial): {x.device}")
x = x.to(device) # Move to GPU if available, else stays on CPU
print(f"Tensor x device (after .to(device)): {x.device}")

# Initialize with default parameters:
separator = demucs.api.Separator(model="htdemucs", segment=3)

youtube_url = "https://youtu.be/DWaB4PXCwFU?si=yzJcBEO9PY5-SCVE"
print("Downloading:", youtube_url)
# Download and save as file.mp3
subprocess.run([
    "yt-dlp", "-x", "--audio-format", "mp3", "-o", "file.%(ext)s", youtube_url
], check=True)

# You can also use other parameters defined
print("Separating stems...")
# Separating an audio file
origin, separated = separator.separate_audio_file("file.mp3")

# Separating a loaded audio
# origin, separated = separator.separate_tensor(audio)

# If you encounter an error like CUDA out of memory, you can use this to change parameters like `segment`:
# separator.update_parameter(segment=smaller_segment)
os.makedirs("sounds", exist_ok=True)

# Save each stem in the sounds folder
for stem, source in separated.items():
    out_path = os.path.join("sounds", f"{stem}_file.wav")
    demucs.api.save_audio(source, out_path, samplerate=separator.samplerate)