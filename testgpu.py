import torch

print("PyTorch version:", torch.__version__)
print("CUDA is available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("Current GPU:", torch.cuda.get_device_name(0))
    print("GPU Memory Available:", torch.cuda.get_device_properties(0).total_memory / 1024**3, "GB")