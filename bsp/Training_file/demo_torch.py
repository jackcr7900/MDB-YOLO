import torch
if torch.cuda.is_available():
    print(f"CUDA is available! Using device: {torch.cuda.current_device()}")
    device = torch.device('cuda')  # Default CUDA device
    x = torch.randn(10, device=device)  # Create a tensor on the GPU
    print(x)
else:
    print("CUDA is not available.")