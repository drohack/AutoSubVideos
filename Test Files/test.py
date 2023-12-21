import torch

# Check if GPU is available
if torch.cuda.is_available():
    # Set GPU device
    torch.cuda.set_device(0)
    gpu_properties = torch.cuda.get_device_properties(0)

    # Check the CUDA version
    cuda_version = torch.version.cuda
    gpu_version = str(gpu_properties.major) + "." + str(gpu_properties.minor)
    print(f"Torch CUDA version: {cuda_version}")
    print(f"GPU   CUDA version: {gpu_version}")

    # Check PyTorch CUDA version compatibility
    required_cuda_version = torch.version.cuda.split('.')[0]
    if int(required_cuda_version) <= float(gpu_version):
        print("Your GPU is compatible with PyTorch.")
    else:
        print("Your GPU may not be compatible with this version of PyTorch.")
else:
    print("No GPU available. PyTorch is running on CPU.")