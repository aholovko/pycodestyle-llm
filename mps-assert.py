import torch

a = torch.rand(1, device='mps')
print("a.item:", a.item())
print("a.half:", a.half())

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print(x)
else:
    print("MPS device not found.")
