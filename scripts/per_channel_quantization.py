import torch

x = torch.tensor([
    [0.1, 0.2, 0.3],
    [2.0, 3.0, 4.0]
])

qmax = 127

# per-tensor quantization

scale_per_tensor = x.abs().max() / qmax
q_per_tensor = torch.round(x / scale_per_tensor)
x_hat_per_tensor = q_per_tensor * scale_per_tensor

#per-channel quantization
scale_per_channel = x.abs().max(dim=1, keepdim=True).values / qmax

q_channel = torch.round(x / scale_per_channel)
x_hat_per_channel = q_channel * scale_per_channel

print("Original tensor:", x)
print("scale (per-tensor):", scale_per_tensor)
print("Quantized tensor (per-tensor):", q_per_tensor)
print("Dequantized tensor (per-tensor):", x_hat_per_tensor)
print("scale (per-channel):", scale_per_channel)
print("Quantized tensor (per-channel):", q_channel)
print("Dequantized tensor (per-channel):", x_hat_per_channel)
print("Error (per-channel):", x - x_hat_per_channel)
print("Error (per-tensor):", x - x_hat_per_tensor)