import torch

x = torch.tensor([
    -1.0,
    -0.5,
    0.0,
    0.5,
    1.0,
])


qmax= 127

scale = x.abs().max() / qmax

q = torch.round(x / scale)

x_hat = q * scale

error = x - x_hat

print("Original tensor:", x)
print("scale:", scale)
print("Quantized tensor:", q)
print("Dequantized tensor:", x_hat)
print("Error:", error)