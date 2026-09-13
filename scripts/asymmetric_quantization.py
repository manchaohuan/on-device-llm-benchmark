import torch

x = torch.tensor([
    -1.0,
    -0.5,
    0.0,
    1.0,
    2.0,
    3.0
])

qmin = 0;
qmax = 255;

scale = (x.max() - x.min()) / (qmax - qmin)

zero_point = torch.round(qmin - x.min() / scale)

q = torch.round(x / scale + zero_point)

q= torch.clamp(q, qmin, qmax)

x_hat = (q - zero_point) * scale

error = x - x_hat

print("Original tensor:", x)
print("scale:", scale)
print("zero_point:", zero_point)
print("Quantized tensor:", q)
print("Dequantized tensor:", x_hat)
print("Error:", error)
