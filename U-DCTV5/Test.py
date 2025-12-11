import torch
import torch.nn as nn
from torchinfo import summary
from thop import profile, clever_format

# 这里假设 globalatt 已经定义在 globalatt.py
from GoLT import globalatt, FullGlobalTransBlock

# --------------------------
# 假输入 (模拟医学图像 512x512)
# --------------------------
B, C, H, W = 1, 32, 128, 128   # 小点分辨率方便测试
k = 17
h_proj = 128

x = torch.randn(B, C, H, W)

# U: (B, h_proj, k)
U = torch.randn(B, h_proj, k)

# S: (B, k)
S = torch.abs(torch.randn(B, C))  # 奇异值非负

# V: (B, C, k)
V = torch.randn(B, C, k)

# G: (HW, h_proj)
G = torch.randn(H*W, h_proj)

# --------------------------
# 模块实例化
# --------------------------

# --------------------------
# 前向计算
# --------------------------

print("输入:", x.shape)
print("U:", U.shape)
print("S:", S.shape)
print("V:", V.shape)
print("G:", G.shape)
# --------------------------
# 模块实例化
# --------------------------
num_layers = 3
block = FullGlobalTransBlock(in_ch=C, H=H, W=W, num_layers=num_layers).to('cuda')

# --------------------------
# 前向计算
# --------------------------
x = x.to('cuda'); U = U.to('cuda'); S = S.to('cuda'); V = V.to('cuda'); G = G.to('cuda')
out = block(x, U, S, V, G)

print("输出:", out.shape)

# --------------------------
# torchinfo summary
# --------------------------
print("\n=== torchinfo summary ===")
summary(block,
        input_data=(x, U, S, V, G),
        col_names=["input_size", "output_size", "num_params", "mult_adds"])

# --------------------------
# thop FLOPs and params
# --------------------------
print("\n=== thop summary ===")
flops, params = profile(block, inputs=(x, U, S, V, G), verbose=False)
flops, params = clever_format([flops, params], "%.3f")
print(f"Parameters: {params}, FLOPs: {flops}")
