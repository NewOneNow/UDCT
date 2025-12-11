import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MLP(nn.Module):
    """简单 MLP，隐藏层扩大 factor"""
    def __init__(self, dim, hidden_factor=2, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_factor * dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_factor * dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, H*W, C)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class MHSAWithMLPBlock(nn.Module):
    """2D 多头自注意力 + LN + MLP"""
    def __init__(self, in_ch, num_heads=4, mlp_hidden_factor=2, dropout=0.0):
        super().__init__()
        assert in_ch % num_heads == 0

        self.in_ch = in_ch
        self.num_heads = num_heads
        self.head_dim = in_ch // num_heads

        # LN 作用在 channel 维度上
        self.ln1 = nn.LayerNorm(in_ch)
        self.ln2 = nn.LayerNorm(in_ch)

        # 1x1 Conv 实现 QKV
        self.qkv = nn.Conv2d(in_ch, in_ch * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(in_ch, in_ch, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout)

        # MLP
        self.mlp = MLP(in_ch, hidden_factor=mlp_hidden_factor, dropout=dropout)

    def forward(self, x):
        B, C, H, W = x.shape

        # -------------------- MHSA --------------------
        # LN
        x_ln = self.ln1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # (B, C, H, W)

        # QKV
        qkv = self.qkv(x_ln)
        q, k, v = torch.chunk(qkv, 3, dim=1)

        q = q.view(B, self.num_heads, self.head_dim, H * W)
        k = k.view(B, self.num_heads, self.head_dim, H * W)
        v = v.view(B, self.num_heads, self.head_dim, H * W)

        attn_scores = torch.matmul(q.transpose(-2, -1), k) / math.sqrt(self.head_dim)
        attn = F.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v.transpose(-2, -1))
        out = out.transpose(-2, -1).contiguous().view(B, C, H, W)
        out = self.proj(out)

        # 残差
        x = x + out

        # -------------------- MLP --------------------
        x_ln2 = self.ln2(x.permute(0, 2, 3, 1))  # (B, H, W, C)
        x_flat = x_ln2.view(B, H*W, C)
        x_mlp = self.mlp(x_flat)
        x_mlp = x_mlp.view(B, H, W, C).permute(0, 3, 1, 2)

        # 残差
        x = x + x_mlp
        return x


# -----------------------------
# 测试
# -----------------------------
if __name__ == "__main__":
    B, C, H, W = 1, 1024,64,64  # batch=2, 序列长度64, embedding dim=128
    x = torch.randn(B, C, H, W)

    mhsa = MHSAWithMLPBlock(in_ch=C, num_heads=1)
    out = mhsa(x)

    print("输入 x:", x.shape)
    print("输出 out:", out.shape)

    # -----------------------------
    # 统计信息
    # -----------------------------
    from torchinfo import summary
    from thop import profile, clever_format
    from fvcore.nn import FlopCountAnalysis, parameter_count

    mhsa.eval()

    # -----------------------------
    # 1. torchinfo
    # -----------------------------
    print("=== torchinfo summary ===")
    summary(mhsa, input_data=x,
            col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"])

    # -----------------------------
    # 2. thop
    # -----------------------------
    print("\n=== thop summary ===")
    flops, params = profile(mhsa, inputs=(x,))
    flops, params = clever_format([flops, params], "%.3f")
    print(f"Parameters: {params}, FLOPs: {flops}")
