import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention2D(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention2D, self).__init__()
        self.in_channels = in_channels
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # 学习缩放因子

    def forward(self, x):
        b, c, h, w = x.size()
        N = h * w  # 空间维度大小

        # 1. 生成 Q, K, V
        proj_query = self.query_conv(x).view(b, -1, N)              # (b, c//8, hw)
        proj_key = self.key_conv(x).view(b, -1, N)                  # (b, c//8, hw)
        proj_value = self.value_conv(x).view(b, -1, N)             # (b, c, hw)

        # 2. 计算注意力矩阵 (b, hw, hw)
        attn = torch.bmm(proj_query.permute(0, 2, 1), proj_key)     # (b, hw, hw)
        attn = F.softmax(attn / (proj_query.size(1) ** 0.5), dim=-1)

        # 3. 应用注意力权重 (b, c, hw)
        out = torch.bmm(proj_value, attn.permute(0, 2, 1))          # (b, c, hw)

        # 4. 恢复空间维度 & 残差连接
        out = out.view(b, c, h, w)
        out = self.gamma * out + x

        return out

# 测试
if __name__ == "__main__":
    from thop import profile

    model = SelfAttention2D(32).cuda()
    dummy_input = torch.randn(2, 32, 128, 128).cuda()
    flops, params = profile(model, inputs=(dummy_input,))

    print(f"FLOPs: {flops / 1e9:.4f} G, Params: {params / 1e6:.4f} M")


    """y = sa(x)
    print(y.shape)  # 结果 (2, 64, 32, 32)"""
