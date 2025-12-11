import torch
import torch.nn as nn
import torch.nn.functional as F
from .TestClusterTransformerblock import FullClusterTransformerBlock

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class UNetOriginalWithTransformer(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, transformer_kwargs=None):
        super().__init__()
        if transformer_kwargs is None:
            transformer_kwargs = {
                "mlp_ratio": 0.50,
                "drop": 0.1,
                "full_layer": 1,
                "clu_head": 1,
                "fastclu_head": 2,
                "fastclu_layer": 3,
                "residual_scale": 1.0
            }

        # UNet 编码器通道设置
        features = [64, 128, 256, 512, 1024]

        # 编码器 + Transformer
        self.encoders = nn.ModuleList()
        self.transformers = nn.ModuleList()
        curr_in_channels = in_channels
        for feat in features:
            self.encoders.append(DoubleConv(curr_in_channels, feat))
            self.transformers.append(FullClusterTransformerBlock(in_ch=feat, cluatt_poolingfactor = 1024//feat, fastcluatt_poolingfactor=1024//feat, **transformer_kwargs))
            curr_in_channels = feat
        self.pool = nn.MaxPool2d(2, 2)

        # 解码器
        self.up_transpose = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        rev_feats = features[:-1][::-1]  # [512,256,128,64]
        for i in range(len(rev_feats)):
            self.up_transpose.append(nn.ConvTranspose2d(features[-(i+1)], rev_feats[i], kernel_size=2, stride=2))
            self.up_convs.append(DoubleConv(rev_feats[i]*2, rev_feats[i]))

        # 输出层
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # 编码器路径
        for i, (conv, trans) in enumerate(zip(self.encoders, self.transformers)):
            x = conv(x)
            x1 = trans(x)
            x = x + x1
            skip_connections.append(x)
            if i != len(self.encoders) - 1:  # 最后一层不池化
                x = self.pool(x)

        # 解码器路径
        skip_connections = skip_connections[::-1]
        for i in range(len(self.up_transpose)):
            x = self.up_transpose[i](x)
            if x.shape[2:] != skip_connections[i+1].shape[2:]:
                x = F.interpolate(x, size=skip_connections[i+1].shape[2:])
            x = torch.cat([skip_connections[i+1], x], dim=1)
            x = self.up_convs[i](x)
        return self.final_conv(x)

