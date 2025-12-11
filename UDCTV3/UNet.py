import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """UNet 双卷积模块"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        """
        完整 UNet
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            features: 编码器每层通道数
        """
        super().__init__()

        # 编码器
        self.encoders = nn.ModuleList()
        in_ch = in_channels
        for feat in features:
            self.encoders.append(DoubleConv(in_ch, feat))
            in_ch = feat
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 解码器
        self.decoders = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        rev_features = features[::-1]
        for i in range(len(rev_features) - 1):
            self.decoders.append(
                nn.ConvTranspose2d(rev_features[i], rev_features[i+1], kernel_size=2, stride=2)
            )
            self.up_convs.append(DoubleConv(rev_features[i], rev_features[i+1]))

        # 最终卷积
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # 编码器
        for enc in self.encoders:
            x = enc(x)
            skip_connections.append(x)
            x = self.pool(x)

        skip_connections = skip_connections[::-1]

        # 解码器
        for i in range(len(self.decoders)):
            x = self.decoders[i](x)
            skip_connection = skip_connections[i+1]

            # 尺寸对齐
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:])

            x = torch.cat([skip_connection, x], dim=1)
            x = self.up_convs[i](x)

        x = self.final_conv(x)
        return x


# -------------------------
# quick test
# -------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, C, H, W = 2, 3, 512, 512
    x = torch.randn(B, C, H, W, device=device)

    model = UNet(in_channels=3, out_channels=2).to(device)
    y = model(x)
    print("Output shape:", y.shape)
