import torch
import torch.nn as nn
from .GoLTV2 import GoLTransBlockLayer

# === Patch Embedding ===
class PatchEmbedding(nn.Module):
    def __init__(self, in_ch, embed_dim, patch_size):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H/patch, W/patch)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # (B,H,W,C)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # (B,C,H,W)
        return x


# === Patch Merging ===
class PatchMerging(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.reduction = nn.Conv2d(in_ch, out_ch,
                                   kernel_size=2, stride=2)
        self.norm = nn.LayerNorm(out_ch)

    def forward(self, x):
        x = self.reduction(x)  # (B, out_ch, H/2, W/2)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


# === Patch Expanding ===
class PatchExpanding(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.expand = nn.ConvTranspose2d(in_ch, out_ch,
                                         kernel_size=2, stride=2)
        self.norm = nn.LayerNorm(out_ch)

    def forward(self, x):
        x = self.expand(x)  # (B, out_ch, H*2, W*2)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x

class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

# === UNet 风格的 GoLTransformer ===
class GoLTransformerUNet(nn.Module):
    def __init__(self, in_ch=3, num_classes=2,imgsize=512,
                 embed_dim=32,patch_size=2,num_layers=[2,2,2,2],
                 mlp_method = 'mlp',):
        super().__init__()
        # patch embedding
        self.patch_embed = PatchEmbedding(in_ch, embed_dim, patch_size)

        # encoder
        self.stage1 = GoLTransBlockLayer(in_ch=embed_dim*1, H=imgsize//patch_size, W=imgsize//patch_size, num_layers=num_layers[0],mlp_hidden_factor=2,mlp_method =mlp_method,
                                   local_kwargs={
                                       'layer': 1,
                                       'clusteratt_num': 1,
                                       'fastclusteratt_num': 2,
                                       'clusteratt_downscale': 32,
                                       'clusteratt_num_heads': 1,
                                       'clusteratt_min_cluster_size': 2,
                                       'fastclusteratt_num_heads': 1,
                                       'fastclusteratt_cluster_stride': 32,
                                       'fastclusteratt_groups': 4
                                   },
                                   global_kwargs={
                                       'num_layers': 2
                                   })
        self.merge1 = PatchMerging(embed_dim, embed_dim*2)

        self.stage2 = GoLTransBlockLayer(in_ch=embed_dim*2, H=(imgsize//2)//patch_size, W=(imgsize//2)//patch_size, num_layers=num_layers[1],mlp_hidden_factor=2,mlp_method =mlp_method,
                                   local_kwargs={
                                       'layer': 1,
                                       'clusteratt_num': 1,
                                       'clusteratt_downscale': 16,
                                       'clusteratt_num_heads': 1,
                                       'clusteratt_min_cluster_size': 2,
                                       'fastclusteratt_num': 2,
                                       'fastclusteratt_num_heads': 1,
                                       'fastclusteratt_cluster_stride': 16,
                                       'fastclusteratt_groups': 4
                                   },
                                   global_kwargs={
                                       'num_layers': 2
                                   })
        self.merge2 = PatchMerging(embed_dim*2, embed_dim*4)

        self.stage3 = GoLTransBlockLayer(in_ch=embed_dim*4, H=(imgsize//4)//patch_size, W=(imgsize//4)//patch_size, num_layers=num_layers[2],mlp_hidden_factor=2,mlp_method =mlp_method,
                                   local_kwargs={
                                       'layer': 1,
                                       'clusteratt_num': 1,
                                       'fastclusteratt_num': 2,
                                       'clusteratt_downscale': 8,
                                       'clusteratt_num_heads': 1,
                                       'clusteratt_min_cluster_size': 2,
                                       'fastclusteratt_num_heads': 1,
                                       'fastclusteratt_cluster_stride': 8,
                                       'fastclusteratt_groups': 4
                                   },
                                   global_kwargs={
                                       'num_layers': 2
                                   })
        self.merge3 = PatchMerging(embed_dim*4, embed_dim*8)

        self.stage4 = GoLTransBlockLayer(in_ch=embed_dim*8, H=(imgsize//8)//patch_size, W=(imgsize//8)//patch_size, num_layers=num_layers[3],mlp_hidden_factor=2,mlp_method =mlp_method,
                                   local_kwargs={
                                       'layer': 1,
                                       'clusteratt_num': 1,
                                       'fastclusteratt_num': 2,
                                       'clusteratt_downscale': 4,
                                       'clusteratt_num_heads': 1,
                                       'clusteratt_min_cluster_size': 2,
                                       'fastclusteratt_num_heads': 1,
                                       'fastclusteratt_cluster_stride': 4,
                                       'fastclusteratt_groups': 4
                                   },
                                   global_kwargs={
                                       'num_layers': 2
                                   })

        self.merge4 = PatchMerging(embed_dim * 8, embed_dim * 16)

        self.bottleneck = GoLTransBlockLayer(in_ch=embed_dim*16, H=(imgsize//16)//patch_size, W=(imgsize//16)//patch_size, num_layers=1, mlp_hidden_factor=1,mlp_method =mlp_method,
                                         local_kwargs={
                                             'layer': 1,
                                             'clusteratt_num': 1,
                                             'fastclusteratt_num': 2,
                                             'clusteratt_downscale': 4,
                                             'clusteratt_num_heads': 1,
                                             'clusteratt_min_cluster_size': 2,
                                             'fastclusteratt_num_heads': 1,
                                             'fastclusteratt_cluster_stride': 4,
                                             'fastclusteratt_groups': 4
                                         },
                                         global_kwargs={
                                             'num_layers': 2
                                         })
        # decoder
        self.expand4 = PatchExpanding(embed_dim * 16, embed_dim * 8)
        self.conv4 = conv_block(embed_dim * 16, embed_dim * 8)
        self.dec4 = GoLTransBlockLayer(in_ch=embed_dim*8, H=(imgsize//8)//patch_size, W=(imgsize//8)//patch_size, num_layers=num_layers[3], mlp_hidden_factor=2,mlp_method =mlp_method,
                                       local_kwargs={
                                           'layer': 1,
                                           'clusteratt_num': 1,
                                           'fastclusteratt_num': 2,
                                           'clusteratt_downscale': 8,
                                           'clusteratt_num_heads': 1,
                                           'clusteratt_min_cluster_size': 2,
                                           'fastclusteratt_num_heads': 1,
                                           'fastclusteratt_cluster_stride': 8,
                                           'fastclusteratt_groups': 4
                                       },
                                       global_kwargs={
                                           'num_layers': 2
                                       })

        self.expand3 = PatchExpanding(embed_dim*8, embed_dim*4)
        self.conv3 = conv_block(embed_dim*8, embed_dim*4)
        self.dec3 = GoLTransBlockLayer(in_ch=embed_dim*4, H=(imgsize//4)//patch_size, W=(imgsize//4)//patch_size, num_layers=num_layers[2],mlp_hidden_factor=2,mlp_method =mlp_method,
                                   local_kwargs={
                                       'layer': 1,
                                       'clusteratt_num': 1,
                                       'fastclusteratt_num': 2,
                                       'clusteratt_downscale': 8,
                                       'clusteratt_num_heads': 1,
                                       'clusteratt_min_cluster_size': 2,
                                       'fastclusteratt_num_heads': 1,
                                       'fastclusteratt_cluster_stride': 8,
                                       'fastclusteratt_groups': 4
                                   },
                                   global_kwargs={
                                       'num_layers': 2
                                   })

        self.expand2 = PatchExpanding(embed_dim*4, embed_dim*2)
        self.conv2 = conv_block(embed_dim*4, embed_dim*2)
        self.dec2 = GoLTransBlockLayer(in_ch=embed_dim*2, H=(imgsize//2)//patch_size, W=(imgsize//2)//patch_size, num_layers=num_layers[1],mlp_hidden_factor=1,mlp_method =mlp_method,
                                   local_kwargs={
                                       'layer': 1,
                                       'clusteratt_num': 1,
                                       'fastclusteratt_num': 2,
                                       'clusteratt_downscale': 8,
                                       'clusteratt_num_heads': 1,
                                       'clusteratt_min_cluster_size': 2,
                                       'fastclusteratt_num_heads': 1,
                                       'fastclusteratt_cluster_stride': 8,
                                       'fastclusteratt_groups': 4
                                   },
                                   global_kwargs={
                                       'num_layers': 2
                                   })

        self.expand1 = PatchExpanding(embed_dim*2, embed_dim)
        self.conv1 = conv_block(embed_dim*2, embed_dim)
        self.dec1 = GoLTransBlockLayer(in_ch=embed_dim*1, H=(imgsize//1)//patch_size, W=(imgsize//1)//patch_size, num_layers=num_layers[0],mlp_hidden_factor=2,mlp_method =mlp_method,
                                   local_kwargs={
                                       'layer': 1,
                                       'clusteratt_num': 1,
                                       'fastclusteratt_num': 2,
                                       'clusteratt_downscale': 16,
                                       'clusteratt_num_heads': 1,
                                       'clusteratt_min_cluster_size': 2,
                                       'fastclusteratt_num_heads': 1,
                                       'fastclusteratt_cluster_stride': 16,
                                       'fastclusteratt_groups': 4
                                   },
                                   global_kwargs={
                                       'num_layers': 2
                                   })
        self.upsample = nn.ConvTranspose2d(embed_dim, embed_dim,patch_size,patch_size)
        # head
        self.head = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, x):
        # encoder
        x = self.patch_embed(x)   # B,embed_dim,H/4,W/4
        x1 = x
        skip1,local1 = self.stage1(x)
        x = self.merge1(skip1)

        x2 = x
        skip2,local2 = self.stage2(x)
        x = self.merge2(skip2)

        x3 = x
        skip3,local3 = self.stage3(x)
        x = self.merge3(skip3)

        x4 = x
        skip4,local4 = self.stage4(x)
        x = self.merge4(skip4)

        x,_ = self.bottleneck(x)  # bottleneck

        # decoder
        x = self.expand4(x)
        x = torch.cat([x, skip4], dim=1)
        x = self.conv4(x)
        x,_ = self.dec4(x)

        x = self.expand3(x)
        x = torch.cat([x, skip3], dim=1)
        x = self.conv3(x)
        x,_ = self.dec3(x)

        x = self.expand2(x)
        x = torch.cat([x, skip2], dim=1)
        x = self.conv2(x)
        x,_ = self.dec2(x)

        x = self.expand1(x)
        x = torch.cat([x,skip1], dim=1)
        x = self.conv1(x)
        x,_ = self.dec1(x)

        x = self.upsample(x)
        out = self.head(x)
        return out
if __name__ == "__main__":
    from torchinfo import summary
    from thop import profile, clever_format
    # 输入图像大小 (B, C, H, W)，这里取 batch=1，输入大小 3×512×512
    model = GoLTransformerUNet(in_ch=3, num_classes=2, embed_dim=64).cuda()

    x = torch.randn(1, 3, 512, 512).cuda()
    out = model(x)

    print("输入:", x.shape)
    print("输出:", out.shape)  # (1, num_classes, H, W)


    # -----------------------------
    # thop FLOPs / Params
    # -----------------------------
    print("\n=== thop summary ===")
    flops, params = profile(model, inputs=(x,), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")
    print(f"Parameters: {params}, FLOPs: {flops}")

    # 打印网络结构
    summary(model, (1,3, 512, 512))
