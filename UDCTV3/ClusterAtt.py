import math
import torch
from torch import nn
from torch.nn import functional as F
from .hdbscanV3ForClustering import hdbscan_fixed_channels_auto_remain_dynamic as cluster
from .SVDfactorizationForFindCategoryNum import RandomProjectionSVD as SVD

device = torch.device('cuda')

def upsample_bilinear(x_pool, factor):
    """
    用双线性插值恢复池化后的特征尺度
    Args:
        x_pool: Tensor, shape (B, C, H_pool, W_pool)
        factor: int, 池化时的缩放因子（stride/scale）
    Returns:
        x_up: Tensor, shape (B, C, H_pool*factor, W_pool*factor)
    """
    B, C, H_pool, W_pool = x_pool.shape
    H_up, W_up = H_pool * factor, W_pool * factor
    x_up = F.interpolate(x_pool, size=(H_up, W_up), mode='bilinear', align_corners=False)
    return x_up


def feature_similarity(x_cluster_att, noise):
    """
    计算两个特征的相似性，通道不同，GPU高效
    Args:
        x_cluster_att: (B, C1, H, W)
        noise: (B, C2, H, W)
    Returns:
        sim_map: (B, 1, H, W) 每个位置的相似度
    """
    B, C1, H, W = x_cluster_att.shape
    B, C2, H, W = noise.shape

    # 线性映射到相同通道
    # 这里用1x1卷积实现，可训练或者固定
    if C1 != C2:
        proj = torch.nn.Conv2d(C2, C1, 1, bias=False).to(device)
        noise_proj = proj(noise)
    else:
        noise_proj = noise

    # L2归一化
    x_norm = F.normalize(x_cluster_att, p=2, dim=1)
    noise_norm = F.normalize(noise_proj, p=2, dim=1)

    # 逐元素内积得到相似性 map
    sim_map = (x_norm * noise_norm).sum(dim=1, keepdim=True)  # (B, 1, H, W)
    sim_map = (sim_map + 1) / 2  # 归一化到 0~1
    return sim_map


class ClusterAtt(nn.Module):
    def __init__(self, in_ch, poolingfactor):
        super(ClusterAtt, self).__init__()
        self.in_ch = in_ch
        self.poolingfactor = poolingfactor
        self.QKVprj = nn.Conv2d(in_ch, in_ch*3, 1, groups=1)  # 改成 groups=1
        self.centering = nn.MaxPool2d(self.poolingfactor, self.poolingfactor)
        self.out_proj = None

    def _get_cluster_embed(self, cluster_key: str, feature: torch.Tensor, device) -> torch.Tensor:
        b, hw, ck = feature.shape
        return nn.Parameter(torch.randn(b, hw, ck, device=device) / math.sqrt(max(1, ck)))

    def forward(self, x):
        b, c, h, w = x.shape
        U_orig, S, V, k = SVD(x)
        # 展平空间维度： (b, c, h*w)
        x_flat = x.view(b, c, -1)
        # V 作用在通道维度上： (b, c, c) @ (b, c, hw) → (b, c, hw)
        x_proj = torch.bmm(V, x_flat)
        # 恢复空间维度： (b, c, h, w)
        x_proj = x_proj.view(b, c, h, w)
        # 和原始 x 相加，结果再送入 cluster
        x_cluster_input = x_proj + x  # (b, c, h, w)
        x_cluster, avg_r, avg_k = cluster(x_cluster_input, max(2,self.in_ch // k))
        if avg_k < 1:
            avg_k = 1
        avg_ck = self.in_ch // avg_k
        noise = x_cluster[:, :avg_ck, :, :]
        x_cluster_pooling = self.centering(x_cluster)

        # QKV
        qkv = self.QKVprj(x_cluster_pooling)
        q = qkv[:, :self.in_ch, :, :]
        k = qkv[:, self.in_ch: self.in_ch*2, :, :]
        v = qkv[:, self.in_ch*2: self.in_ch*3, :, :]

        # 对每个 cluster 做池化并加上可学习嵌入

        x_categories = []

        for i in range(avg_k+1):
            start = avg_ck * i
            end = avg_ck * (i + 1)
            if i == 0:
                continue
            elif start==c:
                pass
            else:
                x_k = x_cluster_pooling[:, start:end, :, :]
                h_now = x_k.shape[2]
                ck_now = x_k.shape[1]
                # 获取可学习嵌入
                cluster_key = f"cluster_{i}_ck_{ck_now}"
                qk = q[:, start:end, :, :].view(b, h_now * h_now, -1)
                emb = self._get_cluster_embed(cluster_key, qk, device=x.device)
            
                qk = qk + emb  # b, hw, ck
                kk = k[:, start:end, :, :].view(b, -1,h_now*h_now) # b, ck, hw
                vk = v[:, start:end, :, :].view(b,h_now*h_now,-1) # b, hw, ck

                # 1. Attention 权重
                attn_weights = torch.matmul(qk, kk) / (ck_now ** 0.5)  # (b, hw, hw)
                # 2. Softmax
                attn_weights = F.softmax(attn_weights, dim=-1)
                # 3. 输出
                out = torch.matmul(attn_weights, vk).view(b,-1,h_now,h_now) + x_k  # (b, hw, ck) +
                x_categories.append(out)
        if len(x_categories) == 1 or len(x_categories) == 0:
            # 若没有任何 cluster，回退到输入特征 x_cluster_pooling，避免 0 通道
            x_cluster_pooling_att = x_cluster_pooling
        else:
            x_cluster_pooling_att = torch.cat(x_categories, dim=1)
        x_cluster_att = upsample_bilinear(x_cluster_pooling_att, self.poolingfactor)
        similar = feature_similarity(x_cluster_att, noise)
        x_cluster_att = (1-similar)*x_cluster_att
        # 动态创建 out_proj，支持变化的通道数
        if self.out_proj is None or self.out_proj.in_channels != x_cluster_att.shape[1]:
            self.out_proj = nn.Conv2d(x_cluster_att.shape[1], self.in_ch,1, 1, bias=False).to(x_cluster_att.device)

        x_cluster_att = self.out_proj(x_cluster_att)
        return x_cluster_att, avg_k, noise

"-------------------------------------快速聚类注意力--------------------------------------------------------"


class OrthoConv1x1Soft(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels) * (1.0 / math.sqrt(max(1, in_channels))))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        # matmul expects (out, in) @ (B, in, HW) -> (out, B, HW)
        x_flat = x.view(B, C, -1)                         # (B, C, HW)
        # we'll perform per-batch matmul via torch.einsum for correct broadcasting
        # W: (out, in)
        y = torch.einsum('oi, b i l -> b o l', self.weight, x_flat)  # (B, out, HW)
        y = y.view(B, -1, H, W)
        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1, 1)
        return y

    def ortho_loss(self) -> torch.Tensor:
        W = self.weight
        WWT = W @ W.T
        I = torch.eye(WWT.size(0), device=WWT.device, dtype=WWT.dtype)
        return F.mse_loss(WWT, I)

class FastclusterAtt(nn.Module):
    def __init__(self, in_ch, poolingfactor=4):
        super(FastclusterAtt, self).__init__()
        self.in_ch = in_ch
        self.poolingfactor = poolingfactor
        self.ClusterConv = None
        self.center = nn.MaxPool2d(poolingfactor,poolingfactor)
        self.QKVprj = None
        self.out_proj = None

    def _get_cluster_embed(self, cluster_key: str, h:int, b: int, device) -> torch.Tensor:
        
        return nn.Parameter(torch.randn(b, h, h, device=device))

    def forward(self, x, avg_k, noise):
        b, c, h, w = x.shape
        self.ClusterConv = OrthoConv1x1Soft(c,avg_k).to(x.device)
        self.QKVprj = nn.Conv2d(avg_k,avg_k*3,1,1,groups=avg_k).to(x.device)

        x_pooling = self.center(x)
        x_pooling_cluster = self.ClusterConv(x_pooling)  #b, avg_k, h,w

        # 计算 ortho_loss
        ortho_loss = self.ClusterConv.ortho_loss()

        qkv = self.QKVprj(x_pooling_cluster)  # b,3*avg_k,h,w
        q = qkv[:, :avg_k, :, :]
        k = qkv[:, avg_k: avg_k * 2, :, :]
        v = qkv[:, avg_k * 2: avg_k * 3, :, :]

        x_categories_att = torch.zeros(b,avg_k, h//self.poolingfactor, w//self.poolingfactor)
        for i in range(avg_k):

            x_k = x_pooling_cluster[:, i, :, :]

            h_now = x_k.shape[1]
            ck_now = 1

            # 获取可学习嵌入
            cluster_key = f'cluster_{i}'
            emb = self._get_cluster_embed(cluster_key, h=h//self.poolingfactor, b=b, device=x.device)
            qk = q[:, i, :, :] + emb  # b, h,w
            kk = k[:, i, :, :]  # b, h,w
            vk = v[:, i, :, :]  # b, h, w

            # 1. Attention 权重
            attn_weights = torch.matmul(qk, kk)  # (b, h, w)
            # 2. Softmax
            attn_weights = F.softmax(attn_weights, dim=-1)
            # 3. 输出
            out = torch.matmul(attn_weights, vk) + x_k  # (b,h,w)
            x_categories_att[:,i,:,:] = out*(1-ortho_loss) #

        x_cluster_att = upsample_bilinear(x_categories_att, self.poolingfactor).to(device)  # b, avg_k, h, w
        similar = feature_similarity(x_cluster_att, noise)
        x_cluster_att = (1 - similar) * x_cluster_att
        # 动态创建 out_proj，支持变化���通道数
        if self.out_proj is None or self.out_proj.in_channels != x_cluster_att.shape[1]:
            self.out_proj = nn.Conv2d(x_cluster_att.shape[1], self.in_ch,1, 1, bias=False).to(x_cluster_att.device)
        x_cluster_att = self.out_proj(x_cluster_att)
        return x_cluster_att
