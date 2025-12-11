import math
import torch
from torch import nn
from torch.nn import functional as F
from .hdbscanV3ForClustering import hdbscan_fixed_channels_auto_remain_dynamic as cluster
from .SVDfactorizationForFindCategoryNum import RandomProjectionSVD as SVD

device = torch.device('cuda')


def upsample_bilinear(x_pool, factor):
    B, C, H_pool, W_pool = x_pool.shape
    H_up, W_up = H_pool * factor, W_pool * factor
    x_up = F.interpolate(x_pool, size=(H_up, W_up), mode='bilinear', align_corners=False)
    return x_up


def feature_similarity(x_cluster_att, noise):
    B, C1, H, W = x_cluster_att.shape
    B, C2, H, W = noise.shape
    if C1 != C2:
        proj = torch.nn.Conv2d(C2, C1, 1, bias=False).to(device)
        noise_proj = proj(noise)
    else:
        noise_proj = noise

    x_norm = F.normalize(x_cluster_att, p=2, dim=1)
    noise_norm = F.normalize(noise_proj, p=2, dim=1)
    sim_map = (x_norm * noise_norm).sum(dim=1, keepdim=True)
    sim_map = (sim_map + 1) / 2
    return sim_map


class ClusterAtt(nn.Module):
    def __init__(self, in_ch, poolingfactor):
        super(ClusterAtt, self).__init__()
        self.in_ch = in_ch
        self.poolingfactor = poolingfactor
        self.QKVprj = nn.Conv2d(in_ch, in_ch * 3, 1, groups=1)
        self.centering = nn.MaxPool2d(self.poolingfactor, self.poolingfactor)
        # 缓存不同通道数的 out_proj
        self.dynamic_out_proj = nn.ModuleDict()

    def _get_cluster_embed(self, cluster_key: str, feature: torch.Tensor, device) -> torch.Tensor:
        b, hw, ck = feature.shape
        return nn.Parameter(torch.randn(b, hw, ck, device=device) / math.sqrt(max(1, ck)))

    def forward(self, x):
        b, c, h, w = x.shape
        U_orig, S, V, k = SVD(x)
        x_flat = x.view(b, c, -1)
        x_proj = torch.bmm(V, x_flat).view(b, c, h, w)
        x_cluster_input = x_proj + x
        x_cluster, avg_r, avg_k = cluster(x_cluster_input, max(2, self.in_ch // k))
        if avg_k < 1:
            avg_k = 1
        avg_ck = self.in_ch // avg_k
        noise = x_cluster[:, :avg_ck, :, :]
        x_cluster_pooling = self.centering(x_cluster)

        qkv = self.QKVprj(x_cluster_pooling)
        q = qkv[:, :self.in_ch, :, :]
        k = qkv[:, self.in_ch: self.in_ch * 2, :, :]
        v = qkv[:, self.in_ch * 2: self.in_ch * 3, :, :]

        x_categories = []
        for i in range(1, avg_k + 1):
            start = avg_ck * i
            end = avg_ck * (i + 1)
            if start == c:
                continue
            x_k = x_cluster_pooling[:, start:end, :, :]
            h_now = x_k.shape[2]
            ck_now = x_k.shape[1]
            cluster_key = f"cluster_{i}_ck_{ck_now}"
            qk = q[:, start:end, :, :].view(b, h_now * h_now, -1)
            emb = self._get_cluster_embed(cluster_key, qk, device=x.device)
            qk = qk + emb
            kk = k[:, start:end, :, :].view(b, -1, h_now * h_now)
            vk = v[:, start:end, :, :].view(b, h_now * h_now, -1)
            attn_weights = torch.matmul(qk, kk) / (ck_now ** 0.5)
            attn_weights = F.softmax(attn_weights, dim=-1)
            out = torch.matmul(attn_weights, vk).view(b, -1, h_now, h_now) + x_k
            x_categories.append(out)

        if len(x_categories) == 0:
            x_cluster_pooling_att = x_cluster_pooling
        else:
            x_cluster_pooling_att = torch.cat(x_categories, dim=1)

        x_cluster_att = upsample_bilinear(x_cluster_pooling_att, self.poolingfactor)
        similar = feature_similarity(x_cluster_att, noise)
        x_cluster_att = (1 - similar) * x_cluster_att

        key = str(x_cluster_att.shape[1])
        if key not in self.dynamic_out_proj:
            self.dynamic_out_proj[key] = nn.Conv2d(
                x_cluster_att.shape[1], self.in_ch, 1, 1, bias=False
            ).to(x.device)

        x_cluster_att = self.dynamic_out_proj[key](x_cluster_att)
        return x_cluster_att, avg_k, noise


# --------------------------- 快速聚类注意力 ---------------------------

class OrthoConv1x1Soft(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels) * (1.0 / math.sqrt(max(1, in_channels))))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1)
        y = torch.einsum('oi, b i l -> b o l', self.weight, x_flat)
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
        self.center = nn.MaxPool2d(poolingfactor, poolingfactor)
        # 缓存 ClusterConv / QKVprj / out_proj
        self.dynamic_proj = nn.ModuleDict()
        self.dynamic_out_proj = nn.ModuleDict()

    def _get_cluster_embed(self, cluster_key: str, h: int, b: int, device) -> torch.Tensor:
        return nn.Parameter(torch.randn(b, h, h, device=device))

    def forward(self, x, avg_k, noise):
        b, c, h, w = x.shape
        key = str(avg_k)

        # ClusterConv
        if f"ClusterConv_{key}" not in self.dynamic_proj:
            self.dynamic_proj[f"ClusterConv_{key}"] = OrthoConv1x1Soft(c, avg_k).to(x.device)
        ClusterConv = self.dynamic_proj[f"ClusterConv_{key}"]

        # QKVprj
        if f"QKV_{key}" not in self.dynamic_proj:
            self.dynamic_proj[f"QKV_{key}"] = nn.Conv2d(avg_k, avg_k * 3, 1, 1, groups=avg_k).to(x.device)
        QKVprj = self.dynamic_proj[f"QKV_{key}"]

        x_pooling = self.center(x)
        x_pooling_cluster = ClusterConv(x_pooling)
        ortho_loss = ClusterConv.ortho_loss()

        qkv = QKVprj(x_pooling_cluster)
        q = qkv[:, :avg_k, :, :]
        k = qkv[:, avg_k: avg_k * 2, :, :]
        v = qkv[:, avg_k * 2: avg_k * 3, :, :]

        x_categories_att = torch.zeros(b, avg_k, h // self.poolingfactor, w // self.poolingfactor, device=x.device)
        for i in range(avg_k):
            x_k = x_pooling_cluster[:, i, :, :]
            emb = self._get_cluster_embed(f'cluster_{i}', h=h // self.poolingfactor, b=b, device=x.device)
            qk = q[:, i, :, :] + emb
            kk = k[:, i, :, :]
            vk = v[:, i, :, :]
            attn_weights = torch.matmul(qk, kk)
            attn_weights = F.softmax(attn_weights, dim=-1)
            out = torch.matmul(attn_weights, vk) + x_k
            x_categories_att[:, i, :, :] = out*(1-ortho_loss)

        x_cluster_att = upsample_bilinear(x_categories_att, self.poolingfactor).to(device)
        similar = feature_similarity(x_cluster_att, noise)
        x_cluster_att = (1 - similar) * x_cluster_att

        out_key = str(x_cluster_att.shape[1])
        if out_key not in self.dynamic_out_proj:
            self.dynamic_out_proj[out_key] = nn.Conv2d(x_cluster_att.shape[1], self.in_ch, 1, 1, bias=False).to(x.device)

        x_cluster_att = self.dynamic_out_proj[out_key](x_cluster_att)
        return x_cluster_att
