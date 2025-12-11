import math
import numpy as np
import torch.nn as nn
import torch
from scipy.linalg import toeplitz
from sklearn.cluster import HDBSCAN
from torch.nn import functional as F, Parameter

#from ValueToCSV import save_feature_heatmap

device = torch.device("cuda")
def compute_knee_point(s: torch.Tensor) -> int:
    """
    计算奇异值序列 s 的“拐点”位置（返回 k，1-based best rank）
    使用二阶差分的最大值作为拐点估计。
    输入 s: 1D torch tensor (非负，降序)
    返回 K:int >=1
    """
    if s.numel() == 0:
        return 1
    cum = torch.cumsum(s, dim=0)
    if cum[-1] == 0:
        return 1
    cum_norm = cum / cum[-1]
    if cum_norm.numel() < 3:
        return 1

    diff2 = cum_norm[2:] - 2 * cum_norm[1:-1] + cum_norm[:-2]
    knee = int(torch.argmax(diff2).item() + 1)  # +1 对应原序列的索引位置
    return max(1, knee)


def SVD(x: torch.Tensor):
    """
    Batch Randomized SVD (approximate, accelerated)
    输入:
        x: (b, c, h, w)
        h_proj: 随机投影维度 (压缩 hw)
        rank: 保留秩 k (k << min(h_proj, c))
    返回:
        U: (b, h_proj, k)
        S: (b, C)
        V: (b, c, k)
    """
    b, c, h, w = x.shape
    hw = h * w
    device = x.device

    # 1. 随机投影 (hw -> h)
    G = torch.randn(hw, h, device=device) / (h ** 0.5)
    X_proj = x.reshape(b, c, hw) @ G  # (b, c, h)
    X_proj = X_proj.transpose(1, 2)  # (b, h, c)
    eps = 1e-6
    X_proj = X_proj + eps * torch.randn_like(X_proj)

    # 2. Batch SVD
    # torch.linalg.svd 支持 batched
        # 2. Batch SVD
    try:
        U, S, Vh = torch.linalg.svd(X_proj, full_matrices=False)
    except RuntimeError:                      # 捕获奇异/数值问题
        # 退化：返回随机正交基 + 零奇异值
        min_dim = min(X_proj.shape[-2:])      # min(h, c)
        U = torch.eye(X_proj.shape[-2], min_dim, device=device).unsqueeze(0).expand(b, -1, -1)
        S = torch.zeros(b, min_dim, device=device)
        Vh = torch.eye(min_dim, X_proj.shape[-1], device=device).unsqueeze(0).expand(b, -1, -1)

    # 3. 截断 (rank k)
    k = compute_knee_point(S.mean(0))
    U_k = U[:, :, :k]
    S_k = S[:, :k]
    V_k = Vh[:, :k, :].transpose(-1, -2)
    return U_k, S_k, V_k, G

def lSVD(x: torch.Tensor, eps: float = 1e-6):
    """
    手写 Batch SVD (近似, GPU friendly)，使用二阶差分计算 k
    输入:
        x: (b, c, h, w)
    返回:
        U_k: (b, h, k)
        S: (b, min(h,c))
        V_k: (b, c, k)
        G: 随机投影矩阵
    """
    b, c, h, w = x.shape
    hw = h * w
    device = x.device

    # 1. 随机投影 (hw -> h)
    G = torch.randn(hw, h, device=device) / (h ** 0.5)
    X_proj = x.reshape(b, c, hw) @ G        # (b, c, h)
    X_proj = X_proj.transpose(1, 2)         # (b, h, c)
    X_proj = X_proj + eps * torch.randn_like(X_proj)

    min_hc = min(h, c)
    if h <= c:
        # 2. 构造 Gram 矩阵 XX^T 并求特征分解
        XXT = X_proj @ X_proj.transpose(1, 2)   # (b, h, h)
        S_eig, U = torch.linalg.eigh(XXT)       # 升序排列
        S_eig = torch.sqrt(torch.clamp(S_eig, min=0.0))
        # 翻转成降序
        S_eig = S_eig.flip(dims=[1])
        U = U.flip(dims=[2])
        # 计算 V
        S_mat = S_eig[:, :min_hc].unsqueeze(1)
        V_k = X_proj.transpose(1, 2) @ U[:, :, :min_hc] / (S_mat + eps)  # (b, c, k)
        U_k = U[:, :, :min_hc]
    else:
        # h > c, 用 X^T X
        XTX = X_proj.transpose(1, 2) @ X_proj   # (b, c, c)
        S_eig, V = torch.linalg.eigh(XTX)
        S_eig = torch.sqrt(torch.clamp(S_eig, min=0.0))
        # 降序
        S_eig = S_eig.flip(dims=[1])
        V = V.flip(dims=[2])
        S_mat = S_eig[:, :min_hc].unsqueeze(1)
        U_k = X_proj @ V[:, :, :min_hc] / (S_mat + eps)
        V_k = V[:, :, :min_hc]

    # 3. 二阶差分计算 k
    S_mean = S_eig.mean(0)
    k = compute_knee_point(S_mean)

    # 4. 截断
    U_k = U_k[:, :, :k]
    V_k = V_k[:, :, :k]
    return U_k, S_eig, V_k, G


class GCNConv(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, X, adj):
        # X: (B, N, C_in)  adj: (N, N)
        B, N, C = X.shape
        outputs = []
        adj = adj.to_dense()
        for i in range(B):
            support = torch.mm(X[i], self.weight)      # (N, C_out)
            out = torch.spmm(adj, support)            # (N, C_out)
            if self.bias is not None:
                out = out + self.bias
            outputs.append(out)
        return torch.stack(outputs, dim=0)            # (B, N, C_out)


# -----------------------------
# GraphExtractor (轻量化)
# -----------------------------
class GCNExtractor(nn.Module):
    def __init__(self, in_channels, knn=8):
        super().__init__()
        c_reduced = max(8, in_channels // 4)
        self.c = in_channels
        self.c_reduced = c_reduced
        self.knn = knn

        self.reduce = nn.Conv1d(in_channels, c_reduced, kernel_size=1, groups=c_reduced,bias=False)
        self.gcn = GCNConv(c_reduced, c_reduced)
        self.expand = nn.Conv1d(c_reduced, in_channels, kernel_size=1, groups=c_reduced,bias=False)

    @staticmethod
    def build_knn_adj(Uk, knn=8):
        # Uk: (H, K)
        device = Uk.device
        H, K = Uk.shape
        Uk = F.normalize(Uk, dim=-1)
        sim = Uk @ Uk.t()  # (H,H)
        k_use = min(knn, H)
        vals, idx = torch.topk(sim, k_use, dim=-1)
        row = torch.arange(H, device=device).unsqueeze(1).repeat(1, k_use).reshape(-1)
        col = idx.reshape(-1)
        edge_index = torch.stack([row, col], dim=0)  # (2, H*k_use)
        edge_weight = vals.reshape(-1)

        # 构造归一化稀疏邻接矩阵
        A = torch.sparse_coo_tensor(edge_index, edge_weight, (H, H), device=device)
        deg = torch.sparse.sum(A, dim=1).to_dense()
        deg_inv_sqrt = deg.pow(-0.5)
        row, col = edge_index
        vals_norm = edge_weight * deg_inv_sqrt[row] * deg_inv_sqrt[col]
        adj = torch.sparse_coo_tensor(edge_index, vals_norm, (H, H), device=device).coalesce()
        return adj

    def forward(self, x, Uk):
        # x: (B, C, H, W), Uk: (B, H, K)
        B, C, H, W = x.shape
        X_node = x.mean(dim=3)                       # (B, C, H)
        X_reduced = self.reduce(X_node)             # (B, C_red, H)
        X_reduced = X_reduced.permute(0, 2, 1)      # (B, H, C_red)

        out_nodes = []
        for i in range(B):
            adj = self.build_knn_adj(Uk[i], self.knn).to(x.device)
            Xi = X_reduced[i:i+1]                     # (1, H, C_red)
            out = F.relu(self.gcn(Xi, adj))          # (1, H, C_red)
            out_nodes.append(out[0])

        out_nodes = torch.stack(out_nodes, dim=0)    # (B, H, C_red)
        out_nodes = out_nodes.permute(0, 2, 1)       # (B, C_red, H)
        out_nodes = self.expand(out_nodes)           # (B, C, H)
        out_map = out_nodes.unsqueeze(-1).expand(-1, -1, -1, W)  # (B, C, H, W)
        return out_map

class globalatt(nn.Module):
    def __init__(self, in_ch, H, W, use_residual=True):
        super().__init__()
        self.in_ch = in_ch
        self.H, self.W = H, W
        self.hw = H * W
        self.use_residual = use_residual

        # ---- 行列注意力所需 ----
        self.w_row = nn.Parameter(torch.randn(H))
        self.w_col = nn.Parameter(torch.randn(W))

        # ---- channel attention ----
        # 无需额外参数，直接用 V 和 S


        # ---- fusion 权重 MLP ----
        # 输入：S (B,k) + S_mean (B,1) + topo_energy (B,1) => (B, k+2)
        self.boast = nn.Parameter(torch.randn(H+W+in_ch))
        self.gcn = GCNExtractor(in_ch)
        # residual scale
        if use_residual:
            self.res_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, U, S, V, G):
        """
        x: (B,C,H,W)
        U: (B, h_proj, k)
        S: (B, k)
        V: (B, C, k)
        G: (HW, h_proj)
        topo: (B,C,H,W)
        """
        device = x.device
        B, C, H, W = x.shape

        # 确保所有输入在同一个 device
        U = U.to(device)
        S = S.to(device)
        V = V.to(device)
        G = G.to(device)


        topo = self.gcn(x, U)
        self.w_row = self.w_row.to(device)
        self.w_col = self.w_col.to(device)
        w_row = self.w_row.unsqueeze(0).expand(B, H)  # (B,H)
        w_col = self.w_col.unsqueeze(0).expand(B, W)  # (B,W)
        # ------------------ 1) U backprojection ------------------
        Gt = G.T  # (h_proj, HW)

        # ------------------ 2) topo spatial guidance ------------------
        topo_map = topo.abs().sum(dim=1)  # (B,H,W)
        topo_map = topo_map / (topo_map.view(B, -1).max(dim=1)[0].view(B, 1, 1).clamp(min=1e-6)) # (B,H,W)

        # ------------------ 3) row attention ------------------

        fusion_row = torch.einsum("bhk,bw->bhwk", U, w_row)  # (B,H,W,k)
        logits_row = fusion_row.sum(dim=-1)  # (B,H,W)
        logits_row = logits_row + 0.5 * topo_map  # (B,H,W)
        att_row = F.softmax(logits_row, dim=-1)  # (B,H,W)
        R = torch.einsum('bhw, bchw -> bch', att_row, x)  # (B,C,H)
        row_out = torch.einsum('bch, hp ->bcp', R, Gt).view(B,C,H,W)  # (B,C,H,W)

        # ------------------ 4) col attention ------------------

        fusion_col = torch.einsum("bhk,bw->bhwk", U, w_col)  # (B,H,W,k)
        logits_col = fusion_col.sum(dim=-1) # (B,H,W)
        logits_col = logits_col + 0.5 * topo_map # (B,H,W)
        att_col = F.softmax(logits_col, dim=1) # (B,H,W)
        Cctx = torch.einsum('bhw, bchw -> bcw', att_col, x)  # (B,C,W)
        col_out = torch.einsum('bcw, wp ->bcp', Cctx, Gt).view(B,C,H,W)  # (B,C,H,W)

        # ------------------ 5) channel attention ------------------

        z_x = x.mean(dim=(2, 3))  # (B,C)
        z_topo = topo.mean(dim=(2, 3))  # (B,C)
        z_x_k = torch.bmm(z_x.unsqueeze(1), V).squeeze(1)  # (B,k)
        z_topo_k = torch.bmm(z_topo.unsqueeze(1), V).squeeze(1)  # (B,k)
        z_k = z_x_k + 0.5 * z_topo_k  # (B,k)
        zS = torch.bmm(z_k.unsqueeze(-1),S.unsqueeze(1))  # (B,k,k)
        ch_rec1 = torch.bmm(V,zS) # (B,C,k)
        ch_rec2 = torch.bmm(V, zS.transpose(1, 2)) # (B,C,k)
        ch_rec = torch.cat([ch_rec1,ch_rec2], dim=-1).mean(dim=-1) # (B,C)
        chan_scale = torch.sigmoid(ch_rec).view(B, C, 1, 1)  # reshape 以便广播
        chan_out = x * chan_scale  # (B,C,H,W)

        # ------------------ topo branch ------------------
        topo_channel = topo.mean(dim=(2, 3),keepdim=True)  # (B,C,1,1)
        topo_channelembed = torch.softmax(topo_channel,dim=1)*topo # (B,C,H,W)
        topo_spatial = topo.mean(dim=1, keepdim=True)  # (B,1,H,W)
        topo_spatialemded = torch.sigmoid(topo_spatial)*topo # (B,C,H,W)
        topo_branch = topo_channelembed+topo_spatialemded # (B,C,H,W)

        # ------------------ 6) fusion ----------------
        s_input = torch.cat([S, w_col, w_row], dim=1) # (B,k+H+W)
        w4 = F.softmax(torch.einsum('bp,c ->bc',s_input,self.boast), dim=-1)  # (B,C+H+W)

        a_row, a_col, a_chan = w4[:, :H],w4[:,H:H+W],w4[:,H+W:]
        a_topo = torch.bmm(a_row.unsqueeze(-1),a_col.unsqueeze(1))  # (B,H,W)
        """
        a_row:B,H
        a_col:B,W
        a_chan:B,C
        """
        out = a_row[:,None,:,None] * row_out + a_col[:,None, None,:] * col_out + a_chan[:,:, None,None] * chan_out + a_topo[:,None,:,:] * topo_branch
        if self.use_residual:
            out = self.res_scale * x + out

        return out


class FullGlobalTransBlock(nn.Module):
    """
    globalatt 循环堆叠模块
    """
    def __init__(self, in_ch, H, W, num_layers=2, use_residual=True):
        """
        Args:
            in_ch: 输入通道数
            H, W: 输入特征图高宽
            num_layers: globalatt 层数
        """
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            globalatt(in_ch=in_ch, H=H, W=W, use_residual=use_residual)
            for _ in range(num_layers)
        ])

    def forward(self, x, U, S, V, G):
        """
        x: (B, C, H, W)
        U: (B, h_proj, k)
        S: (B, C)  # 或 (B,k) 视你的 globalatt 定义
        V: (B, C, k)
        G: (HW, h_proj)
        """
        # 保证所有输入在同一 device
        device = x.device
        U = U.to(device)
        S = S.to(device)
        V = V.to(device)
        G = G.to(device)

        out = x
        for layer in self.layers:
            # globalatt 本身会使用 residual
            out = layer(out, U, S, V, G)
        return out

"-----------------------------------------------------------------------局部注意力----------------------------------------------------------------------------------"
"-----------------------------------------------------------------------局部注意力----------------------------------------------------------------------------------"
"-----------------------------------------------------------------------局部注意力----------------------------------------------------------------------------------"

def feature_similarity(cluster_feats, noise_feats, eps=1e-6):
    """
    cluster_feats: (B, k_max, hw)
    noise_feats: (B, n_noise, hw)
    return: (B, k_max, 1) 相似度
    """
    # 自动展开维度
    if cluster_feats.dim() == 4:
        B, k_max, H, W = cluster_feats.shape
        cluster_feats = cluster_feats.view(B, k_max, H * W)
    else:
        B, k_max, hw = cluster_feats.shape

    if noise_feats.dim() == 4:
        Bn, n_noise, Hn, Wn = noise_feats.shape
        noise_feats = noise_feats.view(Bn, n_noise, Hn * Wn)
    elif noise_feats.dim() == 3:
        Bn, n_noise, hw = noise_feats.shape
    else:
        raise ValueError(f"noise_feats shape 不合法: {noise_feats.shape}")

    if noise_feats.shape[1] == 0:  # 没有噪声
        return torch.zeros(B, k_max, 1, device=cluster_feats.device)

    B, k_max, hw = cluster_feats.shape
    _, n_noise, _ = noise_feats.shape

    if n_noise == 0:
        return torch.zeros(B, k_max, 1, device=cluster_feats.device)

    # L2 归一化
    cluster_norm = F.normalize(cluster_feats, p=2, dim=-1)  # (B, k_max, hw)
    noise_norm = F.normalize(noise_feats, p=2, dim=-1)      # (B, n_noise, hw)

    # 计算余弦相似度并取平均
    # (B, k_max, hw) @ (B, hw, n_noise) -> (B, k_max, n_noise)
    sim = torch.bmm(cluster_norm, noise_norm.transpose(1, 2))

    # 平均化 (降低计算开销)
    sim_mean = sim.mean(dim=-1)  # (B, k_max, 1)

    return sim_mean


# -----------------------------
# HDBSCAN 聚类辅助函数
def hdbscan_batch(x, min_cluster_size=2, downscale=None):
    B, C, H, W = x.shape
    fused = F.adaptive_max_pool2d(x, (H // downscale, W // downscale))

    # (C, HW)  每一行是一个通道的展平特征
    feats = fused.view(B*C, -1).detach().cpu().numpy() # B*C,H*W

    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=2)
    labels_all = clusterer.fit_predict(feats)   # (B*C)
    labels_all = torch.from_numpy(labels_all).to(x.device).long()

    return labels_all # (B*C)


# -----------------------------
# 局部标签注意力
# -----------------------------
class LabelRestrictedSelfAttention(nn.Module):
    def __init__(self, C, num_heads=4):
        super().__init__()
        self.num_heads = num_heads

        # qkv 投影，输入 (B, N, L)，输出 (B, N, 3L)
        self.qkv = nn.Conv2d(C,3*C,1,1,groups=C)#nn.Linear(L, 3 * L, bias=False)

    def forward(self, x, labels):
        """
        x: (B, C, h, w)
        labels: (B*C)
        """
        B, C, h, w = x.shape
        qkv = self.qkv(x)  # (B, 3C, h, w)
        qkv = qkv.reshape(1,3*B*C,h*w)
        q,k,v = qkv.chunk(3, dim=1)  # (1, B*C, h*w)

        mask = labels != -1  # (B*C) bool型
        cls_idx = labels[mask]  # (k,)  类别号
        unique_labels = cls_idx.unique()  # 返回已经排好序的 GPU tensor
        final_out = torch.zeros_like(q)
        for j in unique_labels:
            mask = labels == j
            q_m = q[:, mask, :]  # (1, k, hw)
            k_m_t = k[:, mask, :].transpose(1,2)  # (1, hw, k)
            v_m = v[:, mask, :]  # (1, k, hw)
            weights = F.softmax(torch.bmm(q_m,k_m_t),dim=-1)  # 1,k,k
            xi_att = torch.bmm(weights,v_m)  # 1,k,HW
            final_out[:, mask, :] = xi_att
        return final_out  # (1,B*C,h*w)

# -----------------------------
# 聚类 + 局部注意力
# -----------------------------
class ClusterLocalAtt(nn.Module):
    def __init__(self, H, C, num_heads=4, downscale=4,min_cluster_size=2):
        super().__init__()
        self.downscale = downscale
        self.H =H
        self.H_ds = H//downscale

        self.min_cluster_size = min_cluster_size
        self.down = nn.MaxPool2d(downscale,downscale)
        self.up = nn.ConvTranspose2d(in_channels=C,out_channels=C,kernel_size=downscale,stride=downscale, groups=C)
        self.attn = LabelRestrictedSelfAttention(C, num_heads=num_heads)
        # 反卷积恢复到原始分辨率
        self.bais = nn.Conv1d(C//2, C, 1,1,groups=C//2)


    def forward(self, x):
        """"
        x: B,C,H,W
        labels: B*C
        noise_feats: 1,i,H//down,W//down
        """
        B, C, H, W = x.shape
        labels= hdbscan_batch(x, downscale=self.downscale,min_cluster_size=self.min_cluster_size)  # B*C
        x_down = self.down(x) # B,C,H//down,W//down
        out = self.attn(x_down, labels) # 1, B*C, h*w
        noise_mask = labels.eq(-1)
        noise_feats = x.reshape(1,B*C,-1)[:,noise_mask,:]  # 1, k,HW
        if noise_feats.shape[1]%2 != 0:
            noise_feats = torch.cat([noise_feats,torch.zeros(1,1,H*W).to(device)],dim=1)
        noise_feats = noise_feats.view(B,-1, H, W)
        if noise_feats.size(1) == 0:
            noise_feats = torch.zeros(B, 1, H, W).to(out.device)
        out = out.view(B,C, H//self.downscale, W//self.downscale)  # B,C,h,w
        out = self.up(out)  # B,C,H,W
        similar = feature_similarity(out, noise_feats).view(B,C,1,1)  # B,C,1,1
        out = out + (1 - similar) * x # B,C,H,W
        return out, noise_feats.view(B,-1,H,W)

# -----------------------------



#----------------------------------
#快速聚类注意力
#----------------------------------

"-------------------------------------快速聚类注意力--------------------------------------------------------"

class OrthoConvClustering(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groups: int = 4, bias: bool = True):
        super().__init__()
        assert in_channels % groups == 0
        assert out_channels % groups == 0

        self.groups = groups
        self.in_per_group = in_channels // groups
        self.out_per_group = out_channels // groups

        self.weight = nn.Parameter(
            torch.randn(groups, self.out_per_group, self.in_per_group) *
            (1.0 / math.sqrt(self.in_per_group))
        )
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        B, C, L = x.shape
        x = x.view(B, self.groups, self.in_per_group, L)
        y = torch.einsum('goi,bgil->bgol', self.weight, x)  # (B, groups, Cout_group, L)
        y = y.reshape(B, -1, L)
        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1)
        return y

    def ortho_loss(self) -> torch.Tensor:
        losses = []
        for g in range(self.groups):
            W = self.weight[g]  # (out, in)
            WWT = W @ W.T
            I = torch.eye(WWT.size(0), device=W.device, dtype=W.dtype)
            losses.append(F.mse_loss(WWT, I))
        return torch.stack(losses).mean()


# ---------------------------
# 多头聚类注意力
# ---------------------------
class FastClusterAtt(nn.Module):
    def __init__(self, in_ch, cluster_stride=2, num_heads=4, groups=4, center="Ture"):
        super().__init__()
        assert in_ch % num_heads == 0, "in_ch 必须能整除 num_heads"
        self.in_ch = in_ch
        self.num_heads = num_heads
        self.head_dim = in_ch // num_heads
        self.cluster_stride = cluster_stride
        self.cluster = OrthoConvClustering(in_ch, in_ch, groups=groups)
        self.centering = nn.MaxPool2d(cluster_stride, cluster_stride)
        self.qkv = nn.Conv2d(in_ch, in_ch * 3, 1, 1, groups=in_ch, bias=False)

        self.bais = nn.Conv2d(in_ch, in_ch, 1, 1)

    def forward(self, x, noise=None):
        """
        x: (B, C, H, W)
        noise: (B, C, H, W) or None
        """
        B, C, H, W = x.shape

        # 1) Ortho 投影
        x_flat = x.view(B, C, -1)
        x_clustered_flat = self.cluster(x_flat)
        x_clustered = x_clustered_flat.view(B, C, H, W)

        # 2) 降采样
        x_ds = self.centering(x_clustered)  # (B,C,H',W')
        Hs, Ws = x_ds.shape[2], x_ds.shape[3]
        Ls = Hs * Ws

        # 3) QKV
        qkv = self.qkv(x_ds)  # (B,3C,H',W')
        q, k, v = torch.chunk(qkv, 3, dim=1)

        def to_heads(t):
            return t.view(B, self.num_heads, self.head_dim, Ls)  # (B,H,d,L)

        qh, kh, vh = map(to_heads, (q, k, v))

        # 4) 逐位置注意力（只计算 q[:, :, :, i] 和 k[:, :, :, i]）
        #    scores: (B,H,1,L)，每个位置一个标量权重
        scores = (qh * kh).sum(dim=2, keepdim=True) / math.sqrt(self.head_dim)  # (B,H,1,L)
        attn = torch.softmax(scores, dim=-1)  # (B,H,1,L)

        # 5) 对 V 逐位置加权
        out_heads = (attn * vh).sum(dim=-1)  # (B,H,d)

        # 6) 拼回 channel
        out_heads = out_heads.view(B, -1, 1, 1)      # (B,C,1,1)
        out_sum = out_heads.expand(B, C, Hs, Ws)     # (B,C,H',W')

        # 7) 上采样
        out  = self.bais(F.interpolate(out_sum, scale_factor=self.cluster_stride, mode='bilinear', align_corners=False))  # (B,C,H,W)
        # 8) 残差增强
        if noise is not None:
            sim = feature_similarity(out, noise).unsqueeze(-1).unsqueeze(-1)  # (B,C,1,1)
            ortho_factor = 1.0 - self.cluster.ortho_loss().detach()
            out = out*(1 - sim) + ortho_factor * x
        else:
            ortho_factor = 1.0 - self.cluster.ortho_loss().detach()
            out = out + ortho_factor * x

        return out


# -----------------------------
# 简单 MLP
# -----------------------------
class MLP(nn.Module):
    def __init__(self, dim, hidden_factor, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_factor*dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_factor*dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

# -----------------------------
# FullLocalTransBlock
# -----------------------------
class FullLocalTransBlock(nn.Module):
    def __init__(self,
                 in_ch,
                 H,
                 layer=1,
                 clusteratt_num=1,
                 fastclusteratt_num=1,
                 clusteratt_num_heads=1,
                 clusteratt_downscale = 4,
                 clusteratt_min_cluster_size=2,
                 fastclusteratt_num_heads=4,
                 fastclusteratt_cluster_stride=2,
                 fastclusteratt_groups=4):
        """
        in_ch: 输入通道
        H: 特征图高/宽
        layer: 堆叠层数
        clusteratt_num: 每层 ClusterLocalAtt 堆叠数量
        fastclusteratt_num: 每层 FastClusterAtt 堆叠数量
        clusteratt_num_heads: ClusterLocalAtt 的 heads
        clusteratt_min_cluster_size: ClusterLocalAtt 最小聚类数
        fastclusteratt_num_heads: FastClusterAtt 的 heads
        fastclusteratt_cluster_stride: FastClusterAtt 聚类下采样 stride
        fastclusteratt_groups: FastClusterAtt 的 groups 参数
        """
        super().__init__()
        self.layer = layer
        self.clusteratt_num = clusteratt_num
        self.fastclusteratt_num = fastclusteratt_num
        self.H = H
        self.in_ch = in_ch
        self.layers = nn.ModuleList()
        for l in range(layer):

            sublayer = nn.ModuleDict()
            # ------------------- ClusterLocalAtt stack -------------------
            sublayer['clusteratt'] = nn.ModuleList()
            for _ in range(clusteratt_num):
                block = nn.ModuleDict({
                    'attn': ClusterLocalAtt(H, in_ch, num_heads=clusteratt_num_heads, downscale=clusteratt_downscale,min_cluster_size=clusteratt_min_cluster_size),
                })
                sublayer['clusteratt'].append(block)

            # ------------------- FastClusterAtt stack -------------------
            sublayer['fastclusteratt'] = nn.ModuleList()
            for _ in range(fastclusteratt_num):
                block = nn.ModuleDict({
                    'attn': FastClusterAtt(in_ch, cluster_stride=fastclusteratt_cluster_stride, num_heads=fastclusteratt_num_heads, groups=fastclusteratt_groups),
                })
                sublayer['fastclusteratt'].append(block)
            self.layers.append(sublayer)
    def forward(self, x):
        """
        x: (B, C, H, W)
        noise: (B, C, H, W) 可选，用于 FastClusterAtt 的残差增强
        """
        out = None
        for layer in self.layers:
            # ---------------- ClusterLocalAtt stack ----------------
            noise = None
            out_clu = None
            for block in layer['clusteratt']:
                attn_out, noise = block['attn'](x)
                out_clu = x + attn_out

            # ---------------- FastClusterAtt stack ----------------
            out_fclu = None
            for block in layer['fastclusteratt']:
                out_fclu = block['attn'](out_clu, noise)
                out_fclu = out_clu + out_fclu

            out = out_fclu
        return out


class KANLinear(torch.nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            grid_size=5,  # 网格大小，默认为 5
            spline_order=3,  # 分段多项式的阶数，默认为 3
            scale_noise=0.1,  # 缩放噪声，默认为 0.1
            scale_base=1.0,  # 基础缩放，默认为 1.0
            scale_spline=1.0,  # 分段多项式的缩放，默认为 1.0
            enable_standalone_scale_spline=True,
            base_activation=torch.nn.SiLU,  # 基础激活函数，默认为 SiLU（Sigmoid Linear Unit）
            grid_eps=0.02,
            grid_range=[-1, 1],  # 网格范围，默认为 [-1, 1]
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size  # 设置网格大小和分段多项式的阶数
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size  # 计算网格步长
        grid = (  # 生成网格
            (
                    torch.arange(-spline_order, grid_size + spline_order + 1) * h
                    + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)  # 将网格作为缓冲区注册

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))  # 初始化基础权重和分段多项式权重
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:  # 如果启用独立的分段多项式缩放，则初始化分段多项式缩放参数
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise  # 保存缩放噪声、基础缩放、分段多项式的缩放、是否启用独立的分段多项式缩放、基础激活函数和网格范围的容差
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()  # 重置参数

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)  # 使用 Kaiming 均匀初始化基础权重
        with torch.no_grad():
            noise = (  # 生成缩放噪声
                    (
                            torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                            - 1 / 2
                    )
                    * self.scale_noise
                    / self.grid_size
            )
            self.spline_weight.data.copy_(  # 计算分段多项式权重
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order: -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:  # 如果启用独立的分段多项式缩放，则使用 Kaiming 均匀初始化分段多项式缩放参数
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        """
        计算给定输入张量的 B-样条基函数。
        参数:
        x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。
        返回:
        torch.Tensor: B-样条基函数张量，形状为 (batch_size, in_features, grid_size + spline_order)。
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (  # 形状为 (in_features, grid_size + 2 * spline_order + 1)
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                            (x - grid[:, : -(k + 1)])
                            / (grid[:, k:-1] - grid[:, : -(k + 1)])
                            * bases[:, :, :-1]
                    ) + (
                            (grid[:, k + 1:] - x)
                            / (grid[:, k + 1:] - grid[:, 1:(-k)])
                            * bases[:, :, 1:]
                    )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).
        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        """
        计算插值给定点的曲线的系数。
        参数:
        x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。
        y (torch.Tensor): 输出张量，形状为 (batch_size, in_features, out_features)。
        返回:
        torch.Tensor: 系数张量，形状为 (out_features, in_features, grid_size + spline_order)。
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)
        # 计算 B-样条基函数
        A = self.b_splines(x).transpose(
            0, 1  # 形状为 (in_features, batch_size, grid_size + spline_order)
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features) # 形状为 (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(  # 使用最小二乘法求解线性方程组
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)  # 形状为 (in_features, grid_size + spline_order, out_features)
        result = solution.permute(  # 调整结果的维度顺序
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        """
        获取缩放后的分段多项式权重。
        返回:
        torch.Tensor: 缩放后的分段多项式权重张量，形状与 self.spline_weight 相同。
        """
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):  # 将输入数据通过模型的各个层，经过线性变换和激活函数处理，最终得到模型的输出结果
        """
        前向传播函数。
        参数:
        x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。
        返回:
        torch.Tensor: 输出张量，形状为 (batch_size, out_features)。
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        base_output = F.linear(self.base_activation(x), self.base_weight)  # 计算基础线性层的输出
        spline_output = F.linear(  # 计算分段多项式线性层的输出
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output  # 返回基础线性层输出和分段多项式线性层输出的和

    @torch.no_grad()
    # 更新网格。
    # 参数:
    # x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。
    # margin (float): 网格边缘空白的大小。默认为 0.01。
    # 根据输入数据 x 的分布情况来动态更新模型的网格,使得模型能够更好地适应输入数据的分布特点，从而提高模型的表达能力和泛化能力。
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)  # 计算 B-样条基函数
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)  # 调整维度顺序为 (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)  # 调整维度顺序为 (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]  # 对每个通道单独排序以收集数据分布
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
                torch.arange(
                    self.grid_size + 1, dtype=torch.float32, device=x.device
                ).unsqueeze(1)
                * uniform_step
                + x_sorted[0]
                - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)  # 更新网格和分段多项式权重
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        # 计算正则化损失，用于约束模型的参数，防止过拟合
        """
        Compute the regularization loss.
        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.
        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        """
        计算正则化损失。
        这是对原始 L1 正则化的简单模拟，因为原始方法需要从扩展的（batch, in_features, out_features）中间张量计算绝对值和熵，
        而这个中间张量被 F.linear 函数隐藏起来，如果我们想要一个内存高效的实现。
        现在的 L1 正则化是计算分段多项式权重的平均绝对值。作者的实现也包括这一项，除了基于样本的正则化。
        参数:
        regularize_activation (float): 正则化激活项的权重，默认为 1.0。
        regularize_entropy (float): 正则化熵项的权重，默认为 1.0。
        返回:
        torch.Tensor: 正则化损失。
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
                regularize_activation * regularization_loss_activation
                + regularize_entropy * regularization_loss_entropy
        )


class KAN(torch.nn.Module):  # 封装了一个KAN神经网络模型，可以用于对数据进行拟合和预测。
    def __init__(
            self,
            layers_hidden,
            grid_size=3,
            spline_order=2,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
    ):
        """
        初始化 KAN 模型。
        参数:
            layers_hidden (list): 包含每个隐藏层输入特征数量的列表。
            grid_size (int): 网格大小，默认为 5。
            spline_order (int): 分段多项式的阶数，默认为 3。
            scale_noise (float): 缩放噪声，默认为 0.1。
            scale_base (float): 基础缩放，默认为 1.0。
            scale_spline (float): 分段多项式的缩放，默认为 1.0。
            base_activation (torch.nn.Module): 基础激活函数，默认为 SiLU。
            grid_eps (float): 网格调整参数，默认为 0.02。
            grid_range (list): 网格范围，默认为 [-1, 1]。
        """
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):  # 调用每个KANLinear层的forward方法，对输入数据进行前向传播计算输出。
        """
        前向传播函数。
        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。
            update_grid (bool): 是否更新网格。默认为 False。
        返回:
            torch.Tensor: 输出张量，形状为 (batch_size, out_features)。
        """
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):  # 计算正则化损失的方法，用于约束模型的参数，防止过拟合。
        """
        计算正则化损失。
        参数:
            regularize_activation (float): 正则化激活项的权重，默认为 1.0。
            regularize_entropy (float): 正则化熵项的权重，默认为 1.0。
        返回:
            torch.Tensor: 正则化损失。
        """
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )


class GoLTransBlock(nn.Module):
    """
    单层 GoLTransBlock:
    输入 -> LN(x) -> SVD -> 双分支(Local & Global) -> residual -> LN -> MLP
    """
    def __init__(self, in_ch, H, W,
                 mlp_hidden_factor=2,
                 mlp_method = 'mlp',
                 local_kwargs=None,
                 global_kwargs=None):
        super().__init__()
        self.in_ch = in_ch
        self.H = H
        self.W = W

        # 层归一化
        self.ln1 = nn.LayerNorm(in_ch)
        self.ln2 = nn.LayerNorm(in_ch)

        # MLP
        hidden_dim = in_ch * mlp_hidden_factor
        if mlp_method == 'kan':
            self.mlp=KAN(
                layers_hidden=[in_ch, hidden_dim,in_ch],
                grid_size=5,  # 可以调大调小，影响参数量
                spline_order=3,
                scale_noise=0.1,
                scale_base=1.0,
                scale_spline=1.0,
                base_activation=torch.nn.SiLU,
                grid_eps=0.02,
                grid_range=[-1, 1]
            )
        else:
            self.mlp = nn.Sequential(
            nn.Linear(in_ch, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, in_ch)
        )
        # Local branch
        self.local_block = FullLocalTransBlock(in_ch=in_ch, H=H, **(local_kwargs or {}))

        # Global branch
        self.global_block = FullGlobalTransBlock(in_ch=in_ch, H=H, W=W, **(global_kwargs or {}))

    def forward(self, x):

        # ---------------- LN ----------------
        x_ln = self.ln1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # (B,C,H,W)

        # ---------------- SVD ----------------
        U, S, V, G = SVD(x_ln)  # SVD 使用归一化后的 x

        # ---------------- 双分支 ----------------
        local_out = self.local_block(x_ln)
        global_out = self.global_block(x_ln, U, S, V, G)

        # residual
        x_res = x + local_out + global_out

        # LN + MLP
        x_ln2 = self.ln2(x_res.permute(0, 2, 3, 1))
        B, H, W, C = x_ln2.shape
        x_flat = x_ln2.reshape(B * H * W, C)  # (BHW, C)
        x_kan = self.mlp(x_flat)  # (BHW, C)
        out_mlp = x_kan.view(B, C, H, W)  # B,C,H,W
        out = x_res + out_mlp
        return out,local_out


class GoLTransBlockLayer(nn.Module):
    """
    GoLTransBlock 层堆叠
    输入 -> 多层 GoLTransBlock -> 输出
    """
    def __init__(self, in_ch, H, W, num_layers=2,
                 mlp_hidden_factor=2,
                 mlp_method = 'mlp',
                 local_kwargs=None,
                 global_kwargs=None):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            GoLTransBlock(in_ch=in_ch, H=H, W=W,
                          mlp_hidden_factor=mlp_hidden_factor,
                          local_kwargs=local_kwargs,
                          mlp_method = mlp_method,
                          global_kwargs=global_kwargs)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        out = x
        for layer in self.layers:
            out,local = layer(out)
        return out,local

# -----------------------------
# 测试
# -----------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from torchinfo import summary
    from thop import profile, clever_format
    # 模拟输入
    B, C, H, W = 2, 32, 128, 128
    x = torch.randn(B, C, H, W).cuda()

    GoLT_layer = GoLTransBlockLayer(in_ch=C, H=H, W=W, num_layers=1,mlp_hidden_factor=2,mlp_method = 'mlp',
                                   local_kwargs={
                                       'layer': 1,
                                       'clusteratt_num': 1,
                                       'fastclusteratt_num': 1,
                                       'clusteratt_downscale': 8,
                                       'clusteratt_num_heads': 1,
                                       'clusteratt_min_cluster_size': 2,
                                       'fastclusteratt_num_heads': 1,
                                       'fastclusteratt_cluster_stride': 8,
                                       'fastclusteratt_groups': 4
                                   },
                                   global_kwargs={
                                       'num_layers': 1
                                   }).cuda()

    local = ClusterLocalAtt(H, C, num_heads=4, downscale=4,min_cluster_size=2).cuda()
    local.eval()

    # -----------------------------
    # 前向测试
    # -----------------------------
    with torch.no_grad():
        out,_ = local(x)

    print("输入 shape:", x.shape)
    print("输出 shape:", out.shape)

    # -----------------------------
    # torchinfo summary
    # -----------------------------
    print("\n=== torchinfo summary ===")
    summary(local, input_data=(x,),
            col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"])

    # -----------------------------
    # thop FLOPs / Params
    # -----------------------------
    print("\n=== thop summary ===")
    flops, params = profile(local, inputs=(x,), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")
    print(f"Parameters: {params}, FLOPs: {flops}")
