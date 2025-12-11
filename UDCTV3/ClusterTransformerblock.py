import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .ClusterAttV2 import ClusterAtt, FastclusterAtt


class FullClusterTransBlock(nn.Module):
    """
    FullClusterTransBlock:
      - in_ch: 输入通道数（传入用于初始化子模块）
      - cluatt_poolingfactor: 传给 ClusterAtt 的 poolingfactor
      - fastcluatt_poolingfactor: 传给 FastclusterAtt 的 poolingfactor
      - full_layer: 层数（每层包含 1 x ClusterAtt + fastclu_layer x FastclusterAtt 串联）
      - clu_head: 每层并行的 ClusterAtt 数（head），avg_k/noise 从该层第 0 个 head 获取
      - fastclu_head: 每个 fast 层并行的 FastclusterAtt 数（head）
      - fastclu_layer: 每层串联多少个 FastclusterAtt
      - residual_scale: 控制残差门控强度 (默认 1.0)
    """
    def __init__(
        self,
        in_ch: int,
        cluatt_poolingfactor: int = 4,
        fastcluatt_poolingfactor: int = 4,
        full_layer: int = 1,
        clu_head: int = 1,
        fastclu_head: int = 1,
        fastclu_layer: int = 1,
        residual_scale: float = 1.0,
    ):
        super().__init__()
        assert full_layer >= 1 and fastclu_layer >= 1
        assert clu_head >= 1 and fastclu_head >= 1

        self.in_ch = in_ch
        self.full_layer = full_layer
        self.clu_head = clu_head
        self.fastclu_head = fastclu_head
        self.fastclu_layer = fastclu_layer
        self.residual_scale = residual_scale

        # 构建每一层的模块结构
        self.layers = nn.ModuleList()
        for _ in range(full_layer):
            # cluster heads（clu_head 个并行 ClusterAtt）
            cluster_heads = nn.ModuleList(
                [ClusterAtt(in_ch, poolingfactor=cluatt_poolingfactor) for _ in range(clu_head)]
            )
            # fast 层序列：每一层又包含 fastclu_head 个并行 FastclusterAtt
            fast_layers = nn.ModuleList()
            for _ in range(fastclu_layer):
                fast_heads = nn.ModuleList(
                    [FastclusterAtt(in_ch, poolingfactor=fastcluatt_poolingfactor) for _ in range(fastclu_head)]
                )
                fast_layers.append(fast_heads)

            layer_dict = nn.ModuleDict({
                "cluster_heads": cluster_heads,
                "fast_layers": fast_layers,
            })
            self.layers.append(layer_dict)

    def _merge_heads_mean(self, head_outputs):
        """
        head_outputs: list of tensors with identical shape (B, C, H, W)
        返回均值融合的 tensor (B, C, H, W)
        """
        if len(head_outputs) == 1:
            return head_outputs[0]
        return torch.mean(torch.stack(head_outputs, dim=0), dim=0)

    def forward(self, x: torch.Tensor):
        """
        x: (B, C, H, W)
        返回:
            y: (B, C, H, W)
        """
        y = x
        for layer_idx, layer in enumerate(self.layers):
            cluster_heads = layer["cluster_heads"]
            fast_layers = layer["fast_layers"]

            # ---- 1) 并行 ClusterAtt heads ----
            att_head_outputs = []
            avg_k_layer = None
            noise_layer = None

            for h_idx, ch in enumerate(cluster_heads):
                # ClusterAtt.forward -> (x_cluster_att, avg_k, noise)
                att_out, avg_k, noise = ch(y)  # att_out shape 与 in_ch 相同 (B, C, H, W)
                att_head_outputs.append(att_out)

                if h_idx == 0:
                    # 记录第一个 head 的 avg_k / noise（按你要求）
                    if isinstance(avg_k, torch.Tensor):
                        try:
                            avg_k_layer = int(avg_k.item())
                        except Exception:
                            avg_k_layer = int(avg_k)
                    else:
                        avg_k_layer = int(avg_k)
                    noise_layer = noise

            # 融合多个 ClusterAtt head 的 att（均值）
            att_merged = self._merge_heads_mean(att_head_outputs)  # (B, C, H, W)

            # 门控 + 残差（逐元素）
            gate = torch.sigmoid(att_merged)
            y = y * (1.0 + self.residual_scale * gate)

            # ---- 2) 串联 fastclu_layer 个 FastclusterAtt，每层内部并行 fastclu_head 个 head ----
            # 使用当前层的 avg_k_layer, noise_layer（来自该层第一个 ClusterAtt）
            if avg_k_layer is None or noise_layer is None:
                # 理论上不会发生，但作为保护（若 cluster 返回异常），直接跳过 fast 部分
                continue
            for fast_layer_heads in fast_layers:
                fast_head_outs = []
                for fh in fast_layer_heads:
                    # FastclusterAtt.forward(x, avg_k, noise) -> (B, C, H, W)
                    fast_att_out = fh(y, avg_k_layer, noise_layer)
                    fast_head_outs.append(fast_att_out)

                # 并行 head 融合（均值）
                fast_att_merged = self._merge_heads_mean(fast_head_outs)
                # 门控 + 残差
                gate_f = torch.sigmoid(fast_att_merged)
                y = y * (1.0 + self.residual_scale * gate_f)
            # 当前 layer 完成，进入下一层（若有）
        # end for layers
        
        return y


class FullClusterTransformerBlock(nn.Module):
    def __init__(
        self,
        in_ch,
        mlp_ratio,
        drop,
        cluatt_poolingfactor,
        fastcluatt_poolingfactor,
        full_layer,
        clu_head,
        fastclu_head,
        fastclu_layer,
        residual_scale,
    ):
        """
        Args:
            in_ch: 输入通道数
            mlp_ratio: FFN的扩展倍数（默认4倍）
            drop: Dropout概率
            cluster_kwargs: 传给FullClusterTransBlock的参数
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(in_ch)
        self.attn = FullClusterTransBlock(
            in_ch,
            cluatt_poolingfactor,
            fastcluatt_poolingfactor,
            full_layer,
            clu_head,
            fastclu_head,
            fastclu_layer,
            residual_scale,
        )
        self.norm2 = nn.LayerNorm(in_ch)

        hidden_dim = int(in_ch * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(in_ch, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, in_ch),
            nn.Dropout(drop),
        )

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape

        # 1) LayerNorm + Attention + 残差
        x_reshaped = x.permute(0, 2, 3, 1)  # (B, H, W, C) for LayerNorm
        x_norm = self.norm1(x_reshaped)
        x_attn = self.attn(x_norm.permute(0, 3, 1, 2))  # back to (B, C, H, W)
        x = x + x_attn  # residual

        # 2) LayerNorm + MLP + 残差
        x_reshaped = x.permute(0, 2, 3, 1)
        x_norm = self.norm2(x_reshaped)
        x_mlp = self.mlp(x_norm)
        x = x + x_mlp.permute(0, 3, 1, 2)

        return x


# -------------------------
# quick test (若 cluster / SVD / hdbscan 等能在当前环境运行)
# -------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, C, H, W = 2, 64, 256, 256  # ���试尺寸（较小）
    x = torch.randn(B, C, H, W, device=device)

    block = FullClusterTransformerBlock(
        in_ch=C,
        mlp_ratio= 4.0,
        drop= 0.1,
        cluatt_poolingfactor= 8,
        fastcluatt_poolingfactor= 4,
        full_layer= 2,
        clu_head= 4,
        fastclu_head= 4,
        fastclu_layer= 4,
        residual_scale= 1.0,
    ).to(device)
