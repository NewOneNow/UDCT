import torch
import numpy as np
from sklearn.cluster import HDBSCAN

device = torch.device('cuda')

def hdbscan_fixed_channels_auto_remain_dynamic(x, min_cluster_size=2):
    """
    Args:
      x: Tensor of shape (b, c, hw)
      min_cluster_size: HDBSCAN 参数，控制最小簇大小

    Returns:
      xh: Tensor of shape (b, c, hw)，通道已聚类，余数分配到前 r 个类别
      labels_all: LongTensor (b, c)，每个通道聚类 label
      r_list: list，每个 batch 的余数通道数量
      avg_k: int，平均类别数，可用于初始化模块
    """
    b, c, h, w = x.shape
    device = x.device
    x_np = x.view(b,c,-1)
    labels_all = []
    result_batches = []
    r_list = []
    k_list = []

    for i in range(b):
        feats = x_np[i]  # (c, hw)

        G = torch.randn(h*w, h, device='cuda') / (h ** 0.5)
        feats = (feats @ G).detach().cpu().numpy()  # (c, proj_dim)

        clusterer = HDBSCAN(min_cluster_size=min_cluster_size)
        labels = clusterer.fit_predict(feats)  # shape (c,)
        labels_all.append(labels)

        # 当前 batch 的实际类别数
        k_i = int(labels.max() + 2)
        k_list.append(k_i)
        ck = c // k_i
        r = c % k_i
        r_list.append(r)

        unique_labels = sorted(set(labels))
        if -1 in unique_labels:
            unique_labels.remove(-1)
            unique_labels = [-1] + unique_labels
        else:
            unique_labels = [-1] + unique_labels

        # Build list of cluster indices
        clusters_idx = []
        for lbl in unique_labels:
            idx = np.where(labels == lbl)[0]
            clusters_idx.append(idx)

        # Truncate/pad to exactly k_i clusters
        if len(clusters_idx) > k_i:
            clusters_idx = clusters_idx[:k_i]
        while len(clusters_idx) < k_i:
            clusters_idx.append(np.array([], dtype=int))

        # 分配通道数
        xh_per_batch = []
        for j, idx in enumerate(clusters_idx):
            num_ch = ck + 1 if j < r else ck
            if len(idx) >= num_ch:
                chosen = idx[:num_ch]
            elif len(idx) == 0:
                chosen = np.random.randint(0, c, size=num_ch)
            else:
                repeats = np.random.choice(idx, size=num_ch - len(idx), replace=True)
                chosen = np.concatenate([idx, repeats])
            xh_per_batch.append(x[i, chosen, :])  # (num_ch, hw)

        # Concatenate k_i clusters
        xh_per_batch = torch.cat(xh_per_batch, dim=0)  # (c, hw)
        result_batches.append(xh_per_batch)

    xh = torch.stack(result_batches, dim=0).to(device)  # (b, c, hw)
    avg_k = int(np.mean(k_list))  # 平均类别数
    avg_r = int(np.mean(r_list))
    return xh, avg_r, avg_k


