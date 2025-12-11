import numpy as np
import torch
import torch.nn as nn


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


def safe_svd(mat: torch.Tensor, full_matrices=False, eps=1e-6, driver="gesvd"):
    """
    健壮 SVD 分解，避免 cusolver 不收敛报错
    输入:
        mat: (m, n) 矩阵
        full_matrices: 是否计算 full SVD
        eps: 数值扰动系数
        driver: 调用的底层 LAPACK 接口
    返回:
        U, S, Vh
    """
    device = mat.device
    mat = mat + eps * torch.randn_like(mat)  # 加小扰动避免病态

    try:
        U, S, Vh = torch.linalg.svd(mat, full_matrices=full_matrices, driver=driver)
    except RuntimeError:
        # 如果 CUDA 报错，回退到 CPU
        mat_cpu = mat.cpu()
        U, S, Vh = torch.linalg.svd(mat_cpu, full_matrices=full_matrices, driver=driver)
        U, S, Vh = U.to(device), S.to(device), Vh.to(device)

    return U, S, Vh


def RandomProjectionSVD(x: torch.Tensor):
    """
    x: Tensor (b, c, h, w)
    输出 U、S、V 通道保持 c，同时保留随机投影
    返回:
        U_orig: (b, hw, c)
        S: (b, c)
        V: (b, c, c)
        k_est: 平均估计秩
    """
    device = x.device
    b, c, h, w = x.shape
    hw = h * w
    h_proj = min(h, c)

    # 随机投影矩阵 (hw -> h_proj)
    G = torch.randn(hw, h_proj, device=device) / (h_proj ** 0.5)

    # reshape -> (b, c, hw)
    x_reshaped = x.reshape(b, c, hw)
    X_proj = x_reshaped @ G  # (b, c, h_proj)
    X_proj = X_proj.transpose(1, 2)  # (b, h_proj, c)

    U_list, S_list, V_list, k_list = [], [], [], []

    for i in range(b):
        # 用安全版 SVD
        Ui, Si, Vh = safe_svd(X_proj[i], full_matrices=False)
        k_est = compute_knee_point(Si)
        k_list.append(k_est)

        # U -> c
        W_U = torch.randn(Ui.shape[1], c, device=device) / (Ui.shape[1] ** 0.5)
        U_mapped = Ui @ W_U  # (h_proj, c)

        # V -> c
        Vh_T = Vh.T  # (c, min(h_proj,c))
        W_V = torch.randn(Vh_T.shape[1], c, device=device) / (Vh_T.shape[1] ** 0.5)
        V_mapped = Vh_T @ W_V  # (c, c)

        # S padding
        if Si.numel() < c:
            Si_padded = torch.cat([Si, torch.zeros(c - Si.numel(), device=device)])
        else:
            Si_padded = Si[:c]

        U_list.append(U_mapped)
        S_list.append(Si_padded)
        V_list.append(V_mapped)

    U = torch.stack(U_list, dim=0)  # (b, h_proj, c)
    S = torch.stack(S_list, dim=0)  # (b, c)
    V = torch.stack(V_list, dim=0)  # (b, c, c)

    # 回映射 U -> (b, hw, c)
    W_linear = torch.randn(U.shape[1], hw, device=device) / (U.shape[1] ** 0.5)  # (h_proj -> hw)
    U_orig = torch.einsum('bhc,hd->bdc', U, W_linear)  # (b, hw, c)

    return U_orig, S, V, int(np.mean(k_list))
