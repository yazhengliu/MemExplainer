import torch
from torch import nn


@torch.no_grad()
def lrp_grucell_contrib_full(x: torch.Tensor, h: torch.Tensor, gru: nn.GRUCell, eps: float = 1e-12, return_ratio: bool = False):
  """
  将 GRUCell 的输出逐维分配到输入消息 x 与旧隐状态 h 的逐维贡献。

  返回字典：
    - C_x_full: [B, I, H]  输入每维 → 输出每维 的贡献
    - C_h_full: [B, H_in, H_out] 旧隐状态每维 → 输出每维 的贡献（含候选分支 + 直通项只在对角线累加）
    - C_x: [B, I]   = C_x_full.sum(dim=2)
    - C_h: [B, H]   = C_h_full.sum(dim=1)
    - h_new: [B, H] 手工重算的新隐状态
    - max_cand_err / max_total_err: 守恒误差（标量）
  规则：分母为0 → 贡献为0；否则按比例。
  
  Args:
    return_ratio: 如果为True，返回贡献值除以输出的ratio值；如果为False，返回原始贡献值
  """
  B, I = x.shape
  H = h.shape[1]

  # 权重按 PyTorch GRUCell 顺序 [r, z, n]
  W_r, W_z, W_h = gru.weight_ih.chunk(3, dim=0)
  U_r, U_z, U_h = gru.weight_hh.chunk(3, dim=0)
  if gru.bias_ih is not None:
    b_ir, b_iz, b_ih = gru.bias_ih.chunk(3, dim=0)
  else:
    b_ir = b_iz = b_ih = 0.0
  if gru.bias_hh is not None:
    b_hr, b_hz, b_hh = gru.bias_hh.chunk(3, dim=0)
  else:
    b_hr = b_hz = b_hh = 0.0

  # 手工前向
  z_t = torch.sigmoid(x @ W_z.t() + h @ U_z.t() + (b_iz + b_hz))
  r_t = torch.sigmoid(x @ W_r.t() + h @ U_r.t() + (b_ir + b_hr))
  pre_n_x = x @ W_h.t()
  pre_n_h_lin = h @ U_h.t()
  pre_n = pre_n_x + r_t * (pre_n_h_lin + b_hh) + b_ih
  h_hat = torch.tanh(pre_n)
  h_final = (1.0 - z_t) * h_hat + z_t * h

  # 候选分支 & 分母（不含偏置）
  h_after = (1.0 - z_t) * h_hat
  denom = pre_n_x + r_t * pre_n_h_lin

  # x → 输出逐维
  xW = x.unsqueeze(2) * W_h.t().unsqueeze(0)            # [B, I, H]
  ratio_x = torch.where(denom.unsqueeze(1).abs() > eps,
                        xW / denom.unsqueeze(1),
                        torch.zeros_like(xW))
  C_x_full = ratio_x * h_after.unsqueeze(1)             # [B, I, H]


  # h → 输出逐维（候选部分）
  hUh = h.unsqueeze(2) * U_h.t().unsqueeze(0)           # [B, H, H]
  hUh = hUh * r_t.unsqueeze(1)
  ratio_h = torch.where(denom.unsqueeze(1).abs() > eps,
                        hUh / denom.unsqueeze(1),
                        torch.zeros_like(hUh))
  C_h_cand_full = ratio_h * h_after.unsqueeze(1)        # [B, H, H]

  # 直通项 z*h 加到对角线
  passthrough = torch.zeros_like(C_h_cand_full)
  diag = torch.arange(H, device=h.device)
  passthrough[:, diag, diag] = z_t * h

  C_h_full = C_h_cand_full + passthrough                # [B, H, H]

  if return_ratio:
    # 计算贡献值除以输出的ratio
    # 注意：这里h_final是最终的输出值
    h_final_unsqueezed = h_final.unsqueeze(1)  # [B, 1, H]
    
    # 计算ratio，避免除零
    ratio_C_x = torch.where(
      h_final_unsqueezed != 0,
      C_x_full / h_final_unsqueezed,
      torch.zeros_like(C_x_full)
    )

    ratio_C_h = torch.where(
      h_final_unsqueezed != 0,
      C_h_full / h_final_unsqueezed,
      torch.zeros_like(C_h_full)
    )
    
    return ratio_C_x, ratio_C_h
  else:
    return C_x_full, C_h_full


@torch.no_grad()
def lrp_rnncell_contrib_full(x: torch.Tensor, h: torch.Tensor, rnn: nn.RNNCell, eps: float = 1e-12, return_ratio: bool = False):
  """
  RNNCell（tanh）将输出逐维分配到输入消息 x 与旧隐状态 h 的逐维贡献。
  返回：
    - C_x_full [B,I,H]
    - C_h_full [B,H,H]
  
  Args:
    return_ratio: 如果为True，返回贡献值除以输出的ratio值；如果为False，返回原始贡献值
  """
  W_in = rnn.weight_ih
  W_hh = rnn.weight_hh
  b_in = rnn.bias_ih if rnn.bias_ih is not None else 0.0
  b_hh = rnn.bias_hh if rnn.bias_hh is not None else 0.0

  pre = x @ W_in.t() + h @ W_hh.t() + b_in + b_hh      # [B,H]
  h_new = torch.tanh(pre)

  denom = (x @ W_in.t()) + (h @ W_hh.t())              # [B,H]

  Zx = x.unsqueeze(2) * W_in.t().unsqueeze(0)          # [B,I,H]
  ratio_x = torch.where(denom.unsqueeze(1).abs() > eps,
                        Zx / denom.unsqueeze(1),
                        torch.zeros_like(Zx))
  C_x_full = ratio_x * h_new.unsqueeze(1)

  Zh = h.unsqueeze(2) * W_hh.t().unsqueeze(0)          # [B,H,H]
  ratio_h = torch.where(denom.unsqueeze(1).abs() > eps,
                        Zh / denom.unsqueeze(1),
                        torch.zeros_like(Zh))
  C_h_full = ratio_h * h_new.unsqueeze(1)

  if return_ratio:
    # 计算贡献值除以输出的ratio
    h_new_unsqueezed = h_new.unsqueeze(1)  # [B, 1, H]
    
    # 计算ratio，避免除零
    ratio_C_x = torch.where(
      torch.abs(h_new_unsqueezed) > eps,
      C_x_full / h_new_unsqueezed,
      torch.zeros_like(C_x_full)
    )
    
    ratio_C_h = torch.where(
      torch.abs(h_new_unsqueezed) > eps,
      C_h_full / h_new_unsqueezed,
      torch.zeros_like(C_h_full)
    )
    
    return ratio_C_x, ratio_C_h
  else:
    return C_x_full, C_h_full


