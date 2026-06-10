import numpy as np
import torch


class MergeLayer(torch.nn.Module):
  def __init__(self, dim1, dim2, dim3, dim4):
    super().__init__()
    self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
    self.fc2 = torch.nn.Linear(dim3, dim4)
    self.act = torch.nn.ReLU()

    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)

  def forward(self, x1, x2):
    self.fc1.weight.data = self.fc1.weight.to(dtype=torch.float64)
    self.fc2.weight.data = self.fc2.weight.to(dtype=torch.float64)
    if self.fc1.bias is not None:
      self.fc1.bias.data = self.fc1.bias.data.to(dtype=torch.float64)
    if self.fc2.bias is not None:
      self.fc2.bias.data = self.fc2.bias.data.to(dtype=torch.float64)

    x1 = x1.to(dtype=torch.float64)
    x2 = x2.to(dtype=torch.float64)

    # print('x1.shape',x1.shape)
    # print('x2.shape',x2.shape)

    x = torch.cat([x1, x2], dim=1)
    h = self.act(self.fc1(x))
    return self.fc2(h)

  def linear_contribution(self,weight,input,out):
      # print('weight',weight.shape)
      # print('input', input.shape)
      # print('out',out.shape)
      Z = input.unsqueeze(2) * weight.unsqueeze(0)

      # print('Z',Z.shape)
      S = Z.sum(dim=1)  # [B, D_out]  分母

      den = S.unsqueeze(1)  # [B, 1, D_out] 便于广播
      phi = torch.where(den != 0, Z / den, torch.zeros_like(Z))  # 分母为0 → 0
      # phi =Z / (den+1e-4)

      C = phi
      print('phi',phi.shape)
      # print(phi.sum(dim=(1)))
      #


      ok_mask = torch.isclose(phi.sum(dim=1), torch.ones_like(phi.sum(dim=1)), atol=1e-4)
      print("全部匹配吗:", ok_mask.all().item())
      if not ok_mask.all():
          # 获取不匹配的 (batch_idx, out_idx) 坐标
          mismatch_coords = (~ok_mask).nonzero(as_tuple=False)  # [N_mismatch, 2]
          print("不匹配坐标:\n", mismatch_coords)

          # 还可以查看这些位置的值对比
          for b_idx, o_idx in mismatch_coords:
              pred_val = C.sum(dim=1)[b_idx, o_idx].item()
              true_val = out[b_idx, o_idx].item()
              diff = pred_val - true_val
              print(f"[b={b_idx}, o={o_idx}] 预测={pred_val:.6f}, 真值={true_val:.6f}, 差值={diff:.6e}")


      return phi

  def merge_contribution(self, x1, x2):
    """
    计算MergeLayer的贡献分配

    Args:
        x1: 第一个输入 [B, dim1]
        x2: 第二个输入 [B, dim2]
        output: 最终输出 [B, dim4]

    Returns:
        x1_contributions: x1对输出的贡献 [B, dim1, dim4]
        x2_contributions: x2对输出的贡献 [B, dim2, dim4]
    """
    # 前向传播，保存中间结果
    self.fc1.weight.data = self.fc1.weight.to(dtype=torch.float64)
    self.fc2.weight.data = self.fc2.weight.to(dtype=torch.float64)
    if self.fc1.bias is not None:
      self.fc1.bias.data = self.fc1.bias.data.to(dtype=torch.float64)
    if self.fc2.bias is not None:
      self.fc2.bias.data = self.fc2.bias.data.to(dtype=torch.float64)

    x1 = x1.to(dtype=torch.float64)
    x2 = x2.to(dtype=torch.float64)

    x = torch.cat([x1, x2], dim=1)  # [B, dim1+dim2]
    # h = self.act(self.fc1(x))  # [B, dim3]
    # final_output = self.fc2(h)  # [B, dim4]

    h1 = self.fc1(x)  # [B, 80]
    h1_act = self.act(h1)  # [B, 80]

    final_output = self.fc2(h1_act)  # [B, 10]

    # 计算各层的贡献
    # 第2层：h -> final_output
    C_layer2 = self.linear_contribution(self.fc2.weight.t(),h1_act, final_output)  # [B, dim3, dim4]

    # 第1层：x -> h
    C_layer1 = self.linear_contribution(self.fc1.weight.t(), x, h1)  # [B, dim1+dim2, dim3]

    # 计算输入对最终输出的贡献（链式法则）
    # [B, dim1+dim2, dim3] @ [B, dim3, dim4] = [B, dim1+dim2, dim4]
    input_to_output = torch.bmm(C_layer1, C_layer2)

    B, _= x1.shape

    input_to_output = input_to_output * final_output.unsqueeze(1)

    # 分离x1和x2的贡献
    x1_contributions = input_to_output[:, :x1.shape[1], :]  # [B, dim1, dim4]
    x2_contributions = input_to_output[:, x1.shape[1]:, :]  # [B, dim2, dim4]



    # 验证贡献守恒
    x1_sum = x1_contributions.sum(dim=1)  # [B, dim4]
    x2_sum = x2_contributions.sum(dim=1)  # [B, dim4]

    # print('x1_sum',x1_sum)
    # print('x2_sum', x2_sum)

    total_contrib = x1_sum + x2_sum

    print('total_contrib',total_contrib.shape)
    print('final_output',final_output.shape)

    print(f"MergeLayer贡献守恒验证: {torch.allclose(total_contrib, final_output, atol=1e-4)}")

    return final_output,x1_contributions, x2_contributions

  def forward_with_contributions(self, x1, x2):
    """
    前向传播并计算贡献

    Args:
        x1: 第一个输入 [B, dim1]
        x2: 第二个输入 [B, dim2]

    Returns:
        output: 最终输出 [B, dim4]
        x1_contributions: x1对输出的贡献 [B, dim1, dim4]
        x2_contributions: x2对输出的贡献 [B, dim2, dim4]
    """
    # 计算输出
    # output = self.forward(x1, x2)
    # if x1.dtype != self.fc1.weight.dtype:
    #   x1 = x1.to(dtype=self.fc1.weight.dtype)
    # if x2.dtype != self.fc1.weight.dtype:
    #   x2 = x2.to(dtype=self.fc1.weight.dtype)
    self.fc1.weight.data=self.fc1.weight.to(dtype=torch.float64)
    self.fc2.weight.data = self.fc2.weight.to(dtype=torch.float64)
    if self.fc1.bias is not None:
      self.fc1.bias.data = self.fc1.bias.data.to(dtype=torch.float64)
    if self.fc2.bias is not None:
      self.fc2.bias.data = self.fc2.bias.data.to(dtype=torch.float64)

    x1 = x1.to(dtype=torch.float64)
    x2 = x2.to(dtype=torch.float64)


    output = self.forward(x1, x2)


    # 计算贡献
    _, x1_contributions, x2_contributions = self.merge_contribution(x1, x2)

    return output, x1_contributions, x2_contributions




class MLP(torch.nn.Module):
  def __init__(self, dim, out_dim,drop=0.3,):
    super().__init__()
    self.fc_1 = torch.nn.Linear(dim, dim)
    self.fc_2 = torch.nn.Linear(dim, dim)
    self.fc_3 = torch.nn.Linear(dim, out_dim)
    self.act = torch.nn.ReLU()
    self.dropout = torch.nn.Dropout(p=drop, inplace=False)

  def forward(self, x):
    if x.dtype != self.fc_1.weight.dtype:
      x = x.to(dtype=self.fc_1.weight.dtype)

    x = self.act(self.fc_1(x))
    x = self.dropout(x)
    x = self.act(self.fc_2(x))
    x = self.dropout(x)
    # print('x',self.fc_3(x))
    # print('out_put',self.fc_3(x).squeeze(dim=1))
    return self.fc_3(x).squeeze(dim=1)

  def linear_contribution(self,weight,input,out):
      Z = input.unsqueeze(2) * weight.unsqueeze(0)
      S = Z.sum(dim=1)  # [B, D_out]  分母

      den = S.unsqueeze(1)  # [B, 1, D_out] 便于广播
      phi = torch.where(den != 0, Z / den, torch.zeros_like(Z))  # 分母为0 → 0

      C = phi

      # ok_mask = torch.isclose(C.sum(dim=1), out, atol=1e-4)
      # print("全部匹配吗:", ok_mask.all().item())
      # if not ok_mask.all():
      #     # 获取不匹配的 (batch_idx, out_idx) 坐标
      #     mismatch_coords = (~ok_mask).nonzero(as_tuple=False)  # [N_mismatch, 2]
      #     print("不匹配坐标:\n", mismatch_coords)
      #
      #     # 还可以查看这些位置的值对比
      #     for b_idx, o_idx in mismatch_coords:
      #         pred_val = C.sum(dim=1)[b_idx, o_idx].item()
      #         true_val = out[b_idx, o_idx].item()
      #         diff = pred_val - true_val
      #         print(f"[b={b_idx}, o={o_idx}] 预测={pred_val:.6f}, 真值={true_val:.6f}, 差值={diff:.6e}")
      #
      # # print(C.sum(dim=1))
      # #
      # # print(out)
      #
      # # print('linear 1',torch.allclose(C.sum(dim=1), out, atol=1e-4))
      return C





  def compute_contributions(self, x):
    """
    计算整个MLP的贡献分配（不考虑dropout）

    Args:
        x: 输入张量 [batch_size, input_dim]

    Returns:
        input_contributions: 输入对最终输出的贡献 [batch_size, input_dim, 1]
    """
    batch_size = x.shape[0]

    self.fc_1.weight.data = self.fc_1.weight.to(dtype=torch.float64)
    self.fc_2.weight.data = self.fc_2.weight.to(dtype=torch.float64)
    self.fc_3.weight.data = self.fc_3.weight.to(dtype=torch.float64)
    if self.fc_1.bias is not None:
      self.fc_1.bias.data = self.fc_1.bias.data.to(dtype=torch.float64)
    if self.fc_2.bias is not None:
      self.fc_2.bias.data = self.fc_2.bias.data.to(dtype=torch.float64)
    if self.fc_3.bias is not None:
      self.fc_3.bias.data = self.fc_3.bias.data.to(dtype=torch.float64)

    x = x.to(dtype=torch.float64)

    # if x.dtype != self.fc_1.weight.dtype:
    #   x = x.to(dtype=self.fc_1.weight.dtype)

    B, Din = x.shape

    # 前向传播，保存中间结果（跳过dropout）
    h1 = self.fc_1(x)  # [B, 80]
    h1_act = self.act(h1)  # [B, 80]

    h2 = self.fc_2(h1_act)  # [B, 10]
    h2_act = self.act(h2)  # [B, 10]

    output = self.fc_3(h2_act)  # [B, 1]
    final_output = output.squeeze(dim=1)  # [B]

    # 计算各层的贡献
    # 第3层：h2_act -> output
    C_layer3 = self.linear_contribution(self.fc_3.weight.t(), h2_act, output)  # [B, 10, 1]

    # print('C_layer3.shape',C_layer3.shape)

    # 第2层：h1_act -> h2
    C_layer2 = self.linear_contribution(self.fc_2.weight.t(), h1_act, h2)  # [B, 80, 10]

    # print('C_layer2.shape', C_layer2.shape)

    # 第1层：x -> h1
    C_layer1 = self.linear_contribution(self.fc_1.weight.t(), x, h1)  # [B, input_dim, 80]

    # print('C_layer1.shape', C_layer1.shape)

    # 计算输入对最终输出的贡献（链式法则）
    # C_layer1: [B, input_dim, 80] -> 输入对h1的贡献
    # C_layer2: [B, 80, 10] -> h1对h2的贡献
    # C_layer3: [B, 10, 1] -> h2对输出的贡献

    # 计算输入对h2的贡献
    # [B, input_dim, 80] @ [B, 80, 10] = [B, input_dim, 10]
    input_to_h2 = torch.bmm(C_layer1, C_layer2)

    # 计算输入对最终输出的贡献
    # [B, input_dim, 10] @ [B, 10, 1] = [B, input_dim, 1]
    input_to_output = torch.bmm(input_to_h2, C_layer3)

    # print(input_to_output.shape)
    # print(final_output.shape)

    input_to_output = input_to_output * final_output.unsqueeze(1)

    # print(input_to_output.shape)
    #
    # print(final_output.shape)

    # 验证最终贡献守恒
    final_contrib_sum = input_to_output.sum(dim=1).squeeze(-1)  # [B]


    # print(f"输入贡献总和: {final_contrib_sum}")
    # print(f"最终输出: {final_output}")
    print(f"贡献守恒验证: {torch.allclose(final_contrib_sum, final_output, atol=1e-4)}")

    difference = torch.abs(final_contrib_sum - final_output)
    max_diff = torch.max(difference)
    mean_diff = torch.mean(difference)
    # print(f"最大差异: {max_diff.item():.6f}")
    # print(f"平均差异: {mean_diff.item():.6f}")
    # print(f"差异范围: [{torch.min(difference).item():.6f}, {max_diff.item():.6f}]")

    return input_to_output  # [B, input_dim, 1]

  def forward_with_contributions(self, x,model):
    """
    前向传播并计算贡献（不考虑dropout）

    Args:
        x: 输入张量 [batch_size, input_dim]

    Returns:
        output: 最终输出 [batch_size]
        contributions: 输入贡献 [batch_size, input_dim, 1]
    """
    # 临时禁用dropout
    self.dropout.eval()

    if x.dtype != self.fc_1.weight.dtype:
      x = x.to(dtype=self.fc_1.weight.dtype)

    # 计算输出和贡献
    output = self.forward(x)
    contributions = self.compute_contributions(x)

    # contrib_edge_total, contrib_edge_input = self.edge_contrib_from_dict(model, edge_dict)


    # 恢复dropout训练模式
    # self.dropout.train()

    return output, contributions

class MLP_geography(torch.nn.Module):
  def __init__(self, dim, out_dim,drop=0.3,):
    super().__init__()
    self.fc_1 = torch.nn.Linear(dim, out_dim)
    # self.fc_2 = torch.nn.Linear(dim, dim)
    # self.fc_3 = torch.nn.Linear(dim, out_dim)
    self.act = torch.nn.ReLU()
    self.dropout = torch.nn.Dropout(p=drop, inplace=False)

  def forward(self, x):
    if x.dtype != self.fc_1.weight.dtype:
      x = x.to(dtype=self.fc_1.weight.dtype)

    # x = self.act(self.fc_1(x))
    # x = self.dropout(x)
    # x = self.act(self.fc_2(x))
    # x = self.dropout(x)
    # # print('x',self.fc_3(x))
    # # print('out_put',self.fc_3(x).squeeze(dim=1))
    # return self.act(self.fc_1(x)).squeeze(dim=1)
    return torch.sigmoid(self.fc_1(x)).squeeze(dim=1)

  def linear_contribution(self,weight,input,out):
      Z = input.unsqueeze(2) * weight.unsqueeze(0)
      S = Z.sum(dim=1)  # [B, D_out]  分母

      den = S.unsqueeze(1)  # [B, 1, D_out] 便于广播
      phi = torch.where(den != 0, Z / den, torch.zeros_like(Z))  # 分母为0 → 0

      C = phi

      # ok_mask = torch.isclose(C.sum(dim=1), out, atol=1e-4)
      # print("全部匹配吗:", ok_mask.all().item())
      # if not ok_mask.all():
      #     # 获取不匹配的 (batch_idx, out_idx) 坐标
      #     mismatch_coords = (~ok_mask).nonzero(as_tuple=False)  # [N_mismatch, 2]
      #     print("不匹配坐标:\n", mismatch_coords)
      #
      #     # 还可以查看这些位置的值对比
      #     for b_idx, o_idx in mismatch_coords:
      #         pred_val = C.sum(dim=1)[b_idx, o_idx].item()
      #         true_val = out[b_idx, o_idx].item()
      #         diff = pred_val - true_val
      #         print(f"[b={b_idx}, o={o_idx}] 预测={pred_val:.6f}, 真值={true_val:.6f}, 差值={diff:.6e}")
      #
      # # print(C.sum(dim=1))
      # #
      # # print(out)
      #
      # # print('linear 1',torch.allclose(C.sum(dim=1), out, atol=1e-4))
      return C





  def compute_contributions(self, x):
    """
    计算整个MLP的贡献分配（不考虑dropout）

    Args:
        x: 输入张量 [batch_size, input_dim]

    Returns:
        input_contributions: 输入对最终输出的贡献 [batch_size, input_dim, 1]
    """
    batch_size = x.shape[0]

    self.fc_1.weight.data = self.fc_1.weight.to(dtype=torch.float64)
    self.fc_2.weight.data = self.fc_2.weight.to(dtype=torch.float64)
    self.fc_3.weight.data = self.fc_3.weight.to(dtype=torch.float64)
    if self.fc_1.bias is not None:
      self.fc_1.bias.data = self.fc_1.bias.data.to(dtype=torch.float64)
    if self.fc_2.bias is not None:
      self.fc_2.bias.data = self.fc_2.bias.data.to(dtype=torch.float64)
    if self.fc_3.bias is not None:
      self.fc_3.bias.data = self.fc_3.bias.data.to(dtype=torch.float64)

    x = x.to(dtype=torch.float64)

    # if x.dtype != self.fc_1.weight.dtype:
    #   x = x.to(dtype=self.fc_1.weight.dtype)

    B, Din = x.shape

    # 前向传播，保存中间结果（跳过dropout）
    h1 = self.fc_1(x)  # [B, 80]
    h1_act = self.act(h1)  # [B, 80]

    h2 = self.fc_2(h1_act)  # [B, 10]
    h2_act = self.act(h2)  # [B, 10]

    output = self.fc_3(h2_act)  # [B, 1]
    final_output = output.squeeze(dim=1)  # [B]

    # 计算各层的贡献
    # 第3层：h2_act -> output
    C_layer3 = self.linear_contribution(self.fc_3.weight.t(), h2_act, output)  # [B, 10, 1]

    # print('C_layer3.shape',C_layer3.shape)

    # 第2层：h1_act -> h2
    C_layer2 = self.linear_contribution(self.fc_2.weight.t(), h1_act, h2)  # [B, 80, 10]

    # print('C_layer2.shape', C_layer2.shape)

    # 第1层：x -> h1
    C_layer1 = self.linear_contribution(self.fc_1.weight.t(), x, h1)  # [B, input_dim, 80]

    # print('C_layer1.shape', C_layer1.shape)

    # 计算输入对最终输出的贡献（链式法则）
    # C_layer1: [B, input_dim, 80] -> 输入对h1的贡献
    # C_layer2: [B, 80, 10] -> h1对h2的贡献
    # C_layer3: [B, 10, 1] -> h2对输出的贡献

    # 计算输入对h2的贡献
    # [B, input_dim, 80] @ [B, 80, 10] = [B, input_dim, 10]
    input_to_h2 = torch.bmm(C_layer1, C_layer2)

    # 计算输入对最终输出的贡献
    # [B, input_dim, 10] @ [B, 10, 1] = [B, input_dim, 1]
    input_to_output = torch.bmm(input_to_h2, C_layer3)

    # print(input_to_output.shape)
    # print(final_output.shape)

    input_to_output = input_to_output * final_output.unsqueeze(1)

    # print(input_to_output.shape)
    #
    # print(final_output.shape)

    # 验证最终贡献守恒
    final_contrib_sum = input_to_output.sum(dim=1).squeeze(-1)  # [B]


    # print(f"输入贡献总和: {final_contrib_sum}")
    # print(f"最终输出: {final_output}")
    print(f"贡献守恒验证: {torch.allclose(final_contrib_sum, final_output, atol=1e-4)}")

    difference = torch.abs(final_contrib_sum - final_output)
    max_diff = torch.max(difference)
    mean_diff = torch.mean(difference)
    # print(f"最大差异: {max_diff.item():.6f}")
    # print(f"平均差异: {mean_diff.item():.6f}")
    # print(f"差异范围: [{torch.min(difference).item():.6f}, {max_diff.item():.6f}]")

    return input_to_output  # [B, input_dim, 1]

  def forward_with_contributions(self, x,model):
    """
    前向传播并计算贡献（不考虑dropout）

    Args:
        x: 输入张量 [batch_size, input_dim]

    Returns:
        output: 最终输出 [batch_size]
        contributions: 输入贡献 [batch_size, input_dim, 1]
    """
    # 临时禁用dropout
    self.dropout.eval()

    if x.dtype != self.fc_1.weight.dtype:
      x = x.to(dtype=self.fc_1.weight.dtype)

    # 计算输出和贡献
    output = self.forward(x)
    contributions = self.compute_contributions(x)

    # contrib_edge_total, contrib_edge_input = self.edge_contrib_from_dict(model, edge_dict)


    # 恢复dropout训练模式
    # self.dropout.train()

    return output, contributions

class MLP_geography_v2(torch.nn.Module):
    def __init__(self, dim, out_dim, drop=0.3):
      super().__init__()
      hidden_dim = dim * 2
      self.fc_1 = torch.nn.Linear(dim, hidden_dim)
      self.fc_2 = torch.nn.Linear(hidden_dim, hidden_dim // 2)
      self.fc_3 = torch.nn.Linear(hidden_dim // 2, out_dim)
      self.act = torch.nn.ReLU()
      self.dropout = torch.nn.Dropout(p=drop)

    def forward(self, x):
      if x.dtype != self.fc_1.weight.dtype:
        x = x.to(dtype=self.fc_1.weight.dtype)
      x = self.act(self.fc_1(x))
      x = self.dropout(x)
      x = self.act(self.fc_2(x))
      x = self.dropout(x)
      # 不使用 sigmoid，直接输出并 clamp
      # return torch.clamp(self.fc_3(x), 0.0, 1.0).squeeze(dim=1)
      return self.fc_3(x).squeeze(dim=1)


class MLP_geography_v3(torch.nn.Module):
  def __init__(self, dim, out_dim, drop=0.3):
    super().__init__()
    hidden_dim = dim * 2
    self.fc_1 = torch.nn.Linear(dim, hidden_dim)
    self.fc_2 = torch.nn.Linear(hidden_dim, hidden_dim // 2)
    self.fc_3 = torch.nn.Linear(hidden_dim // 2, out_dim)
    self.act = torch.nn.ReLU()
    self.dropout = torch.nn.Dropout(p=drop)

  def forward(self, x):
    if x.dtype != self.fc_1.weight.dtype:
      x = x.to(dtype=self.fc_1.weight.dtype)
    x = self.act(self.fc_1(x))
    x = self.dropout(x)
    x = self.act(self.fc_2(x))
    x = self.dropout(x)
    # 不使用 sigmoid，直接输出并 clamp
    # return torch.clamp(self.fc_3(x), 0.0, 1.0).squeeze(dim=1)
    return torch.sigmoid(self.fc_3(x)).squeeze(dim=1)

class EarlyStopMonitor(object):
  def __init__(self, max_round=3, higher_better=True, tolerance=1e-10):
    self.max_round = max_round
    self.num_round = 0

    self.epoch_count = 0
    self.best_epoch = 0

    self.last_best = None
    self.higher_better = higher_better
    self.tolerance = tolerance

  def early_stop_check(self, curr_val):
    if not self.higher_better:
      curr_val *= -1
    if self.last_best is None:
      self.last_best = curr_val
    elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
      self.last_best = curr_val
      self.num_round = 0
      self.best_epoch = self.epoch_count
    else:
      self.num_round += 1

    self.epoch_count += 1

    return self.num_round >= self.max_round


class RandEdgeSampler(object):
  def __init__(self, src_list, dst_list, seed=None):
    self.seed = None
    self.src_list = np.unique(src_list)
    self.dst_list = np.unique(dst_list)

    if seed is not None:
      self.seed = seed
      self.random_state = np.random.RandomState(self.seed)

  def sample(self, size):
    if self.seed is None:
      src_index = np.random.randint(0, len(self.src_list), size)
      dst_index = np.random.randint(0, len(self.dst_list), size)
    else:

      src_index = self.random_state.randint(0, len(self.src_list), size)
      dst_index = self.random_state.randint(0, len(self.dst_list), size)
    return self.src_list[src_index], self.dst_list[dst_index]

  def reset_random_state(self):
    self.random_state = np.random.RandomState(self.seed)


def get_neighbor_finder(data, uniform, max_node_idx=None):
  max_node_idx = max(data.sources.max(), data.destinations.max()) if max_node_idx is None else max_node_idx
  # print('max_node_idx',max_node_idx)
  max_node_idx=int(max_node_idx)
  adj_list = [[] for _ in range(max_node_idx + 1)]
  for source, destination, edge_idx, timestamp in zip(data.sources, data.destinations,
                                                      data.edge_idxs,
                                                      data.timestamps):
    # print('source',source)
    adj_list[source].append((destination, edge_idx, timestamp))
    adj_list[destination].append((source, edge_idx, timestamp))

  return NeighborFinder(adj_list, uniform=uniform)


class NeighborFinder:
  def __init__(self, adj_list, uniform=False, seed=None):
    self.node_to_neighbors = []
    self.node_to_edge_idxs = []
    self.node_to_edge_timestamps = []

    for neighbors in adj_list:
      # Neighbors is a list of tuples (neighbor, edge_idx, timestamp)
      # We sort the list based on timestamp
      sorted_neighhbors = sorted(neighbors, key=lambda x: x[2])
      self.node_to_neighbors.append(np.array([x[0] for x in sorted_neighhbors]))
      self.node_to_edge_idxs.append(np.array([x[1] for x in sorted_neighhbors]))
      self.node_to_edge_timestamps.append(np.array([x[2] for x in sorted_neighhbors]))

    self.uniform = uniform

    if seed is not None:
      self.seed = seed
      self.random_state = np.random.RandomState(self.seed)

  def find_before(self, src_idx, cut_time):
    """
    Extracts all the interactions happening before cut_time for user src_idx in the overall interaction graph. The returned interactions are sorted by time.

    Returns 3 lists: neighbors, edge_idxs, timestamps

    """
    i = np.searchsorted(self.node_to_edge_timestamps[src_idx], cut_time)

    return self.node_to_neighbors[src_idx][:i], self.node_to_edge_idxs[src_idx][:i], self.node_to_edge_timestamps[src_idx][:i]

  def get_temporal_neighbor(self, source_nodes, timestamps, n_neighbors=20):
    """
    Given a list of users ids and relative cut times, extracts a sampled temporal neighborhood of each user in the list.

    Params
    ------
    src_idx_l: List[int]
    cut_time_l: List[float],
    num_neighbors: int
    """
    assert (len(source_nodes) == len(timestamps))

    tmp_n_neighbors = n_neighbors if n_neighbors > 0 else 1
    # NB! All interactions described in these matrices are sorted in each row by time
    neighbors = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.int32)  # each entry in position (i,j) represent the id of the item targeted by user src_idx_l[i] with an interaction happening before cut_time_l[i]
    edge_times = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.float32)  # each entry in position (i,j) represent the timestamp of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]
    edge_idxs = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.int32)  # each entry in position (i,j) represent the interaction index of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]

    for i, (source_node, timestamp) in enumerate(zip(source_nodes, timestamps)):
      source_neighbors, source_edge_idxs, source_edge_times = self.find_before(source_node,
                                                   timestamp)  # extracts all neighbors, interactions indexes and timestamps of all interactions of user source_node happening before cut_time

      if len(source_neighbors) > 0 and n_neighbors > 0:
        if self.uniform:  # if we are applying uniform sampling, shuffles the data above before sampling
          sampled_idx = np.random.randint(0, len(source_neighbors), n_neighbors)

          neighbors[i, :] = source_neighbors[sampled_idx]
          edge_times[i, :] = source_edge_times[sampled_idx]
          edge_idxs[i, :] = source_edge_idxs[sampled_idx]

          # re-sort based on time
          pos = edge_times[i, :].argsort()
          neighbors[i, :] = neighbors[i, :][pos]
          edge_times[i, :] = edge_times[i, :][pos]
          edge_idxs[i, :] = edge_idxs[i, :][pos]
        else:
          # Take most recent interactions
          source_edge_times = source_edge_times[-n_neighbors:]
          source_neighbors = source_neighbors[-n_neighbors:]
          source_edge_idxs = source_edge_idxs[-n_neighbors:]

          assert (len(source_neighbors) <= n_neighbors)
          assert (len(source_edge_times) <= n_neighbors)
          assert (len(source_edge_idxs) <= n_neighbors)

          neighbors[i, n_neighbors - len(source_neighbors):] = source_neighbors
          edge_times[i, n_neighbors - len(source_edge_times):] = source_edge_times
          edge_idxs[i, n_neighbors - len(source_edge_idxs):] = source_edge_idxs

    return neighbors, edge_idxs, edge_times

  def find_k_hop(self, k, src_idx_l, cut_time_l, num_neighbors, e_idx_l=None):
    if k == 0:
      return ([], [], [])
    batch = len(src_idx_l)
    layer_i = 0
    x, y, z = self.get_temporal_neighbor(src_idx_l, cut_time_l, num_neighbors,
                                         )  # each: [batch, num_neighbors]
    node_records = [x]
    eidx_records = [y]
    t_records = [z]
    for layer_i in range(1, k):
      ngh_node_est, ngh_e_est, ngh_t_est = node_records[-1], eidx_records[-1], t_records[-1]
      ngh_node_est = ngh_node_est.flatten()
      ngh_e_est = ngh_e_est.flatten()  # [batch * num_neighbors]
      ngh_t_est = ngh_t_est.flatten()
      out_ngh_node_batch, out_ngh_eidx_batch, out_ngh_t_batch = self.get_temporal_neighbor(ngh_node_est,
                                                                                           ngh_t_est,
                                                                                           num_neighbors,
                                                                                           )

      out_ngh_node_batch = out_ngh_node_batch.reshape(batch, -1)  # [batch, num_neighbors* num_neighbors]
      out_ngh_eidx_batch = out_ngh_eidx_batch.reshape(batch, -1)
      out_ngh_t_batch = out_ngh_t_batch.reshape(batch, -1)

      node_records.append(out_ngh_node_batch)
      eidx_records.append(out_ngh_eidx_batch)
      t_records.append(out_ngh_t_batch)

    return (node_records, eidx_records, t_records)

class MLP_video(torch.nn.Module):
    def __init__(self, dim, out_dim, drop=0.3, ):
      super().__init__()
      self.fc_1 = torch.nn.Linear(dim, dim)
      self.fc_2 = torch.nn.Linear(dim, out_dim)
      self.act = torch.nn.ReLU()
      self.dropout = torch.nn.Dropout(p=drop, inplace=False)

    def forward(self, x):
      if x.dtype != self.fc_1.weight.dtype:
        x = x.to(dtype=self.fc_1.weight.dtype)

      x = self.act(self.fc_1(x))
      x = self.dropout(x)
      return self.fc_2(x).squeeze(dim=1)
