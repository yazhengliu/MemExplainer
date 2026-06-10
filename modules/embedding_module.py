import torch
from torch import nn
import numpy as np
import math

from model.temporal_attention import TemporalAttentionLayer
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import MergeLayer

class CustomMultiHeadAttention(nn.Module):
    """
    自定义多头注意力实现，替代 nn.MultiheadAttention
    支持不同的 query、key、value 维度（kdim、vdim）
    """

    def __init__(self, embed_dim, num_heads, kdim=None, vdim=None, dropout=0.1, bias=True):
        """
        Args:
            embed_dim: query 的嵌入维度
            num_heads: 注意力头数
            kdim: key 的嵌入维度（如果为 None，则等于 embed_dim）
            vdim: value 的嵌入维度（如果为 None，则等于 embed_dim）
            dropout: dropout 概率
            bias: 是否使用偏置
        """
        super(CustomMultiHeadAttention, self).__init__()

        # 如果没有指定 kdim 和 vdim，则使用 embed_dim
        if kdim is None:
            kdim = embed_dim
        if vdim is None:
            vdim = embed_dim

        self.embed_dim = embed_dim
        self.kdim = kdim
        self.vdim = vdim
        self.num_heads = num_heads

        # 确保每个头的维度是整数
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) 必须能被 num_heads ({num_heads}) 整除"
        assert kdim % num_heads == 0, f"kdim ({kdim}) 必须能被 num_heads ({num_heads}) 整除"
        assert vdim % num_heads == 0, f"vdim ({vdim}) 必须能被 num_heads ({num_heads}) 整除"

        self.head_dim = embed_dim // num_heads
        self.k_head_dim = kdim // num_heads
        self.v_head_dim = vdim // num_heads

        # Query, Key, Value 线性投影层
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(kdim, embed_dim, bias=bias)  # key 投影到 embed_dim
        self.v_proj = nn.Linear(vdim, embed_dim, bias=bias)  # value 投影到 embed_dim

        # 输出投影层
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # self.merger = MergeLayer(self.embed_dim, n_node_features, n_node_features, output_dimension)

        # 初始化参数
        self._reset_parameters()

        self.double()

    def load_state_from_pytorch_mha(self, pytorch_mha):
        """
        从 PyTorch 的 nn.MultiheadAttention 加载权重参数

        Args:
            pytorch_mha: nn.MultiheadAttention 实例
        """
        with torch.no_grad():
            # 检查是否使用了分离的 q_proj_weight, k_proj_weight, v_proj_weight
            # 当 kdim != embed_dim 或 vdim != embed_dim 时，PyTorch 使用分离的权重
            if hasattr(pytorch_mha, 'q_proj_weight') and pytorch_mha.q_proj_weight is not None:
                # 使用分离的 q, k, v 权重，转换为 float64
                self.q_proj.weight.copy_(pytorch_mha.q_proj_weight.to(dtype=torch.float64))
                self.k_proj.weight.copy_(pytorch_mha.k_proj_weight.to(dtype=torch.float64))
                self.v_proj.weight.copy_(pytorch_mha.v_proj_weight.to(dtype=torch.float64))

                if pytorch_mha.in_proj_bias is not None:
                    embed_dim = self.embed_dim
                    self.q_proj.bias.copy_(pytorch_mha.in_proj_bias[:embed_dim].to(dtype=torch.float64))
                    self.k_proj.bias.copy_(pytorch_mha.in_proj_bias[embed_dim:2 * embed_dim].to(dtype=torch.float64))
                    self.v_proj.bias.copy_(pytorch_mha.in_proj_bias[2 * embed_dim:].to(dtype=torch.float64))
            else:
                # 使用合并的 in_proj_weight（当 kdim == vdim == embed_dim 时）
                in_proj_weight = pytorch_mha.in_proj_weight  # [3 * embed_dim, embed_dim]
                embed_dim = self.embed_dim

                q_proj_weight = in_proj_weight[:embed_dim, :].to(dtype=torch.float64)
                k_proj_weight = in_proj_weight[embed_dim:2 * embed_dim, :].to(dtype=torch.float64)
                v_proj_weight = in_proj_weight[2 * embed_dim:, :].to(dtype=torch.float64)

                self.q_proj.weight.copy_(q_proj_weight)
                self.k_proj.weight.copy_(k_proj_weight)
                self.v_proj.weight.copy_(v_proj_weight)

                if pytorch_mha.in_proj_bias is not None:
                    in_proj_bias = pytorch_mha.in_proj_bias.to(dtype=torch.float64)
                    self.q_proj.bias.copy_(in_proj_bias[:embed_dim])
                    self.k_proj.bias.copy_(in_proj_bias[embed_dim:2 * embed_dim])
                    self.v_proj.bias.copy_(in_proj_bias[2 * embed_dim:])

            # 复制输出投影层的权重，转换为 float64
            self.out_proj.weight.copy_(pytorch_mha.out_proj.weight.to(dtype=torch.float64))
            if pytorch_mha.out_proj.bias is not None and self.out_proj.bias is not None:
                self.out_proj.bias.copy_(pytorch_mha.out_proj.bias.to(dtype=torch.float64))

    def _reset_parameters(self):
        """初始化参数"""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0)
        if self.k_proj.bias is not None:
            nn.init.constant_(self.k_proj.bias, 0)
        if self.v_proj.bias is not None:
            nn.init.constant_(self.v_proj.bias, 0)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None, need_weights=True):
        """
        Args:
            query: [seq_len_q, batch_size, embed_dim] - PyTorch MultiheadAttention 标准格式
            key: [seq_len_k, batch_size, kdim]
            value: [seq_len_v, batch_size, vdim]
            key_padding_mask: [batch_size, seq_len_k], True 表示需要 mask（padding）
            attn_mask: attention mask（可选）
            need_weights: 是否需要返回 attention weights

        Returns:
            output: [seq_len_q, batch_size, embed_dim]
            attn_weights: [batch_size, seq_len_q, seq_len_k] (如果 need_weights=True)
        """


        seq_len_q, batch_size, _ = query.shape
        seq_len_k = key.shape[0]
        seq_len_v = value.shape[0]

        model_dtype = next(self.parameters()).dtype
        model_device = next(self.parameters()).device

        # print('query.shape', query.shape)
        # print('key.shape', key.shape)
        # print('value.shape', value.shape)

        query = query.to(dtype=model_dtype, device=model_device)
        key = key.to(dtype=model_dtype, device=model_device)
        value = value.to(dtype=model_dtype, device=model_device)


        # 转换为 [batch_size, seq_len, embed_dim] 格式
        query_bt = query.transpose(0, 1)  # [batch_size, seq_len_q, embed_dim]
        key_bt = key.transpose(0, 1)  # [batch_size, seq_len_k, kdim]
        value_bt = value.transpose(0, 1)  # [batch_size, seq_len_v, vdim]

        # 线性投影
        Q = self.q_proj(query_bt)  # [batch_size, seq_len_q, embed_dim]
        K = self.k_proj(key_bt)  # [batch_size, seq_len_k, embed_dim]
        V = self.v_proj(value_bt)  # [batch_size, seq_len_v, embed_dim]

        # 重塑为多头格式
        Q = Q.view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        # [batch_size, num_heads, seq_len_q, head_dim]
        K = K.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        # [batch_size, num_heads, seq_len_k, head_dim]
        V = V.view(batch_size, seq_len_v, self.num_heads, self.head_dim).transpose(1, 2)
        # [batch_size, num_heads, seq_len_v, head_dim]

        # Scaled Dot-Product Attention
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # [batch_size, num_heads, seq_len_q, seq_len_k]

        # 处理 key_padding_mask
        if key_padding_mask is not None:
            # key_padding_mask: [batch_size, seq_len_k], True 表示需要 mask
            # 扩展到所有头：[batch_size, 1, 1, seq_len_k]
            key_padding_mask_expanded = key_padding_mask.unsqueeze(1).unsqueeze(2)
            key_padding_mask_expanded = key_padding_mask_expanded.expand(
                batch_size, self.num_heads, seq_len_q, seq_len_k
            )
            attn_weights = attn_weights.masked_fill(key_padding_mask_expanded, float('-inf'))

        # 处理 attn_mask
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_weights = attn_weights.masked_fill(attn_mask, float('-inf'))
            else:
                attn_weights = attn_weights + attn_mask

        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Dropout
        attn_weights = self.dropout(attn_weights)



        # 计算注意力输出
        attn_output = torch.matmul(attn_weights, V)
        # [batch_size, num_heads, seq_len_q, head_dim]

        # 合并所有头
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.embed_dim
        )  # [batch_size, seq_len_q, embed_dim]

        # 输出投影
        output = self.out_proj(attn_output)  # [batch_size, seq_len_q, embed_dim]

        # 转换回 PyTorch 格式：[seq_len, batch_size, embed_dim]
        output = output.transpose(0, 1)  # [seq_len_q, batch_size, embed_dim]

        # 处理 attention weights
        if need_weights:
            # 对所有头求平均
            attn_weights_avg = attn_weights.mean(dim=1)  # [batch_size, seq_len_q, seq_len_k]
            return output, attn_weights_avg
        else:
            return output


    def forward_weights(self, query, key, value, key_padding_mask=None, attn_mask=None, explain_weights=None,need_weights=True):
        """
        Args:
            query: [seq_len_q, batch_size, embed_dim] - PyTorch MultiheadAttention 标准格式
            key: [seq_len_k, batch_size, kdim]
            value: [seq_len_v, batch_size, vdim]
            key_padding_mask: [batch_size, seq_len_k], True 表示需要 mask（padding）
            attn_mask: attention mask（可选）
            need_weights: 是否需要返回 attention weights

        Returns:
            output: [seq_len_q, batch_size, embed_dim]
            attn_weights: [batch_size, seq_len_q, seq_len_k] (如果 need_weights=True)
        """


        seq_len_q, batch_size, _ = query.shape
        seq_len_k = key.shape[0]
        seq_len_v = value.shape[0]

        model_dtype = next(self.parameters()).dtype
        model_device = next(self.parameters()).device

        # print('query.shape', query.shape)
        # print('key.shape', key.shape)
        # print('value.shape', value.shape)

        query = query.to(dtype=model_dtype, device=model_device)
        key = key.to(dtype=model_dtype, device=model_device)
        value = value.to(dtype=model_dtype, device=model_device)


        # 转换为 [batch_size, seq_len, embed_dim] 格式
        query_bt = query.transpose(0, 1)  # [batch_size, seq_len_q, embed_dim]
        key_bt = key.transpose(0, 1)  # [batch_size, seq_len_k, kdim]
        value_bt = value.transpose(0, 1)  # [batch_size, seq_len_v, vdim]

        # 线性投影
        Q = self.q_proj(query_bt)  # [batch_size, seq_len_q, embed_dim]
        K = self.k_proj(key_bt)  # [batch_size, seq_len_k, embed_dim]
        V = self.v_proj(value_bt)  # [batch_size, seq_len_v, embed_dim]

        # 重塑为多头格式
        Q = Q.view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        # [batch_size, num_heads, seq_len_q, head_dim]
        K = K.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        # [batch_size, num_heads, seq_len_k, head_dim]
        V = V.view(batch_size, seq_len_v, self.num_heads, self.head_dim).transpose(1, 2)
        # [batch_size, num_heads, seq_len_v, head_dim]

        # Scaled Dot-Product Attention
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # [batch_size, num_heads, seq_len_q, seq_len_k]

        # 处理 key_padding_mask
        if key_padding_mask is not None:
            # key_padding_mask: [batch_size, seq_len_k], True 表示需要 mask
            # 扩展到所有头：[batch_size, 1, 1, seq_len_k]
            key_padding_mask_expanded = key_padding_mask.unsqueeze(1).unsqueeze(2)
            key_padding_mask_expanded = key_padding_mask_expanded.expand(
                batch_size, self.num_heads, seq_len_q, seq_len_k
            )
            attn_weights = attn_weights.masked_fill(key_padding_mask_expanded, float('-inf'))

        # 处理 attn_mask
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_weights = attn_weights.masked_fill(attn_mask, float('-inf'))
            else:
                attn_weights = attn_weights + attn_mask

        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Dropout
        attn_weights = self.dropout(attn_weights)

        if explain_weights is not None:
            attn_weights=attn_weights*explain_weights

        # 计算注意力输出
        attn_output = torch.matmul(attn_weights, V)
        # [batch_size, num_heads, seq_len_q, head_dim]

        # 合并所有头
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.embed_dim
        )  # [batch_size, seq_len_q, embed_dim]

        # 输出投影
        output = self.out_proj(attn_output)  # [batch_size, seq_len_q, embed_dim]

        # 转换回 PyTorch 格式：[seq_len, batch_size, embed_dim]
        output = output.transpose(0, 1)  # [seq_len_q, batch_size, embed_dim]

        # 处理 attention weights
        if need_weights:
            # 对所有头求平均
            attn_weights_avg = attn_weights.mean(dim=1)  # [batch_size, seq_len_q, seq_len_k]
            return output, attn_weights_avg
        else:
            return output

    def linear_contribution(self, weight, input, out):
        """
        LRP 线性层贡献计算
        Args:
            weight: [D_out, D_in] - 权重矩阵的转置格式（Linear层的weight）
            input: [B, D_in] 或 [B, seq_len, D_in] - 输入
            out: [B, D_out] 或 [B, seq_len, D_out] - 输出
        Returns:
            C: [B, D_in, D_out] 或 [B, seq_len, D_in, D_out] - 贡献矩阵
                表示输入每维对输出每维的贡献
        """
        if input.dim() == 4:  # [B, H, seq_len, D_in] - 4维输入（多头情况）
            B, H, seq_len, D_in = input.shape
            D_out = out.shape[3]

            # 展平为 [B * H * seq_len, D_in] 和 [B * H * seq_len, D_out]
            input_flat = input.view(B * H * seq_len, D_in)
            out_flat = out.view(B * H * seq_len, D_out)

            # LRP 规则
            Z = input_flat.unsqueeze(2) * weight.t().unsqueeze(0)  # [B*H*seq_len, D_in, D_out]
            S = Z.sum(dim=1)  # [B*H*seq_len, D_out] - 分母

            # 计算比例
            den = S.unsqueeze(1)  # [B*H*seq_len, 1, D_out]
            phi = torch.where(den != 0, Z / den, torch.zeros_like(Z))  # 分母为0 → 0

            # 乘以输出得到贡献值
            C = phi * out_flat.unsqueeze(1)  # [B*H*seq_len, D_in, D_out]
            C = C.view(B, H, seq_len, D_in, D_out)  # [B, H, seq_len, D_in, D_out]
        elif input.dim() == 3:  # [B, seq_len, D_in]
            B, seq_len, D_in = input.shape
            D_out = out.shape[2]

            # 展平为 [B * seq_len, D_in] 和 [B * seq_len, D_out]
            input_flat = input.view(B * seq_len, D_in)
            out_flat = out.view(B * seq_len, D_out)

            # LRP 规则：C[i, j] = (input[i] * weight[j, i]) / sum(input * weight[:, i]) * out[j]
            # Z: [B*seq_len, D_in, D_out] = input[i] * weight[j, i]
            Z = input_flat.unsqueeze(2) * weight.t().unsqueeze(0)  # [B*seq_len, D_in, D_out]
            S = Z.sum(dim=1)  # [B*seq_len, D_out] - 分母

            # 计算比例
            den = S.unsqueeze(1)  # [B*seq_len, 1, D_out]
            phi = torch.where(den != 0, Z / den, torch.zeros_like(Z))  # 分母为0 → 0

            # 乘以输出得到贡献值
            C = phi * out_flat.unsqueeze(1)  # [B*seq_len, D_in, D_out]
            C = C.view(B, seq_len, D_in, D_out)  # [B, seq_len, D_in, D_out]
        else:  # [B, D_in]
            B, D_in = input.shape
            D_out = out.shape[1]

            # LRP 规则
            Z = input.unsqueeze(2) * weight.t().unsqueeze(0)  # [B, D_in, D_out]
            S = Z.sum(dim=1)  # [B, D_out] - 分母
            den = S.unsqueeze(1)  # [B, 1, D_out]
            phi = torch.where(den != 0, Z / den, torch.zeros_like(Z))
            C = phi * out.unsqueeze(1)  # [B, D_in, D_out]

        return C

    def linear_contribution_ratio(self, weight, input, out):
        """
        LRP 线性层贡献计算
        Args:
            weight: [D_out, D_in] - 权重矩阵的转置格式（Linear层的weight）
            input: [B, D_in] 或 [B, seq_len, D_in] - 输入
            out: [B, D_out] 或 [B, seq_len, D_out] - 输出
        Returns:
            C: [B, D_in, D_out] 或 [B, seq_len, D_in, D_out] - 贡献矩阵
                表示输入每维对输出每维的贡献
        """
        if input.dim() == 4:  # [B, H, seq_len, D_in] - 4维输入（多头情况）
            B, H, seq_len, D_in = input.shape
            D_out = out.shape[3]

            # 展平为 [B * H * seq_len, D_in] 和 [B * H * seq_len, D_out]
            input_flat = input.view(B * H * seq_len, D_in)
            out_flat = out.view(B * H * seq_len, D_out)

            # LRP 规则
            Z = input_flat.unsqueeze(2) * weight.t().unsqueeze(0)  # [B*H*seq_len, D_in, D_out]
            S = Z.sum(dim=1)  # [B*H*seq_len, D_out] - 分母

            # 计算比例
            den = S.unsqueeze(1)  # [B*H*seq_len, 1, D_out]
            phi = torch.where(den != 0, Z / den, torch.zeros_like(Z))  # 分母为0 → 0

            phi = phi.view(B, H, seq_len, D_in, D_out)


        elif input.dim() == 3:  # [B, seq_len, D_in]
            B, seq_len, D_in = input.shape
            D_out = out.shape[2]

            # 展平为 [B * seq_len, D_in] 和 [B * seq_len, D_out]
            input_flat = input.view(B * seq_len, D_in)
            out_flat = out.view(B * seq_len, D_out)

            # LRP 规则：C[i, j] = (input[i] * weight[j, i]) / sum(input * weight[:, i]) * out[j]
            # Z: [B*seq_len, D_in, D_out] = input[i] * weight[j, i]
            Z = input_flat.unsqueeze(2) * weight.t().unsqueeze(0)  # [B*seq_len, D_in, D_out]
            S = Z.sum(dim=1)  # [B*seq_len, D_out] - 分母

            # 计算比例
            den = S.unsqueeze(1)  # [B*seq_len, 1, D_out]
            phi = torch.where(den != 0, Z / den, torch.zeros_like(Z))  # 分母为0 → 0

            phi = phi.view(B, seq_len, D_in, D_out)  # [B, seq_len, D_in, D_out]
        else:  # [B, D_in]
            B, D_in = input.shape
            D_out = out.shape[1]

            # LRP 规则
            Z = input.unsqueeze(2) * weight.t().unsqueeze(0)  # [B, D_in, D_out]
            S = Z.sum(dim=1)  # [B, D_out] - 分母
            den = S.unsqueeze(1)  # [B, 1, D_out]
            phi = torch.where(den != 0, Z / den, torch.zeros_like(Z))
            C = phi * out.unsqueeze(1)  # [B, D_in, D_out]

        return phi

    def matrix_multiply_contribution(self, X, Y, Z, R_Z, transpose_Y=False, scale_factor=1.0):
        """
        计算矩阵乘法 Z = X @ Y (或 Z = X @ Y^T) 的贡献值分配
        完全按照原始逻辑封装，不改变任何计算步骤

        Args:
            X: 输入矩阵 X，形状 [..., dim_x, dim_shared]
                例如：attn_weights [B, H, seq_len_q, seq_len_k]
            Y: 输入矩阵 Y，形状 [..., dim_shared, dim_y] 或 [..., dim_y, dim_shared] (如果 transpose_Y=True)
                例如：V [B, H, seq_len_k, head_dim]
            Z: 输出矩阵 Z = X @ Y，形状 [..., dim_x, dim_y]
                例如：attn_output_multihead [B, H, seq_len_q, head_dim]
            R_Z: Z 的贡献值，形状与 Z 相同
                例如：R_attn_output [B, H, seq_len_q, head_dim]
            transpose_Y: 是否对 Y 进行转置
            scale_factor: 缩放因子（如 1/sqrt(head_dim)），默认 1.0

        Returns:
            C_X: X 的贡献值，形状 [..., dim_x, dim_shared] (对 dim_y 求和后)
            C_Y: Y 的贡献值，形状 [..., dim_shared, dim_y, dim_y] (使用对角矩阵后，对 dim_x 求和)
        """
        eps = 1e-10

        # 如果需要对 Y 转置，先转置
        if transpose_Y:
            Y = Y.transpose(-2, -1)

        # 获取维度信息
        X_shape = list(X.shape)
        Y_shape = list(Y.shape)
        batch_dims = X_shape[:-2]
        dim_x = X_shape[-2]
        dim_shared = X_shape[-1]
        dim_y = Y_shape[-1]

        # 完全按照原始逻辑：计算 attn_weights 和 V 的贡献
        # 扩展 X: [..., dim_x, dim_shared] -> [..., dim_x, dim_shared, 1]
        X_exp = X.unsqueeze(-1)  # [..., dim_x, dim_shared, 1]

        # 扩展 Y: [..., dim_shared, dim_y] -> [..., 1, dim_shared, dim_y]
        Y_exp = Y.unsqueeze(-3)  # [..., 1, dim_shared, dim_y]

        # 扩展 Z: [..., dim_x, dim_y] -> [..., dim_x, 1, dim_y]
        Z_exp = Z.unsqueeze(-2)  # [..., dim_x, 1, dim_y]

        # 扩展 R_Z: [..., dim_x, dim_y] -> [..., dim_x, 1, dim_y]
        R_Z_exp = R_Z.unsqueeze(-2)  # [..., dim_x, 1, dim_y]

        # 计算 X[i, k] * Y[k, j]
        X_expanded = X_exp.expand(*batch_dims, dim_x, dim_shared, dim_y)
        Y_expanded = Y_exp.expand(*batch_dims, dim_x, dim_shared, dim_y)
        XY_terms = X_expanded * Y_expanded  # [..., dim_x, dim_shared, dim_y]

        # 应用缩放因子
        if scale_factor != 1.0:
            XY_terms = XY_terms * scale_factor

        # 扩展 Z 和 R_Z
        Z_expanded = Z_exp.expand(*batch_dims, dim_x, dim_shared, dim_y)
        R_Z_expanded = R_Z_exp.expand(*batch_dims, dim_x, dim_shared, dim_y)

        # 计算贡献项：C_terms[i, k, j] = X[i, k] * Y[k, j] * R_Z[i, j] / Z[i, j]
        C_terms = torch.where(
            torch.abs(Z_expanded) > eps,
            XY_terms * R_Z_expanded / Z_expanded,
            torch.zeros_like(XY_terms)
        )  # [..., dim_x, dim_shared, dim_y]

        # 完全按照原始逻辑：计算 C_attn_weights（对 head_dim 维度求和）
        C_X = C_terms

        # 完全按照原始逻辑：计算 C_V（使用对角矩阵）
        C_terms_exp = C_terms.unsqueeze(-1)  # [..., dim_x, dim_shared, dim_y, 1]

        # 创建对角矩阵
        eye = torch.eye(dim_y, device=C_terms.device, dtype=C_terms.dtype)  # [dim_y, dim_y]
        eye_shape = [1] * (len(C_terms.shape) - 1) + [dim_y, dim_y]
        eye = eye.view(*eye_shape)  # [1, ..., 1, dim_y, dim_y]

        # 扩展 C_terms
        C_terms_expanded = C_terms_exp.expand(*batch_dims, dim_x, dim_shared, dim_y, dim_y)
        # [..., dim_x, dim_shared, dim_y, dim_y]

        # 应用对角矩阵
        C_Y_full = C_terms_expanded * eye
        # [..., dim_x, dim_shared, dim_y, dim_y]

        # 对 dim_x 维度求和（对应原始代码中的 sum(dim=2)，即对 seq_len_q 求和）
        C_Y = C_Y_full.sum(dim=-4)  # [..., dim_shared, dim_y, dim_y]

        return C_X, C_Y

    def matrix_multiply_contribution_ratio(self, X, Y, Z, R_Z, transpose_Y=False, scale_factor=1.0):
        """
        计算矩阵乘法 Z = X @ Y (或 Z = X @ Y^T) 的贡献值分配
        完全按照原始逻辑封装，不改变任何计算步骤

        Args:
            X: 输入矩阵 X，形状 [..., dim_x, dim_shared]
                例如：attn_weights [B, H, seq_len_q, seq_len_k]
            Y: 输入矩阵 Y，形状 [..., dim_shared, dim_y] 或 [..., dim_y, dim_shared] (如果 transpose_Y=True)
                例如：V [B, H, seq_len_k, head_dim]
            Z: 输出矩阵 Z = X @ Y，形状 [..., dim_x, dim_y]
                例如：attn_output_multihead [B, H, seq_len_q, head_dim]
            R_Z: Z 的贡献值，形状与 Z 相同
                例如：R_attn_output [B, H, seq_len_q, head_dim]
            transpose_Y: 是否对 Y 进行转置
            scale_factor: 缩放因子（如 1/sqrt(head_dim)），默认 1.0

        Returns:
            C_X: X 的贡献值，形状 [..., dim_x, dim_shared] (对 dim_y 求和后)
            C_Y: Y 的贡献值，形状 [..., dim_shared, dim_y, dim_y] (使用对角矩阵后，对 dim_x 求和)
        """
        eps = 1e-10

        # 如果需要对 Y 转置，先转置
        if transpose_Y:
            Y = Y.transpose(-2, -1)

        # 获取维度信息
        X_shape = list(X.shape)
        Y_shape = list(Y.shape)
        batch_dims = X_shape[:-2]
        dim_x = X_shape[-2]
        dim_shared = X_shape[-1]
        dim_y = Y_shape[-1]

        # 完全按照原始逻辑：计算 attn_weights 和 V 的贡献
        # 扩展 X: [..., dim_x, dim_shared] -> [..., dim_x, dim_shared, 1]
        X_exp = X.unsqueeze(-1)  # [..., dim_x, dim_shared, 1]

        # 扩展 Y: [..., dim_shared, dim_y] -> [..., 1, dim_shared, dim_y]
        Y_exp = Y.unsqueeze(-3)  # [..., 1, dim_shared, dim_y]

        # 扩展 Z: [..., dim_x, dim_y] -> [..., dim_x, 1, dim_y]
        Z_exp = Z.unsqueeze(-2)  # [..., dim_x, 1, dim_y]

        # 扩展 R_Z: [..., dim_x, dim_y] -> [..., dim_x, 1, dim_y]
        R_Z_exp = R_Z.unsqueeze(-2)  # [..., dim_x, 1, dim_y]

        # 计算 X[i, k] * Y[k, j]
        X_expanded = X_exp.expand(*batch_dims, dim_x, dim_shared, dim_y)
        Y_expanded = Y_exp.expand(*batch_dims, dim_x, dim_shared, dim_y)
        XY_terms = X_expanded * Y_expanded  # [..., dim_x, dim_shared, dim_y]

        # 应用缩放因子
        if scale_factor != 1.0:
            XY_terms = XY_terms * scale_factor

        # 扩展 Z 和 R_Z
        Z_expanded = Z_exp.expand(*batch_dims, dim_x, dim_shared, dim_y)
        R_Z_expanded = R_Z_exp.expand(*batch_dims, dim_x, dim_shared, dim_y)

        # 计算贡献项：C_terms[i, k, j] = X[i, k] * Y[k, j] * R_Z[i, j] / Z[i, j]
        C_terms = torch.where(
            torch.abs(Z_expanded) > eps,
            XY_terms / Z_expanded,
            torch.zeros_like(XY_terms)
        )  # [..., dim_x, dim_shared, dim_y]

        # 完全按照原始逻辑：计算 C_attn_weights（对 head_dim 维度求和）
        C_X = C_terms

        # 完全按照原始逻辑：计算 C_V（使用对角矩阵）
        C_terms_exp = C_terms.unsqueeze(-1)  # [..., dim_x, dim_shared, dim_y, 1]

        # 创建对角矩阵
        eye = torch.eye(dim_y, device=C_terms.device, dtype=C_terms.dtype)  # [dim_y, dim_y]
        eye_shape = [1] * (len(C_terms.shape) - 1) + [dim_y, dim_y]
        eye = eye.view(*eye_shape)  # [1, ..., 1, dim_y, dim_y]

        # 扩展 C_terms
        C_terms_expanded = C_terms_exp.expand(*batch_dims, dim_x, dim_shared, dim_y, dim_y)
        # [..., dim_x, dim_shared, dim_y, dim_y]

        # 应用对角矩阵
        C_Y_full = C_terms_expanded * eye
        # [..., dim_x, dim_shared, dim_y, dim_y]

        # 对 dim_x 维度求和（对应原始代码中的 sum(dim=2)，即对 seq_len_q 求和）
        C_Y = C_Y_full.sum(dim=-4)  # [..., dim_shared, dim_y, dim_y]

        return C_X, C_Y

    def forward_withcontribution(self, query, key, value, key_padding_mask=None, attn_mask=None, need_weights=True):
        """
        Args:
            query: [seq_len_q, batch_size, embed_dim] - PyTorch MultiheadAttention 标准格式
            key: [seq_len_k, batch_size, kdim]
            value: [seq_len_v, batch_size, vdim]
            key_padding_mask: [batch_size, seq_len_k], True 表示需要 mask（padding）
            attn_mask: attention mask（可选）
            need_weights: 是否需要返回 attention weights

        Returns:
            output: [seq_len_q, batch_size, embed_dim]
            attn_weights: [batch_size, seq_len_q, seq_len_k] (如果 need_weights=True)
        """

        seq_len_q, batch_size, _ = query.shape
        seq_len_k = key.shape[0]
        seq_len_v = value.shape[0]

        model_dtype = next(self.parameters()).dtype
        model_device = next(self.parameters()).device

        print('query.shape',query.shape)
        print('key.shape', key.shape)
        print('value.shape',value.shape)

        query = query.to(dtype=model_dtype, device=model_device)
        key = key.to(dtype=model_dtype, device=model_device)
        value = value.to(dtype=model_dtype, device=model_device)

        # 转换为 [batch_size, seq_len, embed_dim] 格式
        query_bt = query.transpose(0, 1)  # [batch_size, seq_len_q, embed_dim]
        key_bt = key.transpose(0, 1)  # [batch_size, seq_len_k, kdim]
        value_bt = value.transpose(0, 1)  # [batch_size, seq_len_v, vdim]

        print('################')

        print('query.shape', query_bt.shape)
        print('key.shape', key_bt.shape)
        print('value.shape', value_bt.shape)



        # 线性投影
        Q = self.q_proj(query_bt)  # [batch_size, seq_len_q, embed_dim]
        K = self.k_proj(key_bt)  # [batch_size, seq_len_k, embed_dim]
        V = self.v_proj(value_bt)  # [batch_size, seq_len_v, embed_dim]

        Q_original = Q.clone()
        K_original = K.clone()
        V_original = V.clone()

        print('Q.shape',Q.shape)
        print('V.shape',V.shape)
        print('K.shape', K.shape)

        # 重塑为多头格式
        Q = Q.view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        # [batch_size, num_heads, seq_len_q, head_dim]
        K = K.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        # [batch_size, num_heads, seq_len_k, head_dim]
        V = V.view(batch_size, seq_len_v, self.num_heads, self.head_dim).transpose(1, 2)
        # [batch_size, num_heads, seq_len_v, head_dim]

        print('Q.shape', Q.shape)
        print('V.shape', V.shape)
        print('K.shape', K.shape)

        # Scaled Dot-Product Attention
        attn_weights_pre_softmax = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # [batch_size, num_heads, seq_len_q, seq_len_k]
        print('attn_weights_pre_softmax.shape', attn_weights_pre_softmax.shape)



        # 处理 key_padding_mask
        if key_padding_mask is not None:
            key_padding_mask_expanded = key_padding_mask.unsqueeze(1).unsqueeze(2)
            key_padding_mask_expanded = key_padding_mask_expanded.expand(
                batch_size, self.num_heads, seq_len_q, seq_len_k
            )
            attn_weights_pre_softmax = attn_weights_pre_softmax.masked_fill(key_padding_mask_expanded, float('-inf'))

        # 处理 attn_mask
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_weights = attn_weights.masked_fill(attn_mask, float('-inf'))
            else:
                attn_weights = attn_weights + attn_mask

        # Softmax
        attn_weights = F.softmax(attn_weights_pre_softmax, dim=-1)

        # Dropout
        attn_weights = self.dropout(attn_weights)

        print('attn_weights.shape',attn_weights.shape)
        print('v.shape',V.shape)


        # 计算注意力输出
        attn_output = torch.matmul(attn_weights, V)
        # [batch_size, num_heads, seq_len_q, head_dim]

        attn_output_multihead = attn_output.clone()

        print('nomerge attn_output', attn_output.shape)

        # 合并所有头
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.embed_dim
        )  # [batch_size, seq_len_q, embed_dim]

        print('attn_output',attn_output.shape)

        # 输出投影
        output = self.out_proj(attn_output)  # [batch_size, seq_len_q, embed_dim]
        print('before output.shape', output.shape)

        C_attn_to_output = self.linear_contribution_ratio(
            self.out_proj.weight,  # [embed_dim, embed_dim]
            attn_output,  # [batch_size, seq_len_q, embed_dim]
            output  # [batch_size, seq_len_q, embed_dim]
        )

        # print('C_attn_to_output',C_attn_to_output.shape)
        # contrib_sum = C_attn_to_output.sum(dim=2)  # [batch_size, seq_len_q, embed_dim_out]
        #
        # print(f'contrib_sum.shape: {contrib_sum.shape}')  # 应该是 [batch_size, seq_len_q, embed_dim]
        # print(f'output.shape: {output.shape}')  # 应该是 [batch_size, seq_len_q, embed_dim]
        #
        # # ★ 验证是否相等
        # is_conserved = torch.allclose(contrib_sum, output, atol=1e-5, rtol=1e-5)
        # print(f'贡献守恒验证 (attn_output -> output): {is_conserved}')

        C_attn_to_output_multihead = C_attn_to_output.view(
            batch_size, seq_len_q, self.num_heads, self.head_dim, self.embed_dim
        )  # [batch_size, seq_len_q, num_heads, head_dim, embed_dim]

        C_attn_to_output_multihead = C_attn_to_output_multihead.transpose(1, 2)
        # [batch_size, num_heads, seq_len_q, head_dim, embed_dim]

        print('C_attn_to_output_multihead.shape', C_attn_to_output_multihead.shape)



        # 转换回 PyTorch 格式：[seq_len, batch_size, embed_dim]
        output = output.transpose(0, 1)  # [seq_len_q, batch_size, embed_dim]

        C_attn_to_output_sum = C_attn_to_output_multihead.sum(dim=(1, 3))
        # [batch_size, seq_len_q, embed_dim]

        # 转换维度顺序以匹配 output: [seq_len_q, batch_size, embed_dim]
        C_attn_to_output_sum = C_attn_to_output_sum.transpose(0, 1)
        # [seq_len_q, batch_size, embed_dim]

        print('C_attn_to_output_sum.shape', C_attn_to_output_sum.shape)
        print('output.shape', output.shape)

        # 验证是否相等
        is_conserved_multihead = torch.allclose(
            C_attn_to_output_sum,
            output,
            atol=1e-5,
            rtol=1e-5
        )
        print(f'贡献守恒验证 (C_attn_to_output_multihead -> output): {is_conserved_multihead}')

        print('after output.shape',output.shape)

        R_attn_output = C_attn_to_output_multihead.sum(dim=-1)
        # [batch_size, num_heads, seq_len_q, head_dim]

        print('R_attn_output.shape', R_attn_output.shape)

        print('attn_weights.shape',attn_weights.shape)
        print('V',V.shape)

        C_attn_weights, C_V = self.matrix_multiply_contribution_ratio(
            X=attn_weights,  # [B, H, seq_len_q, seq_len_k]
            Y=V,  # [B, H, seq_len_k, head_dim]
            Z=attn_output_multihead,  # [B, H, seq_len_q, head_dim]
            R_Z=R_attn_output,  # [B, H, seq_len_q, head_dim]
            transpose_Y=False,
            scale_factor=1.0,
        )
        print('C_attn_weights', C_attn_weights.shape)
        print('C_V', C_V.shape)
        print('C_attn_to_output_multihead',C_attn_to_output_multihead.shape)

        C_weights_tooutput=torch.einsum('bhjdk,bhjkp->bhjdp', C_attn_weights, C_attn_to_output_multihead)
        C_V_tooutput = torch.einsum('bhjdk,bhjkp->bhjdp', C_V , C_attn_to_output_multihead)

        C_weights_tooutput=C_weights_tooutput/2

        C_V_tooutput=C_V_tooutput/2

        print('C_weights_tooutput.shape',C_weights_tooutput.shape)
        print('C_V_tooutput.shape', C_V_tooutput.shape)


        # is_conserved_attn_weights = torch.allclose(
        #             C_weights_tooutput.sum(dim=(1,2,3)),
        #             output.sum(dim=0),
        #             atol=1e-5,
        #             rtol=1e-5
        #         )
        # print(f'验证2 - C_attn_weights 守恒: {is_conserved_attn_weights}')
        #
        # is_conserved_V = torch.allclose(
        #     C_V_tooutput.sum(dim=(1, 2, 3)),
        #     output.sum(dim=0),
        #     atol=1e-5,
        #     rtol=1e-5
        # )
        # print(f'验证2 - C_V 守恒: {is_conserved_V}')

        Q_attention,K_attention=self.matrix_multiply_contribution_ratio(Q,K,attn_weights_pre_softmax,attn_weights,True, 1/self.head_dim ** 0.5)

        print('Q_attention.shape',Q_attention.shape)
        print('K_attention.shape', K_attention.shape)

        Q_to_output = torch.einsum('bhjdk,bhjkp->bhjdp', Q_attention, C_weights_tooutput)
        K_to_output = torch.einsum('bhjdk,bhjkp->bhjdp', K_attention, C_weights_tooutput)

        K_to_output= K_to_output.permute(0, 1, 3, 2, 4)



        print('Q_to_output.shape', Q_to_output.shape)
        print('K_to_output', K_to_output.shape)

        Q_to_output=Q_to_output/2
        K_to_output=K_to_output/2

        # is_conserved_Q = torch.allclose(
        #     Q_to_output.sum(dim=(1, 2, 3)),
        #     output.sum(dim=0),
        #     atol=1e-5,
        #     rtol=1e-5
        # )
        # print(f'验证2 - Q_to_output 守恒: {is_conserved_Q}')
        #
        # is_conserved_K = torch.allclose(
        #     K_to_output.sum(dim=(1, 2, 3)),
        #     output.sum(dim=0),
        #     atol=1e-5,
        #     rtol=1e-5
        # )
        # print(f'验证2 - K_to_output 守恒: {is_conserved_K}')
        #
        # print('query_bt.shape',query_bt.shape)
        # print('Q_original',Q_original.shape)

        C_query_to_Q = self.linear_contribution_ratio(
            self.q_proj.weight,  # [embed_dim, embed_dim] - Linear层的权重矩阵
            query_bt,  # [batch_size, seq_len_q, embed_dim] - 输入
            Q_original  # [batch_size, seq_len_q, embed_dim] - 输出
        )
        print('C_query_to_Q',C_query_to_Q.shape)
#
        C_query_to_Q_multihead = C_query_to_Q.view(
            batch_size, seq_len_q, self.embed_dim, self.num_heads, self.head_dim
        )
        print('C_query_to_Q_multihead', C_query_to_Q_multihead.shape)

        C_query_to_Q_multihead = C_query_to_Q_multihead.permute(0, 3, 1, 2, 4)

        print('C_query_to_Q_multihead', C_query_to_Q_multihead.shape)

        print('Q_to_output.shape',Q_to_output.shape)


        query_to_output = torch.einsum('bhjdk,bhjkp->bhjdp',C_query_to_Q_multihead, Q_to_output)

        print('query_to_output',query_to_output.shape)

        # is_conserved_query = torch.allclose(
        #     query_to_output.sum(dim=(1, 2, 3)),
        #     output.sum(dim=0),
        #     atol=1e-5,
        #     rtol=1e-5
        # )
        #
        # print(f'验证2 - query_to_output 守恒: {is_conserved_query}')

        C_key_to_K = self.linear_contribution_ratio(
            self.k_proj.weight,  # [embed_dim, embed_dim] - Linear层的权重矩阵
            key_bt,  # [batch_size, seq_len_q, embed_dim] - 输入
            K_original  # [batch_size, seq_len_q, embed_dim] - 输出
        )

        print('key_bt',key_bt.shape)
        print('K_original', K_original.shape)
        print('K',K.shape)

        print('C_key_to_K', C_key_to_K.shape)

        # C_key_to_K_multihead = C_key_to_K.view(
        #     batch_size, seq_len_q, self.embed_dim, self.num_heads, self.head_dim
        # )
        # print('C_key_to_K_multihead', C_key_to_K_multihead.shape)

        kdim = key_bt.shape[2]  # 获取 kdim 的实际值
        C_key_to_K_multihead = C_key_to_K.view(
            batch_size, seq_len_k, kdim, self.num_heads, self.head_dim
        )

        print('C_key_to_K_multihead', C_key_to_K_multihead.shape)

        C_key_to_K_multihead = C_key_to_K_multihead.permute(0, 3, 1,2,4)
        # print('C_key_to_K_multihead (after permute)', C_key_to_K_multihead.shape)
        #
        # print(K_to_output.shape)

        print('C_key_to_K_multihead', C_key_to_K_multihead.shape)

        print('K_to_output',K_to_output.shape)



        key_to_output = torch.einsum('bhjdk,bhjkp->bhjdp',  C_key_to_K_multihead,K_to_output)
        print('key_to_output.shape', key_to_output.shape)

        # is_conserved_key = torch.allclose(
        #     key_to_output.sum(dim=(1, 2, 3)),
        #     output.sum(dim=0),
        #     atol=1e-5,
        #     rtol=1e-5
        # )
        #
        # print(f'验证2 - key_to_output 守恒: {is_conserved_key}')

        C_value_to_V = self.linear_contribution_ratio(
            self.v_proj.weight,  # [embed_dim, embed_dim] - Linear层的权重矩阵
            value_bt,  # [batch_size, seq_len_q, embed_dim] - 输入
            V_original  # [batch_size, seq_len_q, embed_dim] - 输出
        )
        print('C_value_to_V', C_value_to_V.shape)

        vdim = value_bt.shape[2]  # 获取 kdim 的实际值
        C_value_to_V_multihead = C_value_to_V.view(
            batch_size, seq_len_k, kdim, self.num_heads, self.head_dim
        )

        print('C_value_to_V_multihead',C_value_to_V_multihead.shape)

        C_value_to_V_multihead  = C_value_to_V_multihead.permute(0, 3, 1, 2, 4)
        print('C_value_to_V_multihead', C_value_to_V_multihead.shape)
        print('C_V_tooutput.shape',C_V_tooutput.shape)

        value_to_output = torch.einsum('bhjdk,bhjkp->bhjdp', C_value_to_V_multihead, C_V_tooutput)
        print('value_to_output.shape', value_to_output.shape)

        is_conserved_value = torch.allclose(
            value_to_output.sum(dim=(1, 2, 3))+key_to_output.sum(dim=(1, 2, 3))+query_to_output.sum(dim=(1, 2, 3)),
            output.sum(dim=0),
            atol=1e-5,
            rtol=1e-5
        )
        # print('value',value.shape)
        # print('query', query.shape)
        # print('key', key.shape)
        #
        # print('value_to_output.shape', value_to_output.shape)
        # print('query_to_output.shape', query_to_output.shape)
        # print('key_to_output.shape',key_to_output.shape)

        print(f'final 验证 守恒: {is_conserved_value}')

        C_query=query_to_output.sum(dim=1)
        C_query=C_query.permute(1,0,2,3)
        print('C_query',C_query.shape)

        C_key=key_to_output.sum(dim=1)
        C_key =C_key.permute(1,0,2,3)
        print('C_key', C_key.shape)

        C_value= value_to_output.sum(dim=1)
        C_value = C_value.permute(1, 0, 2, 3)
        print('C_value', C_value.shape)

        test_sum=C_value.sum(dim=(0,2)) + C_key.sum(dim=(0,2)) + C_query.sum(dim=(0,2))

        is_conserved_value = torch.allclose(
            test_sum,
            torch.ones_like(test_sum),
            atol=1e-5,
            rtol=1e-5
        )

        print(f'out  验证 守恒: {is_conserved_value}')









        # 处理 attention weights
        if need_weights:
            # 对所有头求平均
            attn_weights_avg = attn_weights.mean(dim=1)  # [batch_size, seq_len_q, seq_len_k]
            return output, attn_weights_avg,C_query,C_key,C_value
        else:
            return output


class EmbeddingModule(nn.Module):
  def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               dropout):
    super(EmbeddingModule, self).__init__()
    self.node_features = node_features
    self.edge_features = edge_features
    # self.memory = memory
    self.neighbor_finder = neighbor_finder
    self.time_encoder = time_encoder
    self.n_layers = n_layers
    self.n_node_features = n_node_features
    self.n_edge_features = n_edge_features
    self.n_time_features = n_time_features
    self.dropout = dropout
    self.embedding_dimension = embedding_dimension
    self.device = device

  def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                        use_time_proj=True):
    return NotImplemented


class IdentityEmbedding(EmbeddingModule):
  def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                        use_time_proj=True):
    return memory[source_nodes, :]


class TimeEmbedding(EmbeddingModule):
  def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               n_heads=2, dropout=0.1, use_memory=True, n_neighbors=1):
    super(TimeEmbedding, self).__init__(node_features, edge_features, memory,
                                        neighbor_finder, time_encoder, n_layers,
                                        n_node_features, n_edge_features, n_time_features,
                                        embedding_dimension, device, dropout)

    class NormalLinear(nn.Linear):
      # From Jodie code
      def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.normal_(0, stdv)
        if self.bias is not None:
          self.bias.data.normal_(0, stdv)

    self.embedding_layer = NormalLinear(1, self.n_node_features)

  def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                        use_time_proj=True):
    source_embeddings = memory[source_nodes, :] * (1 + self.embedding_layer(time_diffs.unsqueeze(1)))

    return source_embeddings


class GraphEmbedding(EmbeddingModule):
  def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               n_heads=2, dropout=0.1, use_memory=True):
    super(GraphEmbedding, self).__init__(node_features, edge_features, memory,
                                         neighbor_finder, time_encoder, n_layers,
                                         n_node_features, n_edge_features, n_time_features,
                                         embedding_dimension, device, dropout)

    self.use_memory = use_memory
    self.device = device

  def compute_embedding_original(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                        use_time_proj=True):
      """Recursive implementation of curr_layers temporal graph attention layers.

      src_idx_l [batch_size]: users / items input ids.
      cut_time_l [batch_size]: scalar representing the instant of the time where we want to extract the user / item representation.
      curr_layers [scalar]: number of temporal convolutional layers to stack.
      num_neighbors [scalar]: number of temporal neighbor to consider in each convolutional layer.
      """

      assert (n_layers >= 0)

      source_nodes_torch = torch.from_numpy(source_nodes).long().to(self.device)
      timestamps_torch = torch.unsqueeze(torch.from_numpy(timestamps).float().to(self.device), dim=1)

      # query node always has the start time -> time span == 0
      source_nodes_time_embedding = self.time_encoder(torch.zeros_like(
          timestamps_torch))

      source_node_features = self.node_features[source_nodes_torch, :]

      if self.use_memory:
          source_node_features = memory[source_nodes, :] + source_node_features

      if n_layers == 0:
          return source_node_features
      else:

          source_node_conv_embeddings = self.compute_embedding_original(memory,
                                                               source_nodes,
                                                               timestamps,
                                                               n_layers=n_layers - 1,
                                                               n_neighbors=n_neighbors)

          neighbors, edge_idxs, edge_times = self.neighbor_finder.get_temporal_neighbor(
              source_nodes,
              timestamps,
              n_neighbors=n_neighbors)

          neighbors_torch = torch.from_numpy(neighbors).long().to(self.device)

          edge_idxs = torch.from_numpy(edge_idxs).long().to(self.device)

          edge_deltas = timestamps[:, np.newaxis] - edge_times

          edge_deltas_torch = torch.from_numpy(edge_deltas).float().to(self.device)

          neighbors = neighbors.flatten()
          neighbor_embeddings = self.compute_embedding_original(memory,
                                                       neighbors,
                                                       np.repeat(timestamps, n_neighbors),
                                                       n_layers=n_layers - 1,
                                                       n_neighbors=n_neighbors)

          effective_n_neighbors = n_neighbors if n_neighbors > 0 else 1
          neighbor_embeddings = neighbor_embeddings.view(len(source_nodes), effective_n_neighbors, -1)
          edge_time_embeddings = self.time_encoder(edge_deltas_torch)

          edge_features = self.edge_features[edge_idxs, :]

          mask = neighbors_torch == 0

          source_embedding = self.aggregate_without_contribution(n_layers, source_node_conv_embeddings,
                                            source_nodes_time_embedding,
                                            neighbor_embeddings,
                                            edge_time_embeddings,
                                            edge_features,
                                            mask)

          return source_embedding



  def aggregate(self, n_layers, source_node_features, source_nodes_time_embedding,
                neighbor_embeddings,
                edge_time_embeddings, edge_features, mask):
    return NotImplemented
  def aggregate_without_contribution(self, n_layers, source_node_features, source_nodes_time_embedding,
                neighbor_embeddings,
                edge_time_embeddings, edge_features, mask):
    return NotImplemented


  def compute_embedding_iterative(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20):
      B = len(source_nodes)
      #print('source nodes',source_nodes)
      source_nodes_torch = torch.from_numpy(source_nodes).long().to(self.device)
      timestamps_torch = torch.unsqueeze(torch.from_numpy(timestamps).float().to(self.device), dim=1)

      raw_source_node_features = self.node_features[source_nodes_torch, :]  # [B, D_node]

      # print('raw_source_node_features',raw_source_node_features.shape)

      # 最底层：静态特征 + memory（可选）
      h = raw_source_node_features.clone()
      if self.use_memory:
          h = h + memory[source_nodes, :]

      # 初始时间编码（始终为0，因为是 query 节点当前时间）
      source_nodes_time_embedding = self.time_encoder(torch.zeros_like(timestamps_torch))  # [B, Dₜ]

      #print('source_nodes_time_embedding',source_nodes_time_embedding)




      for layer in range(n_layers):
          # 采样邻居（每一层用当前 query 时间采样）
          neighbors, edge_idxs, edge_times = self.neighbor_finder.get_temporal_neighbor(
              source_nodes, timestamps, n_neighbors=n_neighbors
          )

          # print('neighbors',neighbors,neighbors.shape)
          # print('edge_idxs',edge_idxs,edge_idxs.shape)
          # print('edge_times',edge_times)

          neighbors_torch = torch.from_numpy(neighbors).long().to(self.device)
          #print('neighbors_torch',neighbors_torch)
          edge_idxs_torch = torch.from_numpy(edge_idxs).long().to(self.device)
          edge_deltas = timestamps[:, np.newaxis] - edge_times

          # print('timestamps',timestamps)

          # print('edge_deltas',edge_deltas,edge_deltas.shape)

          edge_deltas_torch = torch.from_numpy(edge_deltas).float().to(self.device)

          # 获取邻居嵌入（从上一层）
          flat_neighbors = neighbors.flatten()
          # print('flat_neighbors ',len(flat_neighbors) )
          h_neighbors = self.node_features[flat_neighbors, :]
          if self.use_memory:
              h_neighbors = h_neighbors + memory[flat_neighbors, :]
          # print('h_neighbors ', h_neighbors.shape)

          h_neighbors = h_neighbors.view(B, n_neighbors, -1)

          # print('h_neighbors',h_neighbors.shape)

          # 时间编码与边特征
          edge_time_embeddings = self.time_encoder(edge_deltas_torch)  # [B, N, Dₜ]
          # print('edge_time_embeddings',edge_time_embeddings.shape)
          edge_features = self.edge_features[edge_idxs_torch, :]  # [B, N, Dₑ]
          # print('edge_features',edge_features.shape)
          mask = neighbors_torch == 0  # [B, N]


          # 聚合：本层嵌入 = f(hᶫ⁻¹, neighbors)
          h ,C_source_h,C_source_time,C_neighbor_embeddings,\
           C_edge_time_embeddings,C_edge_features= self.aggregate(layer + 1, h,
                             source_nodes_time_embedding,
                             h_neighbors,
                             edge_time_embeddings,
                             edge_features,
                             mask)

          # print('layer+1',layer+1)
      if self.use_memory:
          C_raw_features, C_memory_features = self.allocate_source_contributions(
              C_source_h, raw_source_node_features, memory, source_nodes
          )
          # print(C_raw_features.shape)
          # print(C_memory_features.shape)
          #
          # print(C_source_h.shape)

          total_source_contrib = C_raw_features.sum(dim=1) + C_memory_features.sum(dim=1)
          expected_total = C_source_h.sum(dim=1)
          check = torch.allclose(total_source_contrib, expected_total, atol=1e-8, rtol=1e-8)
          print(f"  分配验证: {'通过' if check else '失败'}")

          C_neighbor_raw_features, C_neighbor_memory_features = self.allocate_neighbor_contributions(
              C_neighbor_embeddings, flat_neighbors, memory
          )

          print('C_neighbor_memory_features',C_neighbor_memory_features.shape)

          total_neighbor_contrib = C_neighbor_raw_features.sum(dim=(1, 2)) + C_neighbor_memory_features.sum(dim=(1, 2))
          expected_neighbor_total = C_neighbor_embeddings.sum(dim=(1, 2))
          neighbor_check = torch.allclose(total_neighbor_contrib, expected_neighbor_total, atol=1e-8, rtol=1e-8)
          print(f"  邻居贡献值分配验证: {'通过' if neighbor_check else '失败'}")

      temporal_edge_contributions= self.map_contributions_to_temporal_edges(C_raw_features,C_source_time,
          C_neighbor_raw_features, C_edge_time_embeddings, C_edge_features,
          source_nodes, neighbors, edge_idxs, timestamps
      )

      # C_neighbor_memory_dict=dict()

      #print('temporal_edge_contributions',temporal_edge_contributions)
      # print('edge_info_list',edge_info_list)

      # 处理邻居贡献值（新增部分）
      C_neighbor_message_to_memory_features = dict()
      C_neighbor_old_memory_to_memory_features = dict()
      neighbor_memory_verify = True
      
      # if C_message is not None and C_memory is not None and self.use_memory:
      #     C_neighbor_memory_features = C_neighbor_memory_features.to(torch.float64)
      #
      #     # 使用最后一层的neighbors信息（已经在循环中获取）
      #     # neighbors: [B, K], edge_idxs: [B, K]
      #
      #     # 遍历每个源节点和其邻居
      #     for b in range(len(source_nodes)):
      #         node_neighbors = neighbors[b]  # 直接使用已有的neighbors
      #         for k, neighbor_node in enumerate(node_neighbors):
      #             edge_idx = edge_idxs[b, k]
      #
      #             C_neighbor_memory_dict[str(source_nodes[b].item())+','+str(neighbor_node.item())+','+str(edge_idx.item())]=C_neighbor_memory_features[b, k]
      #
      #
      #             if neighbor_node in C_message:
      #                 v = C_message[neighbor_node].to(
      #                     dtype=C_neighbor_memory_features.dtype,
      #                     device=C_neighbor_memory_features.device
      #                 )
      #
      #                 u = C_memory[neighbor_node].to(
      #                     dtype=C_neighbor_memory_features.dtype,
      #                     device=C_neighbor_memory_features.device
      #                 )
      #                 # 获取对应的neighbor memory features
      #                 neighbor_memory_feature = C_neighbor_memory_features[b, k]  # [D_n, D_out]
      #
      #                 # print('neighbor_memory_feature',neighbor_memory_feature.shape)
      #                 # print('v.shape',v.shape)
      #                 # print('u.shape', u.shape)
      #
      #                 C_neighbor_message_to_memory_features[edge_idx] = v @ neighbor_memory_feature
      #
      #                 C_neighbor_old_memory_to_memory_features[edge_idx] = u @ neighbor_memory_feature
      #
      #                 # test1=(v @ neighbor_memory_feature).sum(dim=0) + (u @ neighbor_memory_feature).sum(dim=0)
      #                 # test2=neighbor_memory_feature.sum(dim=0)
      #                 #
      #                 # print('test1',test1)
      #                 # print('test2',test2)
      #             else:
      #                 #print('not node',C_neighbor_memory_features[b, k])
      #                 sub = C_neighbor_memory_features[b, k]
      #
      #                 if torch.all(sub == 0):
      #                     pass
      #                 else:
      #                     print("非零元素:", sub[sub != 0])
      #
      #
      #     # for b in range(len(source_nodes)):
      #     #     source_node = source_nodes[b]
      #     #     node_neighbors = neighbors[b]
      #     #
      #     #     for k, neighbor_node in enumerate(node_neighbors):
      #     #             if neighbor_node in C_neighbor_message_to_memory_features:
      #     #                 # 计算test1: message贡献 + old memory贡献
      #     #                 message_contrib = C_neighbor_message_to_memory_features[neighbor_node].sum(dim=0)
      #     #                 old_memory_contrib = C_neighbor_old_memory_to_memory_features[neighbor_node].sum(dim=0)
      #     #                 test1 = message_contrib + old_memory_contrib
      #     #                 # 计算test2: neighbor memory features的总和
      #     #                 test2 = C_neighbor_memory_features[b, k].sum(dim=0)
      #     #
      #     #                 print('test1',test1)
      #     #                 print('test2', test2)
      #     #
      #     #                 # 验证是否守恒
      #     #                 if not torch.allclose(test1, test2, atol=1e-4):
      #     #                     neighbor_memory_verify = False
      #     #                     diff = torch.abs(test1 - test2).max().item()
      #     #
      #     #
      #     # # 打印验证结果
      #     # if neighbor_memory_verify:
      #     #     print(f'邻居守恒验证结果: 通过 ✅')
      #     # else:
      #     #     print(f'邻居守恒验证结果: 失败 ❌')

      total = None
      for idx, mat in temporal_edge_contributions.items():
          for _,second_mat in mat.items():
              if total is None:
                  total = second_mat.clone()
              else:
                  total = total + second_mat



      # sum_dict = torch.stack(list(temporal_edge_contributions.values()), dim=0).sum(dim=0)  # [D_out]

      print('sum_dict',total.shape)

      # 2) 原始三部分贡献矩阵分别对所有轴求和，再相加
      sum_raw = C_neighbor_raw_features.sum(dim=(0, 1, 2))  # [D_out]
      sum_time = C_edge_time_embeddings.sum(dim=(0, 1, 2))  # [D_out]
      sum_feat = C_edge_features.sum(dim=(0, 1, 2))  # [D_out]
      sum_raw_2=C_raw_features.sum(dim=(0, 1))
      sum_source=C_source_time.sum(dim=(0, 1))

      sum_original = sum_raw + sum_time + sum_feat+sum_raw_2 +sum_source # [D_out]

      print('sum_original',sum_original.shape)

      print('sum_original',sum_original)
      print('sum_dict',total)

      # 3) 守恒检查
      check = torch.allclose(total, sum_original, atol=1e-4)
      print("守恒验证:", "通过 ✅" if check else "失败 ❌")

      return (h,  C_memory_features,
               C_neighbor_memory_features,
              temporal_edge_contributions,neighbors,edge_idxs)



  def compute_embedding_iterative_without_contribution(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20):
      B = len(source_nodes)
      # print('source nodes',source_nodes)
      # print('memory.shape',memory.shape)
      source_nodes_torch = torch.from_numpy(source_nodes).long().to(self.device)
      timestamps_torch = torch.unsqueeze(torch.from_numpy(timestamps).float().to(self.device), dim=1)

      raw_source_node_features = self.node_features[source_nodes_torch, :]  # [B, D_node]

      # 最底层：静态特征 + memory（可选）
      h = raw_source_node_features.clone()
      if self.use_memory:
          h = h + memory[source_nodes, :]

      # 初始时间编码（始终为0，因为是 query 节点当前时间）
      source_nodes_time_embedding = self.time_encoder(torch.zeros_like(timestamps_torch))  # [B, Dₜ]


      for layer in range(n_layers):
          # 采样邻居（每一层用当前 query 时间采样）
          neighbors, edge_idxs, edge_times = self.neighbor_finder.get_temporal_neighbor(
              source_nodes, timestamps, n_neighbors=n_neighbors
          )



          neighbors_torch = torch.from_numpy(neighbors).long().to(self.device)
          # print('neighbors_torch',neighbors_torch)
          edge_idxs_torch = torch.from_numpy(edge_idxs).long().to(self.device)
          edge_deltas = timestamps[:, np.newaxis] - edge_times

          # print('timestamps',timestamps)

          # print('edge_deltas',edge_deltas,edge_deltas.shape)

          edge_deltas_torch = torch.from_numpy(edge_deltas).float().to(self.device)

          # 获取邻居嵌入（从上一层）
          flat_neighbors = neighbors.flatten()

          h_neighbors = self.node_features[flat_neighbors, :]
          if self.use_memory:
              h_neighbors = h_neighbors + memory[flat_neighbors, :]


          h_neighbors = h_neighbors.view(B, n_neighbors, -1)


          # 时间编码与边特征
          edge_time_embeddings = self.time_encoder(edge_deltas_torch)  # [B, N, Dₜ]


          edge_features = self.edge_features[edge_idxs_torch, :]  # [B, N, Dₑ]

          mask = neighbors_torch == 0  # [B, N]

          # 聚合：本层嵌入 = f(hᶫ⁻¹, neighbors)
          h = self.aggregate_without_contribution(layer + 1, h,
                                                                       source_nodes_time_embedding,
                                                                       h_neighbors,
                                                                       edge_time_embeddings,
                                                                       edge_features,
                                                                       mask)



      return h

  def allocate_source_contributions(self, C_source_h, raw_source_node_features, memory, source_nodes):
      C_source_h = C_source_h.double()
      raw_features = raw_source_node_features.double()
      mem_features = memory[source_nodes, :].double() if self.use_memory else torch.zeros_like(raw_features,
                                                                                               dtype=torch.float64,
                                                                                               device=raw_features.device)

      h = raw_features + mem_features  # [B, D_sf]

      raw_ratio = torch.zeros_like(h, dtype=torch.float64)
      mem_ratio = torch.zeros_like(h, dtype=torch.float64)

      nonzero_mask = h != 0
      if nonzero_mask.any():
          raw_ratio[nonzero_mask] = raw_features[nonzero_mask] / h[nonzero_mask]
          mem_ratio[nonzero_mask] = mem_features[nonzero_mask] / h[nonzero_mask]

      C_raw_features = raw_ratio.unsqueeze(-1) * C_source_h
      C_memory_features = mem_ratio.unsqueeze(-1) * C_source_h
      return C_raw_features, C_memory_features

  def allocate_neighbor_contributions(self, C_neighbor_embeddings, flat_neighbors, memory):
      """
      将邻居 embedding 的贡献拆分为：来自原始节点特征的贡献 + 来自内存特征的贡献
      Args:
          C_neighbor_embeddings: [B, K, D_n, D_out]
          flat_neighbors:        [B*K]
          memory:                [N, D_n]
      Returns:
          C_neighbor_raw_features:   [B, K, D_n, D_out]
          C_neighbor_memory_features:[B, K, D_n, D_out]
      """

      # ---- 1) 对齐 dtype & device ----
      dev = C_neighbor_embeddings.device
      dtype = C_neighbor_embeddings.dtype

      # 取邻居原始特征与内存特征
      neighbor_raw_features = self.node_features[flat_neighbors, :].to(device=dev, dtype=dtype)  # [B*K, D_n]
      neighbor_memory_features = memory[flat_neighbors, :].to(device=dev, dtype=dtype)  # [B*K, D_n]

      # ---- 2) reshape ----
      B, K, D_n, D_out = C_neighbor_embeddings.shape

      assert flat_neighbors.size == B * K

      neighbor_raw_features = neighbor_raw_features.view(B, K, D_n)
      neighbor_memory_features = neighbor_memory_features.view(B, K, D_n)

      # ---- 3) 计算比例（避免除零 & 保持 dtype）----
      total = neighbor_raw_features + neighbor_memory_features  # [B, K, D_n]
      nonzero = total != 0

      # 直接用 where 构造，不做布尔索引就地赋值（避免 dtype 冲突 & 更稳）
      # 防止除零：对为 0 的位置直接给 0
      raw_ratio = torch.where(
          nonzero,
          neighbor_raw_features / total,
          torch.zeros_like(total, dtype=dtype, device=dev)
      )
      mem_ratio = torch.where(
          nonzero,
          neighbor_memory_features / total,
          torch.zeros_like(total, dtype=dtype, device=dev)
      )



      # ---- 4) 分配贡献 ----
      # 形状对齐：[B,K,D_n,1] * [B,K,D_n,D_out]
      raw_ratio = raw_ratio.unsqueeze(-1)
      mem_ratio = mem_ratio.unsqueeze(-1)

      C_neighbor_raw_features = raw_ratio * C_neighbor_embeddings
      C_neighbor_memory_features = mem_ratio * C_neighbor_embeddings



      return C_neighbor_raw_features, C_neighbor_memory_features

  def map_contributions_to_temporal_edges(self, C_raw_features,C_source_time,C_neighbor_raw_features, C_edge_time_embeddings,
                                          C_edge_features, source_nodes, neighbors, edge_idxs, timestamps):
      """
      将贡献值映射到具体的时序边上

      Args:
          C_neighbor_raw_features: [B, K, D_n, D_out] 邻居原始特征贡献
          C_edge_time_embeddings: [B, K, D_te, D_out] 边时间编码贡献
          C_edge_features: [B, K, D_ef, D_out] 边特征贡献
          source_nodes: [B] 源节点ID列表
          neighbors: [B, K] 邻居节点ID矩阵
          edge_idxs: [B, K] 边索引矩阵
          timestamps: [B] 时间戳列表

      Returns:
          temporal_edge_contributions: [total_edges, total_contrib_dim] 每条时序边的贡献值
          edge_info: [total_edges] 每条边的信息字典列表
      """
      B, K, _, D_out = C_neighbor_raw_features.shape

      # 初始化结果列表
      edge_contributions_dict = {}

      for b in range(B):
          source_node = source_nodes[b]
          timestamp = timestamps[b]

          # print('valid_cnt',valid_cnt)
          if b not in edge_contributions_dict:
              edge_contributions_dict[b]=dict()


          for k in range(K):
              neighbor_node = neighbors[b, k]
              edge_idx = edge_idxs[b, k]
              # print('neighbor edge_idx',edge_idx)

              source_contrib = C_neighbor_raw_features[b, k].sum(dim=0)  # [D_out]

              # print('source_contrib',source_contrib.shape)

              # 2. 边时间编码贡献（对特征维度求和）
              edge_time_contrib = C_edge_time_embeddings[b, k].sum(dim=0)  # [D_out]

              # print('edge_time_contrib', edge_time_contrib.shape)

              # 3. 边特征贡献（对特征维度求和）
              edge_feat_contrib = C_edge_features[b, k].sum(dim=0)  # [D_out]

              # 4. 合并所有贡献值
              edge_total_contrib = source_contrib + edge_time_contrib + edge_feat_contrib  # [D_out]

              # print('edge_feat_contrib',edge_feat_contrib.shape)


              add_from_source = (C_source_time[b].sum(dim=0) + C_raw_features[b].sum(dim=0)) / K
                  # print('add_from_source', add_from_source.shape)
              edge_total_contrib = edge_total_contrib + add_from_source
              

                  # print('edge_total_contrib.shape',edge_total_contrib.shape)

              if edge_idx in edge_contributions_dict[b]:
                  # 如果边标号已存在，贡献值相加
                  edge_contributions_dict[b][edge_idx] += edge_total_contrib

              else:
                  # 新的边标号
                  edge_contributions_dict[b][edge_idx] = edge_total_contrib

      return edge_contributions_dict


class GraphSumEmbedding(GraphEmbedding):
  def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               n_heads=2, dropout=0.1, use_memory=True):
    super(GraphSumEmbedding, self).__init__(node_features=node_features,
                                            edge_features=edge_features,
                                            memory=memory,
                                            neighbor_finder=neighbor_finder,
                                            time_encoder=time_encoder, n_layers=n_layers,
                                            n_node_features=n_node_features,
                                            n_edge_features=n_edge_features,
                                            n_time_features=n_time_features,
                                            embedding_dimension=embedding_dimension,
                                            device=device,
                                            n_heads=n_heads, dropout=dropout,
                                            use_memory=use_memory)
    self.linear_1 = torch.nn.ModuleList([torch.nn.Linear(embedding_dimension + n_time_features +
                                                         n_edge_features, embedding_dimension)
                                         for _ in range(n_layers)])
    self.linear_2 = torch.nn.ModuleList(
      [torch.nn.Linear(embedding_dimension + n_node_features + n_time_features,
                       embedding_dimension) for _ in range(n_layers)])

  def linear_contribution(self,weight,input,out):
      Z = input.unsqueeze(2) * weight.unsqueeze(0)
      S = Z.sum(dim=1)  # [B, D_out]  分母

      den = S.unsqueeze(1)  # [B, 1, D_out] 便于广播
      phi = torch.where(den != 0, Z / den, torch.zeros_like(Z))  # 分母为0 → 0

      C = phi * out.unsqueeze(1)

      ok_mask = torch.isclose(C.sum(dim=1), out, atol=1e-4)
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

      # print(C.sum(dim=1))
      #
      # print(out)

      # print('linear 1',torch.allclose(C.sum(dim=1), out, atol=1e-4))
      return C


  def aggregate(self, n_layer, source_node_features, source_nodes_time_embedding,
                neighbor_embeddings,
                edge_time_embeddings, edge_features, mask):
    source_node_features = source_node_features.double()
    source_nodes_time_embedding = source_nodes_time_embedding.double()
    neighbor_embeddings = neighbor_embeddings.double()
    edge_time_embeddings = edge_time_embeddings.double()
    edge_features = edge_features.double()
    self.linear_1[n_layer - 1] = self.linear_1[n_layer - 1].double()
    self.linear_2[n_layer - 1] = self.linear_2[n_layer - 1].double()

    neighbors_features = torch.cat([neighbor_embeddings, edge_time_embeddings, edge_features],
                                   dim=2)
    # print('neighbors_features',neighbors_features.shape)
    nb_lin = self.linear_1[n_layer - 1](neighbors_features)
    # print('neighbor_embeddings',nb_lin.shape)

    nb_sum_pre = nb_lin.sum(dim=1)  # [B, H1]
    neighbors_sum = torch.nn.functional.relu(nb_sum_pre)  # [B, H1]
    # print('neighbors_sum', neighbors_sum.shape)

    src_time = source_nodes_time_embedding.squeeze()  # [B, D_st]
    source_features = torch.cat([source_node_features, src_time], dim=1)  # [B, D_sf+D_st]
    print('source_features', source_features.shape)


    # source_features = torch.cat([source_node_features,
    #                              source_nodes_time_embedding.squeeze()], dim=1)
    # print('source_features', source_features.shape)

    z = torch.cat([neighbors_sum, source_features], dim=1)

    print('source_embedding', z.shape)

    source_embedding = self.linear_2[n_layer - 1](z)

    print('source_embedding', source_embedding.shape)
    ##############

    W2_T = self.linear_2[n_layer - 1].weight.t()  # [D_z, D_out]
    C_z_to_semb = self.linear_contribution(W2_T, z, source_embedding)  # [B, D_z, D_out]

    print(' C_z_to_semb.shape', C_z_to_semb.shape)

    H1 = neighbors_sum.size(1)
    B, K, D_nf = neighbors_features.shape
    D_out = source_embedding.size(1)

    C_ns_to_semb = C_z_to_semb[:, :H1, :]

    # print('C_ns_to_semb.shape',C_ns_to_semb.shape)

    relu_alpha=torch.where(nb_sum_pre!=0,neighbors_sum/nb_sum_pre,torch.zeros_like(nb_sum_pre))
    # print('relu_alpha.shape',relu_alpha.shape)

    R_nb_sum_pre = C_ns_to_semb * relu_alpha.unsqueeze(-1)  #

    # print('R_nb_sum_pre',R_nb_sum_pre.shape)

    den_sum = nb_sum_pre.unsqueeze(1)  # [B, 1, H1]
    share_sum = torch.where(den_sum != 0,
                            nb_lin / den_sum,  # y/x，其中 y=nb_lin, x=nb_sum_pre
                            torch.zeros_like(nb_lin))

    R_nb_lin = share_sum.unsqueeze(-1) * R_nb_sum_pre.unsqueeze(1)

    # print('R_nb_lin.shape',R_nb_lin.shape)

    # print('R_nb_lin',R_nb_lin.sum(dim=1))
    #
    # print('R_nb_sum_pre',R_nb_sum_pre)

    print('linear 2',torch.allclose(R_nb_lin.sum(dim=1), R_nb_sum_pre, atol=1e-4))

    print('linear 3', torch.allclose(C_ns_to_semb, R_nb_sum_pre, atol=1e-4))

    # print(neighbors_features.shape)

    X_nf = neighbors_features.reshape(B * K, D_nf)

    W1_T = self.linear_1[n_layer - 1].weight.t()

    Z1 = X_nf.unsqueeze(2) * W1_T.unsqueeze(0)  # [BK, D_nf, H1]
    S1 = Z1.sum(dim=1)  # [BK, H1]
    den1 = S1.unsqueeze(1)  # [BK, 1, H1]
    R_nf_to_nb = torch.where(den1 != 0, Z1 / den1, torch.zeros_like(Z1))  # [BK, D_nf, H1]

    R_nf_to_nb = R_nf_to_nb.reshape(B, K, D_nf, H1)


    print('R_nf_to_nb.shape',R_nf_to_nb.shape)

    # R1 = R_nb_lin.reshape(B * K, H1, D_out)
    # print('R1.shape',R1.shape)

    # print(torch.allclose(R1.sum(dim=1), R_nf_to_nb.sum(dim=1), atol=1e-4))

    C_nf_to_semb = torch.einsum('bkfh,bkho->bkfo', R_nf_to_nb, R_nb_lin)

    # print(C_nf_to_semb.shape)

    lhs = C_nf_to_semb.sum(dim=2)  # sum over D_nf → [B,K,D_out]
    rhs = R_nb_lin.sum(dim=2)  # sum over H → [B,K,D_out]
    print('linear 4',torch.allclose(lhs, rhs, atol=1e-4))

    D_n = neighbor_embeddings.size(2)  # 邻居自身特征维
    D_te = edge_time_embeddings.size(2)  # 时间边特征维
    D_ef = edge_features.size(2)  # 其他边特征维

    C_neighbor_embeddings_to_semb = C_nf_to_semb[:, :, :D_n, :]  # [B,K,D_n ,D_out]
    C_edge_time_embeddings_to_semb = C_nf_to_semb[:, :, D_n:D_n + D_te, :]  # [B,K,D_te,D_out]
    C_edge_features_to_semb = C_nf_to_semb[:, :, D_n + D_te:, :]  # [B,K,D_ef,D_out]

    # print('C_neighbor_embeddings_to_semb', C_neighbor_embeddings_to_semb.shape)
    # print('C_edge_time_embeddings_to_semb', C_edge_time_embeddings_to_semb.shape)
    # print('C_edge_features_to_semb', C_edge_features_to_semb.shape)

    D_sf = source_node_features.size(1)
    D_st = src_time.size(1)  # = source_nodes_time_embedding.squeeze().size(1)

    C_srcfeat_to_semb = C_z_to_semb[:, H1:, :]  # [B, D_sf + D_st, D_out]
    C_source_node_features_to_semb = C_srcfeat_to_semb[:, :D_sf, :]  # [B, D_sf, D_out]
    C_source_time_to_semb = C_srcfeat_to_semb[:, D_sf:, :]  # [B, D_st, D_out]

    # print('C_source_node_features_to_semb', C_source_node_features_to_semb.shape)
    # print('C_source_time_to_semb', C_source_time_to_semb.shape)

    nb_sum = (
            C_neighbor_embeddings_to_semb.sum(dim=(1, 2)) +
            C_edge_time_embeddings_to_semb.sum(dim=(1, 2)) +
            C_edge_features_to_semb.sum(dim=(1, 2))
    )  # [B, D_out]

    # 源节点侧聚合
    src_sum = (
            C_source_node_features_to_semb.sum(dim=1) +
            C_source_time_to_semb.sum(dim=1)
    )  # [B, D_out]

    total_contrib = nb_sum + src_sum  # [B, D_out]

    print('final flag',torch.allclose(total_contrib, source_embedding, atol=1e-4))

    # mismatch_mask = ~torch.isclose(total_contrib, source_embedding, atol=1e-4)
    # mismatch_coords = mismatch_mask.nonzero(as_tuple=False)  # [N_mismatch, 2]
    # print("不匹配数量:", mismatch_coords.shape[0])
    # if mismatch_coords.numel() > 0:
    #     for b_idx, o_idx in mismatch_coords:
    #         pred = total_contrib[b_idx, o_idx].item()
    #         true = source_embedding[b_idx, o_idx].item()
    #         print(f"[b={b_idx}, o={o_idx}] pred={pred:.6f}, true={true:.6f}, diff={pred - true:+.3e}")

    return source_embedding,C_source_node_features_to_semb,C_source_time_to_semb,C_neighbor_embeddings_to_semb,\
           C_edge_time_embeddings_to_semb,C_edge_features_to_semb
  def aggregate_without_contribution(self, n_layer, source_node_features, source_nodes_time_embedding,
                neighbor_embeddings,
                edge_time_embeddings, edge_features, mask):
    source_node_features = source_node_features.double()
    source_nodes_time_embedding = source_nodes_time_embedding.double()
    neighbor_embeddings = neighbor_embeddings.double()
    edge_time_embeddings = edge_time_embeddings.double()
    edge_features = edge_features.double()
    self.linear_1[n_layer - 1] = self.linear_1[n_layer - 1].double()
    self.linear_2[n_layer - 1] = self.linear_2[n_layer - 1].double()

    neighbors_features = torch.cat([neighbor_embeddings, edge_time_embeddings, edge_features],
                                   dim=2)

    # print('neighbor_embeddings',neighbor_embeddings.shape)
    #
    # print('neighbors_features',neighbors_features.shape)

    nb_lin = self.linear_1[n_layer - 1](neighbors_features)


    nb_sum_pre = nb_lin.sum(dim=1)  # [B, H1]
    neighbors_sum = torch.nn.functional.relu(nb_sum_pre)  # [B, H1]


    src_time = source_nodes_time_embedding.squeeze()  # [B, D_st]
    source_features = torch.cat([source_node_features, src_time], dim=1)  # [B, D_sf+D_st]





    z = torch.cat([neighbors_sum, source_features], dim=1)



    source_embedding = self.linear_2[n_layer - 1](z)
    return source_embedding

  def init_hidden_embeddings(self, node_list):
      hidden_embeddings, masks = [], []
      for i in range(len(node_list)):
          batch_node_idx = torch.from_numpy(node_list[i]).long().to(self.device)
          hidden_embeddings.append(self.node_features[batch_node_idx])
          masks.append(batch_node_idx == 0)
      return hidden_embeddings, masks

  def retrieve_time_features(self, cut_time, time_list,num_neighbor):
      cut_time = np.concatenate([cut_time, cut_time, cut_time])
      batch = len(cut_time)
      first_time_stamp = np.expand_dims(cut_time, 1)  # [3*bsz, 1]
      # time_features = [self.time_encoder(torch.from_numpy(np.zeros_like(first_time_stamp)).float().to(self.device))]
      time_features = []
      standard_timestamps = np.expand_dims(first_time_stamp, 2)
      for layer_i in range(len(time_list)):
          t_record = time_list[layer_i]
          time_delta = standard_timestamps - t_record.reshape(batch, -1, num_neighbor)
          time_delta = time_delta.reshape(batch, -1)
          time_delta = torch.from_numpy(time_delta).float().to(self.device)
          time_features.append(self.time_encoder(time_delta))
          standard_timestamps = np.expand_dims(t_record, 2)
      return time_features

  def retrieve_edge_features(self, edge_list):
      edge_features = []
      for i in range(len(edge_list)):
          batch_edge_idx = torch.from_numpy(edge_list[i]).long().to(self.device)
          edge_features.append(self.edge_features[batch_edge_idx])
      return edge_features

  def embedding_update_layer(self, memory, node_list, node_features, edge_features, time_features, mask_list,num_neighbor=10,
                             explain_weights=None):
      num_layers = len(node_list)
      ## initial neighboring node feature
      neighbor_node = node_list[-1].flatten()
      neighbor_node_feature = node_features[-1].view(-1, self.n_node_features)
      if self.use_memory:
          neighbor_node_feature = memory[neighbor_node, :] + neighbor_node_feature
      else:
          neighbor_node_feature = neighbor_node_feature

      for i in range(num_layers - 1):  # i=0, 1
          t = num_layers - 1 - i  # 2, 1
          source_node = node_list[t - 1].flatten()
          source_node_feature = node_features[t - 1].view(-1, self.n_node_features)  # [bsz, feature_dim]
          batch_layer = source_node_feature.shape[0]
          if self.use_memory:
              source_node_feature = memory[source_node, :] + source_node_feature
          else:
              source_node_feature = source_node_feature
          source_nodes_time_embedding = self.time_encoder(
              torch.zeros((batch_layer, 1)).to(self.device))  # [bsz, 1, time_dim]
          neighbor_node_feature = neighbor_node_feature.view(batch_layer, num_neighbor,
                                                             -1)  # [bsz, n_neighbor, feature_dim]
          assert neighbor_node_feature.shape[-1] == source_node_feature.shape[-1]
          edge_time_embeddings = time_features[t - 1].view(batch_layer, num_neighbor, -1)
          edgh_feature = edge_features[t - 1].view(batch_layer, num_neighbor, -1)
          mask = mask_list[t].view(batch_layer, -1)
          if explain_weights is not None:
              explain_weight = explain_weights[t - 1].view(batch_layer, -1)
          else:
              explain_weight = None

          assert mask.shape[-1] == num_neighbor

          updated_source_node_feature = self.aggregate_without_contribution(i, source_node_feature, source_nodes_time_embedding,
                                                          neighbor_node_feature, edge_time_embeddings, edgh_feature,
                                                          mask)
          # [bsz, n_node_features]
          neighbor_node_feature = updated_source_node_feature

      return updated_source_node_feature  # [true bsz, feature_dim]



  def embedding_update(self, memory, node_list, edge_list, time_list, cut_time, n_layers, num_neighbor=10,explain_weights=None):
      """Recursive implementation of curr_layers temporal graph attention layers.

      src_idx_l [batch_size]: users / items input ids.
      cut_time_l [batch_size]: scalar representing the instant of the time where we want to extract the user / item representation.
      curr_layers [scalar]: number of temporal convolutional layers to stack.
      num_neighbors [scalar]: number of temporal neighbor to consider in each convolutional layer.
      """

      assert (n_layers >= 0)
      node_features, mask_list = self.init_hidden_embeddings(node_list)
      edge_features = self.retrieve_edge_features(edge_list)
      time_features = self.retrieve_time_features(cut_time, time_list,num_neighbor)

      source_embedding = self.embedding_update_layer(memory, node_list, node_features, edge_features, time_features,
                                                     mask_list, num_neighbor,explain_weights)  # [3*bs, node_features]

      return source_embedding



class GraphAttentionEmbedding(GraphEmbedding):
  def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               n_heads=2, dropout=0.1, use_memory=True):
    super(GraphAttentionEmbedding, self).__init__(node_features, edge_features, memory,
                                                  neighbor_finder, time_encoder, n_layers,
                                                  n_node_features, n_edge_features,
                                                  n_time_features,
                                                  embedding_dimension, device,
                                                  n_heads, dropout,
                                                  use_memory)

    self.attention_models = torch.nn.ModuleList([TemporalAttentionLayer(
      n_node_features=n_node_features,
      n_neighbors_features=n_node_features,
      n_edge_features=n_edge_features,
      time_dim=n_time_features,
      n_head=n_heads,
      dropout=dropout,
      output_dimension=n_node_features)
      for _ in range(n_layers)])

    query_dim = n_node_features + n_time_features
    key_dim = n_node_features + n_edge_features + n_time_features

    self.custom_attention_models = torch.nn.ModuleList([
        CustomMultiHeadAttention(
            embed_dim=query_dim,
            num_heads=n_heads,
            kdim=key_dim,
            vdim=key_dim,
            dropout=dropout
        ) for _ in range(n_layers)
    ])

    self._init_custom_attention_models()



  def _init_custom_attention_models(self):
        """初始化 custom_attention_models 的权重（从 attention_models 复制）"""
        for i in range(len(self.attention_models)):
            self.custom_attention_models[i].load_state_from_pytorch_mha(
                self.attention_models[i].multi_head_target
            )
            self.custom_attention_models[i].eval()


  def aggregate_explainweights(
          self,
          n_layer,
          source_node_features,
          source_nodes_time_embedding,
          neighbor_embeddings,
          edge_time_embeddings,
          edge_features,
          mask,explain_weights=None
  ):
      """Baseline aggregation method for GraphAttentionEmbedding.
      Similar to aggregate_without_contribution but returns (source_embedding, None) format.
      """
      attention_model = self.attention_models[n_layer - 1]

      B = source_node_features.shape[0]  # batch size
      K = neighbor_embeddings.shape[1]  # number of neighbors
      D_n = neighbor_embeddings.shape[2]  # neighbor embedding dim
      D_e = edge_features.shape[2]  # edge feature dim
      D_t = edge_time_embeddings.shape[2]  # time embedding dim
      D_s = source_node_features.shape[1]  # source node feature dim

      src_node_features_unrolled = torch.unsqueeze(source_node_features, dim=1)  # [B, 1, D_s]
      query = torch.cat([src_node_features_unrolled, source_nodes_time_embedding], dim=2)  # [B, 1, D_s + D_t]
      key = torch.cat([neighbor_embeddings, edge_features, edge_time_embeddings], dim=2)  # [B, K, D_n + D_e + D_t]

      query_perm = query.permute([1, 0, 2])  # [1, B, D_s + D_t]
      key_perm = key.permute([1, 0, 2])  # [K, B, D_n + D_e + D_t]

      mask_bool = mask < 0  # True for positions to mask out

      invalid_neighborhood_mask = mask_bool.all(dim=1, keepdim=True)
      mask_processed = mask_bool.clone()
      # If a source node has no valid neighbor, set its first neighbor to be valid
      mask_processed[invalid_neighborhood_mask.squeeze(), 0] = False

      attention_model_test = self.custom_attention_models[n_layer - 1]

      attn_output, attn_output_weights = attention_model_test.forward_weights(
          query=query_perm,
          key=key_perm,
          value=key_perm,
          key_padding_mask=mask_processed,explain_weights=explain_weights
      )

      attn_output = attn_output.squeeze()  # [B, D_s + D_t]
      attn_output_weights = attn_output_weights.squeeze()  # [B, K] or [B, 1, K] -> [B, K]

      final_output = attention_model.merger.forward(
          attn_output, source_node_features
      )

      return final_output, attn_output_weights

  def aggregate(self, n_layer, source_node_features, source_nodes_time_embedding,
                neighbor_embeddings,
                edge_time_embeddings, edge_features, mask):
    attention_model = self.attention_models[n_layer - 1]

    # source_node_features = source_node_features.double()
    # source_nodes_time_embedding = source_nodes_time_embedding.double()
    # neighbor_embeddings = neighbor_embeddings.double()
    # edge_time_embeddings = edge_time_embeddings.double()
    # edge_features = edge_features.double()

    B = source_node_features.shape[0]  # batch size
    K = neighbor_embeddings.shape[1]  # number of neighbors
    D_n = neighbor_embeddings.shape[2]  # neighbor embedding dim
    D_e = edge_features.shape[2]  # edge feature dim
    D_t = edge_time_embeddings.shape[2]  # time embedding dim
    D_s = source_node_features.shape[1]  # source node feature dim

    print('B',B)
    print('K', K)
    print('D_n',D_n)
    print('D_e', D_e)
    print('D_t',D_t)
    print('D_s', D_s)

    print('edge_time_embeddings.shape',edge_time_embeddings.shape)

    src_node_features_unrolled = torch.unsqueeze(source_node_features, dim=1)  # [B, 1, D_s]
    query = torch.cat([src_node_features_unrolled, source_nodes_time_embedding], dim=2)  # [B, 1, D_s + D_t]
    key = torch.cat([neighbor_embeddings, edge_features, edge_time_embeddings], dim=2)  # [B, K, D_n + D_e + D_t]

    query_perm = query.permute([1, 0, 2])  # [1, B, D_s + D_t]
    key_perm = key.permute([1, 0, 2])  # [K, B, D_n + D_e + D_t]

    print('query_perm',query_perm.shape)
    print('key_perm',key_perm.shape)

    invalid_neighborhood_mask = mask.all(dim=1, keepdim=True)
    mask_processed = mask.clone()
    mask_processed[invalid_neighborhood_mask.squeeze(), 0] = False

    # attn_output, attn_output_weights = attention_model.multi_head_target(
    #     query=query_perm,
    #     key=key_perm,
    #     value=key_perm,
    #     key_padding_mask=mask_processed
    # )
    # # print('attn_output.shape',attn_output.shape)
    #
    # print('attn_output.real', attn_output)

    query_dim = D_s + D_t  # query 的维度
    key_dim = D_n + D_e + D_t  # key 的维度

    # 从 attention_model 获取 n_head 和 dropout
    n_head = attention_model.n_head
    dropout = attention_model.multi_head_target.dropout.p if hasattr(attention_model.multi_head_target.dropout,
                                                                     'p') else 0



    # attention_model_test = CustomMultiHeadAttention(
    #     embed_dim=query_dim,
    #     num_heads=n_head,
    #     kdim=key_dim,
    #     vdim=key_dim,
    #     dropout=dropout
    # )
    #
    # attention_model_test.load_state_from_pytorch_mha(attention_model.multi_head_target)
    #
    # attention_model_test.eval()

    attention_model_test = self.custom_attention_models[n_layer - 1]

    attn_output, attn_output_weights,C_query,C_key,C_value = attention_model_test.forward_withcontribution(
        query=query_perm,
        key=key_perm,
        value=key_perm,
        key_padding_mask=mask_processed
    )

    print('attn_output.test',attn_output)


    test_sum=C_value.sum(dim=(0, 2)) + C_key.sum(dim=(0, 2)) + C_query.sum(dim=(0, 2))

    #
    is_conserved_value = torch.allclose(
        test_sum,
        torch.ones_like(test_sum),
        atol=1e-5,
        rtol=1e-5
    )
    # print('C_value',C_value.shape)
    # print('C_key', C_key.shape)
    # print('C_query', C_query.shape)
    #
    print(f'attn_output  验证 守恒: {is_conserved_value}')
    #
    # print('attn_output.custo', attn_output)

    attn_output = attn_output.squeeze()  # [B, D_s + D_t]
    # print('attn_output.shape', attn_output.shape)
    attn_output_weights = attn_output_weights.squeeze()  # [B, K] or [B, 1, K] -> [B, K]

    # print('attn_output_weights', attn_output_weights.shape)



    # 处理无效邻居
    # attn_output = attn_output.masked_fill(invalid_neighborhood_mask, 0)
    # attn_output_weights = attn_output_weights.masked_fill(invalid_neighborhood_mask, 0)

    # attn_output.masked_fill(invalid_neighborhood_mask, 0)
    # attn_output_weights.masked_fill(invalid_neighborhood_mask, 0)
    #
    # C_query.masked_fill(invalid_neighborhood_mask.unsqueeze(-1).unsqueeze(-1), 0)
    # C_key.masked_fill(invalid_neighborhood_mask.unsqueeze(-1).unsqueeze(-1), 0)
    # C_value.masked_fill(invalid_neighborhood_mask.unsqueeze(-1).unsqueeze(-1), 0)


    final_output, C_attn_to_final, C_src_to_final = attention_model.merger.forward_with_contributions(
        attn_output, source_node_features
    )

    is_conserved_value = torch.allclose(
        C_attn_to_final.sum(dim=(1)) + C_src_to_final.sum(dim=(1)),
        final_output,
        atol=1e-5,
        rtol=1e-5
    )

    print(f'test mlp  验证 守恒: {is_conserved_value}')





    # print('attn_output.shape',attn_output.shape)
    #
    print('C_src_to_final.shape',C_src_to_final.shape)
    print('C_attn_to_final',C_attn_to_final.shape)
    #
    # print('final_output', final_output.shape)



    # C_value_to_attn = C_value.sum(dim=2)

    # print('C_value_to_attn',C_value[0].shape)

    C_value_to_final = torch.einsum('hbdk,bko->hbdo', C_value, C_attn_to_final)

    # print('C_value_to_final',C_value_to_final.shape)

    C_key_to_final = torch.einsum('hbdk,bko->hbdo', C_key, C_attn_to_final)
    # print('C_key_to_final', C_key_to_final.shape)

    C_query_to_final = torch.einsum('hbdk,bko->hbdo', C_query, C_attn_to_final)
    # print('C_query_to_final', C_query_to_final.shape)

    is_conserved_value = torch.allclose(
        C_value_to_final.sum(dim=(0, 2)) + C_key_to_final.sum(dim=(0, 2)) + C_query_to_final.sum(dim=(0, 2)),
        final_output,
        atol=1e-5,
        rtol=1e-5
    )

    print(f'without shape  验证 守恒: {is_conserved_value}')

    #
    # print(f'final output  验证 守恒: {is_conserved_value}')
    #
    # print('query_perm', query_perm.shape)
    # print('key_perm', key_perm.shape)

    C_query_to_final_squeezed = C_query_to_final.squeeze(0)  #

    # print('C_query_to_final_squeezed',C_query_to_final_squeezed.shape)

    C_source_node_features_to_final = C_query_to_final_squeezed[  :,:D_s, :]+C_src_to_final




    # source_nodes_time_embedding 的贡献（后 D_t 维）
    C_source_time_to_final = C_query_to_final_squeezed[ :, D_s:, :]  # [10, 300, D_t, 32]



    C_neighbor_embeddings_to_final_from_key = C_key_to_final[:, :, :D_n, :]  # [10, 300, D_n, 32]
    C_neighbor_embeddings_to_final_from_value = C_value_to_final[:, :, :D_n, :]  # [10, 300, D_n, 32]

    C_edge_features_to_final_from_key = C_key_to_final[:, :, D_n:D_n + D_e, :]  # [10, 300, D_e, 32]
    C_edge_features_to_final_from_value = C_value_to_final[:, :, D_n:D_n + D_e, :]  # [10, 300, D_e, 32]

    C_edge_time_to_final_from_key = C_key_to_final[:, :, D_n + D_e:, :]  # [10, 300, D_t, 32]
    C_edge_time_to_final_from_value = C_value_to_final[:, :, D_n + D_e:, :]  # [10, 300, D_t, 32]

    C_neighbor_embeddings_to_final = C_neighbor_embeddings_to_final_from_key + C_neighbor_embeddings_to_final_from_value  # [10, 300, D_n, 32]
    C_edge_features_to_final = C_edge_features_to_final_from_key + C_edge_features_to_final_from_value  # [10, 300, D_e, 32]
    C_edge_time_to_final = C_edge_time_to_final_from_key + C_edge_time_to_final_from_value  # [10, 300, D_t, 32]

    print('C_neighbor_embeddings_to_final',C_neighbor_embeddings_to_final.shape)
    print('C_edge_features_to_final',C_edge_features_to_final.shape)
    print('C_edge_time_to_final',C_edge_time_to_final.shape)
    print('C_source_node_features_to_final', C_source_node_features_to_final.shape)
    print('C_source_time_to_final', C_source_time_to_final.shape)



    C_neighbor_embeddings_to_final =C_neighbor_embeddings_to_final.permute(1,0,2,3)

    C_edge_features_to_final = C_edge_features_to_final.permute(1, 0, 2, 3)

    C_edge_time_to_final = C_edge_time_to_final.permute(1, 0, 2, 3)

    is_conserved_value = torch.allclose(
        C_neighbor_embeddings_to_final.sum(dim=(1, 2)) + C_edge_features_to_final.sum(
            dim=(1, 2)) + C_edge_time_to_final.sum(dim=(1, 2)) \
        + C_source_node_features_to_final.sum(dim=1) + C_source_time_to_final.sum(dim=1),
        final_output,
        atol=1e-4,
        rtol=1e-4
    )

    print(f'contribution output  验证 守恒: {is_conserved_value}')




    return final_output,C_source_node_features_to_final, C_source_time_to_final,\
        C_neighbor_embeddings_to_final,C_edge_time_to_final,C_edge_features_to_final


  def aggregate_without_contribution(self, n_layer, source_node_features, source_nodes_time_embedding,
                                     neighbor_embeddings,
                                     edge_time_embeddings, edge_features, mask):
      """与 aggregate 方法类似，但不返回贡献信息"""
      attention_model = self.attention_models[n_layer - 1]



      B = source_node_features.shape[0]  # batch size
      K = neighbor_embeddings.shape[1]  # number of neighbors
      D_n = neighbor_embeddings.shape[2]  # neighbor embedding dim
      D_e = edge_features.shape[2]  # edge feature dim
      D_t = edge_time_embeddings.shape[2]  # time embedding dim
      D_s = source_node_features.shape[1]  # source node feature dim

      print('B', B)
      print('K', K)
      print('D_n', D_n)
      print('D_e', D_e)
      print('D_t', D_t)
      print('D_s', D_s)

      src_node_features_unrolled = torch.unsqueeze(source_node_features, dim=1)  # [B, 1, D_s]
      query = torch.cat([src_node_features_unrolled, source_nodes_time_embedding], dim=2)  # [B, 1, D_s + D_t]
      key = torch.cat([neighbor_embeddings, edge_features, edge_time_embeddings], dim=2)  # [B, K, D_n + D_e + D_t]

      query_perm = query.permute([1, 0, 2])  # [1, B, D_s + D_t]
      key_perm = key.permute([1, 0, 2])  # [K, B, D_n + D_e + D_t]

      print('query_perm', query_perm.shape)
      print('key_perm', key_perm.shape)

      invalid_neighborhood_mask = mask.all(dim=1, keepdim=True)
      mask_processed = mask.clone()
      mask_processed[invalid_neighborhood_mask.squeeze(), 0] = False

      # attn_output, attn_output_weights = attention_model.multi_head_target(
      #     query=query_perm,
      #     key=key_perm,
      #     value=key_perm,
      #     key_padding_mask=mask_processed
      # )
      # print('attn_output.shape',attn_output.shape)

      # print('attn_output.real', attn_output)

      query_dim = D_s + D_t  # query 的维度
      key_dim = D_n + D_e + D_t  # key 的维度

      # 从 attention_model 获取 n_head 和 dropout
      n_head = attention_model.n_head
      dropout = attention_model.multi_head_target.dropout.p if hasattr(attention_model.multi_head_target.dropout,
                                                                       'p') else 0

      # attention_model_test = CustomMultiHeadAttention(
      #     embed_dim=query_dim,
      #     num_heads=n_head,
      #     kdim=key_dim,
      #     vdim=key_dim,
      #     dropout=dropout
      # )
      #
      # attention_model_test.load_state_from_pytorch_mha(attention_model.multi_head_target)
      #
      # attention_model_test.eval()

      attention_model_test = self.custom_attention_models[n_layer - 1]

      attn_output, attn_output_weights, = attention_model_test.forward(
          query=query_perm,
          key=key_perm,
          value=key_perm,
          key_padding_mask=mask_processed
      )



      attn_output = attn_output.squeeze()  # [B, D_s + D_t]
      # print('attn_output.shape', attn_output.shape)
      attn_output_weights = attn_output_weights.squeeze()  # [B, K] or [B, 1, K] -> [B, K]


      final_output = attention_model.merger.forward(
          attn_output, source_node_features
      )



      return final_output,attn_output_weights


  def compute_embedding_attention(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20):
      B = len(source_nodes)
      # print('source nodes',source_nodes)
      source_nodes_torch = torch.from_numpy(source_nodes).long().to(self.device)
      timestamps_torch = torch.unsqueeze(torch.from_numpy(timestamps).float().to(self.device), dim=1)

      raw_source_node_features = self.node_features[source_nodes_torch, :]  # [B, D_node]

      print('raw_source_node_features', raw_source_node_features.shape)

      # 最底层：静态特征 + memory（可选）
      h = raw_source_node_features.clone()
      if self.use_memory:
          h = h + memory[source_nodes, :]

      # 初始时间编码（始终为0，因为是 query 节点当前时间）
      source_nodes_time_embedding = self.time_encoder(torch.zeros_like(timestamps_torch))  # [B, Dₜ]

      # print('source_nodes_time_embedding',source_nodes_time_embedding)

      for layer in range(n_layers):
          # 采样邻居（每一层用当前 query 时间采样）
          neighbors, edge_idxs, edge_times = self.neighbor_finder.get_temporal_neighbor(
              source_nodes, timestamps, n_neighbors=n_neighbors
          )

          # print('neighbors',neighbors,neighbors.shape)
          # print('edge_idxs',edge_idxs,edge_idxs.shape)
          # print('edge_times',edge_times)

          neighbors_torch = torch.from_numpy(neighbors).long().to(self.device)
          # print('neighbors_torch',neighbors_torch)
          edge_idxs_torch = torch.from_numpy(edge_idxs).long().to(self.device)
          edge_deltas = timestamps[:, np.newaxis] - edge_times

          # print('timestamps',timestamps)

          # print('edge_deltas',edge_deltas,edge_deltas.shape)

          edge_deltas_torch = torch.from_numpy(edge_deltas).float().to(self.device)

          # 获取邻居嵌入（从上一层）
          flat_neighbors = neighbors.flatten()
          print('flat_neighbors ', len(flat_neighbors))
          h_neighbors = self.node_features[flat_neighbors, :]
          if self.use_memory:
              h_neighbors = h_neighbors + memory[flat_neighbors, :]
          print('h_neighbors ', h_neighbors.shape)

          h_neighbors = h_neighbors.view(B, n_neighbors, -1)

          print('h_neighbors', h_neighbors.shape)

          # 时间编码与边特征
          edge_time_embeddings = self.time_encoder(edge_deltas_torch)  # [B, N, Dₜ]
          print('edge_time_embeddings', edge_time_embeddings.shape)
          edge_features = self.edge_features[edge_idxs_torch, :]  # [B, N, Dₑ]
          print('edge_features', edge_features.shape)
          mask = neighbors_torch == 0  # [B, N]

          # 聚合：本层嵌入 = f(hᶫ⁻¹, neighbors)
          h, C_source_h, C_source_time, C_neighbor_embeddings, \
              C_edge_time_embeddings, C_edge_features = self.aggregate(layer + 1, h,
                                                                       source_nodes_time_embedding,
                                                                       h_neighbors,
                                                                       edge_time_embeddings,
                                                                       edge_features,
                                                                       mask)

          print('C_neighbor_embeddings.shape',C_neighbor_embeddings.shape)

          # print('layer+1',layer+1)
      if self.use_memory:
          C_raw_features, C_memory_features = self.allocate_source_contributions(
              C_source_h, raw_source_node_features, memory, source_nodes
          )
          # print(C_raw_features.shape)
          # print(C_memory_features.shape)
          #
          # print(C_source_h.shape)

          total_source_contrib = C_raw_features.sum(dim=1) + C_memory_features.sum(dim=1)
          expected_total = C_source_h.sum(dim=1)
          check = torch.allclose(total_source_contrib, expected_total, atol=1e-8, rtol=1e-8)
          print(f"  分配验证: {'通过' if check else '失败'}")

          C_neighbor_raw_features, C_neighbor_memory_features = self.allocate_neighbor_contributions(
              C_neighbor_embeddings, flat_neighbors, memory
          )

          print('C_neighbor_memory_features', C_neighbor_memory_features.shape)

          total_neighbor_contrib = C_neighbor_raw_features.sum(dim=(1, 2)) + C_neighbor_memory_features.sum(dim=(1, 2))
          expected_neighbor_total = C_neighbor_embeddings.sum(dim=(1, 2))
          neighbor_check = torch.allclose(total_neighbor_contrib, expected_neighbor_total, atol=1e-8, rtol=1e-8)
          print(f"  邻居贡献值分配验证: {'通过' if neighbor_check else '失败'}")

      temporal_edge_contributions = self.map_contributions_to_temporal_edges_attention(C_raw_features, C_source_time,
                                                                             C_neighbor_raw_features,
                                                                             C_edge_time_embeddings, C_edge_features,
                                                                             source_nodes, neighbors, edge_idxs,
                                                                             timestamps
                                                                             )

      # C_neighbor_memory_dict=dict()

      # print('temporal_edge_contributions',temporal_edge_contributions)
      # print('edge_info_list',edge_info_list)

      # 处理邻居贡献值（新增部分）
      C_neighbor_message_to_memory_features = dict()
      C_neighbor_old_memory_to_memory_features = dict()
      neighbor_memory_verify = True

      # if C_message is not None and C_memory is not None and self.use_memory:
      #     C_neighbor_memory_features = C_neighbor_memory_features.to(torch.float64)
      #
      #     # 使用最后一层的neighbors信息（已经在循环中获取）
      #     # neighbors: [B, K], edge_idxs: [B, K]
      #
      #     # 遍历每个源节点和其邻居
      #     for b in range(len(source_nodes)):
      #         node_neighbors = neighbors[b]  # 直接使用已有的neighbors
      #         for k, neighbor_node in enumerate(node_neighbors):
      #             edge_idx = edge_idxs[b, k]
      #
      #             C_neighbor_memory_dict[str(source_nodes[b].item())+','+str(neighbor_node.item())+','+str(edge_idx.item())]=C_neighbor_memory_features[b, k]
      #
      #
      #             if neighbor_node in C_message:
      #                 v = C_message[neighbor_node].to(
      #                     dtype=C_neighbor_memory_features.dtype,
      #                     device=C_neighbor_memory_features.device
      #                 )
      #
      #                 u = C_memory[neighbor_node].to(
      #                     dtype=C_neighbor_memory_features.dtype,
      #                     device=C_neighbor_memory_features.device
      #                 )
      #                 # 获取对应的neighbor memory features
      #                 neighbor_memory_feature = C_neighbor_memory_features[b, k]  # [D_n, D_out]
      #
      #                 # print('neighbor_memory_feature',neighbor_memory_feature.shape)
      #                 # print('v.shape',v.shape)
      #                 # print('u.shape', u.shape)
      #
      #                 C_neighbor_message_to_memory_features[edge_idx] = v @ neighbor_memory_feature
      #
      #                 C_neighbor_old_memory_to_memory_features[edge_idx] = u @ neighbor_memory_feature
      #
      #                 # test1=(v @ neighbor_memory_feature).sum(dim=0) + (u @ neighbor_memory_feature).sum(dim=0)
      #                 # test2=neighbor_memory_feature.sum(dim=0)
      #                 #
      #                 # print('test1',test1)
      #                 # print('test2',test2)
      #             else:
      #                 #print('not node',C_neighbor_memory_features[b, k])
      #                 sub = C_neighbor_memory_features[b, k]
      #
      #                 if torch.all(sub == 0):
      #                     pass
      #                 else:
      #                     print("非零元素:", sub[sub != 0])
      #
      #
      #     # for b in range(len(source_nodes)):
      #     #     source_node = source_nodes[b]
      #     #     node_neighbors = neighbors[b]
      #     #
      #     #     for k, neighbor_node in enumerate(node_neighbors):
      #     #             if neighbor_node in C_neighbor_message_to_memory_features:
      #     #                 # 计算test1: message贡献 + old memory贡献
      #     #                 message_contrib = C_neighbor_message_to_memory_features[neighbor_node].sum(dim=0)
      #     #                 old_memory_contrib = C_neighbor_old_memory_to_memory_features[neighbor_node].sum(dim=0)
      #     #                 test1 = message_contrib + old_memory_contrib
      #     #                 # 计算test2: neighbor memory features的总和
      #     #                 test2 = C_neighbor_memory_features[b, k].sum(dim=0)
      #     #
      #     #                 print('test1',test1)
      #     #                 print('test2', test2)
      #     #
      #     #                 # 验证是否守恒
      #     #                 if not torch.allclose(test1, test2, atol=1e-4):
      #     #                     neighbor_memory_verify = False
      #     #                     diff = torch.abs(test1 - test2).max().item()
      #     #
      #     #
      #     # # 打印验证结果
      #     # if neighbor_memory_verify:
      #     #     print(f'邻居守恒验证结果: 通过 ✅')
      #     # else:
      #     #     print(f'邻居守恒验证结果: 失败 ❌')

      total = None
      for idx, mat in temporal_edge_contributions.items():
          for _, second_mat in mat.items():
              if total is None:
                  total = second_mat.clone()
              else:
                  total = total + second_mat

      # sum_dict = torch.stack(list(temporal_edge_contributions.values()), dim=0).sum(dim=0)  # [D_out]

      print('sum_dict', total.shape)

      # 2) 原始三部分贡献矩阵分别对所有轴求和，再相加
      sum_raw = C_neighbor_raw_features.sum(dim=(0, 1, 2))  # [D_out]
      sum_time = C_edge_time_embeddings.sum(dim=(0, 1, 2))  # [D_out]
      sum_feat = C_edge_features.sum(dim=(0, 1, 2))  # [D_out]
      sum_raw_2 = C_raw_features.sum(dim=(0, 1))
      sum_source = C_source_time.sum(dim=(0, 1))

      sum_original = sum_raw + sum_time + sum_feat + sum_raw_2 + sum_source  # [D_out]

      print('sum_original', sum_original.shape)

      print('sum_original', sum_original)
      print('sum_dict', total)

      # 3) 守恒检查
      check = torch.allclose(total, sum_original, atol=1e-4)
      print("守恒验证:", "通过 ✅" if check else "失败 ❌")

      return (h, C_memory_features,
              C_neighbor_memory_features,
              temporal_edge_contributions, neighbors, edge_idxs)

  def compute_embedding_attention_without_contribution(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20):
      B = len(source_nodes)
      # print('source nodes',source_nodes)
      source_nodes_torch = torch.from_numpy(source_nodes).long().to(self.device)
      timestamps_torch = torch.unsqueeze(torch.from_numpy(timestamps).float().to(self.device), dim=1)

      raw_source_node_features = self.node_features[source_nodes_torch, :]  # [B, D_node]

      print('raw_source_node_features', raw_source_node_features.shape)

      # 最底层：静态特征 + memory（可选）
      h = raw_source_node_features.clone()
      if self.use_memory:
          h = h + memory[source_nodes, :]

      # 初始时间编码（始终为0，因为是 query 节点当前时间）
      source_nodes_time_embedding = self.time_encoder(torch.zeros_like(timestamps_torch))  # [B, Dₜ]

      # print('source_nodes_time_embedding',source_nodes_time_embedding)

      for layer in range(n_layers):
          # 采样邻居（每一层用当前 query 时间采样）
          neighbors, edge_idxs, edge_times = self.neighbor_finder.get_temporal_neighbor(
              source_nodes, timestamps, n_neighbors=n_neighbors
          )

          # print('neighbors',neighbors,neighbors.shape)
          # print('edge_idxs',edge_idxs,edge_idxs.shape)
          # print('edge_times',edge_times)

          neighbors_torch = torch.from_numpy(neighbors).long().to(self.device)
          # print('neighbors_torch',neighbors_torch)
          edge_idxs_torch = torch.from_numpy(edge_idxs).long().to(self.device)
          edge_deltas = timestamps[:, np.newaxis] - edge_times

          # print('timestamps',timestamps)

          # print('edge_deltas',edge_deltas,edge_deltas.shape)

          edge_deltas_torch = torch.from_numpy(edge_deltas).float().to(self.device)

          # 获取邻居嵌入（从上一层）
          flat_neighbors = neighbors.flatten()
          print('flat_neighbors ', len(flat_neighbors))
          h_neighbors = self.node_features[flat_neighbors, :]
          if self.use_memory:
              h_neighbors = h_neighbors + memory[flat_neighbors, :]
          print('h_neighbors ', h_neighbors.shape)

          h_neighbors = h_neighbors.view(B, n_neighbors, -1)

          print('h_neighbors', h_neighbors.shape)

          # 时间编码与边特征
          edge_time_embeddings = self.time_encoder(edge_deltas_torch)  # [B, N, Dₜ]
          print('edge_time_embeddings', edge_time_embeddings.shape)
          edge_features = self.edge_features[edge_idxs_torch, :]  # [B, N, Dₑ]
          print('edge_features', edge_features.shape)
          mask = neighbors_torch == 0  # [B, N]

          # 聚合：本层嵌入 = f(hᶫ⁻¹, neighbors)
          h, _ = self.aggregate_without_contribution(layer + 1, h,
                                                                       source_nodes_time_embedding,
                                                                       h_neighbors,
                                                                       edge_time_embeddings,
                                                                       edge_features,
                                                                       mask)



      return h

  def map_contributions_to_temporal_edges_attention(self, C_raw_features, C_source_time, C_neighbor_raw_features,
                                          C_edge_time_embeddings,
                                          C_edge_features, source_nodes, neighbors, edge_idxs, timestamps):
      """
      将贡献值映射到具体的时序边上

      Args:
          C_neighbor_raw_features: [B, K, D_n, D_out] 邻居原始特征贡献
          C_edge_time_embeddings: [B, K, D_te, D_out] 边时间编码贡献
          C_edge_features: [B, K, D_ef, D_out] 边特征贡献
          source_nodes: [B] 源节点ID列表
          neighbors: [B, K] 邻居节点ID矩阵
          edge_idxs: [B, K] 边索引矩阵
          timestamps: [B] 时间戳列表

      Returns:
          temporal_edge_contributions: [total_edges, total_contrib_dim] 每条时序边的贡献值
          edge_info: [total_edges] 每条边的信息字典列表
      """
      B, K, _, D_out = C_neighbor_raw_features.shape

      # 初始化结果列表
      edge_contributions_dict = {}

      for b in range(B):
          source_node = source_nodes[b]
          timestamp = timestamps[b]

          # print('valid_cnt',valid_cnt)
          if b not in edge_contributions_dict:
              edge_contributions_dict[b] = dict()

          for k in range(K):
              neighbor_node = neighbors[b, k]
              edge_idx = edge_idxs[b, k]
              # print('neighbor edge_idx',edge_idx)

              source_contrib = C_neighbor_raw_features[b, k].sum(dim=0)  # [D_out]

              # print('source_contrib',source_contrib.shape)

              # 2. 边时间编码贡献（对特征维度求和）
              edge_time_contrib = C_edge_time_embeddings[b, k].sum(dim=0)  # [D_out]

              # print('edge_time_contrib', edge_time_contrib.shape)

              # 3. 边特征贡献（对特征维度求和）
              edge_feat_contrib = C_edge_features[b, k].sum(dim=0)  # [D_out]

              # 4. 合并所有贡献值
              edge_total_contrib = source_contrib + edge_time_contrib + edge_feat_contrib  # [D_out]

              # print('edge_feat_contrib',edge_feat_contrib.shape)

              add_from_source = (C_source_time[b].sum(dim=0) + C_raw_features[b].sum(dim=0)) / K
              # print('add_from_source', add_from_source.shape)
              edge_total_contrib = edge_total_contrib + add_from_source

              # print('edge_total_contrib.shape',edge_total_contrib.shape)

              if edge_idx in edge_contributions_dict[b]:
                  # 如果边标号已存在，贡献值相加
                  edge_contributions_dict[b][edge_idx] += edge_total_contrib

              else:
                  # 新的边标号
                  edge_contributions_dict[b][edge_idx] = edge_total_contrib

      return edge_contributions_dict

  def init_hidden_embeddings(self, node_list):
      hidden_embeddings, masks = [], []
      for i in range(len(node_list)):
          batch_node_idx = torch.from_numpy(node_list[i]).long().to(self.device)
          hidden_embeddings.append(self.node_features[batch_node_idx])
          masks.append(batch_node_idx == 0)
      return hidden_embeddings, masks

  def retrieve_time_features(self, cut_time, time_list,num_neighbor):
      cut_time = np.concatenate([cut_time, cut_time, cut_time])
      batch = len(cut_time)
      first_time_stamp = np.expand_dims(cut_time, 1)  # [3*bsz, 1]
      # time_features = [self.time_encoder(torch.from_numpy(np.zeros_like(first_time_stamp)).float().to(self.device))]
      time_features = []
      standard_timestamps = np.expand_dims(first_time_stamp, 2)
      for layer_i in range(len(time_list)):
          t_record = time_list[layer_i]
          time_delta = standard_timestamps - t_record.reshape(batch, -1, num_neighbor)
          time_delta = time_delta.reshape(batch, -1)
          time_delta = torch.from_numpy(time_delta).float().to(self.device)
          time_features.append(self.time_encoder(time_delta))
          standard_timestamps = np.expand_dims(t_record, 2)
      return time_features

  def retrieve_edge_features(self, edge_list):
      edge_features = []
      for i in range(len(edge_list)):
          batch_edge_idx = torch.from_numpy(edge_list[i]).long().to(self.device)
          edge_features.append(self.edge_features[batch_edge_idx])
      return edge_features

  def embedding_update_layer(self, memory, node_list, node_features, edge_features, time_features, mask_list,num_neighbor=10,
                             explain_weights=None):
      num_layers = len(node_list)
      ## initial neighboring node feature
      neighbor_node = node_list[-1].flatten()
      neighbor_node_feature = node_features[-1].view(-1, self.n_node_features)
      if self.use_memory:
          neighbor_node_feature = memory[neighbor_node, :] + neighbor_node_feature
      else:
          neighbor_node_feature = neighbor_node_feature

      for i in range(num_layers - 1):  # i=0, 1
          t = num_layers - 1 - i  # 2, 1
          source_node = node_list[t - 1].flatten()
          source_node_feature = node_features[t - 1].view(-1, self.n_node_features)  # [bsz, feature_dim]
          batch_layer = source_node_feature.shape[0]
          if self.use_memory:
              source_node_feature = memory[source_node, :] + source_node_feature
          else:
              source_node_feature = source_node_feature
          source_nodes_time_embedding = self.time_encoder(
              torch.zeros((batch_layer, 1)).to(self.device))  # [bsz, 1, time_dim]
          neighbor_node_feature = neighbor_node_feature.view(batch_layer, num_neighbor,
                                                             -1)  # [bsz, n_neighbor, feature_dim]
          assert neighbor_node_feature.shape[-1] == source_node_feature.shape[-1]
          edge_time_embeddings = time_features[t - 1].view(batch_layer, num_neighbor, -1)
          edgh_feature = edge_features[t - 1].view(batch_layer, num_neighbor, -1)
          mask = mask_list[t].view(batch_layer, -1)
          if explain_weights is not None:
              explain_weight = explain_weights[t - 1].view(batch_layer, -1)
          else:
              explain_weight = None

          assert mask.shape[-1] == num_neighbor

          updated_source_node_feature,_ = self.aggregate_explainweights(i, source_node_feature, source_nodes_time_embedding,
                                                          neighbor_node_feature, edge_time_embeddings, edgh_feature,
                                                          mask)
          # [bsz, n_node_features]
          neighbor_node_feature = updated_source_node_feature

      return updated_source_node_feature  # [true bsz, feature_dim]



  def embedding_update(self, memory, node_list, edge_list, time_list, cut_time, n_layers, num_neighbor=10,
                       explain_weights=None):
      """Recursive implementation of curr_layers temporal graph attention layers.

      src_idx_l [batch_size]: users / items input ids.
      cut_time_l [batch_size]: scalar representing the instant of the time where we want to extract the user / item representation.
      curr_layers [scalar]: number of temporal convolutional layers to stack.
      num_neighbors [scalar]: number of temporal neighbor to consider in each convolutional layer.
      """

      assert (n_layers >= 0)
      node_features, mask_list = self.init_hidden_embeddings(node_list)
      edge_features = self.retrieve_edge_features(edge_list)
      time_features = self.retrieve_time_features(cut_time, time_list, num_neighbor)

      source_embedding = self.embedding_update_layer(memory, node_list, node_features, edge_features, time_features,
                                                     mask_list, num_neighbor, explain_weights)  # [3*bs, node_features]

      # print('source_embedding',source_embedding.shape)

      return source_embedding



def get_embedding_module(module_type, node_features, edge_features, memory, neighbor_finder,
                         time_encoder, n_layers, n_node_features, n_edge_features, n_time_features,
                         embedding_dimension, device,
                         n_heads=2, dropout=0.1, n_neighbors=None,
                         use_memory=True):
  if module_type == "graph_attention":
    return GraphAttentionEmbedding(node_features=node_features,
                                    edge_features=edge_features,
                                    memory=memory,
                                    neighbor_finder=neighbor_finder,
                                    time_encoder=time_encoder,
                                    n_layers=n_layers,
                                    n_node_features=n_node_features,
                                    n_edge_features=n_edge_features,
                                    n_time_features=n_time_features,
                                    embedding_dimension=embedding_dimension,
                                    device=device,
                                    n_heads=n_heads, dropout=dropout, use_memory=use_memory)
  elif module_type == "graph_sum":
    return GraphSumEmbedding(node_features=node_features,
                              edge_features=edge_features,
                              memory=memory,
                              neighbor_finder=neighbor_finder,
                              time_encoder=time_encoder,
                              n_layers=n_layers,
                              n_node_features=n_node_features,
                              n_edge_features=n_edge_features,
                              n_time_features=n_time_features,
                              embedding_dimension=embedding_dimension,
                              device=device,
                              n_heads=n_heads, dropout=dropout, use_memory=use_memory)

  elif module_type == "identity":
    return IdentityEmbedding(node_features=node_features,
                             edge_features=edge_features,
                             memory=memory,
                             neighbor_finder=neighbor_finder,
                             time_encoder=time_encoder,
                             n_layers=n_layers,
                             n_node_features=n_node_features,
                             n_edge_features=n_edge_features,
                             n_time_features=n_time_features,
                             embedding_dimension=embedding_dimension,
                             device=device,
                             dropout=dropout)
  elif module_type == "time":
    return TimeEmbedding(node_features=node_features,
                         edge_features=edge_features,
                         memory=memory,
                         neighbor_finder=neighbor_finder,
                         time_encoder=time_encoder,
                         n_layers=n_layers,
                         n_node_features=n_node_features,
                         n_edge_features=n_edge_features,
                         n_time_features=n_time_features,
                         embedding_dimension=embedding_dimension,
                         device=device,
                         dropout=dropout,
                         n_neighbors=n_neighbors)
  else:
    raise ValueError("Embedding Module {} not supported".format(module_type))


