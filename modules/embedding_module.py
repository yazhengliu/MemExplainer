import torch
from torch import nn
import numpy as np
import math
from dataclasses import dataclass

from model.temporal_attention import TemporalAttentionLayer
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import MergeLayer

DEBUG_VERBOSE = False

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
                target_dtype = torch.float64
                self.q_proj.weight.copy_(pytorch_mha.q_proj_weight.to(dtype=target_dtype))
                self.k_proj.weight.copy_(pytorch_mha.k_proj_weight.to(dtype=target_dtype))
                self.v_proj.weight.copy_(pytorch_mha.v_proj_weight.to(dtype=target_dtype))

                if pytorch_mha.in_proj_bias is not None:
                    embed_dim = self.embed_dim
                    self.q_proj.bias.copy_(pytorch_mha.in_proj_bias[:embed_dim].to(dtype=target_dtype))
                    self.k_proj.bias.copy_(pytorch_mha.in_proj_bias[embed_dim:2 * embed_dim].to(dtype=target_dtype))
                    self.v_proj.bias.copy_(pytorch_mha.in_proj_bias[2 * embed_dim:].to(dtype=target_dtype))
            else:
                # 使用合并的 in_proj_weight（当 kdim == vdim == embed_dim 时）
                in_proj_weight = pytorch_mha.in_proj_weight  # [3 * embed_dim, embed_dim]
                embed_dim = self.embed_dim
                target_dtype = torch.float64

                q_proj_weight = in_proj_weight[:embed_dim, :].to(dtype=target_dtype)
                k_proj_weight = in_proj_weight[embed_dim:2 * embed_dim, :].to(dtype=target_dtype)
                v_proj_weight = in_proj_weight[2 * embed_dim:, :].to(dtype=target_dtype)

                self.q_proj.weight.copy_(q_proj_weight)
                self.k_proj.weight.copy_(k_proj_weight)
                self.v_proj.weight.copy_(v_proj_weight)

                if pytorch_mha.in_proj_bias is not None:
                    in_proj_bias = pytorch_mha.in_proj_bias.to(dtype=target_dtype)
                    self.q_proj.bias.copy_(in_proj_bias[:embed_dim])
                    self.k_proj.bias.copy_(in_proj_bias[embed_dim:2 * embed_dim])
                    self.v_proj.bias.copy_(in_proj_bias[2 * embed_dim:])

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

        Q_original = Q.clone()
        K_original = K.clone()
        V_original = V.clone()

        # 重塑为多头格式
        Q = Q.view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        # [batch_size, num_heads, seq_len_q, head_dim]
        K = K.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        # [batch_size, num_heads, seq_len_k, head_dim]
        V = V.view(batch_size, seq_len_v, self.num_heads, self.head_dim).transpose(1, 2)
        # [batch_size, num_heads, seq_len_v, head_dim]

        # Scaled Dot-Product Attention
        attn_weights_pre_softmax = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # [batch_size, num_heads, seq_len_q, seq_len_k]



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

        # 计算注意力输出
        attn_output = torch.matmul(attn_weights, V)
        # [batch_size, num_heads, seq_len_q, head_dim]

        attn_output_multihead = attn_output.clone()

        # 合并所有头
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.embed_dim
        )  # [batch_size, seq_len_q, embed_dim]

        # 输出投影
        output = self.out_proj(attn_output)  # [batch_size, seq_len_q, embed_dim]

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

        # 转换回 PyTorch 格式：[seq_len, batch_size, embed_dim]
        output = output.transpose(0, 1)  # [seq_len_q, batch_size, embed_dim]

        C_attn_to_output_sum = C_attn_to_output_multihead.sum(dim=(1, 3))
        # [batch_size, seq_len_q, embed_dim]

        # 转换维度顺序以匹配 output: [seq_len_q, batch_size, embed_dim]
        C_attn_to_output_sum = C_attn_to_output_sum.transpose(0, 1)
        # [seq_len_q, batch_size, embed_dim]

        R_attn_output = C_attn_to_output_multihead.sum(dim=-1)
        # [batch_size, num_heads, seq_len_q, head_dim]

        C_attn_weights, C_V = self.matrix_multiply_contribution_ratio(
            X=attn_weights,  # [B, H, seq_len_q, seq_len_k]
            Y=V,  # [B, H, seq_len_k, head_dim]
            Z=attn_output_multihead,  # [B, H, seq_len_q, head_dim]
            R_Z=R_attn_output,  # [B, H, seq_len_q, head_dim]
            transpose_Y=False,
            scale_factor=1.0,
        )

        C_weights_tooutput=torch.einsum('bhjdk,bhjkp->bhjdp', C_attn_weights, C_attn_to_output_multihead)
        C_V_tooutput = torch.einsum('bhjdk,bhjkp->bhjdp', C_V , C_attn_to_output_multihead)

        C_weights_tooutput=C_weights_tooutput/2

        C_V_tooutput=C_V_tooutput/2



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


        Q_to_output = torch.einsum('bhjdk,bhjkp->bhjdp', Q_attention, C_weights_tooutput)
        K_to_output = torch.einsum('bhjdk,bhjkp->bhjdp', K_attention, C_weights_tooutput)

        K_to_output= K_to_output.permute(0, 1, 3, 2, 4)




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
        C_query_to_Q_multihead = C_query_to_Q.view(
            batch_size, seq_len_q, self.embed_dim, self.num_heads, self.head_dim
        )

        C_query_to_Q_multihead = C_query_to_Q_multihead.permute(0, 3, 1, 2, 4)


        query_to_output = torch.einsum('bhjdk,bhjkp->bhjdp',C_query_to_Q_multihead, Q_to_output)

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


        # C_key_to_K_multihead = C_key_to_K.view(
        #     batch_size, seq_len_q, self.embed_dim, self.num_heads, self.head_dim
        # )
        # print('C_key_to_K_multihead', C_key_to_K_multihead.shape)

        kdim = key_bt.shape[2]  # 获取 kdim 的实际值
        C_key_to_K_multihead = C_key_to_K.view(
            batch_size, seq_len_k, kdim, self.num_heads, self.head_dim
        )

        C_key_to_K_multihead = C_key_to_K_multihead.permute(0, 3, 1,2,4)
        key_to_output = torch.einsum('bhjdk,bhjkp->bhjdp',  C_key_to_K_multihead,K_to_output)

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
        vdim = value_bt.shape[2]  # 获取 kdim 的实际值
        C_value_to_V_multihead = C_value_to_V.view(
            batch_size, seq_len_k, kdim, self.num_heads, self.head_dim
        )

        C_value_to_V_multihead  = C_value_to_V_multihead.permute(0, 3, 1, 2, 4)

        value_to_output = torch.einsum('bhjdk,bhjkp->bhjdp', C_value_to_V_multihead, C_V_tooutput)

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

        C_query=query_to_output.sum(dim=1)
        C_query=C_query.permute(1,0,2,3)

        C_key=key_to_output.sum(dim=1)
        C_key =C_key.permute(1,0,2,3)

        C_value= value_to_output.sum(dim=1)
        C_value = C_value.permute(1, 0, 2, 3)

        test_sum=C_value.sum(dim=(0,2)) + C_key.sum(dim=(0,2)) + C_query.sum(dim=(0,2))

        is_conserved_value = torch.allclose(
            test_sum,
            torch.ones_like(test_sum),
            atol=1e-5,
            rtol=1e-5
        )

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


  def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                        use_time_proj=True):
    """Recursive implementation of curr_layers temporal graph attention layers.

    src_idx_l [batch_size]: users / items input ids.
    cut_time_l [batch_size]: scalar representing the instant of the time where we want to extract the user / item representation.
    curr_layers [scalar]: number of temporal convolutional layers to stack.
    num_neighbors [scalar]: number of temporal neighbor to consider in each convolutional layer.
    """

    assert (n_layers >= 0)
    if DEBUG_VERBOSE:
      print('source_nodes',len(source_nodes),source_nodes)

    source_nodes_torch = torch.from_numpy(source_nodes).long().to(self.device)
    timestamps_torch = torch.unsqueeze(torch.from_numpy(timestamps).float().to(self.device), dim=1)

    # query node always has the start time -> time span == 0
    source_nodes_time_embedding = self.time_encoder(torch.zeros_like(
      timestamps_torch))

    source_node_features = self.node_features[source_nodes_torch, :]
    if DEBUG_VERBOSE:
      print('source_node_features',source_node_features.shape)

    if self.use_memory:
      source_node_features = memory[source_nodes, :] + source_node_features
      # print('memory.shape',memory.shape)
      #
      # print('source_node_features.shape',source_node_features.shape)



    if n_layers == 0:
      return source_node_features
    else:

      source_node_conv_embeddings = self.compute_embedding(memory,
                                                           source_nodes,
                                                           timestamps,
                                                           n_layers=n_layers - 1,
                                                           n_neighbors=n_neighbors)

      if DEBUG_VERBOSE:
        print('source_node_conv_embeddings',source_node_conv_embeddings.shape)

      neighbors, edge_idxs, edge_times = self.neighbor_finder.get_temporal_neighbor(
        source_nodes,
        timestamps,
        n_neighbors=n_neighbors)
      if DEBUG_VERBOSE:
        print('neighbors',neighbors.shape,neighbors)
        print('edge_idxs',edge_idxs.shape ,edge_idxs)
        print('edge_times',edge_times.shape,edge_times)


      neighbors_torch = torch.from_numpy(neighbors).long().to(self.device)

      edge_idxs = torch.from_numpy(edge_idxs).long().to(self.device)

      edge_deltas = timestamps[:, np.newaxis] - edge_times

      edge_deltas_torch = torch.from_numpy(edge_deltas).float().to(self.device)

      neighbors = neighbors.flatten()
      neighbor_embeddings = self.compute_embedding(memory,
                                                   neighbors,
                                                   np.repeat(timestamps, n_neighbors),
                                                   n_layers=n_layers - 1,
                                                   n_neighbors=n_neighbors)

      effective_n_neighbors = n_neighbors if n_neighbors > 0 else 1
      neighbor_embeddings = neighbor_embeddings.view(len(source_nodes), effective_n_neighbors, -1)
      edge_time_embeddings = self.time_encoder(edge_deltas_torch)

      edge_features = self.edge_features[edge_idxs, :]

      mask = neighbors_torch == 0

      source_embedding,C_source_h,C_source_time,C_neighbor_embeddings,\
           C_edge_time_embeddings,C_edge_features = self.aggregate(n_layers, source_node_conv_embeddings,
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
      assert n_layers >= 0

      effective_n_neighbors = n_neighbors if n_neighbors > 0 else 1
      root_batch_size = len(source_nodes)

      nodes_per_depth = [np.asarray(source_nodes)]
      timestamps_per_depth = [np.asarray(timestamps)]
      neighbors_per_depth = []
      edge_idxs_per_depth = []
      masks_per_depth = []
      edge_deltas_per_depth = []
      root_batch_indices_per_depth = [np.arange(root_batch_size)]
      top_neighbor_slots_per_depth = [np.full(root_batch_size, -1, dtype=np.int64)]

      current_nodes = np.asarray(source_nodes)
      current_timestamps = np.asarray(timestamps)
      current_root_batch_indices = np.arange(root_batch_size)
      current_top_neighbor_slots = np.full(root_batch_size, -1, dtype=np.int64)

      for layer_idx in range(n_layers):
          neighbors, edge_idxs, edge_times = self.neighbor_finder.get_temporal_neighbor(
              current_nodes,
              current_timestamps,
              n_neighbors=n_neighbors,
          )
          neighbors_per_depth.append(neighbors)
          edge_idxs_per_depth.append(edge_idxs)
          masks_per_depth.append(neighbors == 0)
          edge_deltas_per_depth.append(current_timestamps[:, np.newaxis] - edge_times)

          current_nodes = neighbors.flatten()
          current_timestamps = np.repeat(current_timestamps, effective_n_neighbors)
          current_root_batch_indices = np.repeat(current_root_batch_indices, effective_n_neighbors)
          if layer_idx == 0:
              current_top_neighbor_slots = np.tile(
                  np.arange(effective_n_neighbors, dtype=np.int64),
                  len(neighbors),
              )
          else:
              current_top_neighbor_slots = np.repeat(current_top_neighbor_slots, effective_n_neighbors)
          nodes_per_depth.append(current_nodes)
          timestamps_per_depth.append(current_timestamps)
          root_batch_indices_per_depth.append(current_root_batch_indices)
          top_neighbor_slots_per_depth.append(current_top_neighbor_slots)

      embeddings = []
      source_time_embeddings = []
      raw_node_features = []
      for depth_nodes, depth_timestamps in zip(nodes_per_depth, timestamps_per_depth):
          depth_nodes_torch = torch.from_numpy(depth_nodes).long().to(self.device)
          depth_timestamps_torch = torch.unsqueeze(
              torch.from_numpy(depth_timestamps).float().to(self.device),
              dim=1,
          )
          raw_features = self.node_features[depth_nodes_torch, :]
          base_embedding = raw_features.clone()
          if self.use_memory:
              base_embedding = base_embedding + memory[depth_nodes, :]
          embeddings.append(base_embedding)
          raw_node_features.append(raw_features)
          source_time_embeddings.append(self.time_encoder(torch.zeros_like(depth_timestamps_torch)))

      layer_caches = [[None for _ in range(n_layers + 1)] for _ in range(n_layers)]
      final_embedding = embeddings[0]
      top_neighbors = np.zeros((root_batch_size, effective_n_neighbors), dtype=np.int64)
      top_edge_idxs = np.zeros((root_batch_size, effective_n_neighbors), dtype=np.int64)

      for remaining_layers in range(1, n_layers + 1):
          for depth in range(0, n_layers - remaining_layers + 1):
              batch_size = len(nodes_per_depth[depth])
              neighbors = neighbors_per_depth[depth]
              edge_idxs = edge_idxs_per_depth[depth]
              mask = torch.from_numpy(masks_per_depth[depth]).to(self.device)
              edge_idxs_torch = torch.from_numpy(edge_idxs).long().to(self.device)
              edge_deltas_torch = torch.from_numpy(edge_deltas_per_depth[depth]).float().to(self.device)

              aggregated = self.aggregate(
                  remaining_layers,
                  embeddings[depth],
                  source_time_embeddings[depth],
                  embeddings[depth + 1].view(batch_size, effective_n_neighbors, -1),
                  self.time_encoder(edge_deltas_torch),
                  self.edge_features[edge_idxs_torch, :],
                  mask,
              )
              embeddings[depth] = aggregated[0]
              layer_caches[depth][remaining_layers] = {
                  'output_embedding': aggregated[0],
                  'source_contrib': aggregated[1],
                  'source_time_contrib': aggregated[2],
                  'neighbor_contrib': aggregated[3],
                  'edge_time_contrib': aggregated[4],
                  'edge_feature_contrib': aggregated[5],
                  'neighbors': neighbors,
                  'edge_idxs': edge_idxs,
                  'timestamps': timestamps_per_depth[depth],
              }

              if depth == 0 and remaining_layers == n_layers:
                  final_embedding = aggregated[0]
                  top_neighbors = neighbors
                  top_edge_idxs = edge_idxs

      output_dim = final_embedding.shape[1]
      output_dtype = final_embedding.dtype
      output_device = final_embedding.device
      final_source_memory_features = torch.zeros(
          root_batch_size,
          raw_node_features[0].shape[1],
          output_dim,
          dtype=output_dtype,
          device=output_device,
      )
      top_neighbor_memory_features = torch.zeros(
          root_batch_size,
          effective_n_neighbors,
          raw_node_features[0].shape[1],
          output_dim,
          dtype=output_dtype,
          device=output_device,
      )
      downstream = [[None for _ in range(n_layers + 1)] for _ in range(n_layers + 1)]
      downstream[0][n_layers] = torch.diag_embed(final_embedding).to(dtype=output_dtype, device=output_device)

      temporal_edge_contributions = {}

      for remaining_layers in range(n_layers, 0, -1):
          for depth in range(0, n_layers - remaining_layers + 1):
              downstream_current = downstream[depth][remaining_layers]
              if downstream_current is None:
                  continue

              cache = layer_caches[depth][remaining_layers]
              if cache is None:
                  continue

              output_embedding = cache['output_embedding']
              output_den_source = output_embedding.unsqueeze(1)
              output_den_neighbor = output_embedding.unsqueeze(1).unsqueeze(1)

              source_share = torch.where(
                  output_den_source != 0,
                  cache['source_contrib'] / output_den_source,
                  torch.zeros_like(cache['source_contrib']),
              )
              source_time_share = torch.where(
                  output_den_source != 0,
                  cache['source_time_contrib'] / output_den_source,
                  torch.zeros_like(cache['source_time_contrib']),
              )
              neighbor_share = torch.where(
                  output_den_neighbor != 0,
                  cache['neighbor_contrib'] / output_den_neighbor,
                  torch.zeros_like(cache['neighbor_contrib']),
              )
              edge_time_share = torch.where(
                  output_den_neighbor != 0,
                  cache['edge_time_contrib'] / output_den_neighbor,
                  torch.zeros_like(cache['edge_time_contrib']),
              )
              edge_feat_share = torch.where(
                  output_den_neighbor != 0,
                  cache['edge_feature_contrib'] / output_den_neighbor,
                  torch.zeros_like(cache['edge_feature_contrib']),
              )

              source_to_top = torch.einsum('bid,bdo->bio', source_share, downstream_current)
              source_time_to_top = torch.einsum('bid,bdo->bio', source_time_share, downstream_current)
              neighbor_to_top = torch.einsum('bkid,bdo->bkio', neighbor_share, downstream_current)
              edge_time_to_top = torch.einsum('bkid,bdo->bkio', edge_time_share, downstream_current)
              edge_feat_to_top = torch.einsum('bkid,bdo->bkio', edge_feat_share, downstream_current)
              root_batch_indices = root_batch_indices_per_depth[depth]

              if remaining_layers > 1:
                  layer_temporal = self.map_leaf_contributions_to_temporal_edges(
                      source_time_to_top,
                      edge_time_to_top,
                      edge_feat_to_top,
                      nodes_per_depth[depth],
                      cache['neighbors'],
                      cache['edge_idxs'],
                      cache['timestamps'],
                  )
                  self._accumulate_temporal_edge_contributions(
                      temporal_edge_contributions,
                      layer_temporal,
                      root_batch_indices,
                  )

              if remaining_layers == 1:
                  source_raw_features, source_memory_features = self.allocate_source_contributions(
                      source_to_top,
                      raw_node_features[depth],
                      memory,
                      nodes_per_depth[depth],
                  )

                  neighbor_raw_features, neighbor_memory_features = self.allocate_neighbor_contributions(
                      neighbor_to_top,
                      cache['neighbors'].flatten(),
                      memory,
                  )

                  source_memory_features = source_memory_features.to(dtype=output_dtype, device=output_device)
                  neighbor_memory_features = neighbor_memory_features.to(dtype=output_dtype, device=output_device)

                  top_neighbor_slots = top_neighbor_slots_per_depth[depth]
                  for batch_idx, root_batch_idx in enumerate(root_batch_indices):
                      root_batch_idx = int(root_batch_idx)
                      if depth == 0:
                          final_source_memory_features[root_batch_idx] += source_memory_features[batch_idx]
                          for neighbor_idx in range(effective_n_neighbors):
                              top_neighbor_memory_features[root_batch_idx, neighbor_idx] += (
                                  neighbor_memory_features[batch_idx, neighbor_idx]
                              )
                      else:
                          top_slot = int(top_neighbor_slots[batch_idx])
                          if top_slot >= 0:
                              top_neighbor_memory_features[root_batch_idx, top_slot] += (
                                  source_memory_features[batch_idx] +
                                  neighbor_memory_features[batch_idx].sum(dim=0)
                              )

                  layer_temporal = self.map_contributions_to_temporal_edges(
                      source_raw_features,
                      source_time_to_top,
                      neighbor_raw_features,
                      edge_time_to_top,
                      edge_feat_to_top,
                      nodes_per_depth[depth],
                      cache['neighbors'],
                      cache['edge_idxs'],
                      cache['timestamps'],
                  )
                  self._accumulate_temporal_edge_contributions(
                      temporal_edge_contributions,
                      layer_temporal,
                      root_batch_indices,
                  )
                  continue

              prev_source = downstream[depth][remaining_layers - 1]
              if prev_source is None:
                  downstream[depth][remaining_layers - 1] = source_to_top.clone()
              else:
                  downstream[depth][remaining_layers - 1] = prev_source + source_to_top

              flat_neighbor_to_top = neighbor_to_top.reshape(-1, neighbor_to_top.shape[2], neighbor_to_top.shape[3])
              prev_neighbor = downstream[depth + 1][remaining_layers - 1]
              if prev_neighbor is None:
                  downstream[depth + 1][remaining_layers - 1] = flat_neighbor_to_top.clone()
              else:
                  downstream[depth + 1][remaining_layers - 1] = prev_neighbor + flat_neighbor_to_top

      return (
          final_embedding,
          final_source_memory_features,
          top_neighbor_memory_features,
          temporal_edge_contributions,
          top_neighbors,
          top_edge_idxs,
      )


  def compute_embedding_iterative_without_contribution(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20):
      assert n_layers >= 0

      effective_n_neighbors = n_neighbors if n_neighbors > 0 else 1
      nodes_per_depth = [np.asarray(source_nodes)]
      timestamps_per_depth = [np.asarray(timestamps)]
      neighbors_per_depth = []
      edge_idxs_per_depth = []
      masks_per_depth = []
      edge_deltas_per_depth = []

      current_nodes = np.asarray(source_nodes)
      current_timestamps = np.asarray(timestamps)
      for _ in range(n_layers):
          neighbors, edge_idxs, edge_times = self.neighbor_finder.get_temporal_neighbor(
              current_nodes,
              current_timestamps,
              n_neighbors=n_neighbors,
          )
          neighbors_per_depth.append(neighbors)
          edge_idxs_per_depth.append(edge_idxs)
          masks_per_depth.append(neighbors == 0)
          edge_deltas_per_depth.append(current_timestamps[:, np.newaxis] - edge_times)

          current_nodes = neighbors.flatten()
          current_timestamps = np.repeat(current_timestamps, effective_n_neighbors)
          nodes_per_depth.append(current_nodes)
          timestamps_per_depth.append(current_timestamps)

      embeddings = []
      source_time_embeddings = []
      for depth_nodes, depth_timestamps in zip(nodes_per_depth, timestamps_per_depth):
          depth_nodes_torch = torch.from_numpy(depth_nodes).long().to(self.device)
          depth_timestamps_torch = torch.unsqueeze(
              torch.from_numpy(depth_timestamps).float().to(self.device),
              dim=1,
          )
          base_embedding = self.node_features[depth_nodes_torch, :].clone()
          if self.use_memory:
              base_embedding = base_embedding + memory[depth_nodes, :]
          embeddings.append(base_embedding)
          source_time_embeddings.append(self.time_encoder(torch.zeros_like(depth_timestamps_torch)))

      for remaining_layers in range(1, n_layers + 1):
          for depth in range(0, n_layers - remaining_layers + 1):
              batch_size = len(nodes_per_depth[depth])
              edge_idxs_torch = torch.from_numpy(edge_idxs_per_depth[depth]).long().to(self.device)
              edge_deltas_torch = torch.from_numpy(edge_deltas_per_depth[depth]).float().to(self.device)
              mask = torch.from_numpy(masks_per_depth[depth]).to(self.device)

              embeddings[depth] = self.aggregate_without_contribution(
                  remaining_layers,
                  embeddings[depth],
                  source_time_embeddings[depth],
                  embeddings[depth + 1].view(batch_size, effective_n_neighbors, -1),
                  self.time_encoder(edge_deltas_torch),
                  self.edge_features[edge_idxs_torch, :],
                  mask,
              )

      return embeddings[0]

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

  def map_leaf_contributions_to_temporal_edges(self, C_source_time, C_edge_time_embeddings,
                                               C_edge_features, source_nodes, neighbors,
                                               edge_idxs, timestamps):
      B, K, _, _ = C_edge_time_embeddings.shape
      edge_contributions_dict = {}

      for b in range(B):
          if b not in edge_contributions_dict:
              edge_contributions_dict[b] = {}

          add_from_source = C_source_time[b].sum(dim=0) / K
          for k in range(K):
              edge_idx = edge_idxs[b, k]
              edge_total_contrib = (
                  C_edge_time_embeddings[b, k].sum(dim=0) +
                  C_edge_features[b, k].sum(dim=0) +
                  add_from_source
              )
              if edge_idx in edge_contributions_dict[b]:
                  edge_contributions_dict[b][edge_idx] += edge_total_contrib
              else:
                  edge_contributions_dict[b][edge_idx] = edge_total_contrib

      return edge_contributions_dict

  def _accumulate_temporal_edge_contributions(self, temporal_edge_contributions, layer_temporal,
                                              root_batch_indices):
      for batch_idx, contribs in layer_temporal.items():
          root_batch_idx = int(root_batch_indices[batch_idx])
          if root_batch_idx not in temporal_edge_contributions:
              temporal_edge_contributions[root_batch_idx] = {}
          for edge_idx, contrib in contribs.items():
              if edge_idx in temporal_edge_contributions[root_batch_idx]:
                  temporal_edge_contributions[root_batch_idx][edge_idx] += contrib
              else:
                  temporal_edge_contributions[root_batch_idx][edge_idx] = contrib.clone()


class GraphSumEmbedding(GraphEmbedding):
  @dataclass
  class GraphSumForwardCache:
    source_node_features: torch.Tensor
    source_nodes_time_embedding: torch.Tensor
    neighbor_embeddings: torch.Tensor
    edge_time_embeddings: torch.Tensor
    edge_features: torch.Tensor
    neighbors_features: torch.Tensor
    nb_lin: torch.Tensor
    nb_sum_pre: torch.Tensor
    neighbors_sum: torch.Tensor
    src_time: torch.Tensor
    source_features: torch.Tensor
    z: torch.Tensor
    source_embedding: torch.Tensor

  @dataclass
  class GraphSumAttributionResult:
    source_embedding: torch.Tensor
    source_node_features_to_semb: torch.Tensor
    source_time_to_semb: torch.Tensor
    neighbor_embeddings_to_semb: torch.Tensor
    edge_time_embeddings_to_semb: torch.Tensor
    edge_features_to_semb: torch.Tensor

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


  def _compute_linear_share(self, weight, input_tensor):
      Z = input_tensor.unsqueeze(2) * weight.unsqueeze(0)
      S = Z.sum(dim=1)
      den = S.unsqueeze(1)
      return torch.where(den != 0, Z / den, torch.zeros_like(Z))

  def _attribute_linear_output(self, weight, input_tensor, output_tensor):
      phi = self._compute_linear_share(weight, input_tensor)
      return phi * output_tensor.unsqueeze(1)

  def _attribute_relu(self, pre_activation, post_activation, downstream_contribution):
      relu_alpha = torch.where(
          pre_activation != 0,
          post_activation / pre_activation,
          torch.zeros_like(pre_activation),
      )
      return downstream_contribution * relu_alpha.unsqueeze(-1)

  def _attribute_sum(self, parts, summed, downstream_contribution):
      den_sum = summed.unsqueeze(1)
      share_sum = torch.where(
          den_sum != 0,
          parts / den_sum,
          torch.zeros_like(parts),
      )
      return share_sum.unsqueeze(-1) * downstream_contribution.unsqueeze(1)

  def _split_concat_contributions(self, contribution, split_sizes, dim):
      return torch.split(contribution, split_sizes, dim=dim)

  def _ensure_graph_sum_dtype(self, n_layer, source_node_features, source_nodes_time_embedding,
                              neighbor_embeddings, edge_time_embeddings, edge_features):
      return (
          source_node_features,
          source_nodes_time_embedding,
          neighbor_embeddings,
          edge_time_embeddings,
          edge_features,
      )

  def _graph_sum_forward(self, n_layer, source_node_features, source_nodes_time_embedding,
                         neighbor_embeddings, edge_time_embeddings, edge_features):
      (
          source_node_features,
          source_nodes_time_embedding,
          neighbor_embeddings,
          edge_time_embeddings,
          edge_features,
      ) = self._ensure_graph_sum_dtype(
          n_layer,
          source_node_features,
          source_nodes_time_embedding,
          neighbor_embeddings,
          edge_time_embeddings,
          edge_features,
      )

      neighbors_features = torch.cat(
          [neighbor_embeddings, edge_time_embeddings, edge_features],
          dim=2,
      )
      nb_lin = self.linear_1[n_layer - 1](neighbors_features)
      nb_sum_pre = nb_lin.sum(dim=1)
      neighbors_sum = torch.nn.functional.relu(nb_sum_pre)

      src_time = source_nodes_time_embedding.squeeze()
      source_features = torch.cat([source_node_features, src_time], dim=1)
      # print('source_features', source_features.shape)

      z = torch.cat([neighbors_sum, source_features], dim=1)
      # print('source_embedding', z.shape)

      source_embedding = self.linear_2[n_layer - 1](z)
      # print('source_embedding', source_embedding.shape)

      return self.GraphSumForwardCache(
          source_node_features=source_node_features,
          source_nodes_time_embedding=source_nodes_time_embedding,
          neighbor_embeddings=neighbor_embeddings,
          edge_time_embeddings=edge_time_embeddings,
          edge_features=edge_features,
          neighbors_features=neighbors_features,
          nb_lin=nb_lin,
          nb_sum_pre=nb_sum_pre,
          neighbors_sum=neighbors_sum,
          src_time=src_time,
          source_features=source_features,
          z=z,
          source_embedding=source_embedding,
      )

  def _attribute_graph_sum_inputs(self, n_layer, cache):
      z64 = cache.z.to(torch.float64)
      source_embedding64 = cache.source_embedding.to(torch.float64)
      neighbors_sum64 = cache.neighbors_sum.to(torch.float64)
      nb_sum_pre64 = cache.nb_sum_pre.to(torch.float64)
      nb_lin64 = cache.nb_lin.to(torch.float64)
      neighbors_features64 = cache.neighbors_features.to(torch.float64)
      source_node_features64 = cache.source_node_features.to(torch.float64)
      src_time64 = cache.src_time.to(torch.float64)
      neighbor_embeddings64 = cache.neighbor_embeddings.to(torch.float64)
      edge_time_embeddings64 = cache.edge_time_embeddings.to(torch.float64)
      edge_features64 = cache.edge_features.to(torch.float64)

      W2_T = self.linear_2[n_layer - 1].weight.t().to(torch.float64)
      C_z_to_semb = self._attribute_linear_output(W2_T, z64, source_embedding64)
      # print(' C_z_to_semb.shape', C_z_to_semb.shape)

      H1 = neighbors_sum64.size(1)
      B, K, D_nf = neighbors_features64.shape

      C_ns_to_semb = C_z_to_semb[:, :H1, :]
      R_nb_sum_pre = self._attribute_relu(
          nb_sum_pre64,
          neighbors_sum64,
          C_ns_to_semb,
      )

      R_nb_lin = self._attribute_sum(nb_lin64, nb_sum_pre64, R_nb_sum_pre)

      # print('linear 2', torch.allclose(R_nb_lin.sum(dim=1), R_nb_sum_pre, atol=1e-4))
      # print('linear 3', torch.allclose(C_ns_to_semb, R_nb_sum_pre, atol=1e-4))

      X_nf = neighbors_features64.reshape(B * K, D_nf)
      W1_T = self.linear_1[n_layer - 1].weight.t().to(torch.float64)
      R_nf_to_nb = self._compute_linear_share(W1_T, X_nf).reshape(B, K, D_nf, H1)
      # print('R_nf_to_nb.shape', R_nf_to_nb.shape)

      C_nf_to_semb = torch.einsum('bkfh,bkho->bkfo', R_nf_to_nb, R_nb_lin)
      lhs = C_nf_to_semb.sum(dim=2)
      rhs = R_nb_lin.sum(dim=2)
      # print('linear 4', torch.allclose(lhs, rhs, atol=1e-4))

      D_n = neighbor_embeddings64.size(2)
      D_te = edge_time_embeddings64.size(2)
      D_ef = edge_features64.size(2)
      (
          C_neighbor_embeddings_to_semb,
          C_edge_time_embeddings_to_semb,
          C_edge_features_to_semb,
      ) = self._split_concat_contributions(C_nf_to_semb, [D_n, D_te, D_ef], dim=2)

      D_sf = source_node_features64.size(1)
      D_st = src_time64.size(1)
      C_srcfeat_to_semb = C_z_to_semb[:, H1:, :]
      (
          C_source_node_features_to_semb,
          C_source_time_to_semb,
      ) = self._split_concat_contributions(C_srcfeat_to_semb, [D_sf, D_st], dim=1)

      nb_sum = (
              C_neighbor_embeddings_to_semb.sum(dim=(1, 2)) +
              C_edge_time_embeddings_to_semb.sum(dim=(1, 2)) +
              C_edge_features_to_semb.sum(dim=(1, 2))
      )
      src_sum = (
              C_source_node_features_to_semb.sum(dim=1) +
              C_source_time_to_semb.sum(dim=1)
      )
      total_contrib = nb_sum + src_sum
      final_ok = torch.allclose(total_contrib, source_embedding64, atol=1e-4)
      if not final_ok:
          abs_err = torch.abs(total_contrib - source_embedding64)
          max_abs_err = abs_err.max().item()
          mean_abs_err = abs_err.mean().item()
          zero_linear_den = (z64.unsqueeze(2) * W2_T.unsqueeze(0)).sum(dim=1) == 0
          zero_neighbor_sum_den = nb_sum_pre64 == 0
          if DEBUG_VERBOSE:
              print(
                  'final flag debug',
                  {
                      'n_layer': int(n_layer),
                      'batch_size': int(cache.source_embedding.shape[0]),
                      'embedding_dim': int(cache.source_embedding.shape[1]),
                      'max_abs_err': max_abs_err,
                      'mean_abs_err': mean_abs_err,
                      'zero_linear_den_count': int(zero_linear_den.sum().item()),
                      'zero_linear_den_total': int(zero_linear_den.numel()),
                      'zero_neighbor_sum_den_count': int(zero_neighbor_sum_den.sum().item()),
                      'zero_neighbor_sum_den_total': int(zero_neighbor_sum_den.numel()),
                  },
              )

      return self.GraphSumAttributionResult(
          source_embedding=cache.source_embedding,
          source_node_features_to_semb=C_source_node_features_to_semb.to(cache.source_embedding.dtype),
          source_time_to_semb=C_source_time_to_semb.to(cache.source_embedding.dtype),
          neighbor_embeddings_to_semb=C_neighbor_embeddings_to_semb.to(cache.source_embedding.dtype),
          edge_time_embeddings_to_semb=C_edge_time_embeddings_to_semb.to(cache.source_embedding.dtype),
          edge_features_to_semb=C_edge_features_to_semb.to(cache.source_embedding.dtype),
      )



  def aggregate(self, n_layer, source_node_features, source_nodes_time_embedding,
                neighbor_embeddings,
                edge_time_embeddings, edge_features, mask):
    cache = self._graph_sum_forward(
        n_layer,
        source_node_features,
        source_nodes_time_embedding,
        neighbor_embeddings,
        edge_time_embeddings,
        edge_features,
    )
    attribution = self._attribute_graph_sum_inputs(n_layer, cache)
    return (
        attribution.source_embedding,
        attribution.source_node_features_to_semb,
        attribution.source_time_to_semb,
        attribution.neighbor_embeddings_to_semb,
        attribution.edge_time_embeddings_to_semb,
        attribution.edge_features_to_semb,
    )
  def aggregate_without_contribution(self, n_layer, source_node_features, source_nodes_time_embedding,
                neighbor_embeddings,
                edge_time_embeddings, edge_features, mask):
    cache = self._graph_sum_forward(
        n_layer,
        source_node_features,
        source_nodes_time_embedding,
        neighbor_embeddings,
        edge_time_embeddings,
        edge_features,
    )
    return cache.source_embedding

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

  def _prepare_attention_mask(self, mask):
      invalid_neighborhood_mask = mask.all(dim=1, keepdim=True)
      mask_processed = mask.clone()
      if invalid_neighborhood_mask.any():
          mask_processed[invalid_neighborhood_mask.squeeze(1), 0] = False
      return mask_processed

  def _prepare_attention_io(self, source_node_features, source_nodes_time_embedding,
                            neighbor_embeddings, edge_time_embeddings, edge_features):
      src_node_features_unrolled = source_node_features.unsqueeze(1)
      query = torch.cat([src_node_features_unrolled, source_nodes_time_embedding], dim=2)
      key = torch.cat([neighbor_embeddings, edge_features, edge_time_embeddings], dim=2)
      return query.permute(1, 0, 2), key.permute(1, 0, 2)

  def _split_attention_contributions(self, C_query, C_key, C_value,
                                     C_attn_to_final, C_src_to_final,
                                     source_feature_dim, neighbor_feature_dim,
                                     edge_feature_dim):
      C_value_to_final = torch.einsum('nbdk,bko->nbdo', C_value, C_attn_to_final)
      C_key_to_final = torch.einsum('nbdk,bko->nbdo', C_key, C_attn_to_final)
      C_query_to_final = torch.einsum('nbdk,bko->nbdo', C_query, C_attn_to_final)

      C_query_to_final_squeezed = C_query_to_final.squeeze(0)
      C_source_node_features_to_final = (
          C_query_to_final_squeezed[:, :source_feature_dim, :] + C_src_to_final
      )
      C_source_time_to_final = C_query_to_final_squeezed[:, source_feature_dim:, :]

      edge_time_start = neighbor_feature_dim + edge_feature_dim
      C_neighbor_embeddings_to_final = (
          C_key_to_final[:, :, :neighbor_feature_dim, :] +
          C_value_to_final[:, :, :neighbor_feature_dim, :]
      ).permute(1, 0, 2, 3)
      C_edge_features_to_final = (
          C_key_to_final[:, :, neighbor_feature_dim:edge_time_start, :] +
          C_value_to_final[:, :, neighbor_feature_dim:edge_time_start, :]
      ).permute(1, 0, 2, 3)
      C_edge_time_to_final = (
          C_key_to_final[:, :, edge_time_start:, :] +
          C_value_to_final[:, :, edge_time_start:, :]
      ).permute(1, 0, 2, 3)

      return (
          C_source_node_features_to_final,
          C_source_time_to_final,
          C_neighbor_embeddings_to_final,
          C_edge_time_to_final,
          C_edge_features_to_final,
      )

  def _run_attention_layer(self, n_layer, source_node_features, source_nodes_time_embedding,
                           neighbor_embeddings, edge_time_embeddings, edge_features,
                           mask, explain_weights=None, with_contributions=False):
      attention_model = self.attention_models[n_layer - 1]
      attention_model_test = self.custom_attention_models[n_layer - 1]
      query_perm, key_perm = self._prepare_attention_io(
          source_node_features,
          source_nodes_time_embedding,
          neighbor_embeddings,
          edge_time_embeddings,
          edge_features,
      )
      mask_processed = self._prepare_attention_mask(mask)

      if explain_weights is not None:
          attn_output, attn_output_weights = attention_model_test.forward_weights(
              query=query_perm,
              key=key_perm,
              value=key_perm,
              key_padding_mask=mask_processed,
              explain_weights=explain_weights,
          )
          attn_output = attn_output.squeeze()
          final_output = attention_model.merger.forward(attn_output, source_node_features)
          return final_output, attn_output_weights.squeeze()

      if not with_contributions:
          attn_output, _ = attention_model_test.forward(
              query=query_perm,
              key=key_perm,
              value=key_perm,
              key_padding_mask=mask_processed,
          )
          attn_output = attn_output.squeeze()
          return attention_model.merger.forward(attn_output, source_node_features)

      attn_output, _, C_query, C_key, C_value = attention_model_test.forward_withcontribution(
          query=query_perm,
          key=key_perm,
          value=key_perm,
          key_padding_mask=mask_processed,
      )
      attn_output = attn_output.squeeze()
      final_output, C_attn_to_final, C_src_to_final = attention_model.merger.forward_with_contributions(
          attn_output,
          source_node_features,
      )
      return (final_output,) + self._split_attention_contributions(
          C_query,
          C_key,
          C_value,
          C_attn_to_final,
          C_src_to_final,
          source_node_features.shape[1],
          neighbor_embeddings.shape[2],
          edge_features.shape[2],
      )

  def _build_unrolled_context(self, source_nodes, timestamps, n_layers, n_neighbors):
      effective_n_neighbors = n_neighbors if n_neighbors > 0 else 1
      root_batch_size = len(source_nodes)

      nodes_per_depth = [np.asarray(source_nodes)]
      timestamps_per_depth = [np.asarray(timestamps)]
      neighbors_per_depth = []
      edge_idxs_per_depth = []
      masks_per_depth = []
      edge_deltas_per_depth = []
      root_batch_indices_per_depth = [np.arange(root_batch_size)]
      top_neighbor_slots_per_depth = [np.full(root_batch_size, -1, dtype=np.int64)]

      current_nodes = np.asarray(source_nodes)
      current_timestamps = np.asarray(timestamps)
      current_root_batch_indices = np.arange(root_batch_size)
      current_top_neighbor_slots = np.full(root_batch_size, -1, dtype=np.int64)

      for layer_idx in range(n_layers):
          neighbors, edge_idxs, edge_times = self.neighbor_finder.get_temporal_neighbor(
              current_nodes,
              current_timestamps,
              n_neighbors=n_neighbors,
          )
          neighbors_per_depth.append(neighbors)
          edge_idxs_per_depth.append(edge_idxs)
          masks_per_depth.append(neighbors == 0)
          edge_deltas_per_depth.append(current_timestamps[:, np.newaxis] - edge_times)

          current_nodes = neighbors.flatten()
          current_timestamps = np.repeat(current_timestamps, effective_n_neighbors)
          current_root_batch_indices = np.repeat(current_root_batch_indices, effective_n_neighbors)
          if layer_idx == 0:
              current_top_neighbor_slots = np.tile(
                  np.arange(effective_n_neighbors, dtype=np.int64),
                  len(neighbors),
              )
          else:
              current_top_neighbor_slots = np.repeat(current_top_neighbor_slots, effective_n_neighbors)

          nodes_per_depth.append(current_nodes)
          timestamps_per_depth.append(current_timestamps)
          root_batch_indices_per_depth.append(current_root_batch_indices)
          top_neighbor_slots_per_depth.append(current_top_neighbor_slots)

      return {
          'effective_n_neighbors': effective_n_neighbors,
          'nodes_per_depth': nodes_per_depth,
          'timestamps_per_depth': timestamps_per_depth,
          'neighbors_per_depth': neighbors_per_depth,
          'edge_idxs_per_depth': edge_idxs_per_depth,
          'masks_per_depth': masks_per_depth,
          'edge_deltas_per_depth': edge_deltas_per_depth,
          'root_batch_indices_per_depth': root_batch_indices_per_depth,
          'top_neighbor_slots_per_depth': top_neighbor_slots_per_depth,
          'root_batch_size': root_batch_size,
      }

  def _initialize_unrolled_embeddings(self, memory, context):
      embeddings = []
      raw_node_features = []
      source_time_embeddings = []
      for depth_nodes, depth_timestamps in zip(
          context['nodes_per_depth'],
          context['timestamps_per_depth'],
      ):
          depth_nodes_torch = torch.from_numpy(depth_nodes).long().to(self.device)
          depth_timestamps_torch = torch.unsqueeze(
              torch.from_numpy(depth_timestamps).float().to(self.device),
              dim=1,
          )
          raw_features = self.node_features[depth_nodes_torch, :]
          base_embedding = raw_features.clone()
          if self.use_memory:
              base_embedding = base_embedding + memory[depth_nodes, :]
          embeddings.append(base_embedding)
          raw_node_features.append(raw_features)
          source_time_embeddings.append(self.time_encoder(torch.zeros_like(depth_timestamps_torch)))
      return embeddings, raw_node_features, source_time_embeddings

  def _map_leaf_contributions_to_temporal_edges_attention(self, C_source_time, C_edge_time_embeddings,
                                                          C_edge_features, edge_idxs):
      B, K, _, _ = C_edge_time_embeddings.shape
      edge_contributions_dict = {}
      for b in range(B):
          edge_contributions_dict[b] = {}
          add_from_source = C_source_time[b].sum(dim=0) / K
          for k in range(K):
              edge_idx = edge_idxs[b, k]
              edge_total_contrib = (
                  C_edge_time_embeddings[b, k].sum(dim=0) +
                  C_edge_features[b, k].sum(dim=0) +
                  add_from_source
              )
              if edge_idx in edge_contributions_dict[b]:
                  edge_contributions_dict[b][edge_idx] += edge_total_contrib
              else:
                  edge_contributions_dict[b][edge_idx] = edge_total_contrib
      return edge_contributions_dict

  def _accumulate_temporal_edge_contributions(self, temporal_edge_contributions, layer_temporal,
                                              root_batch_indices):
      for batch_idx, contribs in layer_temporal.items():
          root_batch_idx = int(root_batch_indices[batch_idx])
          if root_batch_idx not in temporal_edge_contributions:
              temporal_edge_contributions[root_batch_idx] = {}
          for edge_idx, contrib in contribs.items():
              if edge_idx in temporal_edge_contributions[root_batch_idx]:
                  temporal_edge_contributions[root_batch_idx][edge_idx] += contrib
              else:
                  temporal_edge_contributions[root_batch_idx][edge_idx] = contrib.clone()

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
      return self._run_attention_layer(
          n_layer,
          source_node_features,
          source_nodes_time_embedding,
          neighbor_embeddings,
          edge_time_embeddings,
          edge_features,
          mask,
          explain_weights=explain_weights,
      )

  def aggregate(self, n_layer, source_node_features, source_nodes_time_embedding,
                neighbor_embeddings,
                edge_time_embeddings, edge_features, mask):
      return self._run_attention_layer(
          n_layer,
          source_node_features,
          source_nodes_time_embedding,
          neighbor_embeddings,
          edge_time_embeddings,
          edge_features,
          mask,
          with_contributions=True,
      )

  def aggregate_without_contribution(self, n_layer, source_node_features, source_nodes_time_embedding,
                                     neighbor_embeddings,
                                     edge_time_embeddings, edge_features, mask):
      return self._run_attention_layer(
          n_layer,
          source_node_features,
          source_nodes_time_embedding,
          neighbor_embeddings,
          edge_time_embeddings,
          edge_features,
          mask,
          with_contributions=False,
      )

  def compute_embedding_attention(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20):
      assert n_layers >= 0
      context = self._build_unrolled_context(source_nodes, timestamps, n_layers, n_neighbors)
      embeddings, raw_node_features, source_time_embeddings = self._initialize_unrolled_embeddings(
          memory,
          context,
      )

      layer_caches = [[None for _ in range(n_layers + 1)] for _ in range(n_layers)]
      final_embedding = embeddings[0]
      top_neighbors = np.zeros(
          (context['root_batch_size'], context['effective_n_neighbors']),
          dtype=np.int64,
      )
      top_edge_idxs = np.zeros_like(top_neighbors)

      for remaining_layers in range(1, n_layers + 1):
          for depth in range(0, n_layers - remaining_layers + 1):
              batch_size = len(context['nodes_per_depth'][depth])
              edge_idxs = context['edge_idxs_per_depth'][depth]
              edge_idxs_torch = torch.from_numpy(edge_idxs).long().to(self.device)
              edge_deltas_torch = torch.from_numpy(
                  context['edge_deltas_per_depth'][depth]
              ).float().to(self.device)
              mask = torch.from_numpy(context['masks_per_depth'][depth]).to(self.device)

              aggregated = self.aggregate(
                  remaining_layers,
                  embeddings[depth],
                  source_time_embeddings[depth],
                  embeddings[depth + 1].view(batch_size, context['effective_n_neighbors'], -1),
                  self.time_encoder(edge_deltas_torch),
                  self.edge_features[edge_idxs_torch, :],
                  mask,
              )
              embeddings[depth] = aggregated[0]
              layer_caches[depth][remaining_layers] = {
                  'output_embedding': aggregated[0],
                  'source_contrib': aggregated[1],
                  'source_time_contrib': aggregated[2],
                  'neighbor_contrib': aggregated[3],
                  'edge_time_contrib': aggregated[4],
                  'edge_feature_contrib': aggregated[5],
                  'neighbors': context['neighbors_per_depth'][depth],
                  'edge_idxs': edge_idxs,
                  'timestamps': context['timestamps_per_depth'][depth],
              }

              if depth == 0 and remaining_layers == n_layers:
                  final_embedding = aggregated[0]
                  top_neighbors = context['neighbors_per_depth'][depth]
                  top_edge_idxs = edge_idxs

      output_dim = final_embedding.shape[1]
      output_dtype = final_embedding.dtype
      output_device = final_embedding.device
      final_source_memory_features = torch.zeros(
          context['root_batch_size'],
          raw_node_features[0].shape[1],
          output_dim,
          dtype=output_dtype,
          device=output_device,
      )
      top_neighbor_memory_features = torch.zeros(
          context['root_batch_size'],
          context['effective_n_neighbors'],
          raw_node_features[0].shape[1],
          output_dim,
          dtype=output_dtype,
          device=output_device,
      )
      downstream = [[None for _ in range(n_layers + 1)] for _ in range(n_layers + 1)]
      downstream[0][n_layers] = torch.diag_embed(final_embedding).to(
          dtype=output_dtype,
          device=output_device,
      )
      temporal_edge_contributions = {}

      for remaining_layers in range(n_layers, 0, -1):
          for depth in range(0, n_layers - remaining_layers + 1):
              downstream_current = downstream[depth][remaining_layers]
              if downstream_current is None:
                  continue

              cache = layer_caches[depth][remaining_layers]
              if cache is None:
                  continue

              output_embedding = cache['output_embedding']
              output_den_source = output_embedding.unsqueeze(1)
              output_den_neighbor = output_embedding.unsqueeze(1).unsqueeze(1)

              source_share = torch.where(
                  output_den_source != 0,
                  cache['source_contrib'] / output_den_source,
                  torch.zeros_like(cache['source_contrib']),
              )
              source_time_share = torch.where(
                  output_den_source != 0,
                  cache['source_time_contrib'] / output_den_source,
                  torch.zeros_like(cache['source_time_contrib']),
              )
              neighbor_share = torch.where(
                  output_den_neighbor != 0,
                  cache['neighbor_contrib'] / output_den_neighbor,
                  torch.zeros_like(cache['neighbor_contrib']),
              )
              edge_time_share = torch.where(
                  output_den_neighbor != 0,
                  cache['edge_time_contrib'] / output_den_neighbor,
                  torch.zeros_like(cache['edge_time_contrib']),
              )
              edge_feat_share = torch.where(
                  output_den_neighbor != 0,
                  cache['edge_feature_contrib'] / output_den_neighbor,
                  torch.zeros_like(cache['edge_feature_contrib']),
              )

              source_to_top = torch.einsum('bid,bdo->bio', source_share, downstream_current)
              source_time_to_top = torch.einsum('bid,bdo->bio', source_time_share, downstream_current)
              neighbor_to_top = torch.einsum('bkid,bdo->bkio', neighbor_share, downstream_current)
              edge_time_to_top = torch.einsum('bkid,bdo->bkio', edge_time_share, downstream_current)
              edge_feat_to_top = torch.einsum('bkid,bdo->bkio', edge_feat_share, downstream_current)
              root_batch_indices = context['root_batch_indices_per_depth'][depth]

              if remaining_layers > 1:
                  layer_temporal = self._map_leaf_contributions_to_temporal_edges_attention(
                      source_time_to_top,
                      edge_time_to_top,
                      edge_feat_to_top,
                      cache['edge_idxs'],
                  )
                  self._accumulate_temporal_edge_contributions(
                      temporal_edge_contributions,
                      layer_temporal,
                      root_batch_indices,
                  )

              if remaining_layers == 1:
                  source_raw_features, source_memory_features = self.allocate_source_contributions(
                      source_to_top,
                      raw_node_features[depth],
                      memory,
                      context['nodes_per_depth'][depth],
                  )
                  neighbor_raw_features, neighbor_memory_features = self.allocate_neighbor_contributions(
                      neighbor_to_top,
                      cache['neighbors'].flatten(),
                      memory,
                  )

                  source_memory_features = source_memory_features.to(dtype=output_dtype, device=output_device)
                  neighbor_memory_features = neighbor_memory_features.to(dtype=output_dtype, device=output_device)

                  top_neighbor_slots = context['top_neighbor_slots_per_depth'][depth]
                  for batch_idx, root_batch_idx in enumerate(root_batch_indices):
                      root_batch_idx = int(root_batch_idx)
                      if depth == 0:
                          final_source_memory_features[root_batch_idx] += source_memory_features[batch_idx]
                          for neighbor_idx in range(context['effective_n_neighbors']):
                              top_neighbor_memory_features[root_batch_idx, neighbor_idx] += (
                                  neighbor_memory_features[batch_idx, neighbor_idx]
                              )
                      else:
                          top_slot = int(top_neighbor_slots[batch_idx])
                          if top_slot >= 0:
                              top_neighbor_memory_features[root_batch_idx, top_slot] += (
                                  source_memory_features[batch_idx] +
                                  neighbor_memory_features[batch_idx].sum(dim=0)
                              )

                  layer_temporal = self.map_contributions_to_temporal_edges_attention(
                      source_raw_features,
                      source_time_to_top,
                      neighbor_raw_features,
                      edge_time_to_top,
                      edge_feat_to_top,
                      context['nodes_per_depth'][depth],
                      cache['neighbors'],
                      cache['edge_idxs'],
                      cache['timestamps'],
                  )
                  self._accumulate_temporal_edge_contributions(
                      temporal_edge_contributions,
                      layer_temporal,
                      root_batch_indices,
                  )
                  continue

              prev_source = downstream[depth][remaining_layers - 1]
              if prev_source is None:
                  downstream[depth][remaining_layers - 1] = source_to_top.clone()
              else:
                  downstream[depth][remaining_layers - 1] = prev_source + source_to_top

              flat_neighbor_to_top = neighbor_to_top.reshape(
                  -1,
                  neighbor_to_top.shape[2],
                  neighbor_to_top.shape[3],
              )
              prev_neighbor = downstream[depth + 1][remaining_layers - 1]
              if prev_neighbor is None:
                  downstream[depth + 1][remaining_layers - 1] = flat_neighbor_to_top.clone()
              else:
                  downstream[depth + 1][remaining_layers - 1] = prev_neighbor + flat_neighbor_to_top

      return (
          final_embedding,
          final_source_memory_features,
          top_neighbor_memory_features,
          temporal_edge_contributions,
          top_neighbors,
          top_edge_idxs,
      )

  def compute_embedding_attention_without_contribution(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20):
      assert n_layers >= 0
      context = self._build_unrolled_context(source_nodes, timestamps, n_layers, n_neighbors)
      embeddings, _, source_time_embeddings = self._initialize_unrolled_embeddings(memory, context)

      for remaining_layers in range(1, n_layers + 1):
          for depth in range(0, n_layers - remaining_layers + 1):
              batch_size = len(context['nodes_per_depth'][depth])
              edge_idxs_torch = torch.from_numpy(
                  context['edge_idxs_per_depth'][depth]
              ).long().to(self.device)
              edge_deltas_torch = torch.from_numpy(
                  context['edge_deltas_per_depth'][depth]
              ).float().to(self.device)
              mask = torch.from_numpy(context['masks_per_depth'][depth]).to(self.device)

              embeddings[depth] = self.aggregate_without_contribution(
                  remaining_layers,
                  embeddings[depth],
                  source_time_embeddings[depth],
                  embeddings[depth + 1].view(batch_size, context['effective_n_neighbors'], -1),
                  self.time_encoder(edge_deltas_torch),
                  self.edge_features[edge_idxs_torch, :],
                  mask,
              )

      return embeddings[0]

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
