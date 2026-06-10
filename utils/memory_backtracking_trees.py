from dataclasses import dataclass
import math
import os
import pickle
import torch
from typing import Dict, Tuple, List, Any, Optional

def _sorted_times(d: Dict[float, Any]) -> List[float]:
    # d: {timestamp: record}
    return sorted(d.keys())

def _find_time_for_expand(times: List[float], threshold: Optional[float], is_root: bool) -> Optional[float]:
    """
    根节点：取 times 中的最大值
    非根：取 times 中 < threshold 的最大值
    """
    if not times:
        return None
    if is_root:
        return times[-1]
    # 二分也可，这里直接倒序扫描足够清晰
    for t in reversed(times):
        if threshold is None or t < threshold:
            return t
    return None

@dataclass
class TreeNode:
    node: int
    t: float
    side: Optional[str] = None          # 'L' / 'R' / None(根)
    left: Optional["TreeNode"] = None   # source child
    right: Optional["TreeNode"] = None  # destination child

def build_backtrace_tree(
    message_dict: Dict[int, Dict[float, Dict[str, Any]]],
    root_node: int,
    start_time: Optional[float] = None,
    max_depth: int = 10,                 # 根为0层，最多展开到 max_depth 层（不含超过的层）
) -> Optional[TreeNode]:
    if root_node not in message_dict:
        return None

    times_root = _sorted_times(message_dict[root_node])
    if not times_root:
        return None

    is_root = start_time is None
    t0 = _find_time_for_expand(times_root, start_time, is_root=True) if is_root \
         else _find_time_for_expand(times_root, start_time, is_root=False)
    if t0 is None:
        return None

    # 根节点，side=None，根层=0
    root = TreeNode(root_node, t0, side=None)

    def expand(node: TreeNode, is_root_flag: bool, depth: int):
        # 超过或等于最大层数则不再扩展（根为0层）
        if depth >= max_depth:
            return

        node_times = _sorted_times(message_dict.get(node.node, {}))
        if not node_times:
            return

        t_use = _find_time_for_expand(node_times, node.t, is_root=is_root_flag)
        if t_use is None:
            return

        rec = message_dict[node.node][t_use]
        s = int(rec['source_node'])
        d = int(rec['destination_node'])

        # 左=source，右=dest，并标记 side
        node.left = TreeNode(s, t_use, side="L")
        node.right = TreeNode(d, t_use, side="R")

        # 继续向下扩展，深度+1
        if s in message_dict:
            expand(node.left, is_root_flag=False, depth=depth + 1)
        if d in message_dict:
            expand(node.right, is_root_flag=False, depth=depth + 1)

    # 从根开始，根深度视为 0
    expand(root, is_root_flag=True, depth=0)
    return root

def compute_layer_contributions_bottom_up(
    message_dict: Dict[int, Dict[float, Dict[str, Any]]],
    root: TreeNode,
) -> List[Tuple[int, torch.Tensor]]:
    """
    按“层次”输出各层贡献矩阵，但每层的矩阵由“自下向上”的乘法链构成：
      result(L,R) = edge(L,t) @ C_parent @ C_grandparent @ ... @ C_root
    其中 C_* 的选择由到达该层的分支轨迹（L/R）决定：
      若此层是父层的左分支 -> 用父层 L 的 source_node_contribution
      若此层是父层的右分支 -> 用父层 L 的 destination_node_contribution
    返回 [(层描述, 矩阵)]，先左支层序、后右支层序。
    """
    if root.left is None or root.right is None:
        raise ValueError("root 缺少左右子，无法形成第一层 (L, R)。")

    n, dtype, device = _infer_shape_dtype_device(message_dict)
    # print('dtype',dtype)
    I = torch.eye(n, dtype=torch.float64, device=device)
    results: List[Tuple[str, torch.Tensor]] = []

    def layer_name(L: TreeNode, R: TreeNode) -> str:
        tL = int(L.t) if float(L.t).is_integer() else L.t
        tR = int(R.t) if float(R.t).is_integer() else R.t
        return f"L─ {L.node}({tL}) | R─ {R.node}({tR})"

    # ancestors 记录从“第一层(根层)”到“当前层的父层”为止的分支轨迹：
    # 列表元素是 (ancestor_L_node_id, ancestor_time, branch_taken) 其中 branch_taken ∈ {'L','R'}
    def visit_layer(L: Optional[TreeNode], R: Optional[TreeNode],
                    ancestors: List[Tuple[int, float, str]]):
        if L is None or R is None:
            return

        # 1) 计算右侧连乘：按“近祖先→远祖先”的顺序右乘
        right_chain = I
        # ancestors 是从根层到当前父层的顺序；最后一个就是“最近的父层”
        for anc_node_id, anc_t, branch in reversed(ancestors):
            anc_entry = _get_entry(message_dict, anc_node_id, anc_t)
            contrib = (anc_entry["source_node_contribution"]
                       if branch == 'L' else
                       anc_entry["destination_node_contribution"])
            contrib = contrib.to(torch.float64)

            right_chain = right_chain @ contrib

            right_chain=right_chain.to(torch.float64)

        # 2) 当前层 edge 在最左，形成：edge(L,t) @ right_chain
        entry = _get_entry(message_dict, L.node, L.t)
        edge = entry["edge"]
        edge_idx=int(entry['edge_idx'])
        # print('type edge ',type(edge))
        # print('type right',type(right_chain))
        edge=edge.to(torch.float64)
        current = edge @ right_chain
        results.append((edge_idx, current))

        # 3) 继续往下一层（两条支路）
        # 左支下一层由 (L.left, L.right) 组成；祖先新增 (当前层 L, 'L')
        if L.left is not None and L.right is not None:
            anc_left = ancestors + [(L.node, L.t, 'L')]
            visit_layer(L.left, L.right, anc_left)

        # 右支下一层由 (R.left, R.right) 组成；祖先新增 (当前层 L, 'R')
        if R.left is not None and R.right is not None:
            anc_right = ancestors + [(L.node, L.t, 'R')]
            visit_layer(R.left, R.right, anc_right)

    # 第一层（根层）没有祖先
    visit_layer(root.left, root.right, ancestors=[])
    return results

def aggregate_layers_by_edge(layers: List[Tuple[int, torch.Tensor]]) -> Dict[int, torch.Tensor]:
    agg: Dict[int, torch.Tensor] = {}
    for edge_idx, mat in layers:
        if edge_idx in agg:
            agg[edge_idx] = agg[edge_idx] + mat
        else:
            agg[edge_idx] = mat.clone()
    return agg

def _infer_shape_dtype_device(message_dict: Dict[int, Dict[float, Dict[str, Any]]]) -> Tuple[int, torch.dtype, torch.device]:
    for tdict in message_dict.values():
        for rec in tdict.values():
            M = rec["edge"]
            return M.shape[-1], M.dtype, M.device
    raise ValueError("message_dict 为空，无法推断矩阵维度/设备。")

def _get_entry(message_dict, nid: int, t: float) -> Dict[str, Any]:
    try:
        return message_dict[nid][t]
    except KeyError:
        raise KeyError(f"message_dict 中缺少键：node={nid}, time={t}")


def normalize_aggregated_contributions(aggregated):
    """
    对聚合后的贡献矩阵进行列归一化，确保每列和为1
    """
    normalized = {}

    # 计算总的贡献矩阵
    total_mat = None
    for edge_idx, mat in aggregated.items():
        mat = mat.to(dtype=torch.float64)
        if total_mat is None:
            total_mat = mat.clone()
        else:
            total_mat += mat

    # 计算每列的和
    col_sums = total_mat.sum(dim=0, keepdim=True)

    # print('col_sums',col_sums,col_sums.shape)

    num_rows = total_mat.shape[0]



    #对每个边的贡献矩阵进行归一化
    for edge_idx, mat in aggregated.items():
        normalized[edge_idx] = torch.where(
            col_sums >1e-12,  # 条件：列和大于阈值
            mat / col_sums,  # 真值：正常归一化
            torch.ones_like(mat) / num_rows/len(aggregated)
        )

    # for edge_idx, mat in aggregated.items():
    #     normalized[edge_idx] = torch.where(
    #         col_sums > 1e-12,  # 条件：列和大于阈值
    #         mat / col_sums,  # 真值：正常归一化
    #         torch.zeros_like(mat)
    #     )

    return normalized, total_mat