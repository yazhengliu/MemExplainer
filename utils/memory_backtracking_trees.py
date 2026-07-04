import math
import time
import torch
from typing import Dict, Tuple, List, Any, Optional

DEBUG_VERBOSE = False

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

def _as_records(entry: Any) -> List[Dict[str, Any]]:
    return entry if isinstance(entry, list) else [entry]

def aggregate_backtrace_contributions_cached(
    message_dict: Dict[int, Dict[float, Dict[str, Any]]],
    root_node: int,
    max_depth: int,
    start_time: Optional[float] = None,
    child_prune_ratio: float = 1.0,
    progress_every: int = 1000,
    verbose: bool = False,
) -> Dict[int, torch.Tensor]:
    """
    Aggregate edge contribution matrices without materializing the full backtrace tree.

    For each node update record:
      contribution(node) =
        edge +
        contribution(source_child) @ source_node_contribution +
        contribution(destination_child) @ destination_node_contribution

    This is equivalent to the explicit tree traversal, but memoizes repeated
    (node, time-threshold, remaining-depth) subproblems.
    """
    if root_node not in message_dict:
        return {}
    if child_prune_ratio <= 0 or child_prune_ratio > 1:
        raise ValueError(f"child_prune_ratio must be in (0, 1], got {child_prune_ratio}")

    times_cache: Dict[int, List[float]] = {}
    memo: Dict[Tuple[int, Optional[float], int, bool], Dict[int, torch.Tensor]] = {}
    stats = {
        'calls': 0,
        'cache_hits': 0,
        'records': 0,
        'pruned_records': 0,
        'start_time': time.perf_counter(),
    }

    def get_times(node_id: int) -> List[float]:
        if node_id not in times_cache:
            times_cache[node_id] = _sorted_times(message_dict.get(node_id, {}))
        return times_cache[node_id]

    def clone_agg(agg: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        return {edge_idx: mat.clone() for edge_idx, mat in agg.items()}

    def add_to_agg(agg: Dict[int, torch.Tensor], edge_idx: int, mat: torch.Tensor):
        if edge_idx in agg:
            agg[edge_idx] = agg[edge_idx] + mat
        else:
            agg[edge_idx] = mat.clone()

    def merge_child(
        agg: Dict[int, torch.Tensor],
        child_agg: Dict[int, torch.Tensor],
        chain: torch.Tensor,
    ):
        chain = chain.to(torch.float64)
        for child_edge_idx, child_mat in child_agg.items():
            add_to_agg(agg, child_edge_idx, child_mat.to(torch.float64) @ chain)

    def record_indices_to_expand(records: List[Dict[str, Any]]) -> set:
        if child_prune_ratio >= 1.0 or len(records) <= 1:
            return set(range(len(records)))
        keep_count = max(1, int(math.ceil(child_prune_ratio * len(records))))
        scored = []
        for rec_idx, rec in enumerate(records):
            source_score = rec['source_node_contribution'].detach().sum()
            destination_score = rec['destination_node_contribution'].detach().sum()
            score = float((source_score + destination_score).cpu())
            scored.append((score, rec_idx))
        scored.sort(reverse=True)
        return {rec_idx for _, rec_idx in scored[:keep_count]}

    def solve(node_id: int, threshold: Optional[float], remaining_depth: int, is_root: bool) -> Dict[int, torch.Tensor]:
        if remaining_depth <= 0 or node_id not in message_dict:
            return {}

        key = (int(node_id), None if threshold is None else float(threshold), int(remaining_depth), bool(is_root))
        if key in memo:
            stats['cache_hits'] += 1
            return clone_agg(memo[key])

        stats['calls'] += 1
        if verbose and stats['calls'] % progress_every == 0:
            print(
                f'[backtrace-cached] calls={stats["calls"]} cache_hits={stats["cache_hits"]} '
                f'records={stats["records"]} pruned_records={stats["pruned_records"]} '
                f'memo_size={len(memo)} node={node_id} '
                f'remaining_depth={remaining_depth} elapsed={time.perf_counter() - stats["start_time"]:.3f}s',
                flush=True
            )

        times = get_times(node_id)
        t_use = _find_time_for_expand(times, threshold, is_root=is_root)
        if t_use is None:
            memo[key] = {}
            return {}

        result: Dict[int, torch.Tensor] = {}
        records = _as_records(message_dict[node_id][t_use])
        expand_record_indices = record_indices_to_expand(records)

        for rec_idx, rec in enumerate(records):
            stats['records'] += 1
            edge_idx = int(rec['edge_idx'])
            edge = rec['edge'].to(torch.float64)
            add_to_agg(result, edge_idx, edge)

            if remaining_depth <= 1:
                continue
            if rec_idx not in expand_record_indices:
                stats['pruned_records'] += 1
                continue

            source_node = int(rec['source_node'])
            destination_node = int(rec['destination_node'])

            source_child = solve(source_node, t_use, remaining_depth - 1, False)
            if source_child:
                merge_child(result, source_child, rec['source_node_contribution'])

            destination_child = solve(destination_node, t_use, remaining_depth - 1, False)
            if destination_child:
                merge_child(result, destination_child, rec['destination_node_contribution'])

        memo[key] = clone_agg(result)
        return result

    root_threshold = start_time
    aggregated = solve(root_node, root_threshold, max_depth, start_time is None)
    if verbose:
        print(
            f'[backtrace-cached] done root_node={root_node} n_edges={len(aggregated)} '
            f'calls={stats["calls"]} cache_hits={stats["cache_hits"]} '
            f'records={stats["records"]} pruned_records={stats["pruned_records"]} '
            f'child_prune_ratio={child_prune_ratio} memo_size={len(memo)} '
            f'elapsed={time.perf_counter() - stats["start_time"]:.3f}s',
            flush=True
        )
    return aggregated

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
