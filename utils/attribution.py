import math
import time
import torch
from .memory_backtracking_trees import normalize_aggregated_contributions, aggregate_backtrace_contributions_cached

DEBUG_VERBOSE = False


def compute_contribution_with_dtype_check(mat, C_memory_features):
    """
    检查数据类型并计算贡献
    """
    mat_dtype = mat.dtype
    C_memory_dtype = C_memory_features.dtype

    # print('mat',mat.shape)
    # print('C_memory_features',C_memory_features.shape)

    if mat_dtype != C_memory_dtype:
        mat_converted = mat.to(dtype=C_memory_dtype)
        return (mat_converted @ C_memory_features).sum(dim=0)
    else:
        return (mat @ C_memory_features).sum(dim=0)


def verify_raw_backtrace_conservation(target_node, message_dict, C_memory_features, max_depth,
                                      child_prune_ratio=1.0, atol=1e-4, verbose=False):
    """
    Verify whether raw backtracking contributions conserve memory attribution before normalization.

    Returns a dict with two checks:
      - matrix_is_identity: sum(raw edge matrices) is close to identity.
      - projected_is_conserved: sum((raw_edge_mat @ C_memory_features).sum(0)) is close to
        C_memory_features.sum(0), matching the downstream attribution projection.
    """
    if target_node not in message_dict:
        return {
            'target_node': int(target_node),
            'has_message': False,
            'n_edges': 0,
            'matrix_is_identity': False,
            'projected_is_conserved': False,
            'matrix_max_abs_diff': None,
            'projected_max_abs_diff': None,
            'matrix_sum': None,
            'projected_sum': None,
            'target_projected_sum': None,
        }

    aggregated = aggregate_backtrace_contributions_cached(
        message_dict,
        root_node=target_node,
        max_depth=max_depth,
        child_prune_ratio=child_prune_ratio,
        verbose=verbose,
    )

    if not aggregated:
        return {
            'target_node': int(target_node),
            'has_message': True,
            'n_edges': 0,
            'matrix_is_identity': False,
            'projected_is_conserved': False,
            'matrix_max_abs_diff': None,
            'projected_max_abs_diff': None,
            'matrix_sum': None,
            'projected_sum': None,
            'target_projected_sum': C_memory_features.sum(dim=0),
        }

    total_mat = None
    projected_sum = None
    C_memory = C_memory_features.to(dtype=torch.float64)
    for mat in aggregated.values():
        mat = mat.to(dtype=torch.float64, device=C_memory.device)
        total_mat = mat.clone() if total_mat is None else total_mat + mat
        projected = (mat @ C_memory).sum(dim=0)
        projected_sum = projected.clone() if projected_sum is None else projected_sum + projected

    identity = torch.eye(total_mat.shape[0], dtype=total_mat.dtype, device=total_mat.device)
    target_projected_sum = C_memory.sum(dim=0)
    matrix_diff = total_mat - identity
    projected_diff = projected_sum - target_projected_sum

    return {
        'target_node': int(target_node),
        'has_message': True,
        'n_edges': len(aggregated),
        'matrix_is_identity': torch.allclose(total_mat, identity, atol=atol),
        'projected_is_conserved': torch.allclose(projected_sum, target_projected_sum, atol=atol),
        'matrix_max_abs_diff': torch.max(torch.abs(matrix_diff)).item(),
        'matrix_mean_abs_diff': torch.mean(torch.abs(matrix_diff)).item(),
        'projected_max_abs_diff': torch.max(torch.abs(projected_diff)).item(),
        'projected_mean_abs_diff': torch.mean(torch.abs(projected_diff)).item(),
        'matrix_sum': total_mat,
        'projected_sum': projected_sum,
        'target_projected_sum': target_projected_sum,
    }


def compute_edge_memory_contributions(target_node, target_idx, message_dict, C_memory_features, max_depth,
                                      child_prune_ratio=1.0, verbose=False):
    """
    计算目标节点的直接边贡献
    """
    start_time = time.perf_counter()
    final_edge_results = {}

    if target_node not in message_dict:
        if verbose:
            print(
                f'[edge-contrib] target_node={target_node} target_idx={target_idx}: '
                f'not in message_dict, skip',
                flush=True
            )
        return final_edge_results

    # 获取时间戳列表
    ts_list = [t for t in message_dict[target_node] if isinstance(t, (int, float)) and not math.isnan(t)]
    max_ts = max(ts_list)
    if verbose:
        print(
            f'[edge-contrib] start target_node={target_node} target_idx={target_idx} '
            f'max_depth={max_depth} n_times={len(ts_list)} max_ts={max_ts}',
            flush=True
        )

    t0 = time.perf_counter()
    if verbose:
        print(f'[edge-contrib] target_node={target_node}: cached backtrace aggregation...', flush=True)
    aggregated = aggregate_backtrace_contributions_cached(
        message_dict,
        root_node=target_node,
        max_depth=max_depth,
        child_prune_ratio=child_prune_ratio,
        verbose=verbose
    )
    if verbose:
        print(
            f'[edge-contrib] target_node={target_node}: cached aggregation finished '
            f'n_edges={len(aggregated)} elapsed={time.perf_counter() - t0:.3f}s',
            flush=True
        )

    t0 = time.perf_counter()
    if verbose:
        print(f'[edge-contrib] target_node={target_node}: normalizing contributions...', flush=True)
    aggregated, _ = normalize_aggregated_contributions(aggregated)
    if verbose:
        print(
            f'[edge-contrib] target_node={target_node}: normalized '
            f'elapsed={time.perf_counter() - t0:.3f}s',
            flush=True
        )

    total_mat = None

    # 计算每个边的贡献
    t0 = time.perf_counter()
    if verbose:
        print(f'[edge-contrib] target_node={target_node}: projecting edge contributions...', flush=True)
    for idx, mat in aggregated.items():
        if total_mat == None:
            total_mat = mat.clone()
        else:
            total_mat += mat
        if idx in final_edge_results:
            final_edge_results[idx] += compute_contribution_with_dtype_check(
                mat, C_memory_features
            )
        else:
            final_edge_results[idx] = compute_contribution_with_dtype_check(
                mat, C_memory_features
            )
    if verbose:
        print(
            f'[edge-contrib] done target_node={target_node} '
            f'n_result_edges={len(final_edge_results)} '
            f'projection_elapsed={time.perf_counter() - t0:.3f}s '
            f'total_elapsed={time.perf_counter() - start_time:.3f}s',
            flush=True
        )

    # print('total_mat',total_mat.sum(dim=0))

    return final_edge_results


def compute_neighbor_memory_contributions(target_node, target_idx, message_dict,
                                          C_neighbor_memory_features, sample_neighbors,
                                          sample_neighbor_edgeidx, max_depth, child_prune_ratio=1.0,
                                          verbose=False):
    """
    计算目标节点邻居的边贡献
    """
    final_neighbor_results = {}
    target_neighbor_list = sample_neighbors[target_idx]

    for neighbor_idx in range(len(target_neighbor_list)):
        target_neighbor = target_neighbor_list[neighbor_idx].item()

        if target_neighbor not in message_dict:
            continue

        # 计算邻居的边贡献
        neighbor_contributions = compute_edge_memory_contributions(
            target_neighbor, target_idx, message_dict,
            C_neighbor_memory_features[target_idx][neighbor_idx],
            max_depth,
            child_prune_ratio=child_prune_ratio,
            verbose=verbose
        )

        # 累加到结果中
        for idx, contrib in neighbor_contributions.items():
            # print('contrib',contrib)
            if idx in final_neighbor_results:
                final_neighbor_results[idx] += contrib
            else:
                final_neighbor_results[idx] = contrib

    return final_neighbor_results

def merge_contribution_dicts(dicts):
    merged = {}
    for d in dicts:
        for k, v in d.items():
            if k in merged:
                merged[k] = merged[k] + v  # value 相加
            else:
                merged[k] = v.clone()  # 避免引用同一个数组

    return merged
