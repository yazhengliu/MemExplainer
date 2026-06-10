import math
from .memory_backtracking_trees import  build_backtrace_tree,compute_layer_contributions_bottom_up,\
    aggregate_layers_by_edge,normalize_aggregated_contributions

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

def compute_edge_memory_contributions(target_node, target_idx, message_dict, C_memory_features, max_depth):
    """
    计算目标节点的直接边贡献
    """
    final_edge_results = {}

    if target_node not in message_dict:
        return final_edge_results

    # 获取时间戳列表
    ts_list = [t for t in message_dict[target_node] if isinstance(t, (int, float)) and not math.isnan(t)]
    max_ts = max(ts_list)

    # 构建回溯树并计算贡献
    tree = build_backtrace_tree(message_dict, root_node=target_node, max_depth=max_depth)
    layers = compute_layer_contributions_bottom_up(message_dict, tree)
    aggregated = aggregate_layers_by_edge(layers)

    aggregated, _ = normalize_aggregated_contributions(aggregated)

    total_mat = None

    # 计算每个边的贡献
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

    # print('total_mat',total_mat.sum(dim=0))

    return final_edge_results


def compute_neighbor_memory_contributions(target_node, target_idx, message_dict,
                                          C_neighbor_memory_features, sample_neighbors,
                                          sample_neighbor_edgeidx, max_depth):
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
            C_neighbor_memory_features[target_idx][neighbor_idx], max_depth
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