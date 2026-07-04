import torch
import argparse
import logging
import time
import pickle
from pathlib import Path
import os
import math
import numpy as np
from typing import List, Optional, Callable
from model.tgn import TGN
import model.tgn as tgn_debug_module
from utils.utils import get_neighbor_finder, MLP
import utils.utils as utils_debug_module

from typing import Dict, List, Tuple, Any, Optional
import math
import utils.attribution as attribution_debug_module
import utils.memory_backtracking_trees as backtracking_debug_module
import modules.embedding_module as embedding_refactored_debug_module
import cvxpy as cvx
import json

from tqdm import tqdm

from utils.utils import EarlyStopMonitor, RandEdgeSampler, get_neighbor_finder
from utils.linkdata_processing import get_data, compute_time_statistics,get_data_node_classification
import gc



def select_important_edges(select_number, edges_dict, target_logits):
    if select_number <= 0 or not edges_dict:
        return []

    edge_selected = cvx.Variable(len(edges_dict), integer=True)

    sort_key_list = list(edges_dict.keys())

    # print('sort_key_list',sort_key_list)

    tmp_logits = 0

    for i in range(len(sort_key_list)):
        tmp_logits = tmp_logits + edge_selected[i] * edges_dict[sort_key_list[i]][0]

    target_prob = target_logits.sigmoid().item()

    objective = cvx.Minimize(target_prob * target_logits.item() - target_prob * tmp_logits + \
                             cvx.logistic(tmp_logits) - cvx.logistic(target_logits.item()))

    constraints = [sum(edge_selected) == select_number]

    for i in range(len(sort_key_list)):
        constraints.append(0 <= edge_selected[i])
        constraints.append(edge_selected[i] <= 1)
    prob = cvx.Problem(objective, constraints)
    # prob.solve(solver='MOSEK')

    try:
        print("Optimal value", prob.solve(solver='MOSEK')) #SCS
    except Exception as exc:
        print(f"警告: MOSEK 求解失败: {exc}")
        print("提示: 可以使用 'SCS' 进行求解，并且 integer 可以设置为 false。")
        raise
    # print("Optimal var")
    # print('x.value',edge_selected.value)

    edge_res = []

    for i in range(len(sort_key_list)):
        edge_res.append(
            edge_selected[i].value)

    if None in edge_res:
        return []
    else:
        sorted_id = sorted(range(len(edge_res)), key=lambda k: edge_res[k], reverse=True)

        select_edges_list = []

        for i in range(select_number):
            # print(edge_res[sorted_id[i]])
            select_edges_list.append(sort_key_list[sorted_id[i]])

        return select_edges_list
def free_intermediate_states():
    # 1) 清空 message_dict（深度删除，确保断开对张量的引用）
    def clear_nested(obj):
        if isinstance(obj, dict):
            for key in list(obj.keys()):
                clear_nested(obj[key])
                del obj[key]
        elif isinstance(obj, list):
            for item in obj:
                clear_nested(item)
            obj.clear()

    try:
        global message_dict
        if isinstance(message_dict, dict):
            clear_nested(message_dict)
        # 保险：直接置空
        message_dict = {}
    except NameError:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sp(z):
    """Sigmoid function"""
    return np.log(1 + np.exp(z))


def kl_divergence(P, Q):
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    eps = 1e-12
    P = np.clip(P, eps, 1.0)
    Q = np.clip(Q, eps, 1.0)
    P = P / P.sum()
    Q = Q / Q.sum()
    return float(np.sum(P * np.log(P / Q)))


def binary_prob_vector(pos_prob):
    pos_prob = float(np.clip(pos_prob, 1e-12, 1.0 - 1e-12))
    return np.asarray([pos_prob, 1.0 - pos_prob], dtype=np.float64)


def binary_logits_vector(pos_prob):
    pos_prob = float(np.clip(pos_prob, 1e-12, 1.0 - 1e-12))
    return np.asarray([np.log(pos_prob / (1.0 - pos_prob)), 0.0], dtype=np.float64)



def convert_numpy_types(obj):
    """
    递归转换numpy类型为Python原生类型，以便JSON序列化
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def allclose_same_dtype(left, right, **kwargs):
    if torch.is_tensor(left) and torch.is_tensor(right):
        right = right.to(dtype=left.dtype, device=left.device)
    return torch.allclose(left, right, **kwargs)


def resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    dtype_map = {
        'float32': torch.float32,
        'float64': torch.float64,
    }
    if dtype_name not in dtype_map:
        raise ValueError(f'Unsupported explain_dtype: {dtype_name}. Choose from {sorted(dtype_map)}')
    return dtype_map[dtype_name]


def to_explain_dtype(value, dtype: torch.dtype, device=None):
    if torch.is_tensor(value):
        return value.to(dtype=dtype, device=device or value.device)
    if isinstance(value, dict):
        return {key: to_explain_dtype(item, dtype, device=device) for key, item in value.items()}
    if isinstance(value, list):
        return [to_explain_dtype(item, dtype, device=device) for item in value]
    if isinstance(value, tuple):
        return tuple(to_explain_dtype(item, dtype, device=device) for item in value)
    return value


def set_tgn_explain_dtype(tgn, dtype: torch.dtype, device):
    tgn = tgn.to(device=device, dtype=dtype)
    for attr in ('node_raw_features', 'edge_raw_features', 'node_raw_embed', 'edge_raw_embed'):
        if hasattr(tgn, attr) and torch.is_tensor(getattr(tgn, attr)):
            setattr(tgn, attr, getattr(tgn, attr).to(device=device, dtype=dtype))

    embedding_module = getattr(tgn, 'embedding_module', None)
    if embedding_module is not None:
        for attr in ('node_features', 'edge_features'):
            if hasattr(embedding_module, attr) and torch.is_tensor(getattr(embedding_module, attr)):
                setattr(embedding_module, attr, getattr(embedding_module, attr).to(device=device, dtype=dtype))

    if getattr(tgn, 'memory', None) is not None:
        if hasattr(tgn.memory, 'memory') and torch.is_tensor(tgn.memory.memory):
            tgn.memory.memory.data = tgn.memory.memory.data.to(device=device, dtype=dtype)
        if hasattr(tgn.memory, 'last_update') and torch.is_tensor(tgn.memory.last_update):
            tgn.memory.last_update.data = tgn.memory.last_update.data.to(device=device, dtype=dtype)
    return tgn


np.random.seed(0)
torch.manual_seed(0)


def load_config(config_path: str) -> Dict[str, Any]:
    if not config_path:
        return {}
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def apply_config_defaults(parser: argparse.ArgumentParser, config: Dict[str, Any]) -> Dict[str, Any]:
    valid_dests = {action.dest for action in parser._actions}
    config_defaults = {key: value for key, value in config.items() if key in valid_dests}
    unknown_keys = sorted(set(config.keys()) - valid_dests)
    if unknown_keys:
        print(f'警告: 配置文件中存在未使用的参数: {unknown_keys}')
    if config_defaults:
        parser.set_defaults(**config_defaults)
    return config_defaults


def build_dataset_spec(dataset_name: str) -> Dict[str, Any]:
    dataset_specs = {
        'uci': {
            'default_prefix': 'uci',
            'default_message_dim': 100,
            'default_memory_dim': 100,
        },
        'wikipedia': {
            'default_prefix': 'wikipedia',
            'default_message_dim': 172,
            'default_memory_dim': 172,
        },
        'reddit': {
            'default_prefix': 'reddit',
            'default_message_dim': 172,
            'default_memory_dim': 172,
        },
        'enron': {
            'default_prefix': 'enron',
            'default_message_dim': 32,
            'default_memory_dim': 32,
        },
    }
    dataset_name = dataset_name.lower()
    if dataset_name not in dataset_specs:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Choose from {sorted(dataset_specs.keys())}")
    return dataset_specs[dataset_name]


def build_link_prediction_run_name(args: argparse.Namespace, spec: Dict[str, Any]) -> str:
    prefix = args.prefix if args.prefix else f"{spec['default_prefix']}_{args.embedding_module}_l{args.n_layer}"
    return f"{prefix}_{args.memory_updater}_{args.aggregator}_{args.message_function}"


def sanitize_filename_part(value: Any) -> str:
    text = str(value).strip()
    text = text.replace(os.sep, '_')
    if os.altsep:
        text = text.replace(os.altsep, '_')
    return ''.join(ch if ch.isalnum() or ch in ('-', '_', '.') else '_' for ch in text)


def build_explanation_result_path(args: argparse.Namespace) -> str:
    filename_parts = [
        args.result_name or args.data,
        args.embedding_module,
        f'l{args.n_layer}',
        args.memory_updater,
        args.edge_selection_mode,
        f'depth{args.max_depth}',
    ]
    if args.memory_update_at_end:
        filename_parts.insert(-2, 'end')
    filename = '_'.join(sanitize_filename_part(part) for part in filename_parts) + '.json'
    return os.path.join('results', filename)


def setup_logging(args: argparse.Namespace, run_name: str, model_path: str, result_path: str) -> logging.Logger:
    Path('log').mkdir(parents=True, exist_ok=True)
    Path('results').mkdir(parents=True, exist_ok=True)

    log_path = Path('log') / f'explain_link_prediction_{args.data}_{time.time()}.log'

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(args)
    logger.info(f'Run name: {run_name}')
    logger.info(f'Model path: {model_path}')
    logger.info(f'Result path: {result_path}')
    return logger


# 参数设置
parser = argparse.ArgumentParser('TGN link prediction explanation')
parser.add_argument('--config', type=str, default='configs/explain_link_prediction_enron.json',
                    help='Path to a JSON config file. Config values override parser defaults.')
parser.add_argument('-d', '--data', type=str, default='enron',
                    choices=['uci', 'UCI', 'wikipedia', 'reddit', 'enron'])
parser.add_argument('--bs', type=int, default=100)
parser.add_argument('--prefix', type=str, default=None)
parser.add_argument('--n_degree', type=int, default=10)
parser.add_argument('--n_head', type=int, default=2)
parser.add_argument('--n_layer', type=int, default=1)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--node_dim', type=int, default=100)
parser.add_argument('--time_dim', type=int, default=100)
parser.add_argument('--drop_out', type=float, default=0.1)
parser.add_argument('--use_memory', action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('--embedding_module', type=str, default="graph_sum")
parser.add_argument('--message_function', type=str, default="identity")
parser.add_argument('--memory_updater', type=str, default="rnn", choices=["gru", "rnn"])
parser.add_argument('--aggregator', type=str, default="last")
parser.add_argument('--message_dim', type=int, default=None)
parser.add_argument('--memory_dim', type=int, default=None)
parser.add_argument('--uniform', action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('--use_destination_embedding_in_message', action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('--use_source_embedding_in_message', action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('--use_validation', action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('--memory_update_at_end', action=argparse.BooleanOptionalAction, default=False,
                    help='Whether to update memory at the end or at the start of the batch')
parser.add_argument('--explain_dtype', type=str, default='float64', choices=['float32', 'float64'],
                    help='Floating point dtype used by the explanation flow.')
parser.add_argument('--total_batch', type=int, default=10,
                    help='Number of initial train batches used to build memory history.')
parser.add_argument('--explain_epoch', type=int, default=0,
                    help='Start batch index for collecting attribution history. Earlier batches only update memory.')
parser.add_argument('--explain_number_edges', type=int, default=1,
                    help='Number of edges explained after the warm-up batches.')
parser.add_argument('--max_depth', type=int, default=100,
                    help='Maximum depth for tracing memory contributions.')
parser.add_argument('--backtrace_child_prune_ratio', type=float, default=1.0,
                    help='Keep top ratio of child records for deeper memory backtracking. 1.0 disables pruning.')
parser.add_argument('--edge_selection_mode', type=str, default='given_number',
                    choices=['ratio', 'given_number'],
                    help='Select edge count by ratio or by explicitly provided counts.')
parser.add_argument('--select_edge_ratio', type=float, nargs='+',
                    default=[0.01, 0.02, 0.03, 0.04, 0.05],
                    help='Ratios used when edge_selection_mode=ratio.')
parser.add_argument('--given_select_numbers', type=int, nargs='+', default=[1, 2, 3, 4, 5],
                    help='Explicit edge counts used when edge_selection_mode=given_number.')
parser.add_argument('--result_name', type=str, default=None,
                    help='Base name for the output result JSON.')
parser.add_argument('--device', type=str, default='cpu',
                    help='cpu, cuda, cuda:N, or auto.')
parser.add_argument('--verbose_debug', action=argparse.BooleanOptionalAction, default=False,
                    help='Print detailed tensor/contribution debug logs.')


config_probe, _ = parser.parse_known_args()
apply_config_defaults(parser, load_config(config_probe.config))
args = parser.parse_args()
args.data = args.data.lower()
dataset_spec = build_dataset_spec(args.data)
args.message_dim = args.message_dim if args.message_dim is not None else dataset_spec['default_message_dim']
args.memory_dim = args.memory_dim if args.memory_dim is not None else dataset_spec['default_memory_dim']
args.result_name = args.result_name if args.result_name else args.data
EXPLAIN_DTYPE = resolve_torch_dtype(args.explain_dtype)

RUN_NAME = build_link_prediction_run_name(args, dataset_spec)
MODEL_PATH = f'saved_models/{RUN_NAME}-tgn-link-prediction.pth'
RESULT_PATH = build_explanation_result_path(args)
for debug_module in (
    tgn_debug_module,
    utils_debug_module,
    attribution_debug_module,
    backtracking_debug_module,
    embedding_refactored_debug_module,
):
    if hasattr(debug_module, 'DEBUG_VERBOSE'):
        debug_module.DEBUG_VERBOSE = args.verbose_debug
logger = setup_logging(args, RUN_NAME, MODEL_PATH, RESULT_PATH)
print(
    f'Effective explain config: config={args.config}, data={args.data}, '
    f'total_batch={args.total_batch}, explain_epoch={args.explain_epoch}, '
    f'explain_number_edges={args.explain_number_edges}, '
    f'max_depth={args.max_depth}, edge_selection_mode={args.edge_selection_mode}, '
    f'aggregator={args.aggregator}, embedding_module={args.embedding_module}, '
    f'n_layer={args.n_layer}, memory_updater={args.memory_updater}',
    flush=True
)

print(f'[stage] Loading link prediction data: data={args.data}', flush=True)
full_data, node_features, edge_features, train_data, val_data, test_data = \
    get_data_node_classification(args.data, use_validation=args.use_validation)
print(
    f'[stage] Data loaded: train_edges={len(train_data.sources)} '
    f'val_edges={len(val_data.sources) if val_data is not None else 0} '
    f'test_edges={len(test_data.sources)} nodes={len(full_data.unique_nodes)}',
    flush=True
)

max_idx = max(full_data.unique_nodes)
train_ngh_finder = get_neighbor_finder(train_data, uniform=args.uniform, max_node_idx=max_idx)

full_ngh_finder = get_neighbor_finder(full_data, uniform=args.uniform, max_node_idx=max_idx)

train_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)
# val_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=0)
#
# test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=2)

if args.device == 'auto':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device(args.device)
print(f'[stage] Using device: {device}', flush=True)

# 时间统计
mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
    compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)

# 模型初始化
tgn = TGN(neighbor_finder=train_ngh_finder, node_features=node_features,
          edge_features=edge_features, device=device,
          n_layers=args.n_layer,
          n_heads=args.n_head, dropout=args.drop_out, use_memory=args.use_memory,
          message_dimension=args.message_dim, memory_dimension=args.memory_dim,
          memory_update_at_start=not args.memory_update_at_end,
          embedding_module_type=args.embedding_module,
          message_function=args.message_function,
          aggregator_type=args.aggregator, n_neighbors=args.n_degree,
          mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
          mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst,
          use_destination_embedding_in_message=args.use_destination_embedding_in_message,
          use_source_embedding_in_message=args.use_source_embedding_in_message, forbidden_memory_update=False,
          memory_updater_type=args.memory_updater)
tgn = tgn.to(device)

# # Decoder 初始化
# decoder = MLP(node_features.shape[1], drop=args.drop_out).to(device)

# 加载模型
logger.info(f"Loading model from {MODEL_PATH}")
checkpoint = torch.load(MODEL_PATH, map_location=device)
print("Available keys:", list(checkpoint.keys()))
tgn.load_state_dict(checkpoint)
tgn = set_tgn_explain_dtype(tgn, EXPLAIN_DTYPE, device)
#

tgn.eval()

# for name, p in decoder.named_parameters():
#     print(f"{name}: shape={tuple(p.shape)}, numel={p.numel()}, requires_grad={p.requires_grad}")

# memory warm-up
tgn.eval()
if args.use_memory:
    tgn.memory.__init_memory__()
    tgn = set_tgn_explain_dtype(tgn, EXPLAIN_DTYPE, device)

message_dict = dict()
memory_dict = dict()
message_trace_dict = dict()

total_batch = args.total_batch
explain_epoch = args.explain_epoch
max_depth = args.max_depth
explain_number_edges = args.explain_number_edges
backtrace_child_prune_ratio = args.backtrace_child_prune_ratio
select_edge_ratio = args.select_edge_ratio
given_select_numbers = args.given_select_numbers

select_flag = False

final_diff_dict = dict()

original_prob_dict = dict()



with torch.no_grad():
    num_instance = len(train_data.sources)
    num_batch = math.ceil(num_instance / args.bs)

    if total_batch >= num_batch:
        raise ValueError(
            f'total_batch={total_batch} must be smaller than num_batch={num_batch} '
            f'so there are target edges left to explain.'
        )
    if explain_epoch < 0 or explain_epoch > total_batch:
        raise ValueError(
            f'explain_epoch must be in [0, total_batch], got '
            f'explain_epoch={explain_epoch}, total_batch={total_batch}.'
        )

    print(f'[stage] Link train batches: num_batch={num_batch}, batch_size={args.bs}', flush=True)

    ######train

    print(
        f'[stage] Warming up memory: total_batch={total_batch}, '
        f'memory_only_batches=[0, {explain_epoch}), '
        f'attribution_with_memory_batches=[{explain_epoch}, {total_batch})',
        flush=True
    )
    warmup_log_every = max(1, total_batch // 10)
    attribution_log_every = 2
    for k in range(total_batch):  # num_batch
        mode = 'memory_only' if k < explain_epoch else 'with_attribution'
        if mode == 'memory_only':
            should_log = total_batch <= 10 or k == 0 or k == total_batch - 1 or (k + 1) % warmup_log_every == 0
        else:
            attribution_offset = k - explain_epoch
            should_log = (
                attribution_offset == 0
                or k == total_batch - 1
                or (attribution_offset + 1) % attribution_log_every == 0
            )
        if should_log:
            print(f'[stage] Memory warm-up progress: batch={k + 1}/{total_batch} mode={mode}', flush=True)

        s_idx = k * args.bs
        e_idx = min(num_instance, s_idx + args.bs)

        sources_batch = train_data.sources[s_idx:e_idx]
        destinations_batch = train_data.destinations[s_idx:e_idx]
        timestamps_batch = train_data.timestamps[s_idx:e_idx]
        edge_idxs_batch = train_data.edge_idxs[s_idx:e_idx]

        size = len(sources_batch)
        _, negatives_batch = train_rand_sampler.sample(size)

        if k < explain_epoch:
            tgn.compute_temporal_embeddings_without_contributions(
                sources_batch,
                destinations_batch,
                negatives_batch,
                timestamps_batch,
                edge_idxs_batch,
                n_neighbors=args.n_degree
            )
            continue

        # 只 forward，不使用 decoder，也不更新参数；从 explain_epoch 开始记录解释贡献
        source_node_embedding, destination_node_embedding, negative_node_embedding, C_memory_features, \
            C_neighbor_memory_features, temporal_edge_contributions, \
            sample_neighbors, sample_neighbor_edgeidx = tgn.compute_temporal_embeddings(
            sources_batch,
            destinations_batch,
            negatives_batch,  # for memory update
            timestamps_batch,
            edge_idxs_batch, message_dict, memory_dict, message_trace_dict,
            n_neighbors=args.n_degree
        )
        source_node_embedding = to_explain_dtype(source_node_embedding, EXPLAIN_DTYPE)
        destination_node_embedding = to_explain_dtype(destination_node_embedding, EXPLAIN_DTYPE)
        negative_node_embedding = to_explain_dtype(negative_node_embedding, EXPLAIN_DTYPE)
        C_memory_features = to_explain_dtype(C_memory_features, EXPLAIN_DTYPE)
        C_neighbor_memory_features = to_explain_dtype(C_neighbor_memory_features, EXPLAIN_DTYPE)
        temporal_edge_contributions = to_explain_dtype(temporal_edge_contributions, EXPLAIN_DTYPE)

        # sum_dict = torch.stack(list(temporal_edge_contributions.values()), dim=0).sum(dim=0)  # [D_out]
        #
        # # print('sum_dict.shape',sum_dict.shape)
        #
        # total_contrib=C_raw_features.sum(dim=(0,1))+C_memory_features.sum(dim=(0,1))+C_source_time.sum(dim=(0,1))+C_neighbor_memory_features.sum(dim=(0,1,2))\
        # +sum_dict
        #
        # verify_ground_truth=source_node_embedding.sum(dim=(0))+destination_node_embedding.sum(dim=(0))+negative_node_embedding.sum(dim=(0))
        #
        # print('final verify flag', allclose_same_dtype(total_contrib, verify_ground_truth, atol=1e-4))

    #####test
    test_k = total_batch

    s_idx = test_k * args.bs
    e_idx = min(num_instance, s_idx + explain_number_edges)
    print(
        f'[stage] Computing target link explanations: start_batch={test_k} '
        f'edge_range=[{s_idx}, {e_idx})',
        flush=True
    )

    sources_batch = train_data.sources[s_idx:e_idx]
    destinations_batch = train_data.destinations[s_idx:e_idx]
    timestamps_batch = train_data.timestamps[s_idx:e_idx]
    edge_idxs_batch = train_data.edge_idxs[s_idx:e_idx]

    size = len(sources_batch)
    _, negatives_batch = train_rand_sampler.sample(size)

    # 只 forward，不使用 decoder，也不更新参数
    source_node_embedding, destination_node_embedding, negative_node_embedding, C_memory_features, \
        C_neighbor_memory_features, temporal_edge_contributions, \
        sample_neighbors, sample_neighbor_edgeidx = tgn.compute_temporal_embeddings(
        sources_batch,
        destinations_batch,
        negatives_batch,  # for memory update
        timestamps_batch,
        edge_idxs_batch, message_dict, memory_dict, message_trace_dict,
        n_neighbors=args.n_degree
    )
    source_node_embedding = to_explain_dtype(source_node_embedding, EXPLAIN_DTYPE)
    destination_node_embedding = to_explain_dtype(destination_node_embedding, EXPLAIN_DTYPE)
    negative_node_embedding = to_explain_dtype(negative_node_embedding, EXPLAIN_DTYPE)
    C_memory_features = to_explain_dtype(C_memory_features, EXPLAIN_DTYPE)
    C_neighbor_memory_features = to_explain_dtype(C_neighbor_memory_features, EXPLAIN_DTYPE)
    temporal_edge_contributions = to_explain_dtype(temporal_edge_contributions, EXPLAIN_DTYPE)
    # save_message_dict(message_dict)

    # print('temporal_edge_contributions',temporal_edge_contributions)

    final_is_equal = True

    #

    sum_dict = None
    for idx, mat in temporal_edge_contributions.items():
        for _, second_mat in mat.items():
            if sum_dict is None:
                sum_dict = second_mat.clone()
            else:
                sum_dict = sum_dict + second_mat
    if sum_dict is None:
        sum_dict = torch.zeros_like(C_memory_features.sum(dim=(0, 1)))

    # print('sum_dict.shape',sum_dict.shape)

    total_contrib = C_memory_features.sum(dim=(0, 1)) + C_neighbor_memory_features.sum(dim=(0, 1, 2)) \
                    + sum_dict

    verify_ground_truth = source_node_embedding.sum(dim=(0)) + destination_node_embedding.sum(
        dim=(0)) + negative_node_embedding.sum(dim=(0))
    verify_ground_truth = verify_ground_truth.to(dtype=total_contrib.dtype, device=total_contrib.device)

    if args.verbose_debug:
        print('final verify flag', allclose_same_dtype(total_contrib, verify_ground_truth, atol=1e-4))
        print('C_memory_features.shape', C_memory_features.shape)
        print('C_neighbor_memory_features', C_neighbor_memory_features.shape)


    score = tgn.affinity_score(torch.cat([source_node_embedding, source_node_embedding], dim=0),
                               torch.cat([destination_node_embedding,
                                          negative_node_embedding])).squeeze(dim=0)
    n_samples = source_node_embedding.shape[0]

    pos_score = score[:n_samples]
    pos_prob = pos_score.sigmoid()

    print(f'[stage] Target links in explanation batch: n_samples={n_samples}', flush=True)

    total_select_edge_dict = dict()

    for target_idx in range(n_samples):
        edge_id = edge_idxs_batch[target_idx].item()
        original_pos_prob = pos_prob[target_idx].item()

        total_select_edge_dict[edge_id] = dict()
        total_select_edge_dict[edge_id]['original_edge_id'] = edge_id
        total_select_edge_dict[edge_id]['original_prob'] = binary_prob_vector(original_pos_prob)
        total_select_edge_dict[edge_id]['original_logits'] = binary_logits_vector(original_pos_prob)
        total_select_edge_dict[edge_id]['original_pos_prob'] = original_pos_prob
        total_select_edge_dict[edge_id]['label'] = 1
        total_select_edge_dict[edge_id]['original_source_node'] = sources_batch[target_idx].item()
        total_select_edge_dict[edge_id]['original_destination_node'] = destinations_batch[target_idx].item()
        total_select_edge_dict[edge_id]['original_timestamp'] = timestamps_batch[target_idx].item()

        # print('original_source_node',sources_batch[target_idx].item())
        # print('original_destination_node',destinations_batch[target_idx].item())
        # print('timestamp',timestamps_batch[target_idx].item())
        #

    for target_idx in range(n_samples):

        print(
            f'[stage] Attribution target: edge_id={edge_idxs_batch[target_idx].item()} '
            f'source={sources_batch[target_idx].item()} '
            f'destination={destinations_batch[target_idx].item()} '
            f'prob={pos_prob[target_idx].item():.6f}',
            flush=True
        )
        print(
            f'[stage] Computing memory attribution: max_depth={max_depth} '
            f'child_prune_ratio={backtrace_child_prune_ratio}',
            flush=True
        )
        # print('timestamp', timestamps_batch[target_idx].item())

        target_source_node = sources_batch[target_idx].item()
        target_desitination_node = destinations_batch[target_idx].item()

        target_desitination_idx = target_idx + len(sources_batch)

        edge_source_node_reults = attribution_debug_module.compute_edge_memory_contributions(
            target_source_node,
            target_idx,
            message_dict,
            C_memory_features[target_idx],
            max_depth,
            child_prune_ratio=backtrace_child_prune_ratio,
            verbose=args.verbose_debug,
        )

        edge_destination_node_reults = attribution_debug_module.compute_edge_memory_contributions(
            target_desitination_node,
            target_desitination_idx,
            message_dict,
            C_memory_features[target_desitination_idx],
            max_depth,
            child_prune_ratio=backtrace_child_prune_ratio,
            verbose=args.verbose_debug,
        )

        final_edge_sum = None
        for edge_idx, contrib in edge_source_node_reults.items():
            # print('final edge contrib',contrib.shape)
            if final_edge_sum is None:
                final_edge_sum = contrib.clone()
            else:
                final_edge_sum = final_edge_sum + contrib

        target_C_memory = C_memory_features[target_idx]
        if final_edge_sum == None:
            final_edge_sum = torch.zeros_like(target_C_memory)

        # print('final_edge_sum',final_edge_sum)
        # print('target_C_memory.sum(dim=0)',target_C_memory.sum(dim=0))

        is_equal = allclose_same_dtype(final_edge_sum, target_C_memory.sum(dim=0), atol=1e-4)
        if args.verbose_debug:
            print('source edge is_equal', is_equal)

        if not is_equal:
            diff = torch.abs(final_edge_sum - target_C_memory)
            max_diff = torch.max(diff)
            mean_diff = torch.mean(diff)
            print(f'source edge 最大差异值: {max_diff:.6f}')
            print(f'source edge 平均差异值: {mean_diff:.6f}')
            print(f'source edge 差异值之和: {torch.sum(diff):.6f}')
            # print(f'final_edge_sum: {final_edge_sum}')
            # print(f'target_C_memory: {target_C_memory}')

        final_edge_sum = None
        for edge_idx, contrib in edge_destination_node_reults.items():
            # print('final edge contrib',contrib.shape)
            if final_edge_sum is None:
                final_edge_sum = contrib.clone()
            else:
                final_edge_sum = final_edge_sum + contrib

        target_C_memory = C_memory_features[target_desitination_idx]
        if final_edge_sum == None:
            final_edge_sum = torch.zeros_like(target_C_memory)

        is_equal = allclose_same_dtype(final_edge_sum, target_C_memory.sum(dim=0), atol=1e-4)
        if args.verbose_debug:
            print('desitination edge is_equal', is_equal)

        neighbor_source_node_reults = attribution_debug_module.compute_neighbor_memory_contributions(
            target_source_node, target_idx, message_dict, C_neighbor_memory_features,
            sample_neighbors, sample_neighbor_edgeidx, max_depth,
            child_prune_ratio=backtrace_child_prune_ratio,
            verbose=args.verbose_debug,
        )

        neighbor_destination_node_reults = attribution_debug_module.compute_neighbor_memory_contributions(
            target_desitination_node, target_desitination_idx, message_dict, C_neighbor_memory_features,
            sample_neighbors, sample_neighbor_edgeidx, max_depth,
            child_prune_ratio=backtrace_child_prune_ratio,
            verbose=args.verbose_debug,
        )

        # print('neighbor_source_node_reults',neighbor_source_node_reults)

        final_edge_sum = None
        for edge_idx, contrib in neighbor_source_node_reults.items():
            # print('final edge contrib',contrib.shape)
            if final_edge_sum is None:
                final_edge_sum = contrib.clone()
            else:
                final_edge_sum = final_edge_sum + contrib

        target_C_memory = C_neighbor_memory_features[target_idx]
        if final_edge_sum == None:
            final_edge_sum = torch.zeros_like(target_C_memory)

        is_equal = allclose_same_dtype(final_edge_sum, target_C_memory.sum(dim=(0, 1)), atol=1e-4)
        if args.verbose_debug:
            print('neighbor source edge is_equal', is_equal)

        final_edge_sum = None
        for edge_idx, contrib in neighbor_destination_node_reults.items():
            # print('final edge contrib',contrib.shape)
            if final_edge_sum is None:
                final_edge_sum = contrib.clone()
            else:
                final_edge_sum = final_edge_sum + contrib

        target_C_memory = C_neighbor_memory_features[target_desitination_idx]
        if final_edge_sum == None:
            final_edge_sum = torch.zeros_like(target_C_memory)

        is_equal = allclose_same_dtype(final_edge_sum, target_C_memory.sum(dim=(0, 1)), atol=1e-4)
        if args.verbose_debug:
            print('neighbor destination edge is_equal', is_equal)

        final_source_node_results = attribution_debug_module.merge_contribution_dicts(
            [neighbor_source_node_reults, edge_source_node_reults, temporal_edge_contributions[target_idx]]
        )

        final_desitination_node_results = attribution_debug_module.merge_contribution_dicts(
            [neighbor_destination_node_reults, \
             edge_destination_node_reults, temporal_edge_contributions[target_desitination_idx]]
        )

        final_edge_sum = None
        for edge_idx, contrib in final_source_node_results.items():
            # print('final edge contrib',contrib.shape)
            if final_edge_sum is None:
                final_edge_sum = contrib.clone()
            else:
                final_edge_sum = final_edge_sum + contrib

        target_C_memory = source_node_embedding[target_idx].to(dtype=EXPLAIN_DTYPE)
        if final_edge_sum == None:
            final_edge_sum = torch.zeros_like(target_C_memory)

        is_equal = allclose_same_dtype(final_edge_sum, target_C_memory, atol=1e-4)
        if args.verbose_debug:
            print('final source edge is_equal', is_equal)

        if not is_equal:
            diff = torch.abs(final_edge_sum - target_C_memory)
            max_diff = torch.max(diff)
            mean_diff = torch.mean(diff)
            print(f'最大差异值: {max_diff:.6f}')
            print(f'平均差异值: {mean_diff:.6f}')
            print(f'final_edge_sum: {final_edge_sum}')
            print(f'target_C_memory: {target_C_memory}')

        final_edge_sum = None
        for edge_idx, contrib in final_desitination_node_results.items():
            # print('final edge contrib',contrib.shape)
            if final_edge_sum is None:
                final_edge_sum = contrib.clone()
            else:
                final_edge_sum = final_edge_sum + contrib

        target_C_memory = destination_node_embedding[target_idx].to(dtype=EXPLAIN_DTYPE)
        if final_edge_sum == None:
            final_edge_sum = torch.zeros_like(target_C_memory)

        is_equal = allclose_same_dtype(final_edge_sum, target_C_memory, atol=1e-4)
        if args.verbose_debug:
            print('final desitination edge is_equal', is_equal)

        if not is_equal:
            diff = torch.abs(final_edge_sum - target_C_memory)
            max_diff = torch.max(diff)
            mean_diff = torch.mean(diff)
            print(f'最大差异值: {max_diff:.6f}')
            print(f'平均差异值: {mean_diff:.6f}')

        output, x1_contrib, x2_contrib = tgn.affinity_score.forward_with_contributions(source_node_embedding,
                                                                                       destination_node_embedding)

        for key, value in final_source_node_results.items():
            contributions = x1_contrib[target_idx].to(dtype=EXPLAIN_DTYPE)
            value_converted = value.to(dtype=contributions.dtype, device=contributions.device)
            embedding_double = source_node_embedding[target_idx].to(
                dtype=contributions.dtype,
                device=contributions.device,
            )
            value_share = torch.where(
                embedding_double != 0,
                value_converted / embedding_double,
                torch.zeros_like(value_converted),
            )
            final_value = value_share @ contributions
            # print('value',value.shape)
            final_source_node_results[key] = final_value

        for key, value in final_desitination_node_results.items():
            contributions = x2_contrib[target_idx].to(dtype=EXPLAIN_DTYPE)
            value_converted = value.to(dtype=contributions.dtype, device=contributions.device)
            embedding_double = destination_node_embedding[target_idx].to(
                dtype=contributions.dtype,
                device=contributions.device,
            )
            value_share = torch.where(
                embedding_double != 0,
                value_converted / embedding_double,
                torch.zeros_like(value_converted),
            )
            final_value = value_share @ contributions
            # print('value',value.shape)
            final_desitination_node_results[key] = final_value

        final_edge_sum = None
        for edge_idx, contrib in final_source_node_results.items():
            # print('final edge contrib',contrib.shape)
            if final_edge_sum is None:
                final_edge_sum = contrib.clone()
            else:
                final_edge_sum = final_edge_sum + contrib

        for edge_idx, contrib in final_desitination_node_results.items():
            # print('final edge contrib',contrib.shape)
            if final_edge_sum is None:
                final_edge_sum = contrib.clone()
            else:
                final_edge_sum = final_edge_sum + contrib

        if final_edge_sum == None:
            final_edge_sum = torch.zeros_like(target_C_memory)
        # print('final_edge_sum',final_edge_sum)
        target_C_memory = output[target_idx].to(dtype=EXPLAIN_DTYPE)
        is_equal = allclose_same_dtype(final_edge_sum, target_C_memory, atol=1e-4)
        if args.verbose_debug:
            print('mlp edge is_equal', is_equal)

        final_edge_contributions = attribution_debug_module.merge_contribution_dicts(
            [final_source_node_results, final_desitination_node_results])

        final_edge_sum = None
        for edge_idx, contrib in final_edge_contributions.items():
            # print('final edge contrib',contrib.shape)
            if final_edge_sum is None:
                final_edge_sum = contrib.clone()
            else:
                final_edge_sum = final_edge_sum + contrib

        if final_edge_sum == None:
            final_edge_sum = torch.zeros_like(target_C_memory)
        # print('final_edge_sum',final_edge_sum)
        target_C_memory = output[target_idx].to(dtype=EXPLAIN_DTYPE)
        is_equal = allclose_same_dtype(final_edge_sum, target_C_memory, atol=1e-4)
        if args.verbose_debug:
            print('merge mlp edge is_equal', is_equal)

        if not is_equal:
            diff = torch.abs(final_edge_sum - target_C_memory)
            max_diff = torch.max(diff)
            mean_diff = torch.mean(diff)
            print(f'mlp最大差异值: {max_diff:.6f}')
            print(f'mlp平均差异值: {mean_diff:.6f}')

        if 0 in final_edge_contributions:
            final_edge_contributions.pop(0)

        selection_specs = []
        if len(final_edge_contributions) > 0:
            edge_id = edge_idxs_batch[target_idx].item()
            print(
                f'[stage] Edge attribution complete: edge_id={edge_id} '
                f'contributed_edges={len(final_edge_contributions)}',
                flush=True
            )
            if args.edge_selection_mode == 'ratio':
                for ratio in select_edge_ratio:
                    select_number = min(math.ceil(ratio * len(final_edge_contributions)), len(final_edge_contributions))
                    selection_key = str(ratio)
                    selection_specs.append((selection_key, select_number))
            else:
                if not given_select_numbers:
                    raise ValueError('edge_selection_mode=given_number requires non-empty given_select_numbers')
                for given_select_number in given_select_numbers:
                    select_number = min(int(given_select_number), len(final_edge_contributions))
                    selection_key = f'given_{select_number}'
                    selection_specs.append((selection_key, select_number))

            for selection_key, select_number in selection_specs:
                print(
                    f'[stage] Selecting important edges: edge_id={edge_id} '
                    f'key={selection_key} select_number={select_number}',
                    flush=True
                )
                select_edge_list = select_important_edges(
                    select_number,
                    final_edge_contributions,
                    output[target_idx],
                )
                if len(select_edge_list) > 0:
                    total_select_edge_dict[edge_id][f'{selection_key}_select_edge'] = select_edge_list

    free_intermediate_states()

    for target_idx in total_select_edge_dict:
        print(f'[stage] Evaluating selected edges for target edge_id={target_idx}', flush=True)

        selection_items = [
            (key[:-len('_select_edge')], value)
            for key, value in total_select_edge_dict[target_idx].items()
            if key.endswith('_select_edge')
        ]

        for selection_key, select_edge_list in selection_items:
                print(
                    f'[stage] Evaluating selected-edge subgraph: target={target_idx} '
                    f'key={selection_key} n_selected_edges={len(select_edge_list)}',
                    flush=True
                )

                edge_mask = np.zeros(len(train_data.edge_idxs), dtype=bool)

                for edge_idx in select_edge_list:
                    mask_positions = np.where(train_data.edge_idxs == edge_idx)[0]
                    edge_mask[mask_positions] = True

                target_positions = np.where(train_data.edge_idxs == target_idx)[0]
                edge_mask[target_positions] = True

                if args.use_memory:
                    tgn.memory.__init_memory__()
                    tgn = set_tgn_explain_dtype(tgn, EXPLAIN_DTYPE, device)

                matched_edges = int(np.sum(edge_mask))
                for k in range(total_batch):

                    s_idx = k * args.bs
                    e_idx = min(num_instance, s_idx + args.bs)

                    # 获取当前batch的数据
                    sources_batch = train_data.sources[s_idx:e_idx]
                    destinations_batch = train_data.destinations[s_idx:e_idx]
                    timestamps_batch = train_data.timestamps[s_idx:e_idx]
                    edge_idxs_batch = train_data.edge_idxs[s_idx:e_idx]

                    # 创建当前batch的mask
                    batch_mask = edge_mask[s_idx:e_idx]

                    # 如果当前batch中所有边都被mask了，跳过这个batch
                    if not np.any(batch_mask):
                        continue

                    # 只使用未被mask的边
                    valid_indices = np.where(batch_mask)[0]
                    if len(valid_indices) == 0:
                        continue

                    # 获取有效的边数据
                    valid_sources = sources_batch[valid_indices]
                    valid_destinations = destinations_batch[valid_indices]
                    valid_timestamps = timestamps_batch[valid_indices]
                    valid_edge_idxs = edge_idxs_batch[valid_indices]

                    size = len(valid_sources)
                    _, negatives_batch = train_rand_sampler.sample(size)

                    if args.verbose_debug:
                        print(f"Batch {k}: Using {len(valid_indices)} out of {len(sources_batch)} edges")

                    # print(len(valid_destinations))
                    # print(len(valid_sources))
                    # print(len(len(valid_timestamps)))

                    # print(f"valid_sources length: {len(valid_sources)}")
                    # print(f"valid_destinations length: {len(valid_destinations)}")
                    # print(f"negatives_batch length: {len(negatives_batch)}")
                    # print(f"valid_timestamps length: {len(valid_timestamps)}")
                    # print(f"valid_edge_idxs length: {len(valid_edge_idxs)}")

                    # 使用mask后的数据重新运行compute_temporal_embeddings
                    source_node_embedding, destination_node_embedding, negative_node_embedding = tgn.compute_temporal_embeddings_without_contributions(
                        valid_sources,
                        valid_destinations,
                        negatives_batch,  # for memory update
                        valid_timestamps,
                        valid_edge_idxs,
                        n_neighbors=args.n_degree
                    )

                test_k = total_batch

                s_idx = test_k * args.bs
                e_idx = min(num_instance, s_idx + explain_number_edges)

                sources_batch = train_data.sources[s_idx:e_idx]
                destinations_batch = train_data.destinations[s_idx:e_idx]
                timestamps_batch = train_data.timestamps[s_idx:e_idx]
                edge_idxs_batch = train_data.edge_idxs[s_idx:e_idx]

                # 创建测试batch的mask
                test_batch_mask = edge_mask[s_idx:e_idx]
                target_idx_in_valid = None

                if np.any(test_batch_mask):
                    # 获取有效的测试边
                    valid_test_indices = np.where(test_batch_mask)[0]
                    valid_test_sources = sources_batch[valid_test_indices]
                    valid_test_destinations = destinations_batch[valid_test_indices]
                    valid_test_timestamps = timestamps_batch[valid_test_indices]
                    valid_test_edge_idxs = edge_idxs_batch[valid_test_indices]

                    for test_i, edge_id in enumerate(valid_test_edge_idxs):
                        # print('test_i',test_i)
                        if edge_id.item() == target_idx:
                            target_idx_in_valid = test_i
                            break

                    # print('valid_test_edge_idxs',valid_test_edge_idxs)
                    # print('targrt_idx',target_idx)

                    size = len(valid_test_sources)
                    _, negatives_batch = train_rand_sampler.sample(size)

                    if args.verbose_debug:
                        print(f"Test: Using {len(valid_test_indices)} out of {len(sources_batch)} edges")

                    # print(f"valid_sources length: {len(valid_sources)}")
                    # print(f"valid_destinations length: {len(valid_destinations)}")
                    # print(f"negatives_batch length: {len(negatives_batch)}")
                    # print(f"valid_timestamps length: {len(valid_timestamps)}")
                    # print(f"valid_edge_idxs length: {len(valid_edge_idxs)}")

                    pos_prob, neg_prob = tgn.compute_edge_probabilities(valid_test_sources, valid_test_destinations,
                                                                        negatives_batch,
                                                                        valid_test_timestamps, valid_test_edge_idxs,
                                                                        args.n_degree)
                    # print('target_idx_in_valid',target_idx_in_valid)
                    #
                    # print('final mask prob',pos_prob[target_idx_in_valid])

                    if target_idx_in_valid is not None:
                        selected_edges_pos_prob = pos_prob[target_idx_in_valid].item()
                        selected_edges_prob = binary_prob_vector(selected_edges_pos_prob)
                        selected_edges_logits = binary_logits_vector(selected_edges_pos_prob)
                        prob_kl_original_to_selected = kl_divergence(
                            total_select_edge_dict[target_idx]['original_prob'],
                            selected_edges_prob,
                        )
                        positive_prob_abs_diff = float(
                            abs(selected_edges_pos_prob - total_select_edge_dict[target_idx]['original_pos_prob'])
                        )
                        total_select_edge_dict[target_idx][f'{selection_key}_selected_edges_prob'] = selected_edges_prob
                        total_select_edge_dict[target_idx][f'{selection_key}_selected_edges_logits'] = selected_edges_logits
                        total_select_edge_dict[target_idx][f'{selection_key}_kl_original_to_selected_edges'] = prob_kl_original_to_selected
                        total_select_edge_dict[target_idx][f'{selection_key}_positive_prob_abs_diff'] = positive_prob_abs_diff
                        total_select_edge_dict[target_idx][f'{selection_key}_true_class_prob_abs_diff'] = positive_prob_abs_diff
                        total_select_edge_dict[target_idx][f'{selection_key}_matched_edges'] = matched_edges
                        print(
                            f'[stage] Selected-edge evaluation complete: target={target_idx} '
                            f'key={selection_key} matched_edges={matched_edges} '
                            f'kl={prob_kl_original_to_selected:.6f} '
                            f'positive_abs_diff={positive_prob_abs_diff:.6f}',
                            flush=True
                        )

                if args.use_memory:
                    tgn.memory.__init_memory__()
                    tgn = set_tgn_explain_dtype(tgn, EXPLAIN_DTYPE, device)

    save_result_dict = {
        sample_idx: result
        for sample_idx, (_, result) in enumerate(total_select_edge_dict.items())
    }
    print(save_result_dict)

    with open(RESULT_PATH, 'w', encoding='utf-8') as f:
        json.dump(convert_numpy_types(save_result_dict), f)
    print(f"Results successfully saved to: {RESULT_PATH}")
