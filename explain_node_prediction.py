import argparse
import gc
import json
import logging
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

import model.tgn as tgn_debug_module
import modules.embedding_module as embedding_refactored_debug_module
import utils.attribution as attribution_debug_module
import utils.memory_backtracking_trees as backtracking_debug_module
import utils.utils as utils_debug_module
from model.tgn import TGN
from utils.linkdata_processing import compute_time_statistics
from utils.utils import MLP, get_neighbor_finder


class SimpleData:
    pass


def load_config(config_path: str) -> Dict[str, Any]:
    if not config_path:
        return {}
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_config_defaults(parser: argparse.ArgumentParser, config: Dict[str, Any]) -> Dict[str, Any]:
    valid_dests = {action.dest for action in parser._actions}
    config_defaults = {key: value for key, value in config.items() if key in valid_dests}
    unknown_keys = sorted(set(config.keys()) - valid_dests)
    if unknown_keys:
        print(f"Warning: unused config keys: {unknown_keys}")
    if config_defaults:
        parser.set_defaults(**config_defaults)
    return config_defaults


def build_dataset_spec(dataset_name: str) -> Dict[str, Any]:
    dataset_specs = {
        "tgbn-genre": {
            "default_prefix": "tgbn_genre",
            "default_message_dim": 32,
            "default_memory_dim": 32,
            "default_target_time_idx": 17,
        },
    }
    dataset_name = dataset_name.lower()
    if dataset_name not in dataset_specs:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Choose from {sorted(dataset_specs.keys())}")
    return dataset_specs[dataset_name]


def resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    dtype_map = {
        "float32": torch.float32,
        "float64": torch.float64,
    }
    if dtype_name not in dtype_map:
        raise ValueError(f"Unsupported explain_dtype: {dtype_name}. Choose from {sorted(dtype_map)}")
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
    for attr in ("node_raw_features", "edge_raw_features", "node_raw_embed", "edge_raw_embed"):
        if hasattr(tgn, attr) and torch.is_tensor(getattr(tgn, attr)):
            setattr(tgn, attr, getattr(tgn, attr).to(device=device, dtype=dtype))

    embedding_module = getattr(tgn, "embedding_module", None)
    if embedding_module is not None:
        for attr in ("node_features", "edge_features"):
            if hasattr(embedding_module, attr) and torch.is_tensor(getattr(embedding_module, attr)):
                setattr(embedding_module, attr, getattr(embedding_module, attr).to(device=device, dtype=dtype))

    if getattr(tgn, "memory", None) is not None:
        if hasattr(tgn.memory, "memory") and torch.is_tensor(tgn.memory.memory):
            tgn.memory.memory.data = tgn.memory.memory.data.to(device=device, dtype=dtype)
        if hasattr(tgn.memory, "last_update") and torch.is_tensor(tgn.memory.last_update):
            tgn.memory.last_update.data = tgn.memory.last_update.data.to(device=device, dtype=dtype)
    return tgn


def set_decoder_explain_dtype(decoder, dtype: torch.dtype, device):
    return decoder.to(device=device, dtype=dtype)


def allclose_same_dtype(left, right, **kwargs):
    if torch.is_tensor(left) and torch.is_tensor(right):
        right = right.to(dtype=left.dtype, device=left.device)
    return torch.allclose(left, right, **kwargs)


def make_time_subset(full_data, target_time):
    src = np.asarray(full_data.sources, dtype=np.int32)
    dst = np.asarray(full_data.destinations, dtype=np.int32)
    ts = full_data.timestamps
    eidx = np.asarray(full_data.edge_idxs, dtype=np.int32)

    mask = ts < int(target_time)
    src_s = src[mask]
    dst_s = dst[mask]
    ts_s = ts[mask]
    eidx_s = eidx[mask]

    sd = SimpleData()
    sd.sources = src_s
    sd.destinations = dst_s
    sd.timestamps = ts_s
    sd.edge_idxs = eidx_s
    sd.unique_nodes = np.unique(np.concatenate([src_s, dst_s])) if len(src_s) > 0 else np.array([], dtype=np.int64)
    return sd


def build_full_data(raw_full_data):
    full_data = SimpleData()
    full_data.sources = raw_full_data["sources"].astype(int)
    full_data.destinations = raw_full_data["destinations"].astype(int)
    full_data.edge_idxs = raw_full_data["edge_idxs"].astype(int)

    timestamps = raw_full_data["timestamps"].copy()
    edge_idxs = raw_full_data["edge_idxs"]
    full_data.timestamps = timestamps + (edge_idxs - edge_idxs.min()) * 1e-4
    full_data.unique_nodes = np.unique(
        np.concatenate([raw_full_data["sources"], raw_full_data["destinations"]])
    )
    return full_data


def resolve_target_time(raw_ds, target_time_idx):
    target_key_list = sorted(raw_ds.full_data["node_label_dict"].keys())
    if target_time_idx < 0 or target_time_idx >= len(target_key_list):
        raise ValueError(
            f"target_time_idx={target_time_idx} is out of range for "
            f"{len(target_key_list)} label timestamps"
        )
    return target_key_list[target_time_idx], target_key_list


def extract_nodes_and_probs(target_label_dict):
    if not target_label_dict:
        return [], np.array([])

    nodes = sorted(target_label_dict.keys())
    first_probs = target_label_dict[nodes[0]]

    if torch.is_tensor(first_probs):
        first_probs = first_probs.detach().cpu().numpy()
    elif isinstance(first_probs, list):
        first_probs = np.array(first_probs)

    num_classes = len(first_probs)
    prob_matrix = np.zeros((len(nodes), num_classes), dtype=np.float32)

    for i, node_id in enumerate(nodes):
        probs = target_label_dict[node_id]
        if torch.is_tensor(probs):
            probs = probs.detach().cpu().numpy()
        elif isinstance(probs, list):
            probs = np.array(probs)

        probs = probs.astype(np.float32)
        if probs.sum() > 0:
            probs = probs / probs.sum()
        prob_matrix[i] = probs

    return nodes, prob_matrix


def softmax_np(x):
    x = np.asarray(x, dtype=np.float64)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def kl_divergence(p, q):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    eps = 1e-12
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def select_important_edges(select_number, edges_dict, target_logits, solver="MOSEK"):
    if select_number <= 0 or not edges_dict:
        return []

    try:
        import cvxpy as cvx
    except ImportError as exc:
        raise ImportError(
            "cvxpy is required for edge selection. Install cvxpy or skip selected-edge evaluation."
        ) from exc

    edge_selected = cvx.Variable(len(edges_dict), integer=False)
    sort_key_list = list(edges_dict.keys())
    tmp_logits = 0

    for i, edge_key in enumerate(sort_key_list):
        tmp_logits = tmp_logits + edge_selected[i] * edges_dict[edge_key].detach().cpu().numpy()

    target_prob = softmax_np(target_logits.detach().cpu().numpy())
    objective = cvx.Minimize(-target_prob @ tmp_logits + cvx.atoms.log_sum_exp(tmp_logits))
    constraints = [sum(edge_selected) == select_number]
    for i in range(len(sort_key_list)):
        constraints.append(0 <= edge_selected[i])
        constraints.append(edge_selected[i] <= 1)

    problem = cvx.Problem(objective, constraints)
    try:
        print("Optimal value", problem.solve(solver=solver))
    except Exception as exc:
        if solver != "SCS":
            print(f"Warning: {solver} failed: {exc}. Retrying with SCS.")
            problem.solve(solver="SCS")
        else:
            raise

    if edge_selected.value is None:
        return []
    edge_scores = list(edge_selected.value)
    sorted_id = sorted(range(len(edge_scores)), key=lambda k: edge_scores[k], reverse=True)
    return [sort_key_list[sorted_id[i]] for i in range(min(select_number, len(sort_key_list)))]


def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj


def sanitize_filename_part(value: Any) -> str:
    text = str(value).strip()
    text = text.replace(os.sep, "_")
    if os.altsep:
        text = text.replace(os.altsep, "_")
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in text)


def build_run_name(args: argparse.Namespace, spec: Dict[str, Any], target_time):
    prefix = args.prefix if args.prefix else f"{spec['default_prefix']}_{args.embedding_module}_l{args.n_layer}"
    return (
        f"{prefix}_{args.memory_updater}_{args.aggregator}_{args.message_function}"
        f"_tidx{args.target_time_idx}_t{target_time}"
    )


def build_result_path(args: argparse.Namespace) -> str:
    filename_parts = [
        args.result_name or args.data,
        args.embedding_module,
        f"l{args.n_layer}",
        args.memory_updater,
        args.edge_selection_mode,
        f"depth{args.max_depth}",
        f"tidx{args.target_time_idx}",
    ]
    filename = "_".join(sanitize_filename_part(part) for part in filename_parts) + ".json"
    return os.path.join("results", filename)


def setup_logging(args: argparse.Namespace, run_name: str, model_path: str, decoder_path: str, result_path: str):
    Path("log").mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(parents=True, exist_ok=True)

    log_path = Path("log") / f"explain_node_prediction_{args.data}_{time.time()}.log"
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(args)
    logger.info(f"Run name: {run_name}")
    logger.info(f"TGN model path: {model_path}")
    logger.info(f"Decoder model path: {decoder_path}")
    logger.info(f"Result path: {result_path}")
    return logger


def build_parser():
    parser = argparse.ArgumentParser("TGN node prediction explanation for tgbn-genre")
    parser.add_argument("--config", type=str, default="configs/explain_node_prediction_genre.json",
                        help="Path to a JSON config file. Config values override parser defaults.")
    parser.add_argument("-d", "--data", type=str, choices=["tgbn-genre"], default="tgbn-genre")
    parser.add_argument("--dataset_root", type=str, default="datasets")
    parser.add_argument("--target_time_idx", type=int, default=None,
                        help="Index into sorted node label timestamps; all nodes at this time are explained.")
    parser.add_argument("--bs", type=int, default=300)
    parser.add_argument("--prefix", type=str, default=None)
    parser.add_argument("--n_degree", type=int, default=10)
    parser.add_argument("--n_head", type=int, default=2)
    parser.add_argument("--n_layer", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--node_dim", type=int, default=32)
    parser.add_argument("--time_dim", type=int, default=32)
    parser.add_argument("--drop_out", type=float, default=0.1)
    parser.add_argument("--use_memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--embedding_module", type=str, default="graph_sum",
                        choices=["graph_attention", "graph_sum"])
    parser.add_argument("--message_function", type=str, default="identity", choices=["mlp", "identity"])
    parser.add_argument("--memory_updater", type=str, default="rnn", choices=["gru", "rnn"])
    parser.add_argument("--aggregator", type=str, default="last")
    parser.add_argument("--memory_update_at_end", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--message_dim", type=int, default=None)
    parser.add_argument("--memory_dim", type=int, default=None)
    parser.add_argument("--uniform", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_destination_embedding_in_message", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_source_embedding_in_message", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dyrep", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--device", type=str, default="cpu", help="cpu, cuda, cuda:N, or auto.")
    parser.add_argument("--explain_dtype", type=str, default="float64", choices=["float32", "float64"])
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--backtrace_child_prune_ratio", type=float, default=1.0)
    parser.add_argument("--edge_selection_mode", type=str, default="given_number",
                        choices=["ratio", "given_number"])
    parser.add_argument("--select_edge_ratio", type=float, nargs="+",
                        default=[0.02, 0.04, 0.06, 0.08, 0.1])
    parser.add_argument("--given_select_numbers", type=int, nargs="+", default=[5, 6, 7, 8, 9, 10])
    parser.add_argument("--selection_solver", type=str, default="MOSEK")
    parser.add_argument("--topk_classes", type=int, default=10,
                        help="Use top-k predicted class logits for convex edge selection. 0 means all classes.")
    parser.add_argument("--result_name", type=str, default=None)
    parser.add_argument("--verbose_debug", action=argparse.BooleanOptionalAction, default=False)
    return parser


def instantiate_tgn(args, node_features, edge_features, train_ngh_finder, mean_stats, device, message_dim, memory_dim):
    mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = mean_stats
    return TGN(
        neighbor_finder=train_ngh_finder,
        node_features=node_features,
        edge_features=edge_features,
        device=device,
        n_layers=args.n_layer,
        n_heads=args.n_head,
        dropout=args.drop_out,
        use_memory=args.use_memory,
        message_dimension=message_dim,
        memory_dimension=memory_dim,
        memory_update_at_start=not args.memory_update_at_end,
        embedding_module_type=args.embedding_module,
        message_function=args.message_function,
        aggregator_type=args.aggregator,
        memory_updater_type=args.memory_updater,
        n_neighbors=args.n_degree,
        mean_time_shift_src=mean_time_shift_src,
        std_time_shift_src=std_time_shift_src,
        mean_time_shift_dst=mean_time_shift_dst,
        std_time_shift_dst=std_time_shift_dst,
        use_destination_embedding_in_message=args.use_destination_embedding_in_message,
        use_source_embedding_in_message=args.use_source_embedding_in_message,
        dyrep=args.dyrep,
    )


def main():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    parser = build_parser()
    config_probe, _ = parser.parse_known_args()
    apply_config_defaults(parser, load_config(config_probe.config))
    args = parser.parse_args()
    args.data = args.data.lower()

    spec = build_dataset_spec(args.data)
    args.target_time_idx = (
        args.target_time_idx
        if args.target_time_idx is not None
        else spec["default_target_time_idx"]
    )
    args.message_dim = args.message_dim if args.message_dim is not None else spec["default_message_dim"]
    args.memory_dim = args.memory_dim if args.memory_dim is not None else spec["default_memory_dim"]
    args.result_name = args.result_name if args.result_name else args.data.replace("-", "_")
    explain_dtype = resolve_torch_dtype(args.explain_dtype)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    for debug_module in (
        tgn_debug_module,
        utils_debug_module,
        attribution_debug_module,
        backtracking_debug_module,
        embedding_refactored_debug_module,
    ):
        if hasattr(debug_module, "DEBUG_VERBOSE"):
            debug_module.DEBUG_VERBOSE = args.verbose_debug

    print(f"[stage] Loading node prediction data: data={args.data}", flush=True)
    from tgb.nodeproppred.dataset_pyg import PyGNodePropPredDataset

    dataset = PyGNodePropPredDataset(name=args.data, root=args.dataset_root)
    raw_ds = dataset.dataset
    full_data = build_full_data(raw_ds.full_data)
    target_time, target_key_list = resolve_target_time(raw_ds, args.target_time_idx)
    train_data = make_time_subset(full_data, target_time)
    if len(train_data.sources) == 0:
        raise ValueError(f"No training interactions before target_time={target_time}")

    target_label_dict = raw_ds.label_dict[target_time]
    target_nodes, prob_matrix = extract_nodes_and_probs(target_label_dict)
    if len(target_nodes) == 0:
        raise ValueError(f"No target nodes for target_time={target_time}")
    target_nodes = np.asarray(target_nodes, dtype=np.int64)

    run_name = build_run_name(args, spec, target_time)
    model_path = f"saved_models/{run_name}-tgn-node-prediction.pth"
    decoder_path = f"saved_models/{run_name}-decoder-node-prediction.pth"
    result_path = build_result_path(args)
    logger = setup_logging(args, run_name, model_path, decoder_path, result_path)

    print(
        f"[stage] Effective explain config: config={args.config}, data={args.data}, "
        f"target_time_idx={args.target_time_idx}, target_time={target_time}, "
        f"target_nodes={len(target_nodes)}, max_depth={args.max_depth}, "
        f"edge_selection_mode={args.edge_selection_mode}",
        flush=True,
    )

    max_idx = max(full_data.unique_nodes)
    temporal_data = dataset.get_TemporalData()
    num_edges = temporal_data.msg.size(0)
    raw_edge_dim = int(temporal_data.msg.size(-1))
    edge_dim = max(raw_edge_dim, 2)
    num_nodes = int(full_data.unique_nodes.max()) + 1
    node_features = np.zeros((num_nodes, args.node_dim), dtype=np.float32)
    edge_features = np.zeros((num_edges + 1, edge_dim), dtype=np.float32)
    raw_edge_features = temporal_data.msg.detach().cpu().numpy().astype(np.float32)
    edge_features[1:, :raw_edge_dim] = raw_edge_features
    if edge_dim > raw_edge_dim:
        edge_features[1:, raw_edge_dim:] = raw_edge_features[:, -1:]

    mean_stats = compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)
    train_ngh_finder = get_neighbor_finder(train_data, uniform=args.uniform, max_node_idx=max_idx)

    tgn = instantiate_tgn(
        args,
        node_features,
        edge_features,
        train_ngh_finder,
        mean_stats,
        device,
        args.message_dim,
        args.memory_dim,
    ).to(device)
    decoder = MLP(node_features.shape[1], prob_matrix.shape[1], drop=args.drop_out).to(device)

    logger.info(f"Loading TGN model from {model_path}")
    logger.info(f"Loading decoder model from {decoder_path}")
    tgn.load_state_dict(torch.load(model_path, map_location=device))
    decoder.load_state_dict(torch.load(decoder_path, map_location=device))
    tgn = set_tgn_explain_dtype(tgn, explain_dtype, device)
    decoder = set_decoder_explain_dtype(decoder, explain_dtype, device)
    tgn.eval()
    decoder.eval()

    if args.use_memory:
        tgn.memory.__init_memory__()
        tgn = set_tgn_explain_dtype(tgn, explain_dtype, device)

    message_dict = {}
    memory_dict = {}
    message_trace_dict = {}

    num_instance = len(train_data.sources)
    num_batch = math.ceil(num_instance / args.bs)
    print(f"[stage] Replaying full history with attribution: num_batch={num_batch}", flush=True)

    with torch.no_grad():
        for k in range(num_batch):
            if k == 0 or k == num_batch - 1 or (k + 1) % max(1, num_batch // 10) == 0:
                print(f"[stage] History replay progress: batch={k + 1}/{num_batch}", flush=True)

            s_idx = k * args.bs
            e_idx = min(num_instance, s_idx + args.bs)
            (
                source_node_embedding,
                destination_node_embedding,
                negative_node_embedding,
                C_memory_features,
                C_neighbor_memory_features,
                temporal_edge_contributions,
                sample_neighbors,
                sample_neighbor_edgeidx,
            ) = tgn.compute_temporal_embeddings(
                train_data.sources[s_idx:e_idx],
                train_data.destinations[s_idx:e_idx],
                train_data.destinations[s_idx:e_idx],
                train_data.timestamps[s_idx:e_idx],
                train_data.edge_idxs[s_idx:e_idx],
                message_dict,
                memory_dict,
                message_trace_dict,
                n_neighbors=args.n_degree,
            )

        dummy_eidx = np.zeros_like(target_nodes, dtype=np.int64)
        ts_np = np.full(len(target_nodes), target_time, dtype=np.int64)
        print(f"[stage] Computing target node explanations: n_targets={len(target_nodes)}", flush=True)
        (
            source_node_embedding,
            destination_node_embedding,
            negative_node_embedding,
            C_memory_features,
            C_neighbor_memory_features,
            temporal_edge_contributions,
            sample_neighbors,
            sample_neighbor_edgeidx,
        ) = tgn.compute_temporal_embeddings(
            target_nodes,
            target_nodes,
            target_nodes,
            ts_np,
            dummy_eidx,
            message_dict,
            memory_dict,
            message_trace_dict,
            n_neighbors=args.n_degree,
        )

        source_node_embedding = to_explain_dtype(source_node_embedding, explain_dtype, device)
        destination_node_embedding = to_explain_dtype(destination_node_embedding, explain_dtype, device)
        negative_node_embedding = to_explain_dtype(negative_node_embedding, explain_dtype, device)
        C_memory_features = to_explain_dtype(C_memory_features, explain_dtype, device)
        C_neighbor_memory_features = to_explain_dtype(C_neighbor_memory_features, explain_dtype, device)
        temporal_edge_contributions = to_explain_dtype(temporal_edge_contributions, explain_dtype, device)

        if args.verbose_debug:
            sum_dict = None
            for _, mat in temporal_edge_contributions.items():
                for _, second_mat in mat.items():
                    sum_dict = second_mat.clone() if sum_dict is None else sum_dict + second_mat
            if sum_dict is None:
                sum_dict = torch.zeros_like(C_memory_features.sum(dim=(0, 1)))

            total_contrib = (
                C_memory_features.sum(dim=(0, 1))
                + C_neighbor_memory_features.sum(dim=(0, 1, 2))
                + sum_dict
            )
            verify_ground_truth = (
                source_node_embedding.sum(dim=0)
                + destination_node_embedding.sum(dim=0)
                + negative_node_embedding.sum(dim=0)
            )
            verify_ground_truth = verify_ground_truth.to(
                dtype=total_contrib.dtype,
                device=total_contrib.device,
            )
            print("final verify flag", torch.allclose(total_contrib, verify_ground_truth, atol=1e-4))

        output, decoder_contributions = decoder.forward_with_contributions(source_node_embedding, decoder)
        output = to_explain_dtype(output, explain_dtype, device)
        decoder_contributions = to_explain_dtype(decoder_contributions, explain_dtype, device)
        output_np = output.detach().cpu().numpy()
        prob_np = np.asarray([softmax_np(row) for row in output_np])

        total_select_edge_dict = {}
        for target_idx, node_id in enumerate(target_nodes):
            original_prob = prob_np[target_idx]
            original_logits = output_np[target_idx]
            ground_truth = prob_matrix[target_idx]
            true_class = int(np.argmax(ground_truth))
            pred_class = int(np.argmax(original_prob))
            total_select_edge_dict[int(target_idx)] = {
                "original_node_id": int(node_id),
                "original_timestamp": int(target_time),
                "original_prob": original_prob,
                "original_logits": original_logits,
                "ground_truth": ground_truth,
                "true_class": true_class,
                "pred_class": pred_class,
                "original_true_class_prob": float(original_prob[true_class]),
            }

        for target_idx, node_id in enumerate(target_nodes):
            print(
                f"[stage] Attribution target: target_idx={target_idx} node={int(node_id)} "
                f"pred_class={total_select_edge_dict[int(target_idx)]['pred_class']}",
                flush=True,
            )

            edge_source_node_results = attribution_debug_module.compute_edge_memory_contributions(
                int(node_id),
                target_idx,
                message_dict,
                C_memory_features[target_idx],
                args.max_depth,
                child_prune_ratio=args.backtrace_child_prune_ratio,
                verbose=args.verbose_debug,
            )
            neighbor_source_node_results = attribution_debug_module.compute_neighbor_memory_contributions(
                int(node_id),
                target_idx,
                message_dict,
                C_neighbor_memory_features,
                sample_neighbors,
                sample_neighbor_edgeidx,
                args.max_depth,
                child_prune_ratio=args.backtrace_child_prune_ratio,
                verbose=args.verbose_debug,
            )
            final_source_node_results = attribution_debug_module.merge_contribution_dicts(
                [
                    neighbor_source_node_results,
                    edge_source_node_results,
                    temporal_edge_contributions[target_idx],
                ]
            )

            final_edge_sum = None
            for contrib in final_source_node_results.values():
                final_edge_sum = contrib.clone() if final_edge_sum is None else final_edge_sum + contrib
            if final_edge_sum is None:
                final_edge_sum = torch.zeros_like(source_node_embedding[target_idx])
            if args.verbose_debug:
                print(
                    "final source edge is_equal",
                    allclose_same_dtype(final_edge_sum, source_node_embedding[target_idx], atol=1e-4),
                )

            for key, value in list(final_source_node_results.items()):
                contributions = decoder_contributions[target_idx]
                value_converted = value.to(dtype=contributions.dtype, device=contributions.device)
                embedding = source_node_embedding[target_idx].to(
                    dtype=contributions.dtype,
                    device=contributions.device,
                )
                value_share = torch.where(
                    embedding != 0,
                    value_converted / embedding,
                    torch.zeros_like(value_converted),
                )
                final_source_node_results[key] = value_share @ contributions

            if 0 in final_source_node_results:
                final_source_node_results.pop(0)

            selection_source = final_source_node_results
            target_logits = output[target_idx]
            if args.topk_classes and args.topk_classes > 0 and args.topk_classes < target_logits.numel():
                _, topk_indices = torch.topk(target_logits, args.topk_classes)
                target_logits = target_logits[topk_indices]
                selection_source = {
                    edge_idx: vector[topk_indices]
                    for edge_idx, vector in final_source_node_results.items()
                }

            selection_specs = []
            if len(selection_source) > 0:
                if args.edge_selection_mode == "ratio":
                    for ratio in args.select_edge_ratio:
                        select_number = min(math.ceil(ratio * len(selection_source)), len(selection_source))
                        selection_specs.append((str(ratio), select_number))
                else:
                    for given_select_number in args.given_select_numbers:
                        select_number = min(int(given_select_number), len(selection_source))
                        selection_specs.append((f"given_{select_number}", select_number))

            for selection_key, select_number in selection_specs:
                print(
                    f"[stage] Selecting important edges: target_idx={target_idx} "
                    f"key={selection_key} select_number={select_number}",
                    flush=True,
                )
                select_edge_list = select_important_edges(
                    select_number,
                    selection_source,
                    target_logits,
                    solver=args.selection_solver,
                )
                if select_edge_list:
                    total_select_edge_dict[int(target_idx)][f"{selection_key}_select_edge"] = select_edge_list

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("[stage] Evaluating selected-edge subgraphs", flush=True)
        for target_idx, result in total_select_edge_dict.items():
            selection_items = [
                (key[:-len("_select_edge")], value)
                for key, value in result.items()
                if key.endswith("_select_edge")
            ]
            for selection_key, select_edge_list in selection_items:
                edge_mask = np.zeros(len(train_data.edge_idxs), dtype=bool)
                for edge_idx in select_edge_list:
                    edge_mask[np.where(train_data.edge_idxs == edge_idx)[0]] = True

                if args.use_memory:
                    tgn.memory.__init_memory__()
                    tgn = set_tgn_explain_dtype(tgn, explain_dtype, device)

                matched_edges = int(np.sum(edge_mask))
                for k in range(num_batch):
                    s_idx = k * args.bs
                    e_idx = min(num_instance, s_idx + args.bs)
                    batch_mask = edge_mask[s_idx:e_idx]
                    if not np.any(batch_mask):
                        continue
                    valid_indices = np.where(batch_mask)[0]
                    tgn.compute_temporal_embeddings_without_contributions(
                        train_data.sources[s_idx:e_idx][valid_indices],
                        train_data.destinations[s_idx:e_idx][valid_indices],
                        train_data.destinations[s_idx:e_idx][valid_indices],
                        train_data.timestamps[s_idx:e_idx][valid_indices],
                        train_data.edge_idxs[s_idx:e_idx][valid_indices],
                        n_neighbors=args.n_degree,
                    )

                source_embedding_masked, _, _ = tgn.compute_temporal_embeddings_without_contributions(
                    target_nodes,
                    target_nodes,
                    target_nodes,
                    ts_np,
                    dummy_eidx,
                    n_neighbors=args.n_degree,
                )
                source_embedding_masked = to_explain_dtype(source_embedding_masked, explain_dtype, device)
                masked_output = decoder(source_embedding_masked)
                masked_logits = masked_output[int(target_idx)].detach().cpu().numpy()
                masked_prob = softmax_np(masked_logits)
                original_prob = np.asarray(result["original_prob"], dtype=np.float64)
                true_class = int(result["true_class"])
                result[f"{selection_key}_selected_edges_prob"] = masked_prob
                result[f"{selection_key}_selected_edges_logits"] = masked_logits
                result[f"{selection_key}_kl_original_to_selected_edges"] = kl_divergence(original_prob, masked_prob)
                result[f"{selection_key}_true_class_prob_abs_diff"] = float(
                    abs(masked_prob[true_class] - original_prob[true_class])
                )
                result[f"{selection_key}_matched_edges"] = matched_edges
                print(
                    f"[stage] Selected-edge evaluation complete: target_idx={target_idx} "
                    f"key={selection_key} matched_edges={matched_edges}",
                    flush=True,
                )

    save_result_dict = {
        sample_idx: result
        for sample_idx, result in total_select_edge_dict.items()
    }
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(convert_numpy_types(save_result_dict), f)
    print(f"Results successfully saved to: {result_path}")


if __name__ == "__main__":
    main()
