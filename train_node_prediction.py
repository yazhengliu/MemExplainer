import argparse
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from sklearn.metrics import mean_squared_error, ndcg_score
from tgb.nodeproppred.dataset_pyg import PyGNodePropPredDataset

from model.tgn import TGN
from utils.linkdata_processing import compute_time_statistics
from utils.utils import EarlyStopMonitor, MLP, get_neighbor_finder


DATA_EVAL_METRIC_DICT = {
    "tgbn-trade": "ndcg",
    "tgbn-genre": "ndcg",
    "tgbn-reddit": "ndcg",
    "tgbn-token": "ndcg",
}


class SimpleData:
    pass


class Evaluator(object):
    """Evaluator for node property prediction."""

    def __init__(self, name: str):
        self.name = name
        self.valid_metric_list = ["mse", "rmse", "ndcg"]

        if self.name not in DATA_EVAL_METRIC_DICT:
            raise NotImplementedError("Dataset not supported")

    def _parse_and_check_input(self, input_dict):
        if "eval_metric" not in input_dict:
            raise RuntimeError("Missing key of eval_metric")

        for eval_metric in input_dict["eval_metric"]:
            if eval_metric not in self.valid_metric_list:
                print("ERROR: The evaluation metric should be in:", self.valid_metric_list)
                raise ValueError("Undefined eval metric %s " % eval_metric)

            if "y_true" not in input_dict:
                raise RuntimeError("Missing key of y_true")
            if "y_pred" not in input_dict:
                raise RuntimeError("Missing key of y_pred")

            y_true, y_pred = input_dict["y_true"], input_dict["y_pred"]

            if torch is not None and isinstance(y_true, torch.Tensor):
                y_true = y_true.detach().cpu().numpy()
            if torch is not None and isinstance(y_pred, torch.Tensor):
                y_pred = y_pred.detach().cpu().numpy()

            if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
                raise RuntimeError(
                    "Arguments to Evaluator need to be either numpy ndarray or torch tensor!"
                )

            if not y_true.shape == y_pred.shape:
                raise RuntimeError("Shape of y_true and y_pred must be the same!")

        self.eval_metric = input_dict["eval_metric"]
        return y_true, y_pred

    def _compute_metrics(self, y_true, y_pred):
        perf_dict = {}
        for eval_metric in self.eval_metric:
            if eval_metric == "mse":
                perf_dict = {
                    "mse": mean_squared_error(y_true, y_pred),
                    "rmse": math.sqrt(mean_squared_error(y_true, y_pred)),
                }
            elif eval_metric == "ndcg":
                perf_dict = {"ndcg": ndcg_score(y_true, y_pred, k=10)}
        return perf_dict

    def eval(self, input_dict, verbose=False):
        y_true, y_pred = self._parse_and_check_input(input_dict)
        perf_dict = self._compute_metrics(y_true, y_pred)

        if verbose:
            print("INFO: Evaluation Results:")
            for eval_metric in input_dict["eval_metric"]:
                print(f"\t>>> {eval_metric}: {perf_dict[eval_metric]:.4f}")
        return perf_dict


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
        "tgbn-reddit": {
            "default_prefix": "tgbn_reddit",
            "default_message_dim": 32,
            "default_memory_dim": 32,
            "default_target_time_idx": 17,
        },
        "tgbn-token": {
            "default_prefix": "tgbn_token",
            "default_message_dim": 32,
            "default_memory_dim": 32,
            "default_target_time_idx": 17,
        },
        "tgbn-trade": {
            "default_prefix": "tgbn_trade",
            "default_message_dim": 32,
            "default_memory_dim": 32,
            "default_target_time_idx": 17,
        },
    }
    dataset_name = dataset_name.lower()
    if dataset_name not in dataset_specs:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Choose from {sorted(dataset_specs.keys())}")
    return dataset_specs[dataset_name]


def build_parser():
    parser = argparse.ArgumentParser("TGN node prediction training")
    parser.add_argument("--config", type=str, default="configs/train_node_prediction_genre.json",
                        help="Path to a JSON config file. Config values override parser defaults.")
    parser.add_argument("-d", "--data", type=str,
                        choices=["tgbn-genre", "tgbn-reddit", "tgbn-token", "tgbn-trade"],
                        default="tgbn-genre", help="TGB node property prediction dataset name")
    parser.add_argument("--dataset_root", type=str, default="datasets", help="Dataset root directory")
    parser.add_argument("--target_time_idx", type=int, default=None,
                        help="Index into sorted node label timestamps.")
    parser.add_argument("--bs", type=int, default=300, help="Batch size")
    parser.add_argument("--prefix", type=str, default=None, help="Prefix to name checkpoints and models")
    parser.add_argument("--n_degree", type=int, default=10, help="Number of neighbors to sample")
    parser.add_argument("--n_head", type=int, default=2, help="Number of heads used in attention layer")
    parser.add_argument("--n_epoch", type=int, default=200, help="Number of epochs")
    parser.add_argument("--n_layer", type=int, default=1, help="Number of network layers")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--patience", type=int, default=20, help="Patience for early stopping")
    parser.add_argument("--n_runs", type=int, default=1, help="Number of runs")
    parser.add_argument("--drop_out", type=float, default=0.1, help="Dropout probability")
    parser.add_argument("--gpu", type=int, default=0, help="Idx for the gpu to use")
    parser.add_argument("--node_dim", type=int, default=32, help="Dimensions of the node embedding")
    parser.add_argument("--time_dim", type=int, default=32, help="Dimensions of the time embedding")
    parser.add_argument("--backprop_every", type=int, default=1,
                        help="Every how many batches to backpropagate")
    parser.add_argument("--use_memory", action=argparse.BooleanOptionalAction, default=True,
                        help="Whether to augment the model with a node memory")
    parser.add_argument("--embedding_module", type=str, default="graph_sum",
                        choices=["graph_attention", "graph_sum", "identity", "time"],
                        help="Type of embedding module")
    parser.add_argument("--message_function", type=str, default="identity",
                        choices=["mlp", "identity"], help="Type of message function")
    parser.add_argument("--memory_updater", type=str, default="rnn",
                        choices=["gru", "rnn"], help="Type of memory updater")
    parser.add_argument("--aggregator", type=str, default="last", help="Type of message aggregator")
    parser.add_argument("--memory_update_at_end", action=argparse.BooleanOptionalAction, default=False,
                        help="Whether to update memory at the end or at the start of the batch")
    parser.add_argument("--message_dim", type=int, default=None, help="Dimensions of the messages")
    parser.add_argument("--memory_dim", type=int, default=None, help="Dimensions of the memory for each node")
    parser.add_argument("--uniform", action=argparse.BooleanOptionalAction, default=False,
                        help="Take uniform sampling from temporal neighbors")
    parser.add_argument("--use_destination_embedding_in_message", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Whether to use destination embedding as part of the message")
    parser.add_argument("--use_source_embedding_in_message", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Whether to use source embedding as part of the message")
    parser.add_argument("--dyrep", action=argparse.BooleanOptionalAction, default=False,
                        help="Whether to run the dyrep model")
    return parser


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


def main():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    parser = build_parser()
    config_args, _ = parser.parse_known_args()
    apply_config_defaults(parser, load_config(config_args.config))
    args = parser.parse_args()

    args.data = args.data.lower()
    dataset_spec = build_dataset_spec(args.data)

    batch_size = args.bs
    num_neighbors = args.n_degree
    num_epoch = args.n_epoch
    num_heads = args.n_head
    drop_out = args.drop_out
    data_name = args.data
    num_layer = args.n_layer
    learning_rate = args.lr
    use_memory = args.use_memory
    message_dim = args.message_dim if args.message_dim is not None else dataset_spec["default_message_dim"]
    memory_dim = args.memory_dim if args.memory_dim is not None else dataset_spec["default_memory_dim"]
    target_time_idx = (
        args.target_time_idx
        if args.target_time_idx is not None
        else dataset_spec["default_target_time_idx"]
    )

    Path("./saved_models/").mkdir(parents=True, exist_ok=True)
    Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)
    Path("log/").mkdir(parents=True, exist_ok=True)
    Path("results/").mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler("log/{}.log".format(str(time.time())))
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(args)

    device_string = "cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu"
    device = torch.device(device_string)

    dataset = PyGNodePropPredDataset(name=data_name, root=args.dataset_root)
    raw_ds = dataset.dataset
    full_data = build_full_data(raw_ds.full_data)
    target_time, target_key_list = resolve_target_time(raw_ds, target_time_idx)
    train_data = make_time_subset(full_data, target_time)

    prefix = args.prefix if args.prefix is not None else f"{dataset_spec['default_prefix']}_{args.embedding_module}_l{args.n_layer}"
    run_name = (
        f"{prefix}_{args.memory_updater}_{args.aggregator}_{args.message_function}"
        f"_tidx{target_time_idx}_t{target_time}"
    )
    model_save_path = f"./saved_models/{run_name}-tgn-node-prediction.pth"
    decoder_save_path = f"./saved_models/{run_name}-decoder-node-prediction.pth"
    get_checkpoint_path = lambda epoch: f"./saved_checkpoints/{run_name}-{epoch}-node-prediction.pth"
    results_path = f"results/{run_name}.pkl"

    logger.info(f"Resolved run name: {run_name}")
    logger.info(f"Target time idx: {target_time_idx}")
    logger.info(f"Target time: {target_time}")
    logger.info(f"Num label timestamps: {len(target_key_list)}")
    logger.info(f"TGN save path: {model_save_path}")
    logger.info(f"Decoder save path: {decoder_save_path}")

    if len(train_data.sources) == 0:
        raise ValueError(f"No training interactions before target_time={target_time}")

    eval_metric = dataset.eval_metric
    max_idx = max(full_data.unique_nodes)
    num_instance = len(train_data.sources)
    num_batch = math.ceil(num_instance / batch_size)

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

    mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
        compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)

    train_ngh_finder = get_neighbor_finder(train_data, uniform=args.uniform, max_node_idx=max_idx)

    target_label_dict = raw_ds.label_dict[target_time]
    target_nodes, prob_matrix = extract_nodes_and_probs(target_label_dict)
    if len(target_nodes) == 0:
        raise ValueError(f"No target nodes for target_time={target_time}")

    target_nodes = np.asarray(target_nodes, dtype=np.int64)
    target_probs = torch.tensor(prob_matrix, dtype=torch.float32, device=device)

    for run_idx in range(args.n_runs):
        run_results_path = results_path if run_idx == 0 else f"results/{run_name}_{run_idx}.pkl"

        tgn = TGN(
            neighbor_finder=train_ngh_finder,
            node_features=node_features,
            edge_features=edge_features,
            device=device,
            n_layers=num_layer,
            n_heads=num_heads,
            dropout=drop_out,
            use_memory=use_memory,
            message_dimension=message_dim,
            memory_dimension=memory_dim,
            memory_update_at_start=not args.memory_update_at_end,
            embedding_module_type=args.embedding_module,
            message_function=args.message_function,
            aggregator_type=args.aggregator,
            memory_updater_type=args.memory_updater,
            n_neighbors=num_neighbors,
            mean_time_shift_src=mean_time_shift_src,
            std_time_shift_src=std_time_shift_src,
            mean_time_shift_dst=mean_time_shift_dst,
            std_time_shift_dst=std_time_shift_dst,
            use_destination_embedding_in_message=args.use_destination_embedding_in_message,
            use_source_embedding_in_message=args.use_source_embedding_in_message,
            dyrep=args.dyrep,
        )
        tgn = tgn.to(device)

        decoder = MLP(node_features.shape[1], prob_matrix.shape[1], drop=drop_out).to(device)
        optimizer = torch.optim.Adam(list(tgn.parameters()) + list(decoder.parameters()), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        evaluator = Evaluator(name=data_name)
        early_stopper = EarlyStopMonitor(max_round=args.patience)

        train_losses = []
        metric_values = []
        epoch_times = []

        logger.info("num of training instances: {}".format(num_instance))
        logger.info("num of batches per epoch: {}".format(num_batch))
        print("num_batch", num_batch)
        print("target_time_idx", target_time_idx)
        print("target_time", target_time)
        print("target_nodes", len(target_nodes))
        print("prob_matrix", prob_matrix.shape)

        for epoch in range(num_epoch):
            start_epoch = time.time()
            if use_memory:
                tgn.memory.__init_memory__()

            tgn.train()
            decoder.train()

            for k in range(num_batch):
                s_idx = k * batch_size
                e_idx = min(num_instance, s_idx + batch_size)

                sources_batch = train_data.sources[s_idx:e_idx]
                destinations_batch = train_data.destinations[s_idx:e_idx]
                timestamps_batch = train_data.timestamps[s_idx:e_idx]
                edge_idxs_batch = train_data.edge_idxs[s_idx:e_idx]

                tgn.compute_temporal_embeddings_without_contributions(
                    sources_batch,
                    destinations_batch,
                    destinations_batch,
                    timestamps_batch,
                    edge_idxs_batch,
                    num_neighbors,
                )

            dummy_eidx = np.zeros_like(target_nodes, dtype=np.int64)
            ts_np = np.full(len(target_nodes), target_time, dtype=np.int64)
            source_embedding, _, _ = tgn.compute_temporal_embeddings_without_contributions(
                target_nodes,
                target_nodes,
                target_nodes,
                ts_np,
                dummy_eidx,
                num_neighbors,
            )

            optimizer.zero_grad()
            output = decoder(source_embedding)
            loss = criterion(output, target_probs)
            loss.backward()
            optimizer.step()

            result_dict = evaluator.eval({
                "y_true": prob_matrix,
                "y_pred": output.detach().cpu().numpy(),
                "eval_metric": [eval_metric],
            })
            metric_value = result_dict[eval_metric]
            train_losses.append(float(loss.detach().cpu()))
            metric_values.append(metric_value)
            epoch_times.append(time.time() - start_epoch)

            print(result_dict)
            print(f"epoch {epoch} | loss {train_losses[-1]:.4f}")
            logger.info(f"epoch: {epoch} loss: {train_losses[-1]} {eval_metric}: {metric_value}")

            torch.save(
                {
                    "tgn": tgn.state_dict(),
                    "decoder": decoder.state_dict(),
                    "args": vars(args),
                    "target_time_idx": target_time_idx,
                    "target_time": target_time,
                },
                get_checkpoint_path(epoch),
            )

            if early_stopper.early_stop_check(metric_value):
                logger.info("No improvement over {} epochs, stop training".format(early_stopper.max_round))
                break

        torch.save(tgn.state_dict(), model_save_path)
        torch.save(decoder.state_dict(), decoder_save_path)
        with open(run_results_path, "wb") as f:
            import pickle
            pickle.dump({
                "metric": eval_metric,
                "metric_values": metric_values,
                "train_losses": train_losses,
                "epoch_times": epoch_times,
                "target_time_idx": target_time_idx,
                "target_time": target_time,
            }, f)

        logger.info("TGN model saved")
        logger.info("Decoder model saved")


if __name__ == "__main__":
    main()
