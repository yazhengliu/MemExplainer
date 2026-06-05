import os
import glob
import random
import time
import argparse
import sys
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import torch
import torch.nn as nn

from model.tgn import TGN
from utils.prepare_video_data import TemporalGraphDataLoader
from utils.utils import MLP


def load_all_videos_no_split(root_dir: str, classes_list=None):
    if classes_list is None:
        classes_list = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

    all_data = {}
    for cls in classes_list:
        cls_dir = os.path.join(root_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        for video_name in sorted(os.listdir(cls_dir)):
            vid_dir = os.path.join(cls_dir, video_name)
            if not os.path.isdir(vid_dir):
                continue
            frame_files = sorted(glob.glob(os.path.join(vid_dir, 'frame_*.joblib')))
            if not frame_files:
                continue
            batches = [joblib.load(fp) for fp in frame_files]
            all_data[f'{cls}/{video_name}'] = batches
    return all_data


def split_dataset_by_class(all_data: dict,
                           train_ratio: float = 0.8,
                           test_ratio: float = 0.2,
                           seed: int = 42):
    assert 0 < train_ratio <= 1 and 0 < test_ratio <= 1 and train_ratio + test_ratio <= 1.0
    rng = random.Random(seed)

    class_to_keys = {}
    for key in all_data.keys():
        cls = key.split('/')[0]
        class_to_keys.setdefault(cls, []).append(key)

    train_keys, test_keys = [], []
    stats = {}

    for cls, keys in class_to_keys.items():
        keys_sorted = sorted(keys)
        rng.shuffle(keys_sorted)
        n = len(keys_sorted)
        n_train = int(round(n * train_ratio))
        n_test = int(round(n * test_ratio))
        n_train = min(n_train, n)
        n_test = min(n_test, n - n_train)

        cls_train = keys_sorted[:n_train]
        cls_test = keys_sorted[n_train:n_train + n_test]

        train_keys.extend(cls_train)
        test_keys.extend(cls_test)

        stats[cls] = {'total': n, 'train': len(cls_train), 'test': len(cls_test)}

    train_data = {k: all_data[k] for k in train_keys}
    test_data = {k: all_data[k] for k in test_keys}

    stats['overall'] = {
        'total_videos': len(all_data),
        'train_videos': len(train_data),
        'test_videos': len(test_data),
    }
    return train_data, test_data, stats


def balance_dataset_by_undersampling(train_data: dict, class_to_label: dict, seed: int = 42):
    rng = random.Random(seed)

    class_to_keys = {}
    for key in train_data.keys():
        cls = key.split('/')[0]
        if cls in class_to_label:
            class_to_keys.setdefault(cls, []).append(key)

    min_count = min(len(keys) for keys in class_to_keys.values())
    print(f"\n类别统计（采样前）:")
    for cls, keys in sorted(class_to_keys.items()):
        print(f"  {cls} (label {class_to_label[cls]}): {len(keys)} 个视频")
    print(f"\n最少类别数量: {min_count}")
    print(f"将每个类别采样到 {min_count} 个视频")

    balanced_keys = []
    sampling_stats = {}

    for cls, keys in class_to_keys.items():
        original_count = len(keys)
        if original_count > min_count:
            sampled_keys = rng.sample(keys, min_count)
        else:
            sampled_keys = keys

        balanced_keys.extend(sampled_keys)

        sampling_stats[cls] = {
            'original': original_count,
            'sampled': len(sampled_keys),
            'removed': original_count - len(sampled_keys)
        }
        print(f"  {cls}: 原始 {original_count} 个 -> 采样 {len(sampled_keys)} 个")

    balanced_train_data = {key: train_data[key] for key in balanced_keys}

    print(f"\n平衡后的数据集:")
    print(f"  总视频数: {len(balanced_train_data)}")
    print(f"  每个类别: {min_count} 个视频")
    print(f"  类别数: {len(class_to_keys)}")
    print("=" * 50)

    return balanced_train_data, sampling_stats


def make_windows(batches, T):
    windows = []
    n = len(batches)
    for i in range(0, n, T):
        windows.append(batches[i:i + T])
    return windows


def build_window_super_batch_dynamic_edges(window_frames):
    assert len(window_frames) > 0

    sources = []
    destinations = []
    edge_idxs = []
    timestamps = []
    edge_feats = []

    for fr in window_frames:
        sources.append(np.asarray(fr['sources'], dtype=np.int64))
        destinations.append(np.asarray(fr['destinations'], dtype=np.int64))
        edge_idxs.append(np.asarray(fr['edge_idxs'], dtype=np.int64))
        timestamps.append(np.asarray(fr['timestamps'], dtype=np.float64))
        edge_feats.append(np.asarray(fr['edge_features'], dtype=np.float32))

    sources = np.concatenate(sources, axis=0)
    destinations = np.concatenate(destinations, axis=0)
    edge_idxs = np.concatenate(edge_idxs, axis=0)
    timestamps = np.concatenate(timestamps, axis=0)
    edge_feats = np.concatenate(edge_feats, axis=0)

    order = np.argsort(timestamps, kind='mergesort')
    sources = sources[order]
    destinations = destinations[order]
    edge_idxs = edge_idxs[order]
    timestamps = timestamps[order]
    edge_feats = edge_feats[order]

    return {
        'n_nodes': int(window_frames[0]['n_nodes']),
        'sources': sources,
        'destinations': destinations,
        'edge_idxs': edge_idxs,
        'timestamps': timestamps,
        'edge_features': edge_feats,
    }


class EdgeFeatureProjector(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        return self.projection(x)


class NodeFeatureProjector(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        return self.projection(x)


def build_common_parser():
    parser = argparse.ArgumentParser('TGN Video Classification Training')
    parser.add_argument('--dataset', type=str, choices=['penn', 'hmdb'], required=True)
    parser.add_argument('-d', '--data', type=str, default=None, help='Dataset root directory')
    parser.add_argument('--bs', type=int, default=50, help='Batch_size')
    parser.add_argument('--prefix', type=str, default=None, help='Prefix to name the checkpoints')
    parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
    parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
    parser.add_argument('--n_epoch', type=int, default=200, help='Number of epochs')
    parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
    parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
    parser.add_argument('--drop_out', type=float, default=0.5, help='Dropout probability')
    parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
    parser.add_argument('--node_dim', type=int, default=100, help='Dimensions of the node embedding')
    parser.add_argument('--time_dim', type=int, default=100, help='Dimensions of the time embedding')
    parser.add_argument('--backprop_every', type=int, default=1, help='Every how many batches to backprop')
    parser.add_argument('--use_memory', action='store_true', default='True',
                        help='Whether to augment the model with a node memory')
    parser.add_argument('--embedding_module', type=str, default="graph_sum",
                        choices=["graph_attention", "graph_sum", "identity", "time"],
                        help='Type of embedding module')
    parser.add_argument('--message_function', type=str, default="identity",
                        choices=["mlp", "identity"], help='Type of message function')
    parser.add_argument('--aggregator', type=str, default="last", help='Type of message aggregator')
    parser.add_argument('--memory_update_at_end', action='store_true',
                        help='Whether to update memory at the end or at the start of the batch')
    parser.add_argument('--message_dim', type=int, default=None)
    parser.add_argument('--memory_dim', type=int, default=None)
    parser.add_argument('--different_new_nodes', action='store_true',
                        help='Whether to use disjoint set of new nodes for train and val')
    parser.add_argument('--uniform', action='store_true',
                        help='take uniform sampling from temporal neighbors')
    parser.add_argument('--randomize_features', action='store_true',
                        help='Whether to randomize node features')
    parser.add_argument('--use_destination_embedding_in_message', action='store_true',
                        help='Whether to use the embedding of the destination node as part of the message')
    parser.add_argument('--use_source_embedding_in_message', action='store_true',
                        help='Whether to use the embedding of the source node as part of the message')
    parser.add_argument('--n_neg', type=int, default=1)
    parser.add_argument('--use_validation', action='store_true',
                        help='Whether to use a validation set')
    parser.add_argument('--new_node', action='store_true', help='model new node')
    parser.add_argument('--train_T', type=int, default=4, help='Window size for training')
    return parser


def build_dataset_spec(args):
    if args.dataset == 'penn':
        return {
            'name': 'penn',
            'classes_list': [
                'baseball_pitch', 'clean_and_jerk', 'golf_swing', 'pullup',
                'tennis_forehand'
            ],
            'default_root': 'data/penn_action_processed',
            'default_prefix': 'penn_action',
            'default_message_dim': 32,
            'default_memory_dim': 32,
            'use_balancing': False,
            'decoder_input_dim_kind': 'memory_dim',
            'node_feat_input_dim_kind': 'n_nodes',
            'target_nodes_kind': 'n_nodes',
            'log_filename_prefix': 'penn_action',
            'print_dataset_split_stats': True,
            'print_class_mapping': True,
            'print_model_init_stats': True,
            'print_preprocess_header': True,
            'print_target_nodes': True,
            'print_removed_keys_list': False,
        }

    if args.dataset == 'hmdb':
        return {
            'name': 'hmdb',
            'classes_list': ['pullup', 'climb', 'run', 'walk', 'situp'],
            'default_root': 'data/hmdb51_processed_data',
            'default_prefix': 'hmdb51',
            'default_message_dim': 64,
            'default_memory_dim': 64,
            'use_balancing': True,
            'decoder_input_dim_kind': 'message_dim',
            'node_feat_input_dim_kind': 'fixed_17',
            'target_nodes_kind': 'fixed_17',
            'log_filename_prefix': 'hmdb51',
            'print_dataset_split_stats': False,
            'print_class_mapping': False,
            'print_model_init_stats': False,
            'print_preprocess_header': False,
            'print_target_nodes': True,
            'print_removed_keys_list': True,
        }

    raise ValueError(f"Unsupported dataset: {args.dataset}")


def setup_logging(args, spec, model_save_path, decoder_save_path):
    log_dir = Path("log")
    saved_models_dir = Path("saved_models")
    saved_checkpoints_dir = Path("saved_checkpoints")
    log_dir.mkdir(parents=True, exist_ok=True)
    saved_models_dir.mkdir(parents=True, exist_ok=True)
    saved_checkpoints_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / f"{spec['log_filename_prefix']}_{time.time()}.log"

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
    logger.info(f"Model save path: {model_save_path}")
    logger.info(f"Decoder save path: {decoder_save_path}")
    return logger


def resolve_runtime_config(args, spec):
    resolved = {
        'BATCH_SIZE': args.bs,
        'NUM_NEIGHBORS': args.n_degree,
        'NUM_NEG': 1,
        'NUM_EPOCH': args.n_epoch,
        'NUM_HEADS': args.n_head,
        'DROP_OUT': args.drop_out,
        'GPU': args.gpu,
        'UNIFORM': args.uniform,
        'NEW_NODE': args.new_node,
        'SEQ_LEN': args.n_degree,
        'DATA': args.data if args.data is not None else spec['default_root'],
        'NUM_LAYER': args.n_layer,
        'LEARNING_RATE': args.lr,
        'NODE_LAYER': 1,
        'NODE_DIM': args.node_dim,
        'TIME_DIM': args.time_dim,
        'USE_MEMORY': args.use_memory,
        'MESSAGE_DIM': args.message_dim if args.message_dim is not None else spec['default_message_dim'],
        'MEMORY_DIM': args.memory_dim if args.memory_dim is not None else spec['default_memory_dim'],
        'TRAIN_T': args.train_T,
        'PREFIX': args.prefix if args.prefix is not None else spec['default_prefix'],
    }
    resolved['MODEL_SAVE_PATH'] = f"./saved_models/{resolved['PREFIX']}-tgn-classification.pth"
    resolved['DECODER_SAVE_PATH'] = f"./saved_models/{resolved['PREFIX']}-decoder-classification.pth"
    resolved['NODE_PROJECTOR_SAVE_PATH'] = f"./saved_models/{resolved['PREFIX']}-node_projector.pth"
    resolved['EDGE_PROJECTOR_SAVE_PATH'] = f"./saved_models/{resolved['PREFIX']}-edge_projector.pth"
    resolved['get_checkpoint_path'] = lambda epoch: (
        f"./saved_checkpoints/{resolved['PREFIX']}-{epoch}-classification.pth"
    )
    return resolved


def prepare_dataset(spec, runtime_cfg):
    classes_list = spec['classes_list']
    root = runtime_cfg['DATA']

    all_data = load_all_videos_no_split(root, classes_list)

    if spec['name'] == 'penn':
        print(f'加载的视频数: {len(all_data)}')
    else:
        print(f'视频数: {len(all_data)}')

    if len(all_data) == 0:
        print("错误: 未加载到任何数据，请检查数据路径和类别列表")
        sys.exit(1)

    train_data, test_data, stats = split_dataset_by_class(
        all_data,
        train_ratio=0.7,
        test_ratio=0.3,
        seed=42
    )

    if spec['print_dataset_split_stats']:
        print("\n数据集划分统计:")
        for cls, cls_stats in stats.items():
            if cls != 'overall':
                print(f"  {cls}: 总计 {cls_stats['total']}, 训练 {cls_stats['train']}, 测试 {cls_stats['test']}")
        print(f"  总体: {stats['overall']}")

    class_to_label = {c: i for i, c in enumerate(classes_list)}

    if spec['print_class_mapping']:
        print(f"\n类别标签映射: {class_to_label}")

    if spec['use_balancing']:
        balanced_train_data, sampling_stats = balance_dataset_by_undersampling(
            train_data,
            class_to_label,
            seed=42
        )
        train_data = balanced_train_data

        class_distribution = {}
        for key in train_data.keys():
            cls = key.split('/')[0]
            class_distribution[cls] = class_distribution.get(cls, 0) + 1
        for cls, count in sorted(class_distribution.items()):
            print(f"  {cls} (label {class_to_label[cls]}): {count} 个视频")
        print("=" * 50)

    return {
        'classes_list': classes_list,
        'all_data': all_data,
        'train_data': train_data,
        'test_data': test_data,
        'stats': stats,
        'class_to_label': class_to_label,
    }


def initialize_model_state(args, spec, runtime_cfg, train_data, device):
    first_key = next(iter(train_data.keys()))
    first_batches = train_data[first_key]

    builder = TemporalGraphDataLoader(device=device)
    first_neighbor_finder = builder.create_neighbor_finder(first_batches)

    edge_feat_dim = int(first_batches[0]['edge_features'].shape[1])
    node_feat_dim = int(first_batches[0]['node_features'].shape[1])
    n_nodes = int(first_batches[0]['n_nodes'])

    if spec['print_model_init_stats']:
        print(f"\n模型初始化参数:")
        print(f"  节点数: {n_nodes}")
        print(f"  节点特征维度: {node_feat_dim}")
        print(f"  边特征维度: {edge_feat_dim}")

    edge_idxs_list = []
    edge_feats_list = []
    for b in first_batches:
        edge_idxs_list.append(np.asarray(b['edge_idxs'], dtype=np.int64))
        edge_feats_list.append(np.asarray(b['edge_features'], dtype=np.float32))
    edge_idxs_cat = np.concatenate(edge_idxs_list, axis=0)
    edge_feats_cat = np.concatenate(edge_feats_list, axis=0)
    order = np.argsort(edge_idxs_cat, kind='mergesort')
    first_edge_features = edge_feats_cat[order]

    edge_feat_input_dim = 4
    edge_feat_output_dim = runtime_cfg['MESSAGE_DIM']
    edge_projector = EdgeFeatureProjector(edge_feat_input_dim, edge_feat_output_dim).to(device)

    if spec['node_feat_input_dim_kind'] == 'n_nodes':
        node_feat_input_dim = n_nodes
    elif spec['node_feat_input_dim_kind'] == 'fixed_17':
        node_feat_input_dim = 17
    else:
        raise ValueError(spec['node_feat_input_dim_kind'])

    node_feat_output_dim = runtime_cfg['MESSAGE_DIM']
    node_projector = NodeFeatureProjector(node_feat_input_dim, node_feat_output_dim).to(device)

    node_features_projected = node_projector(
        torch.from_numpy(np.eye(n_nodes, dtype=np.float32)).to(device)
    )
    edge_features_projected = edge_projector(
        torch.ones((len(first_edge_features), edge_feat_input_dim), device=device)
    )

    node_features_projected_np = node_features_projected.detach().cpu().numpy()
    edge_features_projected_np = edge_features_projected.detach().cpu().numpy()

    tgn = TGN(
        neighbor_finder=first_neighbor_finder,
        node_features=node_features_projected_np,
        edge_features=edge_features_projected_np,
        device=device,
        n_layers=runtime_cfg['NUM_LAYER'],
        n_heads=runtime_cfg['NUM_HEADS'],
        dropout=runtime_cfg['DROP_OUT'],
        use_memory=runtime_cfg['USE_MEMORY'],
        message_dimension=runtime_cfg['MESSAGE_DIM'],
        memory_dimension=runtime_cfg['MEMORY_DIM'],
        memory_update_at_start=not args.memory_update_at_end,
        embedding_module_type=args.embedding_module,
        message_function=args.message_function,
        aggregator_type=args.aggregator,
        n_neighbors=runtime_cfg['NUM_NEIGHBORS'],
        use_destination_embedding_in_message=args.use_destination_embedding_in_message,
        use_source_embedding_in_message=args.use_source_embedding_in_message
    ).to(device)

    if spec['decoder_input_dim_kind'] == 'memory_dim':
        decoder_input_dim = runtime_cfg['MEMORY_DIM']
    elif spec['decoder_input_dim_kind'] == 'message_dim':
        decoder_input_dim = runtime_cfg['MESSAGE_DIM']
    else:
        raise ValueError(spec['decoder_input_dim_kind'])

    out_dim = len(spec['classes_list'])
    decoder = MLP(decoder_input_dim, out_dim, drop=runtime_cfg['DROP_OUT']).to(device)
    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        list(tgn.parameters()) +
        list(decoder.parameters()) +
        list(edge_projector.parameters()) +
        list(node_projector.parameters()),
        lr=runtime_cfg['LEARNING_RATE']
    )

    if spec['target_nodes_kind'] == 'n_nodes':
        target_nodes = list(range(n_nodes))
    elif spec['target_nodes_kind'] == 'fixed_17':
        target_nodes = list(range(17))
    else:
        raise ValueError(spec['target_nodes_kind'])

    if spec['print_target_nodes']:
        if spec['name'] == 'penn':
            print(f"目标节点: {target_nodes}")
        else:
            print(target_nodes)

    return {
        'tgn': tgn,
        'decoder': decoder,
        'edge_projector': edge_projector,
        'node_projector': node_projector,
        'criterion': criterion,
        'optimizer': optimizer,
        'target_nodes': target_nodes,
        'n_nodes': n_nodes,
    }


def preprocess_video_features(spec, train_data):
    video_global_edge_features = {}
    valid_keys = []
    removed_keys = []

    if spec['print_preprocess_header']:
        print("\n预处理视频数据...")

    for key, batches in train_data.items():
        edge_idxs_list = []
        edge_feats_list = []
        for b in batches:
            edge_idxs_list.append(np.asarray(b['edge_idxs'], dtype=np.int64))
            edge_feats_list.append(np.asarray(b['edge_features'], dtype=np.float32))
        edge_idxs_cat = np.concatenate(edge_idxs_list, axis=0)
        edge_feats_cat = np.concatenate(edge_feats_list, axis=0)
        order = np.argsort(edge_idxs_cat, kind='mergesort')

        if np.allclose(edge_feats_cat[order], 0.0):
            removed_keys.append(key)
            print(f'警告: 视频 {key} 的 edge_features 全为 0，已从训练集中移除')
            continue

        video_global_edge_features[key] = edge_feats_cat[order]
        valid_keys.append(key)

    train_data_filtered = {key: train_data[key] for key in valid_keys}

    print(f'\n数据过滤统计:')
    print(f'  原始视频数: {len(train_data)}')
    print(f'  有效视频数: {len(train_data_filtered)}')
    print(f'  移除视频数: {len(removed_keys)}')
    if spec['print_removed_keys_list'] and removed_keys:
        print(f'  移除的视频列表: {removed_keys[:10]}...' if len(
            removed_keys) > 10 else f'  移除的视频列表: {removed_keys}')

    return train_data_filtered, video_global_edge_features


def save_best_and_checkpoint(runtime_cfg, model_state, epoch, accuracy, avg_loss, best_accuracy):
    improved = accuracy > best_accuracy
    if improved:
        best_accuracy = accuracy
        torch.save(model_state['tgn'].state_dict(), runtime_cfg['MODEL_SAVE_PATH'])
        torch.save(model_state['decoder'].state_dict(), runtime_cfg['DECODER_SAVE_PATH'])
        torch.save(model_state['node_projector'].state_dict(), runtime_cfg['NODE_PROJECTOR_SAVE_PATH'])
        torch.save(model_state['edge_projector'].state_dict(), runtime_cfg['EDGE_PROJECTOR_SAVE_PATH'])
        print(f'  -> 保存最佳模型 (Accuracy: {best_accuracy:.4f})')

    if (epoch + 1) % 10 == 0:
        checkpoint_path = runtime_cfg['get_checkpoint_path'](epoch + 1)
        torch.save({
            'epoch': epoch + 1,
            'tgn_state_dict': model_state['tgn'].state_dict(),
            'decoder_state_dict': model_state['decoder'].state_dict(),
            'node_projector_state_dict': model_state['node_projector'].state_dict(),
            'edge_projector_state_dict': model_state['edge_projector'].state_dict(),
            'accuracy': accuracy,
            'loss': avg_loss,
        }, checkpoint_path)

    return best_accuracy


def run_training_loop(args, spec, runtime_cfg, dataset_state, model_state, device):
    train_data = dataset_state['train_data']
    class_to_label = dataset_state['class_to_label']
    video_global_edge_features = dataset_state['video_global_edge_features']

    tgn = model_state['tgn']
    decoder = model_state['decoder']
    edge_projector = model_state['edge_projector']
    node_projector = model_state['node_projector']
    criterion = model_state['criterion']
    optimizer = model_state['optimizer']
    target_nodes = model_state['target_nodes']
    n_nodes = model_state['n_nodes']

    tgn.train()
    decoder.train()
    edge_projector.train()
    node_projector.train()

    best_accuracy = 0.0
    patience_counter = 0

    for epoch in range(runtime_cfg['NUM_EPOCH']):
        total_loss = 0.0
        total_videos = 0
        correct_predictions = 0
        total_predictions = 0

        accumulated_loss = 0.0
        batch_count = 0
        batch_correct = 0
        batch_total = 0

        train_items = list(train_data.items())
        random.shuffle(train_items)

        optimizer.zero_grad()

        for video_idx, (key, batches) in enumerate(train_items):
            cls = key.split('/')[0]
            label = class_to_label[cls]
            label_t = torch.tensor([label], dtype=torch.long, device=device)

            builder = TemporalGraphDataLoader(device=device)
            tgn.neighbor_finder = builder.create_neighbor_finder(batches)

            raw_edge_features = torch.from_numpy(
                video_global_edge_features[key].astype(np.float32)
            ).to(device)
            projected_edge_features = edge_projector(raw_edge_features)

            tgn.edge_raw_features = projected_edge_features
            tgn.embedding_module.edge_features = projected_edge_features

            node_features_static_raw = torch.from_numpy(np.eye(n_nodes, dtype=np.float32)).to(device)
            projected_node_features = node_projector(node_features_static_raw)
            tgn.node_raw_features = projected_node_features
            tgn.embedding_module.node_features = projected_node_features

            if tgn.use_memory:
                tgn.memory.__init_memory__()

            windows = make_windows(batches, runtime_cfg['TRAIN_T'])
            for window in windows:
                super_batch = build_window_super_batch_dynamic_edges(window)

                sources = super_batch['sources']
                destinations = super_batch['destinations']
                timestamps = super_batch['timestamps']
                edge_idxs = super_batch['edge_idxs']
                neg = np.zeros_like(sources)

                src_emb, _, _ = tgn.compute_temporal_embeddings_without_contributions(
                    sources, destinations, neg, timestamps, edge_idxs,
                    n_neighbors=runtime_cfg['NUM_NEIGHBORS']
                )

            if spec['name'] == 'penn':
                nodes_np = np.array(target_nodes, dtype=np.int64)
                dummy_eidx = np.zeros_like(nodes_np, dtype=np.int64)
                target_time = float(np.max(timestamps))
                ts_np = np.full(len(target_nodes), target_time, dtype=np.float64)
            else:
                nodes_np = target_nodes
                dummy_eidx = np.zeros_like(nodes_np, dtype=np.int64)
                target_time = max(timestamps)
                ts_np = np.full(len(target_nodes), target_time, dtype=np.int64)

            source_embedding, _, _ = tgn.compute_temporal_embeddings_without_contributions(
                nodes_np, nodes_np, nodes_np, ts_np, dummy_eidx, runtime_cfg['NUM_NEIGHBORS']
            )

            out_put = decoder(source_embedding)
            out_put = out_put.mean(dim=0)
            out_put = out_put.unsqueeze(0)
            loss = criterion(out_put, label_t)

            with torch.no_grad():
                pred = torch.argmax(out_put, dim=1)
                correct = (pred == label_t).item()
                correct_predictions += correct
                total_predictions += 1
                batch_correct += correct
                batch_total += 1

            accumulated_loss += loss
            batch_count += 1

            if batch_count >= runtime_cfg['BATCH_SIZE'] or video_idx == len(train_items) - 1:
                avg_batch_loss = accumulated_loss / batch_count
                batch_accuracy = batch_correct / max(batch_total, 1)

                optimizer.zero_grad()
                avg_batch_loss.backward()
                optimizer.step()

                total_loss += float(avg_batch_loss.detach().cpu()) * batch_count
                total_videos += batch_count

                if (epoch + 1) % 10 == 0 or batch_count > 0:
                    print(f'Epoch {epoch + 1} | Batch updated | videos in batch: {batch_count} | '
                          f'Batch Loss: {float(avg_batch_loss.detach().cpu()):.4f} | '
                          f'Batch Accuracy: {batch_accuracy:.4f} ({batch_correct}/{batch_total})')

                batch_correct = 0
                batch_total = 0
                accumulated_loss = 0.0
                batch_count = 0

            if tgn.use_memory:
                tgn.memory.__init_memory__()

        avg_loss = total_loss / max(total_videos, 1)
        accuracy = correct_predictions / max(total_predictions, 1)

        print(f'Epoch {epoch + 1}/{runtime_cfg["NUM_EPOCH"]} | Loss: {avg_loss:.4f} | '
              f'Accuracy: {accuracy:.4f} ({correct_predictions}/{total_predictions})')

        best_accuracy = save_best_and_checkpoint(
            runtime_cfg, model_state, epoch, accuracy, avg_loss, best_accuracy
        )

    print(f"\n训练完成!")
    print(f"最佳准确率: {best_accuracy:.4f}")
    print(f"模型已保存到: {runtime_cfg['MODEL_SAVE_PATH']}")
    print(f"解码器已保存到: {runtime_cfg['DECODER_SAVE_PATH']}")


def run_dataset(args):
    spec = build_dataset_spec(args)
    runtime_cfg = resolve_runtime_config(args, spec)
    setup_logging(args, spec, runtime_cfg['MODEL_SAVE_PATH'], runtime_cfg['DECODER_SAVE_PATH'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if spec['name'] == 'penn':
        print(f"使用设备: {device}")

    dataset_state = prepare_dataset(spec, runtime_cfg)
    model_state = initialize_model_state(
        args, spec, runtime_cfg, dataset_state['train_data'], device
    )

    filtered_train_data, video_global_edge_features = preprocess_video_features(
        spec, dataset_state['train_data']
    )
    dataset_state['train_data'] = filtered_train_data
    dataset_state['video_global_edge_features'] = video_global_edge_features

    run_training_loop(args, spec, runtime_cfg, dataset_state, model_state, device)


if __name__ == "__main__":
    parser = build_common_parser()
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    run_dataset(args)
