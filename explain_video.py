# 保存：单个视频 → 逐帧 .joblib（无 train 目录）
import os, json, glob, numpy as np, joblib
from pathlib import Path
from typing import List, Dict, Any, Optional, Sequence
import random
import torch
import logging
from model.tgn import TGN
import model.tgn as tgn_debug_module
import time
import argparse
import sys
from utils.prepare_video_data import TemporalGraphDataLoader

from utils.utils import  MLP
import utils.utils as utils_debug_module
import torch.nn as nn
import cv2
import math
import copy
from utils.attribution import compute_edge_memory_contributions,compute_neighbor_memory_contributions,\
    merge_contribution_dicts, verify_raw_backtrace_conservation
import utils.attribution as attribution_debug_module
import utils.memory_backtracking_trees as backtracking_debug_module
import modules.embedding_module as embedding_refactored_debug_module


import cvxpy as cvx
# from tools.compare_graph_attention_outputs import (
#     compare_graph_attention_current_refactored_tgn_with_legacy
# )

def load_config(config_path: str) -> Dict[str, Any]:
    if not config_path:
        return {}
    if not os.path.exists(config_path):
        return {}
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def apply_config_defaults(parser: argparse.ArgumentParser, config: Dict[str, Any]) -> Dict[str, Any]:
    valid_dests = {action.dest for action in parser._actions}
    config_defaults = {k: v for k, v in config.items() if k in valid_dests}
    unknown_keys = sorted(set(config.keys()) - valid_dests)
    if unknown_keys:
        print(f'警告: 配置文件中存在未使用的参数: {unknown_keys}')
    if config_defaults:
        parser.set_defaults(**config_defaults)
    return config_defaults

def sanitize_filename_part(value: Any) -> str:
    text = str(value).strip()
    text = text.replace(os.sep, '_')
    if os.altsep:
        text = text.replace(os.altsep, '_')
    return ''.join(ch if ch.isalnum() or ch in ('-', '_', '.') else '_' for ch in text)

def build_explanation_result_path(args: argparse.Namespace, max_depth: int) -> str:
    filename_parts = [
        args.result_name,
        args.embedding_module,
        f'l{args.n_layer}',
        args.memory_updater,
        args.edge_selection_mode,
        f'depth{max_depth}',
    ]
    if args.memory_update_at_end:
        filename_parts.insert(-2, 'end')
    filename = '_'.join(sanitize_filename_part(part) for part in filename_parts) + '.json'
    return os.path.join('results', filename)

def build_video_model_run_name(args: argparse.Namespace) -> str:
    prefix = args.prefix
    embedding_layer_suffix = f'_{args.embedding_module}_l{args.n_layer}'
    if not prefix.endswith(embedding_layer_suffix):
        prefix = f'{prefix}{embedding_layer_suffix}'
    return f'{prefix}_{args.memory_updater}_{args.aggregator}_{args.message_function}'

# Penn Action 骨架定义（13个节点）
# PENN_ACTION_SKELETON = [
#     # 头部到颈部
#     [0, 1],  # head -> neck
#     # 颈部到肩膀
#     [1, 2],  # neck -> left_shoulder
#     [1, 3],  # neck -> right_shoulder
#     # 左臂
#     [2, 4],  # left_shoulder -> left_elbow
#     [4, 6],  # left_elbow -> left_wrist
#     # 右臂
#     [3, 5],  # right_shoulder -> right_elbow
#     [5, 7],  # right_elbow -> right_wrist
#     # 躯干
#     [2, 8],  # left_shoulder -> left_hip
#     [3, 9],  # right_shoulder -> right_hip
#     [8, 9],  # left_hip <-> right_hip
#     # 左腿
#     [8, 10],  # left_hip -> left_knee
#     [10, 12], # left_knee -> left_ankle
#     # 右腿
#     [9, 11],  # right_hip -> right_knee
#     # [11, 12], # right_knee -> right_ankle (注意：13个节点时，right_ankle是12)
# ]
PENN_ACTION_SKELETON = [[0,1],[0,2],[2,4],[4,6],[2,8],[8,10],[10,12],[1,3],[3,5],[1,7],[7,9],[9,11]]
COCO_SKELETON = [
    [0,1],[0,2],[1,3],[2,4],
    [5,6],
    [5,7],[7,9],
    [6,8],[8,10],
    [5,11],[6,12],
    [11,12],
    [11,13],[13,15],
    [12,14],[14,16],
]
SKELETONS = {
    'penn_action': PENN_ACTION_SKELETON,
    'penn': PENN_ACTION_SKELETON,
    'coco': COCO_SKELETON,
    'hmdb51': COCO_SKELETON,
    'hmdb': COCO_SKELETON,
}
CURRENT_SKELETON_NAME = 'penn_action'

def set_current_skeleton(skeleton_name: str):
    global CURRENT_SKELETON_NAME
    if skeleton_name not in SKELETONS:
        raise ValueError(f'Unknown skeleton: {skeleton_name}. Available: {sorted(SKELETONS.keys())}')
    CURRENT_SKELETON_NAME = skeleton_name

def get_skeleton_edges(skeleton_name: str = None):
    return SKELETONS[skeleton_name or CURRENT_SKELETON_NAME]

def _make_edges(bidir=True, skeleton_name: str = None):
    e=[]
    for u,v in get_skeleton_edges(skeleton_name):
        e.append([u,v])
        if bidir: e.append([v,u])
    return np.asarray(e, dtype=np.int64)

def infer_n_nodes(skeleton_name: str) -> int:
    skeleton = get_skeleton_edges(skeleton_name)
    return max(max(u, v) for u, v in skeleton) + 1

def parse_classes_list(value):
    if value is None:
        return None
    if isinstance(value, str):
        return [item.strip() for item in value.split(',') if item.strip()]
    return list(value)

def select_video_items(train_items, video_indices: Sequence[int] = None, video_keys: Sequence[str] = None):
    if video_keys:
        item_by_key = {key: value for key, value in train_items}
        missing = [key for key in video_keys if key not in item_by_key]
        if missing:
            raise KeyError(f'video_keys not found in filtered train data: {missing}')
        return [(key, item_by_key[key]) for key in video_keys]
    if video_indices:
        selected = []
        for idx in video_indices:
            if idx < 0 or idx >= len(train_items):
                raise IndexError(f'video index out of range: {idx}, train_items={len(train_items)}')
            selected.append(train_items[idx])
        return selected
    if not train_items:
        return []
    return [train_items[0]]

def _load_json(p):
    with open(p,'r') as f: return json.load(f)

def _ts(frame_idx:int, fps: Optional[int])->float:
    return float(frame_idx) if not fps or fps<=0 else float(frame_idx)/float(fps)


def _feat_xy(frame_json: Dict[str,Any], use_normalized=True)->np.ndarray:
    n=13; feat=np.zeros((n,2), dtype=np.float32)
    key='keypoints_normalized' if use_normalized else 'keypoints_pixel'
    kp=frame_json.get(key, None)
    if kp is None: return feat
    if isinstance(kp, list) and len(kp)==n and isinstance(kp[0], dict):
        for it in kp:
            i=int(it['index']); feat[i]=[float(it['x']), float(it['y'])]
        return feat
    arr=np.asarray(kp, dtype=np.float32)
    if arr.shape[0]==n and arr.shape[1]>=2:
        feat[:,0]=arr[:,0]; feat[:,1]=arr[:,1]
    return feat

def save_single_video_no_split(
    video_dir: str,
    out_root: str,
    use_normalized: bool=True,
    bidirectional_edges: bool=True,
    edge_feature_dim: int=1
)->Dict[str,Any]:
    assert os.path.isdir(video_dir), f'not found: {video_dir}'
    meta_path=os.path.join(video_dir,'metadata.json'); assert os.path.exists(meta_path)
    meta=_load_json(meta_path); vi=meta.get('video_info',{})
    fps=vi.get('fps', None); video_class=meta.get('class','unknown')
    video_name=meta.get('video_name', os.path.basename(video_dir))
    frame_files=sorted(glob.glob(os.path.join(video_dir,'frame_*.json')))
    assert frame_files, f'no frames in {video_dir}'

    edges=_make_edges(bidirectional_edges); E=edges.shape[0]
    src=edges[:,0].astype(np.int64); dst=edges[:,1].astype(np.int64)
    efeat=np.ones((E, edge_feature_dim), dtype=np.float32)

    save_dir=os.path.join(out_root, video_class, video_name)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    n_nodes=17; saved=0
    for rank, fp in enumerate(frame_files):
        fj=_load_json(fp); fidx=int(fj.get('frame_index', -1)); t=_ts(fidx, fps)
        nfeat=_feat_xy(fj, use_normalized=use_normalized)  # [17,2]
        eidx = (rank*E) + np.arange(E, dtype=np.int64)     # 每帧唯一 edge_idxs

        batch={
            'n_nodes': n_nodes,
            'sources': src.copy(),
            'destinations': dst.copy(),
            'edge_idxs': eidx,
            'timestamps': np.full((E,), t, dtype=np.float64),
            'node_features': nfeat.astype(np.float32),
            'edge_features': efeat.copy(),
            'frame_index': fidx,
            'video_name': video_name,
            'class': video_class,
        }
        joblib.dump(batch, os.path.join(save_dir, f'frame_{fidx:06d}.joblib'))
        saved+=1

    summary={
        'video_dir': video_dir,
        'output_dir': save_dir,
        'class': video_class,
        'video_name': video_name,
        'n_frames': saved,
        'E_per_frame': E,
        'edge_feature_dim': edge_feature_dim,
        'node_feature_dim': 2,
        'bidirectional_edges': bidirectional_edges,
        'use_normalized_xy': use_normalized,
        'fps': fps
    }
    joblib.dump(summary, os.path.join(save_dir,'summary.joblib'))
    with open(os.path.join(save_dir,'summary.json'),'w') as f: json.dump(summary,f,indent=2,ensure_ascii=False)
    print(f'保存: {video_class}/{video_name} 帧数={saved} 输出={save_dir}')
    return summary
def load_single_video_batches(root_dir: str, cls_name: str, video_name: str):
    """
    读取 {root}/{class}/{video_name}/frame_*.joblib
    返回按帧排序的 batch 列表
    """
    video_dir=os.path.join(root_dir, cls_name, video_name)
    frame_files=sorted(glob.glob(os.path.join(video_dir, 'frame_*.joblib')))
    return [joblib.load(fp) for fp in frame_files]

def save_all_videos_no_split(
    in_root: str = 'data/hmdb51_keypoints',
    out_root: str = 'data/video_daily_data_tgn',
    classes_list: list = None,
    use_normalized: bool = True,
    bidirectional_edges: bool = True,
    edge_feature_dim: int = 1
):
    """
    批量处理 in_root 下的所有类别/视频：
      - 输入：{in_root}/{class}/{video}/frame_xxx.json + metadata.json
      - 输出：{out_root}/{class}/{video}/frame_xxx.joblib
    """
    if classes_list is None:
        # 如需限定类别，传入 ['pullup','climb','run','walk','situp']
        classes_list = [d for d in os.listdir(in_root) if os.path.isdir(os.path.join(in_root, d))]

    processed = 0
    failed = []

    for cls_name in classes_list:
        cls_dir = os.path.join(in_root, cls_name)
        if not os.path.isdir(cls_dir):
            print(f'跳过（非目录）: {cls_dir}')
            continue

        # 遍历该类别下的视频目录
        for video_name in sorted(os.listdir(cls_dir)):
            video_dir = os.path.join(cls_dir, video_name)
            if not os.path.isdir(video_dir):
                continue

            # 必须包含 metadata.json 才认为是有效视频目录
            meta_path = os.path.join(video_dir, 'metadata.json')
            if not os.path.exists(meta_path):
                # 可能是杂项文件夹，跳过
                continue

            try:
                save_single_video_no_split(
                    video_dir=video_dir,
                    out_root=out_root,
                    use_normalized=use_normalized,
                    bidirectional_edges=bidirectional_edges,
                    edge_feature_dim=edge_feature_dim
                )
                processed += 1
            except Exception as e:
                print(f'错误: 处理失败 {video_dir}: {e}')
                failed.append(video_dir)

    print(f'\n批量处理完成: 成功 {processed} 个视频, 失败 {len(failed)} 个。输出根目录: {out_root}')
    if failed:
        print('失败列表(前10):')
        for p in failed[:10]:
            print('  -', p)
    return {'processed': processed, 'failed': failed, 'out_root': out_root}

def load_all_videos_no_split(root_dir: str, classes_list=None):
    """
    扫描 {root}/{class}/{video}/frame_*.joblib
    返回: dict { 'class/video_name': [batches...] }
    """
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
    """
    按类别分层划分训练/测试集。
    - all_data: { 'class/video_name': [batches...] }
    - train_ratio + test_ratio 应 <= 1.0
    返回:
      train_data: { 'class/video_name': [batches...] }
      test_data:  { 'class/video_name': [batches...] }
      stats: { 'class': {'total':N, 'train':n1, 'test':n2}, 'overall': {...} }
    """
    assert 0 < train_ratio <= 1 and 0 < test_ratio <= 1 and train_ratio + test_ratio <= 1.0
    rng = random.Random(seed)

    # 1) 按类别聚合视频键
    class_to_keys = {}
    for key in all_data.keys():
        cls = key.split('/')[0]
        class_to_keys.setdefault(cls, []).append(key)

    train_keys, test_keys = [], []
    stats = {}

    # 2) 每个类别内部独立打乱并按比例切分
    for cls, keys in class_to_keys.items():
        keys_sorted = sorted(keys)
        rng.shuffle(keys_sorted)
        n = len(keys_sorted)
        n_train = int(round(n * train_ratio))
        n_test = int(round(n * test_ratio))
        # 防止四舍五入导致溢出
        n_train = min(n_train, n)
        n_test = min(n_test, n - n_train)

        cls_train = keys_sorted[:n_train]
        cls_test  = keys_sorted[n_train:n_train + n_test]

        train_keys.extend(cls_train)
        test_keys.extend(cls_test)

        stats[cls] = {'total': n, 'train': len(cls_train), 'test': len(cls_test)}

    # 3) 组装字典
    train_data = {k: all_data[k] for k in train_keys}
    test_data  = {k: all_data[k] for k in test_keys}

    # overall
    stats['overall'] = {
        'total_videos': len(all_data),
        'train_videos': len(train_data),
        'test_videos': len(test_data),
    }
    return train_data, test_data, stats

def build_tgn_for_video(batches, device):
    builder = TemporalGraphDataLoader(device=device)
    neighbor_finder = builder.create_neighbor_finder(batches)
    # 这里不直接用 builder.prepare_tgn_data 返回的 node_features，因为我们每帧都要动态更新
    # 为了初始化 TGN，需要一个占位的 node_features 与 edge_features（形状一致即可）

    edge_feat_dim = int(batches[0]['edge_features'].shape[1])

    n_nodes = int(batches[0]['n_nodes'])
    node_features_init = np.ones((n_nodes, edge_feat_dim), dtype=np.float32)  # [n_nodes, 1]
    feat_dim =edge_feat_dim  # 2 (x,y)


    edge_idxs_list = []
    edge_feats_list = []

    for b in batches:
        ei = np.asarray(b['edge_idxs'], dtype=np.int64)  # [E]
        ef = np.asarray(b['edge_features'], dtype=np.float32)  # [E, F]
        assert ei.shape[0] == ef.shape[0], "edge_idxs 与 edge_features 行数不一致"
        edge_idxs_list.append(ei)
        edge_feats_list.append(ef)

    # 2) 拼接并按 edge_idxs 排序（保证全局一致顺序）
    edge_idxs_cat = np.concatenate(edge_idxs_list, axis=0)  # [sum_E]
    edge_feats_cat = np.concatenate(edge_feats_list, axis=0)  # [sum_E, F]

    order = np.argsort(edge_idxs_cat, kind='mergesort')
    edge_idxs_sorted = edge_idxs_cat[order]  # [sum_E]
    edge_features_sorted = edge_feats_cat[order]

    # tgn = TGN(
    #     neighbor_finder=neighbor_finder,
    #     node_features=node_features_init,
    #     edge_features=edge_features,
    #     device=device,
    #     n_layers=n_layers,
    #     n_heads=n_heads,
    #     dropout=dropout,
    #     use_memory=use_memory,
    #     message_dimension=message_dim,
    #     memory_dimension=memory_dim,
    #     memory_update_at_start=True,
    #     embedding_module_type="graph_sum",
    #     message_function="identity",
    #     aggregator_type="last",
    #     n_neighbors=n_degree,
    #     use_destination_embedding_in_message=False,
    #     use_source_embedding_in_message=False
    # ).to(device)
    tgn = TGN(neighbor_finder=neighbor_finder, node_features=node_features_init,
              edge_features=edge_features_sorted, device=device,
              n_layers=NUM_LAYER,
              n_heads=NUM_HEADS, dropout=DROP_OUT, use_memory=USE_MEMORY,
              message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM,
              memory_update_at_start=not args.memory_update_at_end,
              memory_updater_type=args.memory_updater,
              embedding_module_type=args.embedding_module,
              message_function=args.message_function,
              aggregator_type=args.aggregator, n_neighbors=NUM_NEIGHBORS,
              use_destination_embedding_in_message=args.use_destination_embedding_in_message,
              use_source_embedding_in_message=args.use_source_embedding_in_message)

    # print(f"TGN 初始化后:")
    # print(f"  n_node_features: {tgn.n_node_features}")
    # print(f"  n_edge_features: {tgn.n_edge_features}")
    # print(f"  embedding_dimension: {tgn.embedding_dimension}")
    # print(f"  time_encoder.dimension: {tgn.time_encoder.dimension}")
    # print(f"  期望 linear_1 输入维度: {2 * tgn.n_node_features + tgn.n_edge_features}")
    return tgn

def make_windows(batches, T):
    # 将帧按顺序分成长度为 T 的窗口（不足 T 的尾窗也保留）
    windows = []
    n = len(batches)
    for i in range(0, n, T):
        windows.append(batches[i:i+T])
    return windows

def build_window_super_batch_dynamic_edges(window_frames):
    """
    将 window 内多帧合为一次 batch：
    - edge_features: 动态，拼接每帧 [x_src, y_src, x_dst, y_dst]
    返回 super_batch
    """
    import numpy as np

    assert len(window_frames) > 0

    sources = []
    destinations = []
    edge_idxs = []
    timestamps = []
    edge_feats = []

    for fr in window_frames:
        ef = np.asarray(fr['edge_features'], dtype=np.float64)
        # 如果该帧的 edge_features 全为 0 或为空，则跳过
        if ef.size == 0 or np.allclose(ef, 0.0):
            continue

        sources.append(np.asarray(fr['sources'], dtype=np.int64))
        destinations.append(np.asarray(fr['destinations'], dtype=np.int64))
        edge_idxs.append(np.asarray(fr['edge_idxs'], dtype=np.int64))
        timestamps.append(np.asarray(fr['timestamps'], dtype=np.float64))
        edge_feats.append(np.asarray(fr['edge_features'], dtype=np.float64))

    if len(edge_idxs) == 0:
        n_nodes = int(window_frames[0].get('n_nodes', 0))
        return {
            'n_nodes': n_nodes,
            'sources': np.array([], dtype=np.int64),
            'destinations': np.array([], dtype=np.int64),
            'edge_idxs': np.array([], dtype=np.int64),
            'timestamps': np.array([], dtype=np.float64),
            'edge_features': np.array([], dtype=np.float64),
        }

    sources = np.concatenate(sources, axis=0)
    destinations = np.concatenate(destinations, axis=0)
    edge_idxs = np.concatenate(edge_idxs, axis=0)
    timestamps = np.concatenate(timestamps, axis=0)
    edge_feats = np.concatenate(edge_feats, axis=0)

    # 3) 按时间排序
    order = np.argsort(timestamps, kind='mergesort')
    sources = sources[order]
    destinations = destinations[order]
    edge_idxs = edge_idxs[order]
    timestamps = timestamps[order]
    edge_feats = edge_feats[order]

    super_batch = {
        'n_nodes': int(window_frames[0]['n_nodes']),
        'sources': sources,
        'destinations': destinations,
        'edge_idxs': edge_idxs,
        'timestamps': timestamps,
        'edge_features': edge_feats,  # 动态特征
    }
    return super_batch

def balance_dataset_by_undersampling(train_data: dict, class_to_label: dict, seed: int = 42):
    """
    通过下采样平衡数据集：找到最少类别数量，然后每个类别都采样相同数量

    Parameters:
    train_data: { 'class/video_name': [batches...] }
    class_to_label: { 'class': label }
    seed: 随机种子

    Returns:
    balanced_train_data: 平衡后的训练数据
    sampling_stats: 采样统计信息
    """
    import random
    rng = random.Random(seed)

    # 1. 按类别分组视频
    class_to_keys = {}
    for key in train_data.keys():
        cls = key.split('/')[0]
        if cls in class_to_label:  # 只处理已知类别
            class_to_keys.setdefault(cls, []).append(key)

    # 2. 找到最少类别的数量
    min_count = min(len(keys) for keys in class_to_keys.values())
    print(f"\n类别统计（采样前）:")
    for cls, keys in sorted(class_to_keys.items()):
        print(f"  {cls} (label {class_to_label[cls]}): {len(keys)} 个视频")
    print(f"\n最少类别数量: {min_count}")
    print(f"将每个类别采样到 {min_count} 个视频")

    # 3. 对每个类别进行采样
    balanced_keys = []
    sampling_stats = {}

    for cls, keys in class_to_keys.items():
        original_count = len(keys)

        # 如果该类别数量多于最少数量，进行随机采样
        if original_count > min_count:
            sampled_keys = rng.sample(keys, min_count)
        else:
            # 如果数量等于最少数量，全部使用
            sampled_keys = keys

        balanced_keys.extend(sampled_keys)

        sampling_stats[cls] = {
            'original': original_count,
            'sampled': len(sampled_keys),
            'removed': original_count - len(sampled_keys)
        }
        print(f"  {cls}: 原始 {original_count} 个 -> 采样 {len(sampled_keys)} 个")

    # 4. 构建平衡后的 train_data
    balanced_train_data = {key: train_data[key] for key in balanced_keys}

    print(f"\n平衡后的数据集:")
    print(f"  总视频数: {len(balanced_train_data)}")
    print(f"  每个类别: {min_count} 个视频")
    print(f"  类别数: {len(class_to_keys)}")
    print("=" * 50)

    return balanced_train_data, sampling_stats

class EdgeFeatureProjector(nn.Module):
    """将边特征投影到更高维度"""

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
    """将节点特征投影到更高维度"""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        return self.projection(x)

def _load_json(p):
    with open(p, 'r') as f:
        return json.load(f)
def _collect_frames_json(video_dir: str):
    frames = []
    for fp in sorted(glob.glob(os.path.join(video_dir, 'frame_*.json'))):
        frames.append(_load_json(fp))
    return frames


def _draw_skeleton(frame, pts_xy, draw_points=True, draw_edges=True,
                   point_color=(0, 255, 0), edge_color=(0, 0, 255), radius=4, thickness=2,
                   highlight_edges=None, highlight_edge_color=(0, 255, 255), highlight_thickness=5):
    """
    绘制骨架，支持高亮特定边

    参数:
    - highlight_edges: set of (u, v) 或 set of local_edge_idx，要高亮的边
    - highlight_edge_color: 高亮边的颜色 (BGR)
    - highlight_thickness: 高亮边的粗细
    """
    if draw_edges:
        edges = _make_edges(bidir=True)  # 获取所有边（包括双向）
        for local_idx, (u, v) in enumerate(edges):
            x1, y1 = int(round(pts_xy[u, 0])), int(round(pts_xy[u, 1]))
            x2, y2 = int(round(pts_xy[v, 0])), int(round(pts_xy[v, 1]))

            # 判断是否高亮（可以通过 local_idx 或 (u,v) 判断）
            is_highlight = False
            if highlight_edges is not None:
                # 支持两种格式：local_idx 或 (u, v) 元组
                if local_idx in highlight_edges or (u, v) in highlight_edges:
                    is_highlight = True

            if is_highlight:
                # 先画粗的高亮边（更明显）
                cv2.line(frame, (x1, y1), (x2, y2), highlight_edge_color, highlight_thickness, cv2.LINE_AA)
            else:
                # 普通边
                cv2.line(frame, (x1, y1), (x2, y2), edge_color, thickness, cv2.LINE_AA)

    if draw_points:
        for i in range(pts_xy.shape[0]):
            x, y = int(round(pts_xy[i, 0])), int(round(pts_xy[i, 1]))
            cv2.circle(frame, (x, y), radius, point_color, -1, cv2.LINE_AA)




def visualize_video_with_keypoints_from_joblib(
        video_data_dir: str,  # 处理后的数据目录，包含 joblib 文件
        output_video_dir: str = None,
        output_video_name: str = None,
        output_frames_dir: str = None,
        original_frames_dir: str = 'data/Penn_Action/frames',  # 原始图片目录
        draw_points: bool = True,
        draw_edges: bool = True,
        select_edge_list: List[int] = None,
        bidirectional_edges: bool = True,
        highlight_edge_color: tuple = (0, 255, 255),
        highlight_thickness: int = 5,
        output_fps: float = None
):
    """
    从 joblib 文件中可视化 Penn Action 视频的关键点，支持从原始图片目录加载背景
    """
    assert os.path.isdir(video_data_dir), f'not found: {video_data_dir}'
    meta_path = os.path.join(video_data_dir, 'metadata.json')
    if not os.path.exists(meta_path):
        meta_path = os.path.join(video_data_dir, 'summary.json')
    assert os.path.exists(meta_path), f'not found: {meta_path}'
    meta = _load_json(meta_path)

    # 读取视频信息
    video_info = meta.get('video_info', {})
    W = int(video_info.get('width', 480))
    H = int(video_info.get('height', 360))
    fps = int(video_info.get('fps', 30))

    # 获取视频ID用于查找原始图片
    video_id = meta.get('video_id', None)
    if video_id is None:
        video_name = meta.get('video_name', os.path.basename(video_data_dir))
        video_id = video_name.replace('video_', '')

    # 加载第一个 joblib 文件获取边连接信息
    frame_files = sorted(glob.glob(os.path.join(video_data_dir, 'frame_*.joblib')))
    if not frame_files:
        print(f'未找到帧文件: {video_data_dir}')
        return

    first_batch = joblib.load(frame_files[0])
    edges = _make_edges(bidirectional_edges)
    E = len(edges)
    n_nodes = first_batch['n_nodes']

    # 构建帧到高亮边的映射
    frame_to_highlight_edges = {}
    if select_edge_list is not None:
        for global_eidx in select_edge_list:
            frame_rank = global_eidx // E
            local_eidx = global_eidx % E
            frame_to_highlight_edges.setdefault(frame_rank, set()).add(local_eidx)
        print(f'高亮边统计: 共 {len(select_edge_list)} 条边，分布在 {len(frame_to_highlight_edges)} 帧中')

    # 加载原始图片文件
    original_frames_path = os.path.join(original_frames_dir, video_id)
    if meta.get('video_dir') and os.path.isdir(meta['video_dir']):
        original_frames_path = meta['video_dir']
    use_original_frames = False
    original_img_files = []

    if os.path.exists(original_frames_path):
        # 尝试多种图片格式
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            img_files = sorted(glob.glob(os.path.join(original_frames_path, ext)))
            if img_files:
                original_img_files = img_files
                use_original_frames = True
                print(f'找到原始图片目录: {original_frames_path}, 共 {len(original_img_files)} 张图片')
                break

    if not use_original_frames:
        print(f'警告: 未找到原始图片目录 {original_frames_path}，将使用黑色背景')

    # 确定视频尺寸（从第一张原始图片或使用默认值）
    if use_original_frames and original_img_files:
        first_img = cv2.imread(original_img_files[0])
        if first_img is not None:
            H, W = first_img.shape[:2]
    else:
        # 使用 metadata 中的尺寸或默认值
        if W <= 0 or H <= 0:
            W, H = 640, 480

    if output_fps is None:
        output_fps = fps

    # 创建输出目录
    output_video_path = None
    if output_video_dir and output_video_name:
        Path(output_video_dir).mkdir(parents=True, exist_ok=True)
        output_video_path = os.path.join(output_video_dir, output_video_name)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_video_path, fourcc, output_fps, (W, H))
        if not writer.isOpened():
            print(f'错误: 无法创建视频文件 {output_video_path}')
            writer = None
        else:
            print(f'创建输出视频: {output_video_path}, 分辨率: {W}x{H}, 帧率: {output_fps} fps')
    else:
        writer = None
        if output_video_dir or output_video_name:
            print('警告: 仅提供了目录或文件名之一，未生成视频。')

    if output_frames_dir:
        Path(output_frames_dir).mkdir(parents=True, exist_ok=True)

    saved_image_count = 0
    saved_video_count = 0

    def get_pts_xy_from_batch(batch):
        """
        从 batch 的 edge_features 中重建节点坐标
        edge_features: [E, 4] = [src_x, src_y, dst_x, dst_y]
        """
        edge_features = batch['edge_features']  # [E, 4]
        sources = batch['sources']  # [E]
        destinations = batch['destinations']  # [E]

        # 初始化节点坐标数组
        pts_xy = np.zeros((n_nodes, 2), dtype=np.float32)
        count = np.zeros(n_nodes, dtype=np.int32)  # 用于平均

        # 从边特征中提取节点坐标
        for i, (s, d) in enumerate(zip(sources, destinations)):
            # src 坐标
            pts_xy[s, 0] += edge_features[i, 0]
            pts_xy[s, 1] += edge_features[i, 1]
            count[s] += 1

            # dst 坐标
            pts_xy[d, 0] += edge_features[i, 2]
            pts_xy[d, 1] += edge_features[i, 3]
            count[d] += 1

        # 平均化（因为每个节点可能出现在多条边中）
        for i in range(n_nodes):
            if count[i] > 0:
                pts_xy[i, 0] /= count[i]
                pts_xy[i, 1] /= count[i]

        # 如果关键点是归一化的，需要转换到像素坐标
        # 检查是否归一化（坐标在 [0, 1] 范围内）
        if np.all(pts_xy >= 0) and np.all(pts_xy <= 1):
            pts_xy[:, 0] *= W
            pts_xy[:, 1] *= H

        return pts_xy

    # 处理每一帧
    for idx, frame_file in enumerate(frame_files):
        batch = joblib.load(frame_file)
        frame_idx = batch.get('frame_index', idx)
        pts = get_pts_xy_from_batch(batch)

        if pts is None:
            continue

        highlight_edges = frame_to_highlight_edges.get(idx, None)
        should_save_image = True
        if select_edge_list is not None and highlight_edges is None:
            should_save_image = False

        # 加载原始图片作为背景
        if use_original_frames and original_img_files:
            # 尝试通过索引匹配
            if frame_idx < len(original_img_files):
                frame = cv2.imread(original_img_files[frame_idx])
            elif idx < len(original_img_files):
                # 如果 frame_idx 超出范围，使用 idx
                frame = cv2.imread(original_img_files[idx])
            else:
                # 如果都超出范围，使用最后一张图片
                frame = cv2.imread(original_img_files[-1])

            if frame is None:
                # 如果图片加载失败，创建黑色背景
                frame = np.zeros((H, W, 3), dtype=np.uint8)
            else:
                # 如果图片尺寸不匹配，调整尺寸
                if frame.shape[0] != H or frame.shape[1] != W:
                    frame = cv2.resize(frame, (W, H))
        else:
            # 创建黑色背景
            frame = np.zeros((H, W, 3), dtype=np.uint8)

        # 在背景上绘制骨架
        _draw_skeleton(
            frame, pts,
            draw_points=draw_points,
            draw_edges=draw_edges,
            highlight_edges=highlight_edges,
            highlight_edge_color=highlight_edge_color,
            highlight_thickness=highlight_thickness
        )

        # 写入视频
        if writer is not None:
            writer.write(frame)
            saved_video_count += 1

        # 保存单帧图片
        if output_frames_dir and should_save_image:
            frame_filename = f'frame_{frame_idx:06d}.jpg'
            cv2.imwrite(os.path.join(output_frames_dir, frame_filename), frame)
            saved_image_count += 1

    # 释放资源
    if writer is not None:
        writer.release()

    total_frames = len(frame_files)
    if select_edge_list is not None:
        selected_frames = len(frame_to_highlight_edges)
        print(f'可视化完成。输出: {output_video_path or output_frames_dir}')
        print(f'视频: 保存了 {saved_video_count} 帧（所有帧）')
        print(f'图片: 保存了 {saved_image_count} 帧（包含选中边的帧），共 {total_frames} 帧中的 {selected_frames} 帧')
    else:
        print(f'可视化完成。输出: {output_video_path or output_frames_dir}')
        print(f'视频: 保存了 {saved_video_count} 帧')
        print(f'图片: 保存了 {saved_image_count} 帧')

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def kl_divergence_np(p, q, eps=1e-12):
    """Compute KL(p || q) for probability vectors."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))

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

def select_important_edges(select_number, edges_dict, target_logits, verbose=False):
    edge_selected = cvx.Variable(len(edges_dict), integer=True)
    sort_key_list = list(edges_dict.keys())

    # print('sort_key_list',sort_key_list)
    tmp_logits = 0

    for i in range(len(sort_key_list)):
        # print(edges_dict[sort_key_list[i]])
        tmp_logits = tmp_logits + edge_selected[i] * edges_dict[sort_key_list[i]].detach().numpy()

    # print('target_logits',target_logits.detach().numpy())
    tmp_logits = tmp_logits/ target_logits.shape[0]

    target_prob = softmax(target_logits.detach().numpy())

    # print(len(target_prob))
    # # print(tmp_logits.shape)

    d = 0
    for i in range(0, len(target_prob)):
        d = d + tmp_logits[i] * target_prob[i]



    objective = cvx.Minimize(-d+cvx.atoms.log_sum_exp(tmp_logits))

    constraints = [sum(edge_selected) == select_number]

    for i in range(len(sort_key_list)):
        constraints.append(0 <= edge_selected[i])
        constraints.append(edge_selected[i] <= 1)
    prob = cvx.Problem(objective, constraints)
    # prob.solve(solver='MOSEK')

    optimal_value = prob.solve(solver='MOSEK')
    if verbose:
        print("Optimal value", optimal_value)
    # print("Optimal var")
    # print('x.value',edge_selected.value)

    edge_res = []

    for i in range(len(sort_key_list)):
        edge_res.append(
            edge_selected[i].value)

    sorted_id = sorted(range(len(edge_res)), key=lambda k: edge_res[k], reverse=True)

    select_edges_list = []

    for i in range(select_number):
        # print(edge_res[sorted_id[i]])
        select_edges_list.append(sort_key_list[sorted_id[i]])

    return select_edges_list

def filter_super_batch_by_edge_indices(super_batch, selected_edges):
    """
    仅保留 super_batch 中 edge_idxs 属于 selected_edges 集合的样本。
    super_batch: dict with keys ['sources','destinations','timestamps','edge_idxs']
                 每个 value 为等长的一维 numpy 数组
    selected_edges: 可迭代的边id集合（list/set/np.array）
    返回：同结构的过滤后 super_batch；若无保留样本则四个数组均为空
    """
    import numpy as np

    src = np.asarray(super_batch['sources'])
    dst = np.asarray(super_batch['destinations'])
    ts  = np.asarray(super_batch['timestamps'])
    eids= np.asarray(super_batch['edge_idxs'])

    if len(eids) == 0:
        return {
            'sources': src[:0],
            'destinations': dst[:0],
            'timestamps': ts[:0],
            'edge_idxs': eids[:0],
            'test_true': 0,
        }

    sel = np.isin(eids, list(selected_edges))
    # print('sel_test',sel)
    if not np.any(sel):
        return {'sources': src[:0], 'destinations': dst[:0], 'timestamps': ts[:0], 'edge_idxs': eids[:0], 'test_true': np.sum(sel)}

    return {
        'sources':      src[sel],
        'destinations': dst[sel],
        'timestamps':   ts[sel],
        'edge_idxs':    eids[sel],
        'test_true': np.sum(sel)
    }

def filter_super_batch_excluding_edge_indices(super_batch, excluded_edges):
    import numpy as np
    src = np.asarray(super_batch['sources'])
    dst = np.asarray(super_batch['destinations'])
    ts  = np.asarray(super_batch['timestamps'])
    eids= np.asarray(super_batch['edge_idxs'])
    if len(eids) == 0:
        return {'sources': src[:0], 'destinations': dst[:0], 'timestamps': ts[:0], 'edge_idxs': eids[:0]}
    keep = ~np.isin(eids, list(excluded_edges))  # 关键：取反，保留不在选中集合的边
    if not np.any(keep):
        return {'sources': src[:0], 'destinations': dst[:0], 'timestamps': ts[:0], 'edge_idxs': eids[:0]}
    return {'sources': src[keep], 'destinations': dst[keep], 'timestamps': ts[keep], 'edge_idxs': eids[keep]}

def visualize_video_with_keypoints_from_folder(
        video_dir: str,
        output_video_dir: str = None,     # 新：视频保存目录
        output_video_name: str = None,    # 新：视频文件名（如 'xxx.mp4'）
        output_frames_dir: str = None,
        input_video_path: str = None,
        use_keypoints_pixel_first: bool = True,
        draw_points: bool = True,
        draw_edges: bool = True,
        select_edge_list: List[int] = None,
        bidirectional_edges: bool = True,
        highlight_edge_color: tuple = (0, 255, 255),
        highlight_thickness: int = 5,
        original_video_root: str = 'data/hmdb51_data/hmdb51_sta',
        output_fps: float = None
):
    """
    ...
    - output_video_dir: 视频输出目录。如果提供，将自动创建
    - output_video_name: 输出文件名（含扩展名），必须在提供 output_video_dir 时指定
    - output_frames_dir: 导出逐帧 JPG 的目录（同样会自动创建）
    ...
    """
    assert os.path.isdir(video_dir), f'not found: {video_dir}'
    meta_path = os.path.join(video_dir, 'metadata.json')
    assert os.path.exists(meta_path), f'not found: {meta_path}'
    meta = _load_json(meta_path)

    # 读取视频信息
    video_info = meta.get('video_info', {})
    W = int(video_info.get('width', 0))
    H = int(video_info.get('height', 0))
    fps = int(video_info.get('fps', 25))

    edges = _make_edges(bidirectional_edges)
    E = len(edges)

    # print('E',E)
    # print('edges',edges)

    frame_to_highlight_edges = {}
    if select_edge_list is not None:
        for global_eidx in select_edge_list:
            frame_rank = global_eidx // E
            local_eidx = global_eidx % E
            frame_to_highlight_edges.setdefault(frame_rank, set()).add(local_eidx)
        print(f'高亮边统计: 共 {len(select_edge_list)} 条边，分布在 {len(frame_to_highlight_edges)} 帧中')


    # print('frame_to_highlight_edges',frame_to_highlight_edges)

    # 原始视频定位
    video_path = input_video_path or meta.get('video_path', None)
    if video_path and not os.path.exists(video_path):
        video_name = meta.get('video_name', os.path.basename(video_dir))
        cls = meta.get('class', os.path.basename(os.path.dirname(video_dir)))
        video_path = os.path.join(original_video_root, cls, f'{video_name}.avi')
        if not os.path.exists(video_path):
            video_path = None
    if video_path is None:
        cls = os.path.basename(os.path.dirname(video_dir))
        video_name = os.path.basename(video_dir)
        video_path = os.path.join(original_video_root, cls, f'{video_name}.avi')
        if not os.path.exists(video_path):
            video_path = None

    cap = None
    writer = None

    frame_jsons = _collect_frames_json(video_dir)
    if not frame_jsons:
        print(f'未找到帧 JSON: {video_dir}')
        return

    def get_pts_xy(fj):
        if use_keypoints_pixel_first and fj.get('keypoints_pixel') is not None:
            kps = fj['keypoints_pixel']
            if isinstance(kps[0], dict):
                pts = np.array([[kp['x'], kp['y']] for kp in kps], dtype=np.float32)
            else:
                arr = np.asarray(kps, dtype=np.float32)
                pts = arr[:, :2]
            return pts
        kpsn = fj.get('keypoints_normalized', None)
        if kpsn is None:
            return None
        if isinstance(kpsn[0], dict):
            pts = np.array([[kp['x'] * W, kp['y'] * H] for kp in kpsn], dtype=np.float32)
        else:
            arr = np.asarray(kpsn, dtype=np.float32)
            pts = np.stack([arr[:, 0] * W, arr[:, 1] * H], axis=1)
        return pts

    if video_path and os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        if W == 0 or H == 0:
            W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if output_fps is None:
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            fps = max(1, int(original_fps)) if original_fps > 0 else fps
        else:
            fps = output_fps
        print(f'使用原始视频: {video_path}')
        print(f'原始帧率: {cap.get(cv2.CAP_PROP_FPS):.2f} fps, 输出帧率: {fps} fps')
    else:
        print(f'未找到原始视频，将在空白画布上绘制。')
        if video_path:
            print(f'  尝试的路径: {video_path}')
        fps = output_fps or max(1, fps)

    output_video_path = None
    if output_video_dir and output_video_name:
        Path(output_video_dir).mkdir(parents=True, exist_ok=True)
        output_video_path = os.path.join(output_video_dir, output_video_name)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        if W <= 0 or H <= 0:
            W = max(W, 640)
            H = max(H, 480)
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (W, H))
        print(f'创建输出视频: {output_video_path}, 分辨率: {W}x{H}, 帧率: {fps} fps')
    elif output_video_dir or output_video_name:
        print('警告: 仅提供了目录或文件名之一，未生成视频。')

    if output_frames_dir:
        Path(output_frames_dir).mkdir(parents=True, exist_ok=True)

    saved_image_count = 0
    saved_video_count = 0

    # print('frame_jsons',frame_jsons)

    for idx, fj in enumerate(frame_jsons):
        pts = get_pts_xy(fj)
        if pts is None:
            if cap is not None:
                ret, _ = cap.read()
                if not ret:
                    break
            if writer is not None:
                blank_frame = np.zeros((H, W, 3), dtype=np.uint8)
                writer.write(blank_frame)
                saved_video_count += 1
            continue

        highlight_edges = frame_to_highlight_edges.get(idx, None)
        should_save_image = True
        if select_edge_list is not None and highlight_edges is None:
            should_save_image = False

        if cap is not None:
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((H, W, 3), dtype=np.uint8)
        else:
            frame = np.zeros((H, W, 3), dtype=np.uint8)

        # print('frame',frame)
        # print('highlight_edges',highlight_edges)

        _draw_skeleton(
            frame, pts,
            draw_points=draw_points,
            draw_edges=draw_edges,
            highlight_edges=highlight_edges,
            highlight_edge_color=highlight_edge_color,
            highlight_thickness=highlight_thickness
        )

        if writer is not None:
            writer.write(frame)
            saved_video_count += 1

        if output_frames_dir and should_save_image:
            cv2.imwrite(os.path.join(output_frames_dir, f'frame_{idx:06d}.jpg'), frame)
            saved_image_count += 1

    if cap is not None:
        cap.release()
    if writer is not None:
        writer.release()

    total_frames = len(frame_jsons)
    if select_edge_list is not None:
        selected_frames = len(frame_to_highlight_edges)
        print(f'可视化完成。输出: {output_video_path or output_frames_dir}')
        print(f'视频: 保存了 {saved_video_count} 帧（所有帧）')
        print(f'图片: 保存了 {saved_image_count} 帧（包含选中边的帧），共 {total_frames} 帧中的 {selected_frames} 帧')
    else:
        print(f'可视化完成。输出: {output_video_path or output_frames_dir}')
        print(f'视频: 保存了 {saved_video_count} 帧')
        print(f'图片: 保存了 {saved_image_count} 帧')

def visualize_selected_edges_for_dataset(args, root_dir, keypoint_root, index_key, video_idx, selection_key, select_edge_list):
    if not args.enable_visualization or args.visualization_type == 'none':
        return

    dataset_name = sanitize_filename_part(args.result_name or args.dataset)
    output_video_dir = f'results/visualizations/{dataset_name}/generate_video/{selection_key}'
    output_frames_dir = f'results/visualizations/{dataset_name}/frames/{selection_key}/{video_idx}'

    if args.visualization_type == 'penn_joblib':
        video_data_dir = os.path.join(root_dir, index_key)
        visualize_video_with_keypoints_from_joblib(
            video_data_dir=video_data_dir,
            output_video_dir=output_video_dir,
            output_video_name=f'{video_idx}.mp4',
            output_frames_dir=output_frames_dir,
            original_frames_dir=args.original_frames_dir,
            draw_points=True,
            draw_edges=True,
            select_edge_list=select_edge_list,
            bidirectional_edges=True,
            highlight_edge_color=(0, 255, 255),
            highlight_thickness=5,
            output_fps=args.visualization_output_fps,
        )
        return

    if args.visualization_type == 'hmdb_folder':
        video_dir = os.path.join(keypoint_root, index_key)
        visualize_video_with_keypoints_from_folder(
            video_dir=video_dir,
            output_video_dir=output_video_dir,
            output_video_name=f'{video_idx}.mp4',
            output_frames_dir=output_frames_dir,
            input_video_path=None,
            use_keypoints_pixel_first=True,
            draw_points=True,
            draw_edges=True,
            select_edge_list=select_edge_list,
            bidirectional_edges=True,
            highlight_edge_color=(0, 255, 255),
            highlight_thickness=5,
            original_video_root=args.original_video_root,
            output_fps=args.visualization_output_fps,
        )
        return

    raise ValueError(f'Unsupported visualization_type: {args.visualization_type}')
if __name__ == "__main__":

    parser = argparse.ArgumentParser('TGN Video Classification Explanation')
    parser.add_argument('--config', type=str, default='configs/explain_video_penn.json',
                        help='Path to a JSON config file. Config values override parser defaults.')
    parser.add_argument('--dataset', type=str, default='penn',
                        help='Dataset name, e.g. penn or hmdb51.')
    parser.add_argument('--classes_list', type=str, nargs='+', default=None,
                        help='Class names to explain. If omitted, classes are discovered from --data.')
    parser.add_argument('--skeleton', type=str, default='penn_action',
                        choices=sorted(SKELETONS.keys()), help='Skeleton layout used by this dataset.')
    parser.add_argument('--n_nodes', type=int, default=None,
                        help='Expected node count. If omitted, inferred from the skeleton/data.')
    parser.add_argument('-d', '--data', type=str, help='Dataset root directory',
                        default='data/penn_action_processed')
    parser.add_argument('--bs', type=int, default=50, help='Batch_size')
    parser.add_argument('--prefix', type=str, default='penn_action', help='Prefix to name the checkpoints')
    parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
    parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
    parser.add_argument('--n_layer', type=int, default=2, help='Number of network layers')
    parser.add_argument('--drop_out', type=float, default=0.5, help='Dropout probability')
    parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
    parser.add_argument('--node_dim', type=int, default=100, help='Dimensions of the node embedding')
    parser.add_argument('--time_dim', type=int, default=100, help='Dimensions of the time embedding')
    parser.add_argument('--use_memory', action='store_true', default='True',
                        help='Whether to augment the model with a node memory')
    parser.add_argument('--embedding_module', type=str, default="graph_attention", choices=[
        "graph_attention", "graph_sum", "identity", "time"], help='Type of embedding module')
    parser.add_argument('--message_function', type=str, default="identity", choices=[
        "mlp", "identity"], help='Type of message function')
    parser.add_argument('--aggregator', type=str, default="last", choices=["last", "mean"],
                        help='Type of message aggregator')
    parser.add_argument('--memory_updater', type=str, default="gru", choices=["gru", "rnn"],
                        help='Type of memory updater')
    parser.add_argument('--memory_update_at_end', action='store_true',
                        help='Whether to update memory at the end or at the start of the batch')
    parser.add_argument('--message_dim', type=int, default=32, help='Dimensions of the messages')
    parser.add_argument('--memory_dim', type=int, default=32, help='Dimensions of the memory for each user')
    parser.add_argument('--use_destination_embedding_in_message', action='store_true',
                        help='Whether to use the embedding of the destination node as part of the message')
    parser.add_argument('--use_source_embedding_in_message', action='store_true',
                        help='Whether to use the embedding of the source node as part of the message')
    parser.add_argument('--train_T', type=int, default=4, help='Window size for training')
    parser.add_argument('--keypoint_root', type=str, default='data/penn_action_keypoints',
                        help='Root directory for keypoint JSON files')
    parser.add_argument('--uniform', action='store_true',
                        help='take uniform sampling from temporal neighbors')
    parser.add_argument('--new_node', action='store_true', help='model new node')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--edge_selection_mode', type=str, default='ratio',
                        choices=['ratio', 'given_number'],
                        help='Select edge count by ratio or by explicitly provided counts')
    parser.add_argument('--select_edge_ratio', type=float, nargs='+',
                        default=[0.01, 0.02, 0.03, 0.04, 0.05],
                        help='Ratios used when edge_selection_mode=ratio')
    parser.add_argument('--given_select_numbers', type=int, nargs='+', default=[],
                        help='Explicit edge counts used when edge_selection_mode=given_number')
    parser.add_argument('--max_depth', type=int, default=2,
                        help='Maximum depth for tracing memory contributions')
    parser.add_argument('--backtrace_child_prune_ratio', type=float, default=1.0,
                        help='Keep top ratio of child records for deeper memory backtracking. 1.0 disables pruning.')
    parser.add_argument('--verbose_debug', action='store_true',
                        help='Print detailed tensor/contribution debug logs.')
    parser.add_argument('--result_name', type=str, default='penn_action',
                        help='Base name for the output result JSON')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Train split ratio used before selecting explanation videos.')
    parser.add_argument('--test_ratio', type=float, default=0.3,
                        help='Test split ratio used before selecting explanation videos.')
    parser.add_argument('--split_seed', type=int, default=42,
                        help='Random seed for train/test split.')
    parser.add_argument('--balance_train_data', action='store_true',
                        help='Whether to balance train data by undersampling before explanation.')
    parser.add_argument('--device', type=str, default='cpu',
                        help='cpu, cuda, cuda:N, or auto.')
    parser.add_argument('--video_indices', type=int, nargs='+', default=None,
                        help='Filtered train item indices to explain. Default: first item.')
    parser.add_argument('--video_keys', type=str, nargs='+', default=None,
                        help='Exact class/video keys to explain. Overrides --video_indices.')
    parser.add_argument('--enable_visualization', action='store_true',
                        help='Generate highlighted videos/frames for selected edges.')
    parser.add_argument('--visualization_type', type=str, default='none',
                        choices=['none', 'penn_joblib', 'hmdb_folder'],
                        help='Visualization backend.')
    parser.add_argument('--original_frames_dir', type=str, default='data/Penn_Action/frames',
                        help='Penn Action original frames root.')
    parser.add_argument('--original_video_root', type=str, default='data/hmdb51_data/hmdb51_sta',
                        help='HMDB original video root.')
    parser.add_argument('--visualization_output_fps', type=float, default=10,
                        help='FPS for generated visualization videos.')

    config_probe, _ = parser.parse_known_args()
    config = load_config(config_probe.config)
    loaded_config_defaults = apply_config_defaults(parser, config)

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    if loaded_config_defaults:
        print(f'Loaded config from {args.config}: {sorted(loaded_config_defaults.keys())}')
    for debug_module in (
        tgn_debug_module,
        utils_debug_module,
        attribution_debug_module,
        backtracking_debug_module,
        embedding_refactored_debug_module,
    ):
        if hasattr(debug_module, 'DEBUG_VERBOSE'):
            debug_module.DEBUG_VERBOSE = args.verbose_debug
    print(
        f'Effective explain config: config={args.config}, dataset={args.dataset}, '
        f'data={args.data}, skeleton={args.skeleton}, max_depth={args.max_depth}, '
        f'backtrace_child_prune_ratio={args.backtrace_child_prune_ratio}, '
        f'verbose_debug={args.verbose_debug}, '
        f'aggregator={args.aggregator}, embedding_module={args.embedding_module}, '
        f'n_layer={args.n_layer}, memory_updater={args.memory_updater}',
        flush=True
    )

    BATCH_SIZE = args.bs
    NUM_NEIGHBORS = args.n_degree
    NUM_NEG = 1
    NUM_HEADS = args.n_head
    DROP_OUT = args.drop_out
    GPU = args.gpu
    UNIFORM = args.uniform
    NEW_NODE = args.new_node
    SEQ_LEN = NUM_NEIGHBORS
    DATA = args.data
    NUM_LAYER = args.n_layer
    LEARNING_RATE = args.lr
    NODE_LAYER = 1
    NODE_DIM = args.node_dim
    TIME_DIM = args.time_dim
    USE_MEMORY = args.use_memory
    MESSAGE_DIM = args.message_dim
    MEMORY_DIM = args.memory_dim
    TRAIN_T = args.train_T

    Path("./saved_models/").mkdir(parents=True, exist_ok=True)
    Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler('log/{}.log'.format(str(time.time())))
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(args)

    set_current_skeleton(args.skeleton)
    classes_list = parse_classes_list(args.classes_list)
    if classes_list is None:
        classes_list = [
            d for d in sorted(os.listdir(args.data))
            if os.path.isdir(os.path.join(args.data, d))
        ]
    ROOT = args.data
    # CLASSES = ['pullup']#['pullup', 'climb', 'run', 'walk', 'situp']  # 或 None 自动发现
    all_data = load_all_videos_no_split(ROOT, classes_list)
    print(f'视频数: {len(all_data)}')
    if not all_data:
        raise RuntimeError(f'No videos loaded from data={ROOT}, classes={classes_list}')

    # device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    train_data, test_data, stats = split_dataset_by_class(
        all_data,
        train_ratio=args.train_ratio,
        test_ratio=args.test_ratio,
        seed=args.split_seed
    )
    # 1. 创建一个全局共享的 TGN（使用第一个视频的结构初始化）
    class_to_label = {c: i for i, c in enumerate(classes_list)}

    if args.balance_train_data:
        train_data, _ = balance_dataset_by_undersampling(
            train_data,
            class_to_label,
            seed=args.split_seed
        )



    class_distribution = {}
    for key in train_data.keys():
        cls = key.split('/')[0]
        class_distribution[cls] = class_distribution.get(cls, 0) + 1
    for cls, count in sorted(class_distribution.items()):
        print(f"  {cls} (label {class_to_label[cls]}): {count} 个视频")
    print("=" * 50)

    first_key = next(iter(train_data.keys()))
    first_batches = train_data[first_key]

    # 获取全局 edge_features（第一个视频，用于初始化维度）
    builder = TemporalGraphDataLoader(device=device)
    first_neighbor_finder = builder.create_neighbor_finder(first_batches)

    edge_feat_dim = int(first_batches[0]['edge_features'].shape[1])
    n_nodes = int(first_batches[0]['n_nodes'])
    inferred_n_nodes = infer_n_nodes(args.skeleton)
    if args.n_nodes is not None and args.n_nodes != n_nodes:
        raise ValueError(f'Config n_nodes={args.n_nodes} but loaded data n_nodes={n_nodes}')
    if inferred_n_nodes != n_nodes:
        print(
            f'警告: skeleton={args.skeleton} 推断节点数为 {inferred_n_nodes}, '
            f'但数据中 n_nodes={n_nodes}。将以数据为准初始化模型。',
            flush=True
        )

    # 收集第一个视频的全局 edge_features（用于初始化 TGN）
    edge_idxs_list = []
    edge_feats_list = []
    for b in first_batches:
        edge_idxs_list.append(np.asarray(b['edge_idxs'], dtype=np.int64))
        edge_feats_list.append(np.asarray(b['edge_features'], dtype=np.float32))
    edge_idxs_cat = np.concatenate(edge_idxs_list, axis=0)
    edge_feats_cat = np.concatenate(edge_feats_list, axis=0)
    order = np.argsort(edge_idxs_cat, kind='mergesort')
    first_edge_features = edge_feats_cat[order]

    edge_feat_input_dim = 4  # [x_src, y_src, x_dst, y_dst]
    edge_feat_output_dim = args.message_dim
    edge_projector = EdgeFeatureProjector(edge_feat_input_dim, edge_feat_output_dim).to(device)

    node_feat_input_dim = n_nodes  # 13 (one-hot编码)
    node_feat_output_dim = args.message_dim
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
        n_layers=NUM_LAYER,
        n_heads=NUM_HEADS,
        dropout=DROP_OUT,
        use_memory=USE_MEMORY,
        message_dimension=MESSAGE_DIM,
        memory_dimension=MEMORY_DIM,
        memory_update_at_start=not args.memory_update_at_end,
        memory_updater_type=args.memory_updater,
        embedding_module_type=args.embedding_module,
        message_function=args.message_function,
        aggregator_type=args.aggregator,
        n_neighbors=NUM_NEIGHBORS,
        use_destination_embedding_in_message=args.use_destination_embedding_in_message,
        use_source_embedding_in_message=args.use_source_embedding_in_message
    ).to(device)

    out_dim = len(classes_list)
    decoder = MLP(MEMORY_DIM, out_dim, drop=DROP_OUT)
    decoder = decoder.to(device)

    target_nodes = list(range(n_nodes))
    print(f"目标节点: {target_nodes}")

    video_global_edge_features = {}
    valid_keys = []
    removed_keys = []

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

    RUN_NAME = build_video_model_run_name(args)
    MODEL_SAVE_PATH = f'saved_models/{RUN_NAME}-tgn-classification.pth'
    DECODER_SAVE_PATH = f'saved_models/{RUN_NAME}-decoder-classification.pth'
    EDGE_PROJECTOR_PATH = f'saved_models/{RUN_NAME}-edge_projector.pth'
    NODE_PROJECTOR_PATH = f'saved_models/{RUN_NAME}-node_projector.pth'
    print(f'加载模型 run_name: {RUN_NAME}')
    print(f'加载 TGN: {MODEL_SAVE_PATH}')
    print(f'加载 decoder: {DECODER_SAVE_PATH}')

    tgn.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    decoder.load_state_dict(torch.load(DECODER_SAVE_PATH, map_location=device))
    edge_projector.load_state_dict(torch.load(EDGE_PROJECTOR_PATH, map_location=device))
    node_projector.load_state_dict(torch.load(NODE_PROJECTOR_PATH, map_location=device))

    tgn.eval()
    decoder = decoder.eval()
    edge_projector.eval()
    node_projector.eval()

    train_items = list(train_data_filtered.items())
    selected_train_items = select_video_items(
        train_items,
        video_indices=args.video_indices,
        video_keys=args.video_keys,
    )
    print(
        f'[stage] Selected videos for explanation: {len(selected_train_items)} '
        f'of {len(train_items)} filtered train videos',
        flush=True
    )

    max_depth = args.max_depth
    backtrace_child_prune_ratio = args.backtrace_child_prune_ratio
    select_edge_ratio = args.select_edge_ratio
    given_select_numbers = args.given_select_numbers

    KEYPOINT_ROOT = args.keypoint_root

    save_result_dict = dict()


    for video_idx, (index_key, batches) in enumerate(selected_train_items):
        # print('video_idx', video_idx)
        # print('key',index_key)
        #

        cls = index_key.split('/')[0]
        label = class_to_label[cls]
        label_t = torch.tensor([label], dtype=torch.long, device=device)
        windows = make_windows(batches, TRAIN_T)
        print(
            f'[video] idx={video_idx} key={index_key} class={cls} label={label} '
            f'frames={len(batches)} windows={len(windows)} train_T={TRAIN_T}',
            flush=True
        )


        builder = TemporalGraphDataLoader(device=device)
        tgn.neighbor_finder = builder.create_neighbor_finder(batches)

        # current_edge_features = torch.from_numpy(
        #     video_global_edge_features[index_key].astype(np.float32)
        # ).to(device)

        raw_edge_features = torch.from_numpy(
            video_global_edge_features[index_key].astype(np.float32)
        ).to(device)  # [E, 4]

        # 通过投影层投影到 32 维
        projected_edge_features = edge_projector(raw_edge_features)  # [E, 32]

        tgn.edge_raw_features = projected_edge_features
        tgn.embedding_module.edge_features = projected_edge_features

        tgn.edge_raw_embed = projected_edge_features


        node_features_static_raw = torch.from_numpy(np.eye(n_nodes, dtype=np.float32)).to(device)  # [17, 17] one-hot
        projected_node_features = node_projector(node_features_static_raw)  # [17, 32]
        tgn.node_raw_features = projected_node_features
        tgn.embedding_module.node_features = projected_node_features
        tgn.node_raw_embed = projected_node_features


        message_dict = dict()
        memory_dict = dict()
        message_trace_dict = dict()

        if tgn.use_memory:
            tgn.memory.__init_memory__()

        target_time=0
        target_time_verify=0
        total_num_edge=0

        if len(windows)<7:
            print(f'[stage] Short video window count: windows={len(windows)}', flush=True)
        print(f'[stage] Building memory history from temporal windows...', flush=True)
        window_log_every = max(1, len(windows) // 10)
        processed_windows = 0
        processed_events = 0
        for window_idx, window in enumerate(windows, start=1):
            super_batch = build_window_super_batch_dynamic_edges(window)
            # print('super_batch',super_batch)
            sources = super_batch['sources']
            # print('sources',sources)
            destinations = super_batch['destinations']
            timestamps = super_batch['timestamps']
            edge_idxs = super_batch['edge_idxs']
            neg = np.zeros_like(sources)
            processed_windows += 1
            processed_events += len(sources)

            if len(windows) <= 10 or window_idx == 1 or window_idx == len(windows) or window_idx % window_log_every == 0:
                print(
                    f'[stage] Memory history progress: window={window_idx}/{len(windows)} '
                    f'events={len(sources)} cumulative_events={processed_events}',
                    flush=True
                )

            if len(sources)>0:
                _, _, _, _, \
                    _, _, _, _ = tgn.compute_temporal_embeddings(
                    sources,
                    destinations,
                    neg,  # for memory update
                    timestamps,
                    edge_idxs, message_dict, memory_dict, message_trace_dict,
                    n_neighbors=args.n_degree
                )

                target_time = max(timestamps)

                target_time_verify = max(timestamps)

                total_num_edge = max(edge_idxs)




        nodes_np = torch.tensor(target_nodes)
        dummy_eidx = np.zeros_like(nodes_np, dtype=np.int64)


        ts_np = np.full(len(target_nodes), target_time, dtype=np.int64)

        print(f'[stage] Built memory history from {len(windows)} windows for video={index_key}', flush=True)

        print(
            '[stage] Computing topology attribution inputs '
            '(node embeddings, memory contributions, neighbor contributions, temporal edge contributions)...',
            flush=True
        )


        # source_embedding, _, _ = tgn.compute_temporal_embeddings_without_contributions(
        #     nodes_np, nodes_np, nodes_np, ts_np, dummy_eidx, NUM_NEIGHBORS
        # )

        source_node_embedding, destination_node_embedding, negative_node_embedding, C_memory_features, \
            C_neighbor_memory_features, temporal_edge_contributions, \
            sample_neighbors, sample_neighbor_edgeidx = tgn.compute_temporal_embeddings(
            nodes_np,
            nodes_np,
            nodes_np,  # for memory update
            ts_np,
            dummy_eidx, message_dict, memory_dict, message_trace_dict,
            n_neighbors=args.n_degree
        )
        print('[stage] Topology attribution inputs computed', flush=True)

        output, contributions = decoder.forward_with_contributions(source_node_embedding, decoder)

        original_prob = softmax(output.mean(dim=0).detach().numpy())
        pred_label= np.argmax(original_prob)
        print(
            f'[stage] Original prediction ready: label={label_t.item()} '
            f'pred={pred_label} prob_true={original_prob[label_t.item()]:.6f}',
            flush=True
        )

        total_select_edge_dict = dict()
        total_select_edge_dict['original_prob'] = original_prob
        total_select_edge_dict['original_logits'] = output.mean(dim=0).detach().numpy()
        total_select_edge_dict['label'] = label_t.item()




        final_is_equal = True

        sum_dict = None
        for idx, mat in temporal_edge_contributions.items():
            for _, second_mat in mat.items():
                if sum_dict is None:
                    sum_dict = second_mat.clone()
                else:
                    sum_dict = sum_dict + second_mat

        # print('sum_dict.shape',sum_dict.shape)

        total_contrib = C_memory_features.sum(dim=(0, 1)) + C_neighbor_memory_features.sum(dim=(0, 1, 2)) \
                        + sum_dict

        verify_ground_truth = source_node_embedding.sum(dim=(0)) + destination_node_embedding.sum(
            dim=(0)) + negative_node_embedding.sum(dim=(0))

        if args.verbose_debug:
            print(
                'topology attribution 守恒验证',
                torch.allclose(
                    total_contrib.to(dtype=torch.float64, device=verify_ground_truth.device),
                    verify_ground_truth.to(dtype=torch.float64),
                    atol=1e-4,
                ),
            )


        contrib_all_edges=dict()
        print(
            f'[stage] Computing memory attribution tree: target_nodes={len(nodes_np)} '
            f'max_depth={max_depth} child_prune_ratio={backtrace_child_prune_ratio}',
            flush=True
        )

        for target_idx in range(len(nodes_np)):  #
            target_source_node = nodes_np[target_idx].item()
            if args.verbose_debug:
                print(f'[debug] Attribution target_node={target_source_node} ({target_idx + 1}/{len(nodes_np)})')
                raw_conservation = verify_raw_backtrace_conservation(
                    target_source_node,
                    message_dict,
                    C_memory_features[target_idx],
                    max_depth,
                    child_prune_ratio=1.0,
                    atol=1e-4,
                )
                print(
                    '[debug] raw backtrace conservation',
                    {
                        'matrix_is_identity': raw_conservation['matrix_is_identity'],
                        'projected_is_conserved': raw_conservation['projected_is_conserved'],
                        'matrix_max_abs_diff': raw_conservation['matrix_max_abs_diff'],
                        'projected_max_abs_diff': raw_conservation['projected_max_abs_diff'],
                        'n_edges': raw_conservation['n_edges'],
                    },
                    flush=True,
                )

            edge_source_node_reults = compute_edge_memory_contributions(target_source_node, target_idx, message_dict,
                                                                        C_memory_features[target_idx], max_depth,
                                                                        child_prune_ratio=backtrace_child_prune_ratio,
                                                                        verbose=args.verbose_debug)

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

            is_equal = torch.allclose(final_edge_sum, target_C_memory.sum(dim=0), atol=1e-4)
            if args.verbose_debug:
                print('source edge is_equal', is_equal)

            # if not is_equal:
            #     print(f'final_edge_sum: {final_edge_sum}')
            #     print(f'target_C_memory: {target_C_memory.sum(dim=0)}')

            neighbor_source_node_reults = compute_neighbor_memory_contributions(
                target_source_node, target_idx, message_dict, C_neighbor_memory_features,
                sample_neighbors, sample_neighbor_edgeidx, max_depth,
                child_prune_ratio=backtrace_child_prune_ratio,
                verbose=args.verbose_debug
            )

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

            is_equal = torch.allclose(final_edge_sum, target_C_memory.sum(dim=(0, 1)), atol=1e-4)
            if args.verbose_debug:
                print('neighbor source edge is_equal', is_equal)

            final_source_node_results = merge_contribution_dicts(
                [neighbor_source_node_reults, edge_source_node_reults, temporal_edge_contributions[target_idx]]
            )

            final_edge_sum = None
            for edge_idx, contrib in final_source_node_results.items():
                # print('final edge contrib',contrib.shape)
                if final_edge_sum is None:
                    final_edge_sum = contrib.clone()
                else:
                    final_edge_sum = final_edge_sum + contrib

            target_C_memory = source_node_embedding[target_idx].double()
            if final_edge_sum == None:
                final_edge_sum = torch.zeros_like(target_C_memory)

            is_equal = torch.allclose(final_edge_sum, target_C_memory, atol=1e-4)
            if args.verbose_debug:
                print('final source edge is_equal', is_equal)

            for key, value in final_source_node_results.items():
                value = value.to(dtype=torch.float64)
                target_embedding = source_node_embedding[target_idx].to(dtype=torch.float64)
                contributions_current = contributions[target_idx].to(dtype=torch.float64)
                normalized_value = torch.where(
                    target_embedding != 0,
                    value / target_embedding,
                    torch.zeros_like(value),
                )
                final_value = normalized_value @ contributions_current
                # print('value',value.shape)
                final_source_node_results[key] = final_value

            final_edge_sum = None
            for edge_idx, contrib in final_source_node_results.items():
                # print('edge_idx',edge_idx)
                # print('final edge contrib',contrib.shape)
                if final_edge_sum is None:
                    final_edge_sum = contrib.clone()
                else:
                    final_edge_sum = final_edge_sum + contrib

                if edge_idx not in contrib_all_edges.keys():
                    contrib_all_edges[edge_idx]=contrib
                else:
                    contrib_all_edges[edge_idx] += contrib

            target_C_memory = output[target_idx].double()
            is_equal = torch.allclose(final_edge_sum, target_C_memory, atol=1e-4)
            if args.verbose_debug:
                print('mlp edge is_equal', is_equal)

            # print('output.shape',output.shape)

        final_edge_sum = None
        for edge_idx, contrib in contrib_all_edges.items():
            if final_edge_sum is None:
                final_edge_sum = contrib.clone()
            else:
                final_edge_sum = final_edge_sum + contrib

        is_equal = torch.allclose(final_edge_sum.to(dtype=torch.float32, device=output.device),
                                  output.sum(dim=0).to(dtype=torch.float32, device=output.device),
                                  atol=1e-4)
        print(
            f'[stage] Memory attribution complete: contributed_edges={len(contrib_all_edges)} '
            f'graph_verify={bool(is_equal)}',
            flush=True
        )
        # compare_graph_attention_current_refactored_tgn_with_legacy(
        #     tgn=tgn,
        #     memory=tgn.memory.memory.data.clone(),
        #     nodes=np.asarray(target_nodes, dtype=np.int64),
        #     timestamps=np.full(len(target_nodes), target_time, dtype=np.float64),
        #     n_neighbors=args.n_degree,
        #     atol=1e-5,
        #     rtol=1e-5,
        # )
        selection_specs = []
        if args.edge_selection_mode == 'ratio':
            for ratio in select_edge_ratio:
                select_number = min(math.ceil(ratio * len(contrib_all_edges)), len(contrib_all_edges))
                print(f'[stage] Selecting important edges: ratio={ratio} select_number={select_number}', flush=True)

                select_edge_list = select_important_edges(select_number,
                                                          contrib_all_edges,
                                                          output.mean(dim=0),
                                                          verbose=args.verbose_debug)

                selection_key = str(ratio)
                total_select_edge_dict[f"{selection_key}_select_edge"] = select_edge_list
                selection_specs.append((selection_key, select_edge_list))
        else:
            if not given_select_numbers:
                raise ValueError('edge_selection_mode=given_number requires non-empty given_select_numbers')
            for given_select_number in given_select_numbers:
                select_number = min(int(given_select_number), len(contrib_all_edges))
                print(f'[stage] Selecting important edges: select_number={select_number}', flush=True)

                select_edge_list = select_important_edges(select_number,
                                                          contrib_all_edges,
                                                          output.mean(dim=0),
                                                          verbose=args.verbose_debug)

                selection_key = f'given_{select_number}'
                total_select_edge_dict[f"{selection_key}_select_edge"] = select_edge_list
                selection_specs.append((selection_key, select_edge_list))

        print(f'[stage] Edge selection complete: specs={len(selection_specs)} total_edges={total_num_edge}', flush=True)

        for selection_key, select_edge_list in selection_specs:
                print(
                    f'[stage] Evaluating selected-edge subgraph: key={selection_key} '
                    f'n_selected_edges={len(select_edge_list)}',
                    flush=True
                )

                if args.use_memory:
                    tgn.memory.__init_memory__()

                selected_edges = set(select_edge_list)

                # excluded_edges=set(select_edge_list)
                mask_windows = make_windows(batches, TRAIN_T)
                total_true=0

                for mask_window in mask_windows:
                    super_batch = build_window_super_batch_dynamic_edges(mask_window)
                    # print('super_batch',super_batch)

                    filtered_sb = filter_super_batch_by_edge_indices(super_batch, selected_edges)


                    total_true+= filtered_sb['test_true']

                    # filtered_sb = filter_super_batch_excluding_edge_indices(super_batch, excluded_edges)

                    sources = filtered_sb['sources']
                    # print('sources',sources)
                    destinations = filtered_sb['destinations']
                    timestamps = filtered_sb['timestamps']
                    edge_idxs = filtered_sb['edge_idxs']
                    neg = np.zeros_like(sources)



                    if len(sources)>0:
                        # print('sources', sources)
                        # print('destinations',destinations)
                        # print('edge_idxs',edge_idxs)
                        _, _, _ = tgn.compute_temporal_embeddings_without_contributions(
                            sources,
                            destinations,
                            neg,  # for memory update
                            timestamps,
                            edge_idxs,
                            n_neighbors=args.n_degree
                        )

                nodes_np = torch.tensor(target_nodes)
                dummy_eidx = np.zeros_like(nodes_np, dtype=np.int64)



                ts_np = np.full(len(target_nodes), target_time_verify, dtype=np.int64)

                # source_embedding, _, _ = tgn.compute_temporal_embeddings_without_contributions(
                #     nodes_np, nodes_np, nodes_np, ts_np, dummy_eidx, NUM_NEIGHBORS
                # )

                source_node_embedding, _,_ = tgn.compute_temporal_embeddings_without_contributions(
                    nodes_np,
                    nodes_np,
                    nodes_np,  # for memory update
                    ts_np,
                    dummy_eidx,
                    n_neighbors=args.n_degree
                )

                selected_edges_output = decoder.forward(source_node_embedding)
                selected_edges_output = selected_edges_output.mean(dim=0)

                selected_edges_prob = softmax(selected_edges_output.detach().numpy())
                prob_kl_original_to_selected = kl_divergence_np(original_prob, selected_edges_prob)
                true_class_prob_abs_diff = float(
                    abs(selected_edges_prob[label_t.item()] - original_prob[label_t.item()])
                )
                print(
                    f'[stage] Selected-edge evaluation complete: key={selection_key} '
                    f'matched_edges={total_true} kl={prob_kl_original_to_selected:.6f} '
                    f'true_class_abs_diff={true_class_prob_abs_diff:.6f}',
                    flush=True
                )

                total_select_edge_dict[f'{selection_key}_selected_edges_prob'] = selected_edges_prob
                total_select_edge_dict[f'{selection_key}_selected_edges_logits'] = selected_edges_output.detach().numpy()
                total_select_edge_dict[f'{selection_key}_kl_original_to_selected_edges'] = prob_kl_original_to_selected
                total_select_edge_dict[f'{selection_key}_true_class_prob_abs_diff'] = true_class_prob_abs_diff

                if args.use_memory:
                    tgn.memory.__init_memory__()
                visualize_selected_edges_for_dataset(
                    args=args,
                    root_dir=ROOT,
                    keypoint_root=KEYPOINT_ROOT,
                    index_key=index_key,
                    video_idx=video_idx,
                    selection_key=selection_key,
                    select_edge_list=select_edge_list,
                )

        save_result_dict[video_idx]=total_select_edge_dict
        if args.use_memory:
            tgn.memory.__init_memory__()

    save_path = build_explanation_result_path(args, max_depth)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(convert_numpy_types(save_result_dict), f)
    print(f"Results successfully saved to: {save_path}")
