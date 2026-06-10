import numpy as np
import torch
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import matplotlib.dates as mdates
import datetime
import json
import time
import os
from scipy import sparse
import joblib
import pickle
from pathlib import Path
class Data:
    """
    数据容器类，兼容TGN模型
    """

    def __init__(self, sources, destinations, timestamps, edge_idxs, labels):
        self.sources = sources
        self.destinations = destinations
        self.timestamps = timestamps
        self.edge_idxs = edge_idxs
        self.labels = labels
        self.n_interactions = len(sources)
        self.unique_nodes = set(sources) | set(destinations)
        self.n_unique_nodes = len(self.unique_nodes)

class SimpleNeighborFinder:
    """
    简化的邻居查找器，兼容TGN模型
    """

    def __init__(self, adj_list, uniform=False, seed=None):
        self.adj_list = adj_list
        self.uniform = uniform
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    def get_temporal_neighbor(self, source_nodes, timestamps, n_neighbors=20):
        """
        获取时序邻居
        """
        batch_size = len(source_nodes)
        neighbors = np.zeros((batch_size, n_neighbors), dtype=int)
        edge_idxs = np.zeros((batch_size, n_neighbors), dtype=int)
        edge_times = np.zeros((batch_size, n_neighbors), dtype=float)

        for i, (source, timestamp) in enumerate(zip(source_nodes, timestamps)):
            if source < len(self.adj_list):
                # 获取该节点的所有邻居
                node_neighbors = self.adj_list[source]

                # 过滤时间戳小于等于当前时间的邻居
                valid_neighbors = [
                    (neighbor, edge_idx, edge_time)
                    for neighbor, edge_idx, edge_time in node_neighbors
                    if edge_time <= timestamp
                ]

                # 按时间排序，选择最近的邻居
                valid_neighbors.sort(key=lambda x: x[2], reverse=True)
                selected_neighbors = valid_neighbors[:n_neighbors]

                # 填充结果
                for j, (neighbor, edge_idx, edge_time) in enumerate(selected_neighbors):
                    neighbors[i, j] = neighbor
                    edge_idxs[i, j] = edge_idx
                    edge_times[i, j] = edge_time

        return neighbors, edge_idxs, edge_times

class TemporalGraphDataLoader:
    """
    时序图数据加载器，用于将邻接矩阵和节点时序特征封装成TGN模型可接受的格式
    """

    def __init__(self, device='cpu'):
        self.device = device

    def prepare_daily_batches_from_timerange(self,
                                             adjacency_matrix: np.ndarray,
                                             node_features_dict: Dict[str, Dict[str, List]],
                                             start_time: str,
                                             end_time: str,
                                             runoff_label_dict: Optional[Dict] = None,
                                             runoff_label_normalized_dict: Optional[Dict] = None,
                                             runoff_label_normalized_minmax_dict: Optional[Dict] = None,  # 新增
                                             save_dir: str = 'daily_batches',
                                             save_format: str = 'joblib',
                                             data_type: str = 'train'
                                             ) -> List[Dict]:
        """
        从指定时间范围内准备每日批次数据

        参数:
        - adjacency_matrix: 静态邻接矩阵，形状为 [n_nodes, n_nodes]
        - node_features_dict: 节点特征字典 {node_idx: {date_str: [f1,f2,f3,f4]}}
        - start_time: 开始时间 'YYYY-MM-DD'
        - end_time: 结束时间 'YYYY-MM-DD'
        - runoff_label_dict: 可选的径流标签字典

        返回:
        - daily_batches: 每天的批次数据列表
        """
        save_path = Path(save_dir) / data_type
        save_path.mkdir(parents=True, exist_ok=True)


        print('save_path',save_path)

        daily_batches = []

        # 解析时间范围
        start_date = datetime.datetime.strptime(start_time, '%Y-%m-%d')
        end_date = datetime.datetime.strptime(end_time, '%Y-%m-%d')

        # 获取节点数量
        n_nodes = adjacency_matrix.shape[0]

        print('n_nodes',n_nodes)

        # 从邻接矩阵提取边信息（静态图结构）
        sources, destinations, edge_weights = self._extract_edges_from_adjacency(adjacency_matrix)

        if len(sources) == 0:
            print("警告: 邻接矩阵中没有边")
            return daily_batches

        # 遍历每一天
        current_date = start_date
        day_idx = 0

        while current_date <= end_date:
            date_str = current_date.strftime('%Y-%m-%d')
            timestamp = current_date.timestamp()

            # 收集当天所有节点的特征
            node_features_today = self._get_node_features_for_date(
                node_features_dict, date_str, n_nodes
            )

            print('node_features_today',node_features_today)

            # 如果当天没有足够的节点特征，跳过这一天


            # 创建边索引（每天使用相同的图结构，但时间戳不同）
            edge_idxs = np.arange(len(sources)) + day_idx * len(sources)

            # 创建时间戳数组（所有边使用当天的时间戳）
            edge_timestamps = np.full(len(sources), timestamp)

            # 使用边权重作为边特征
            day_edge_features = edge_weights.reshape(-1, 1) if len(edge_weights) > 0 else np.ones((len(sources), 1))


            # 获取径流标签（如果有的话）
            runoff_labels = None
            if runoff_label_dict:
                runoff_labels = self._get_runoff_labels_for_date(
                    runoff_label_dict, current_date
                )
            print('runoff_labels',runoff_labels)

            if runoff_label_normalized_dict:
                runoff_normalized_labels = self._get_runoff_labels_for_date(
                    runoff_label_normalized_dict, current_date
                )
            print('runoff_labels',runoff_labels)

            runoff_normalized_minmax_labels = None
            if runoff_label_normalized_minmax_dict:
                runoff_normalized_minmax_labels = self._get_runoff_labels_for_date(
                    runoff_label_normalized_minmax_dict, current_date
                )



            # 创建批次数据
            batch_data = {
                'sources': sources.copy(),
                'destinations': destinations.copy(),
                'timestamps': edge_timestamps,
                'edge_idxs': edge_idxs,
                'edge_features': day_edge_features,
                'node_features': node_features_today,
                'runoff_labels': runoff_labels,
                'runoff_labels_normalized': runoff_normalized_labels,
                'runoff_labels_normalized_minmax': runoff_normalized_minmax_labels,  # 新增：min-max归一化

                'date': date_str,
                'day_idx': day_idx,
                'n_nodes': n_nodes,
                'n_edges': len(sources),
                'timestamp': timestamp
            }

            file_path = self._save_batch_immediately(
                batch_data, save_path, data_type, save_format
            )

            daily_batches.append(batch_data)

            # 移动到下一天
            current_date += datetime.timedelta(days=1)
            day_idx += 1

        print(f"成功创建了 {len(daily_batches)} 个每日批次，时间范围: {start_time} 到 {end_time}")
        return daily_batches

    def _get_node_features_for_date(self,
                                    node_features_dict: Dict[str, Dict[str, List]],
                                    date_str: str,
                                    n_nodes: int) -> Optional[np.ndarray]:
        """
        获取指定日期所有节点的特征

        参数:
        - node_features_dict: 节点特征字典
        - date_str: 日期字符串 'YYYY-MM-DD'
        - n_nodes: 节点总数

        返回:
        - node_features: 节点特征矩阵 [n_nodes, n_features] 或 None
        """
        node_features_list = []
        missing_nodes = []

        for node_idx in range(n_nodes):
            node_key = str(node_idx)

            if node_key in node_features_dict and date_str in node_features_dict[node_key]:
                features = node_features_dict[node_key][date_str]
                features = [float(f)  for f in features]
                node_features_list.append(features)
                # 确保特征是数值类型

        return np.array(node_features_list, dtype=np.float32)

    def _get_runoff_labels_for_date(self,
                                    runoff_label_dict: Dict,
                                    date: datetime.datetime) -> Optional[Dict]:
        """
        获取指定日期的径流标签

        参数:
        - runoff_label_dict: 径流标签字典
        - date: 日期对象

        返回:
        - runoff_labels: 径流标签字典或None
        """
        runoff_labels = {}

        for node_idx, time_series in runoff_label_dict.items():
            if date in time_series:
                runoff_labels[node_idx] = time_series[date]

        return runoff_labels if runoff_labels else None

    def _extract_edges_from_adjacency(self, adj_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        从邻接矩阵中提取边信息

        参数:
        - adj_matrix: 邻接矩阵，形状为 [n_nodes, n_nodes]

        返回:
        - sources: 源节点数组
        - destinations: 目标节点数组
        - weights: 边权重数组
        """
        # 找到非零元素的位置
        sources, destinations = np.nonzero(adj_matrix)
        weights = adj_matrix[sources, destinations]

        return sources, destinations, weights

    def create_neighbor_finder(self, daily_batches: List[Dict]) -> 'SimpleNeighborFinder':
        """
        为所有批次数据创建邻居查找器

        参数:
        - daily_batches: 每天的批次数据列表

        返回:
        - neighbor_finder: TGN使用的邻居查找器
        """
        # 收集所有边信息
        all_sources = []
        all_destinations = []
        all_timestamps = []
        all_edge_idxs = []

        for batch in daily_batches:
            all_sources.extend(batch['sources'])
            all_destinations.extend(batch['destinations'])
            all_timestamps.extend(batch['timestamps'])
            all_edge_idxs.extend(batch['edge_idxs'])

        # 转换为numpy数组
        all_sources = np.array(all_sources)
        all_destinations = np.array(all_destinations)
        all_timestamps = np.array(all_timestamps)
        all_edge_idxs = np.array(all_edge_idxs)

        # 创建邻接列表
        max_node_idx = max(all_sources.max(), all_destinations.max())
        adj_list = [[] for _ in range(max_node_idx + 1)]

        for source, destination, edge_idx, timestamp in zip(
                all_sources, all_destinations, all_edge_idxs, all_timestamps
        ):
            adj_list[source].append((destination, edge_idx, timestamp))
            adj_list[destination].append((source, edge_idx, timestamp))

        return SimpleNeighborFinder(adj_list)

    def prepare_tgn_data(self, daily_batches: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """
        准备TGN模型需要的节点特征和边特征

        参数:
        - daily_batches: 每天的批次数据列表

        返回:
        - node_features: 节点特征矩阵（使用最后一天的特征）
        - edge_features: 合并的边特征矩阵
        """
        if not daily_batches:
            raise ValueError("daily_batches不能为空")

        # 使用最后一天的节点特征作为静态特征
        # 或者您可以选择使用平均特征、第一天特征等
        node_features = daily_batches[-1]['node_features']

        # 收集所有边特征
        all_edge_features = []
        for batch in daily_batches:
            all_edge_features.append(batch['edge_features'])

        edge_features = np.vstack(all_edge_features)

        return node_features, edge_features

    def create_data_object(self, batch_data: Dict) -> 'Data':
        """
        为单个批次创建Data对象

        参数:
        - batch_data: 单个批次的数据字典

        返回:
        - data: Data对象
        """
        return Data(
            sources=batch_data['sources'],
            destinations=batch_data['destinations'],
            timestamps=batch_data['timestamps'],
            edge_idxs=batch_data['edge_idxs'],
            labels=np.ones(len(batch_data['sources']))  # 默认标签，可根据需要修改
        )

    def _save_batch_immediately(self,
                                batch_data: Dict,
                                save_path: Path,
                                data_type: str,
                                save_format: str) -> Path:
        """
        立即保存单个批次数据

        参数:
        - batch_data: 批次数据
        - save_path: 保存路径
        - data_type: 数据类型
        - save_format: 保存格式

        返回:
        - file_path: 保存的文件路径
        """

        # 生成文件名
        date_str = batch_data['date']
        day_idx = batch_data['day_idx']
        filename = f"{data_type}_day_{day_idx:04d}_{date_str}"

        try:
            if save_format == 'joblib':
                file_path = save_path / f"{filename}.joblib"
                joblib.dump(batch_data, file_path)

            elif save_format == 'pickle':
                file_path = save_path / f"{filename}.pkl"
                with open(file_path, 'wb') as f:
                    pickle.dump(batch_data, f)

            elif save_format == 'numpy':
                # 创建子目录保存numpy格式数据
                batch_dir = save_path / filename
                batch_dir.mkdir(exist_ok=True)

                # 保存numpy数组
                np.save(batch_dir / 'sources.npy', batch_data['sources'])
                np.save(batch_dir / 'destinations.npy', batch_data['destinations'])
                np.save(batch_dir / 'timestamps.npy', batch_data['timestamps'])
                np.save(batch_dir / 'edge_idxs.npy', batch_data['edge_idxs'])
                np.save(batch_dir / 'edge_features.npy', batch_data['edge_features'])
                np.save(batch_dir / 'node_features.npy', batch_data['node_features'])

                # 保存其他数据为JSON
                metadata = {
                    'date': batch_data['date'],
                    'day_idx': batch_data['day_idx'],
                    'n_nodes': batch_data['n_nodes'],
                    'n_edges': batch_data['n_edges'],
                    'timestamp': batch_data['timestamp']
                }

                if batch_data.get('runoff_labels') is not None:
                    metadata['runoff_labels'] = batch_data['runoff_labels']

                with open(batch_dir / 'metadata.json', 'w') as f:
                    json.dump(metadata, f, indent=2)

                file_path = batch_dir

            else:
                raise ValueError(f"不支持的保存格式: {save_format}")

            return file_path

        except Exception as e:
            print(f"保存批次数据失败 {date_str}: {e}")
            raise

    def get_batches_for_timerange(self,
                                  adjacency_matrix: np.ndarray,
                                  node_features_dict: Dict[str, Dict[str, List]],
                                  start_time: str,
                                  end_time: str,
                                  runoff_label_dict: Optional[Dict] = None,
                                  runoff_label_normalized_dict: Optional[Dict] = None,
                                  runoff_label_normalized_minmax_dict: Optional[Dict] = None,  # 新增
                                  save_dir: str = 'daily_batches',
                                             save_format: str = 'joblib',
                                             data_type: str = 'train') -> Tuple[
        List[Dict], 'SimpleNeighborFinder', np.ndarray, np.ndarray]:
        """
        一站式获取指定时间范围的所有数据

        参数:
        - adjacency_matrix: 邻接矩阵
        - node_features_dict: 节点特征字典
        - start_time: 开始时间
        - end_time: 结束时间
        - runoff_label_dict: 径流标签字典

        返回:
        - daily_batches: 每日批次列表
        - neighbor_finder: 邻居查找器
        - node_features: 节点特征矩阵
        - edge_features: 边特征矩阵
        """
        # 准备每日批次
        daily_batches = self.prepare_daily_batches_from_timerange(
            adjacency_matrix=adjacency_matrix,
            node_features_dict=node_features_dict,
            start_time=start_time,
            end_time=end_time,
            runoff_label_dict=runoff_label_dict,save_dir=save_dir,save_format=save_format,data_type=data_type,runoff_label_normalized_dict=runoff_label_normalized_dict,
            runoff_label_normalized_minmax_dict=runoff_label_normalized_minmax_dict,  # 新增
        )

        if not daily_batches:
            raise ValueError(f"在时间范围 {start_time} 到 {end_time} 内没有找到有效数据")

        # 创建邻居查找器
        neighbor_finder = self.create_neighbor_finder(daily_batches)

        # 准备TGN数据
        node_features, edge_features = self.prepare_tgn_data(daily_batches)

        return daily_batches, neighbor_finder, node_features, edge_features