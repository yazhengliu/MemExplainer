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
class DailyBatchLoader:
    """
    用于加载已保存的每日批次数据的工具类
    """

    def __init__(self, save_dir: str = 'data/geography_daily_data'):
        """
        初始化加载器

        参数:
        - save_dir: 保存数据的目录
        """
        self.save_dir = Path(save_dir)
        self.train_dir = self.save_dir / 'train'
        self.test_dir = self.save_dir / 'test'

    def load_single_batch(self,
                          date_str: str,
                          data_type: str = 'train',
                          save_format: str = 'joblib') -> Optional[Dict]:
        """
        加载单个日期的批次数据

        参数:
        - date_str: 日期字符串 'YYYY-MM-DD'
        - data_type: 数据类型 ('train' 或 'test')
        - save_format: 保存格式 ('joblib', 'pickle', 'numpy')

        返回:
        - batch_data: 批次数据字典或None
        """

        # 选择搜索目录
        search_dir = self.train_dir if data_type == 'train' else self.test_dir

        if not search_dir.exists():
            print(f"目录不存在: {search_dir}")
            return None

        try:
            if save_format == 'joblib':
                # 查找匹配的joblib文件
                pattern = f"{data_type}_day_*_{date_str}.joblib"
                matching_files = list(search_dir.glob(pattern))

                if not matching_files:
                    print(f"未找到日期 {date_str} 的 {data_type} 数据文件")
                    return None

                file_path = matching_files[0]
                batch_data = joblib.load(file_path)
                print(f"成功加载 {date_str} 的 {data_type} 数据: {batch_data['n_edges']} 条边")
                return batch_data

            elif save_format == 'pickle':
                # 查找匹配的pickle文件
                pattern = f"{data_type}_day_*_{date_str}.pkl"
                matching_files = list(search_dir.glob(pattern))

                if not matching_files:
                    print(f"未找到日期 {date_str} 的 {data_type} 数据文件")
                    return None

                file_path = matching_files[0]
                with open(file_path, 'rb') as f:
                    batch_data = pickle.load(f)
                print(f"成功加载 {date_str} 的 {data_type} 数据: {batch_data['n_edges']} 条边")
                return batch_data

            elif save_format == 'numpy':
                # 查找匹配的numpy目录
                pattern = f"{data_type}_day_*_{date_str}"
                matching_dirs = list(search_dir.glob(pattern))

                if not matching_dirs:
                    print(f"未找到日期 {date_str} 的 {data_type} 数据目录")
                    return None

                batch_dir = matching_dirs[0]

                # 加载numpy数组
                batch_data = {
                    'sources': np.load(batch_dir / 'sources.npy'),
                    'destinations': np.load(batch_dir / 'destinations.npy'),
                    'timestamps': np.load(batch_dir / 'timestamps.npy'),
                    'edge_idxs': np.load(batch_dir / 'edge_idxs.npy'),
                    'edge_features': np.load(batch_dir / 'edge_features.npy'),
                    'node_features': np.load(batch_dir / 'node_features.npy')
                }

                # 加载元数据
                with open(batch_dir / 'metadata.json', 'r') as f:
                    metadata = json.load(f)
                    batch_data.update(metadata)

                print(f"成功加载 {date_str} 的 {data_type} 数据: {batch_data['n_edges']} 条边")
                return batch_data

            else:
                raise ValueError(f"不支持的加载格式: {save_format}")

        except Exception as e:
            print(f"加载数据失败 {date_str}: {e}")
            return None

    def load_date_range(self,
                        start_date: str,
                        end_date: str,
                        data_type: str = 'train',
                        save_format: str = 'joblib') -> List[Dict]:
        """
        加载指定日期范围的批次数据

        参数:
        - start_date: 开始日期 'YYYY-MM-DD'
        - end_date: 结束日期 'YYYY-MM-DD'
        - data_type: 数据类型 ('train' 或 'test')
        - save_format: 保存格式

        返回:
        - batches: 批次数据列表
        """

        start_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.datetime.strptime(end_date, '%Y-%m-%d')

        batches = []
        missing_dates = []
        current_dt = start_dt

        print(f"开始加载 {data_type} 数据，日期范围: {start_date} 到 {end_date}")

        while current_dt <= end_dt:
            date_str = current_dt.strftime('%Y-%m-%d')

            batch = self.load_single_batch(date_str, data_type, save_format)

            if batch is not None:
                batches.append(batch)
            else:
                missing_dates.append(date_str)

            current_dt += datetime.timedelta(days=1)

        print(f"成功加载 {len(batches)} 个 {data_type} 批次")
        if missing_dates:
            print(f"缺失的日期: {len(missing_dates)} 个 - {missing_dates[:5]}{'...' if len(missing_dates) > 5 else ''}")

        return batches

    def get_available_dates(self,
                            data_type: str = 'train',
                            save_format: str = 'joblib') -> List[str]:
        """
        获取可用的日期列表

        参数:
        - data_type: 数据类型 ('train' 或 'test')
        - save_format: 保存格式

        返回:
        - dates: 可用日期列表
        """

        search_dir = self.train_dir if data_type == 'train' else self.test_dir

        if not search_dir.exists():
            print(f"目录不存在: {search_dir}")
            return []

        dates = []

        if save_format in ['joblib', 'pickle']:
            # 查找文件
            extension = '.joblib' if save_format == 'joblib' else '.pkl'
            pattern = f"{data_type}_day_*{extension}"
            files = list(search_dir.glob(pattern))

            for file_path in files:
                # 从文件名中提取日期
                filename = file_path.stem  # 去掉扩展名
                parts = filename.split('_')

                # 查找日期部分 (格式: YYYY-MM-DD)
                for part in parts:
                    if len(part) == 10 and part.count('-') == 2:
                        try:
                            datetime.datetime.strptime(part, '%Y-%m-%d')
                            dates.append(part)
                            break
                        except ValueError:
                            continue

        elif save_format == 'numpy':
            # 查找目录
            pattern = f"{data_type}_day_*"
            dirs = [d for d in search_dir.iterdir() if d.is_dir() and d.name.startswith(f"{data_type}_day_")]

            for dir_path in dirs:
                # 从目录名中提取日期
                dirname = dir_path.name
                parts = dirname.split('_')

                # 查找日期部分
                for part in parts:
                    if len(part) == 10 and part.count('-') == 2:
                        try:
                            datetime.datetime.strptime(part, '%Y-%m-%d')
                            dates.append(part)
                            break
                        except ValueError:
                            continue

        return sorted(list(set(dates)))

    def get_data_summary(self) -> Dict:
        """
        获取保存数据的摘要信息

        返回:
        - summary: 数据摘要字典
        """

        summary = {
            'save_directory': str(self.save_dir),
            'train_data': {},
            'test_data': {}
        }

        # 检查训练数据
        if self.train_dir.exists():
            train_dates = self.get_available_dates('train', 'joblib')
            summary['train_data'] = {
                'available_days': len(train_dates),
                'date_range': (train_dates[0], train_dates[-1]) if train_dates else None,
                'sample_dates': train_dates[:5] if train_dates else []
            }

        # 检查测试数据
        if self.test_dir.exists():
            test_dates = self.get_available_dates('test', 'joblib')
            summary['test_data'] = {
                'available_days': len(test_dates),
                'date_range': (test_dates[0], test_dates[-1]) if test_dates else None,
                'sample_dates': test_dates[:5] if test_dates else []
            }

        return summary

    def load_batch_by_day_index(self,
                                day_idx: int,
                                data_type: str = 'train',
                                save_format: str = 'joblib') -> Optional[Dict]:
        """
        根据天数索引加载批次数据

        参数:
        - day_idx: 天数索引
        - data_type: 数据类型
        - save_format: 保存格式

        返回:
        - batch_data: 批次数据或None
        """

        search_dir = self.train_dir if data_type == 'train' else self.test_dir

        if not search_dir.exists():
            print(f"目录不存在: {search_dir}")
            return None

        try:
            if save_format == 'joblib':
                pattern = f"{data_type}_day_{day_idx:04d}_*.joblib"
                matching_files = list(search_dir.glob(pattern))

                if not matching_files:
                    print(f"未找到天数索引 {day_idx} 的 {data_type} 数据")
                    return None

                file_path = matching_files[0]
                batch_data = joblib.load(file_path)
                print(f"成功加载第 {day_idx} 天的 {data_type} 数据: {batch_data['date']}")
                return batch_data

            # 类似地处理其他格式...

        except Exception as e:
            print(f"加载第 {day_idx} 天数据失败: {e}")
            return None

    def create_neighbor_finder_from_batches(self, batches: List[Dict]) -> 'SimpleNeighborFinder':
        """
        从加载的批次数据创建邻居查找器

        参数:
        - batches: 批次数据列表

        返回:
        - neighbor_finder: 邻居查找器
        """

        # 收集所有边信息
        all_sources = []
        all_destinations = []
        all_timestamps = []
        all_edge_idxs = []

        for batch in batches:
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