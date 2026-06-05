import os
import json
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Any, List, Optional
from scipy.io import loadmat
import cv2

PENN_ACTION_SKELETON = [[0,1],[0,2],[2,4],[4,6],[2,8],[8,10],[10,12],[1,3],[3,5],[1,7],[7,9],[9,11]]


def _make_edges_penn_action(bidirectional_edges=True, n_nodes=13):
    """
    创建 Penn Action 13 个关键点的边连接

    参数:
        bidir: 是否双向边
        n_nodes: 节点数量（13 或 12）

    返回:
        edges: [E, 2] 边数组
    """
    # if n_nodes == 13:
    #     skeleton = PENN_ACTION_SKELETON
    # else:
    #     # 如果只有12个点，过滤掉超出范围的边
    #     skeleton = [e for e in PENN_ACTION_SKELETON_12 if e[1] < n_nodes]
    skeleton = PENN_ACTION_SKELETON

    edges = []
    for u, v in skeleton:
        if u < n_nodes and v < n_nodes:
            edges.append([u, v])
            if bidirectional_edges:
                edges.append([v, u])

    return np.asarray(edges, dtype=np.int64)


def normalize_keypoints(keypoints: np.ndarray, bbox: Optional[np.ndarray] = None,
                        img_width: int = 640, img_height: int = 480) -> np.ndarray:
    """
    将关键点坐标归一化到 [0, 1]

    参数:
        keypoints: [n_nodes, 2] - 像素坐标
        bbox: [4] - 边界框 [x, y, w, h] (可选)
        img_width: 图像宽度
        img_height: 图像高度

    返回:
        normalized: [n_nodes, 2] - 归一化坐标 [0, 1]
    """
    if bbox is not None:
        x, y, w, h = bbox
        # 相对于边界框归一化
        normalized = keypoints.copy()
        normalized[:, 0] = (normalized[:, 0] - x) / w if w > 0 else 0
        normalized[:, 1] = (normalized[:, 1] - y) / h if h > 0 else 0
    else:
        # 相对于整个图像归一化
        normalized = keypoints.copy()
        normalized[:, 0] = normalized[:, 0] / img_width if img_width > 0 else 0
        normalized[:, 1] = normalized[:, 1] / img_height if img_height > 0 else 0

    return np.clip(normalized, 0, 1)


def process_single_video_penn_action(
        video_id: str,
        penn_frames_dir: str,
        penn_label_path: str,
        out_root: str,
        use_normalized: bool = True,
        img_width: int = None,
        img_height: int = None,
        bidirectional_edges: bool = True,
        skip_abnormal: bool = True,  # 新增参数：是否跳过异常视频
        max_abnormal_ratio: float = 0.01  # 新增参数：最大异常点比例阈值
) -> Dict[str, Any]:
    """
    处理单个 Penn Action 视频（保持13个关键点）

    参数:
        video_id: 视频ID (如 '0001')
        penn_frames_dir: Penn Action frames 目录
        penn_label_path: Penn Action label 文件路径
        out_root: 输出根目录
        use_normalized: 是否使用归一化坐标
        img_width: 图像宽度
        img_height: 图像高度
        bidirectional_edges: 是否双向边
        edge_feature_dim: 边特征维度

    返回:
        summary: 处理摘要信息
    """
    # 1. 加载 .mat 文件
    try:
        mat_data = loadmat(penn_label_path, simplify_cells=True)
        if 'annotation' in mat_data:
            ann = mat_data['annotation']
        else:
            ann = {k: v for k, v in mat_data.items() if not k.startswith('__')}
            if len(ann) == 1:
                ann = list(ann.values())[0]
    except Exception as e:
        print(f"错误: 无法加载 {penn_label_path}: {e}")
        return None

    # 提取数据
    action = ann.get('action', 'unknown')
    x_coords = ann.get('x', None)  # [nframes, n_kp]
    y_coords = ann.get('y', None)  # [nframes, n_kp]
    visibility = ann.get('visibility', None)  # [nframes, n_kp]
    bbox = ann.get('bbox', None)  # [nframes, 4]
    nframes = ann.get('nframes', 0)
    train_label = ann.get('train', 0)

    print('x_coords',x_coords.shape)
    print('x_coords',x_coords)
    print('y_coords', y_coords)
    print('bbox',bbox)

    if x_coords is None or y_coords is None:
        print(f"警告: {video_id} 缺少关键点数据")
        return None

    # 确保是 numpy 数组
    x_coords = np.asarray(x_coords, dtype=np.float32)
    y_coords = np.asarray(y_coords, dtype=np.float32)

    # 确定实际的关键点数量
    if len(x_coords.shape) == 2:
        n_nodes = x_coords.shape[1]
    else:
        n_nodes = 13  # 默认值

    if visibility is not None:
        visibility = np.asarray(visibility, dtype=bool)
    else:
        visibility = np.ones((nframes, n_nodes), dtype=bool)

    print(f"视频 {video_id}: {n_nodes} 个关键点, {nframes} 帧")

    if img_width is None or img_height is None:
        frame_files = sorted([f for f in os.listdir(penn_frames_dir) if f.endswith('.jpg')])
        if frame_files:
            try:
                from PIL import Image
                first_img_path = os.path.join(penn_frames_dir, frame_files[0])
                img = Image.open(first_img_path)
                detected_width, detected_height = img.size

                if img_width is None:
                    img_width = detected_width
                if img_height is None:
                    img_height = detected_height

                print(f"视频 {video_id}: 从图片读取尺寸 {img_width}x{img_height}")
            except Exception as e:
                print(f"警告: 无法读取图片尺寸，使用默认值 640x480: {e}")
                if img_width is None:
                    img_width = 640
                if img_height is None:
                    img_height = 480
        else:
            print(f"警告: 找不到图片文件，使用默认尺寸 640x480")
            if img_width is None:
                img_width = 640
            if img_height is None:
                img_height = 480

    print(f"视频 {video_id}: {n_nodes} 个关键点, {nframes} 帧, 图像尺寸: {img_width}x{img_height}")

    if skip_abnormal:
        print(f"视频 {video_id}: 检查异常值...")
        abnormal_frames = []

        for frame_idx in range(nframes):
            # 提取当前帧的关键点
            if len(x_coords.shape) == 2:
                frame_x = x_coords[frame_idx]
                frame_y = y_coords[frame_idx]
            else:
                frame_x = x_coords
                frame_y = y_coords

            penn_kp = np.stack([frame_x, frame_y], axis=1).astype(np.float32)

            # 处理可见性
            if len(visibility.shape) > 1:
                frame_vis = visibility[frame_idx]
            else:
                frame_vis = visibility

            # 检查异常
            is_abnormal, abnormal_count, reason = check_frame_abnormal(
                penn_kp, img_width, img_height,
                threshold_ratio=max_abnormal_ratio
            )

            if is_abnormal:
                abnormal_frames.append({
                    'frame_idx': frame_idx,
                    'abnormal_count': abnormal_count,
                    'reason': reason
                })

        # 如果发现异常帧，跳过整个视频
        if abnormal_frames:
            print(f"视频 {video_id}: 检测到 {len(abnormal_frames)} 帧异常，跳过处理")
            for af in abnormal_frames[:5]:  # 只打印前5个异常帧的信息
                print(f"  异常帧 {af['frame_idx']}: {af['reason']}")
            if len(abnormal_frames) > 5:
                print(f"  ... 还有 {len(abnormal_frames) - 5} 个异常帧")
            return None

        print(f"视频 {video_id}: 所有帧检查通过，开始处理")

    # 2. 创建输出目录
    video_name = f"video_{video_id}"
    save_dir = os.path.join(out_root, action, video_name)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # 3. 创建边连接
    edges = _make_edges_penn_action(bidirectional_edges=bidirectional_edges, n_nodes=n_nodes)
    E = edges.shape[0]
    src = edges[:, 0].astype(np.int64)
    dst = edges[:, 1].astype(np.int64)

    # 4. 处理每一帧
    saved = 0
    frame_files = sorted([f for f in os.listdir(penn_frames_dir) if f.endswith('.jpg')])

    # 假设 30 FPS（可根据实际情况调整）
    fps = 30

    for frame_idx in range(nframes):
        if frame_idx >= len(frame_files):
            break

        # 提取当前帧的关键点
        if len(x_coords.shape) == 2:
            frame_x = x_coords[frame_idx]  # [n_nodes]
            frame_y = y_coords[frame_idx]  # [n_nodes]
        else:
            frame_x = x_coords
            frame_y = y_coords

        # 组合为 [n_nodes, 2]
        penn_kp = np.stack([frame_x, frame_y], axis=1).astype(np.float32)  # [n_nodes, 2]

        # 处理可见性
        if len(visibility.shape) > 1:
            frame_vis = visibility[frame_idx]
        else:
            frame_vis = visibility

        # 将不可见的关键点坐标设为 0（或保持原值，根据需求）
        # 这里我们保持原值，只用于归一化

        # 归一化（如果需要）
        if use_normalized:
            normalized_kp = normalize_keypoints(penn_kp,None, img_width, img_height)

        else:
            normalized_kp =penn_kp

        node_features = np.eye(n_nodes, dtype=np.float32)  # [n_nodes, n_nodes] = [13, 13]
        edge_features = np.zeros((E, 4), dtype=np.float32)  # [E, 4]
        for i, (s, d) in enumerate(zip(src, dst)):
            edge_features[i, 0] = normalized_kp[s, 0]  # src_x
            edge_features[i, 1] = normalized_kp[s, 1]  # src_y
            edge_features[i, 2] = normalized_kp[d, 0]  # dst_x
            edge_features[i, 3] = normalized_kp[d, 1]  # dst_y

        # 创建时间戳
        timestamp = float(frame_idx) / fps if fps > 0 else float(frame_idx)

        # 创建 edge_idxs
        eidx = (frame_idx * E) + np.arange(E, dtype=np.int64)

        # 创建 batch 字典
        batch = {
            'n_nodes': n_nodes,
            'sources': src.copy(),
            'destinations': dst.copy(),
            'edge_idxs': eidx,
            'timestamps': np.full((E,), timestamp, dtype=np.float64),
            'node_features': node_features.astype(np.float32),
            'edge_features': edge_features.astype(np.float32),  # [E, 4] [src_x, src_y, dst_x, dst_y]
            'frame_index': frame_idx,
            'video_name': video_name,
            'class': action,
        }

        # 保存为 joblib
        joblib.dump(batch, os.path.join(save_dir, f'frame_{frame_idx:06d}.joblib'))
        saved += 1

    # 5. 创建 metadata.json
    metadata = {
        'video_info': {
            'fps': fps,
            'width': img_width,
            'height': img_height,
        },
        'class': action,
        'video_name': video_name,
        'video_id': video_id,
        'train': bool(train_label),
        'nframes': saved,
        'n_nodes': n_nodes,
    }

    with open(os.path.join(save_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # 6. 创建 summary
    summary = {
        'video_dir': penn_frames_dir,
        'output_dir': save_dir,
        'class': action,
        'video_name': video_name,
        'video_id': video_id,
        'n_frames': saved,
        'n_nodes': n_nodes,
        'E_per_frame': E,
        'edge_feature_dim': 4,
        'node_feature_dim': n_nodes,
        'bidirectional_edges': bidirectional_edges,
        'use_normalized_xy': use_normalized,
        'fps': fps,
        'train': bool(train_label),
    }

    joblib.dump(summary, os.path.join(save_dir, 'summary.joblib'))
    with open(os.path.join(save_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f'处理完成: {action}/{video_name} 帧数={saved} 节点数={n_nodes} 边数={E} 输出={save_dir}')
    return summary


def process_all_penn_action(
        penn_root: str = 'data/Penn_Action',
        out_root: str = 'data/penn_action_processed',
        use_normalized: bool = True,
        bidirectional_edges: bool = True,
        classes_list: Optional[List[str]] = None
):
    """
    批量处理所有 Penn Action 视频

    参数:
        penn_root: Penn Action 数据集根目录
        out_root: 输出根目录
        use_normalized: 是否使用归一化坐标
        bidirectional_edges: 是否双向边
        edge_feature_dim: 边特征维度
        classes_list: 要处理的类别列表（None 表示处理所有）
    """
    frames_dir = os.path.join(penn_root, 'frames')
    labels_dir = os.path.join(penn_root, 'labels')

    if not os.path.exists(frames_dir) or not os.path.exists(labels_dir):
        print(f"错误: 找不到 frames 或 labels 目录")
        return

    # 获取所有视频ID
    video_ids = sorted([d for d in os.listdir(frames_dir) if os.path.isdir(os.path.join(frames_dir, d))])

    print(len(video_ids))

    processed = 0
    failed = []

    for video_id in video_ids:
        video_frames_dir = os.path.join(frames_dir, video_id)
        label_file = os.path.join(labels_dir, f'{video_id}.mat')

        if not os.path.exists(label_file):
            print(f"跳过: {video_id} - 找不到 label 文件")
            continue

        try:
            summary = process_single_video_penn_action(
                video_id=video_id,
                penn_frames_dir=video_frames_dir,
                penn_label_path=label_file,
                out_root=out_root,
                use_normalized=use_normalized,
                bidirectional_edges=bidirectional_edges,
            )

            if summary:
                # 检查是否需要过滤类别
                if classes_list is None or summary['class'] in classes_list:
                    processed += 1
                else:
                    # 删除不需要的类别
                    import shutil
                    shutil.rmtree(summary['output_dir'])
        except Exception as e:
            print(f"错误: 处理 {video_id} 失败: {e}")
            import traceback
            traceback.print_exc()
            failed.append(video_id)

    print(f'\n批量处理完成: 成功 {processed} 个视频, 失败 {len(failed)} 个')
    if failed:
        print('失败列表(前10):')
        for v in failed[:10]:
            print('  -', v)

    return {'processed': processed, 'failed': failed, 'out_root': out_root}


def get_keypoints_from_batch(batch):
    """
    从 batch 的 edge_features 中直接提取节点坐标
    由于坐标是从同一个数组复制的，可以直接从任意一条包含该节点的边中取坐标
    """
    edge_features = batch['edge_features']  # [E, 4]
    sources = batch['sources']  # [E]
    destinations = batch['destinations']  # [E]
    n_nodes = batch['n_nodes']

    # 初始化节点坐标数组
    keypoints = np.zeros((n_nodes, 2), dtype=np.float32)
    visited = np.zeros(n_nodes, dtype=bool)  # 标记哪些节点已经提取过坐标

    # 遍历所有边，提取每个节点的坐标（每个节点只取一次）
    for i, (s, d) in enumerate(zip(sources, destinations)):
        # 如果源节点还没提取过坐标，从这条边提取
        if not visited[s]:
            keypoints[s, 0] = edge_features[i, 0]  # src_x
            keypoints[s, 1] = edge_features[i, 1]  # src_y
            visited[s] = True

        # 如果目标节点还没提取过坐标，从这条边提取
        if not visited[d]:
            keypoints[d, 0] = edge_features[i, 2]  # dst_x
            keypoints[d, 1] = edge_features[i, 3]  # dst_y
            visited[d] = True

        # 如果所有节点都已提取，可以提前退出
        if np.all(visited):
            break

    return keypoints

def visualize_keypoints_on_frame(frame_img, keypoints, edges, visibility=None,
                                 normalized=True, img_width=640, img_height=480):
    """
    在单帧图像上绘制关键点和骨架连接

    参数:
        frame_img: numpy array [H, W, 3] - 图像数组
        keypoints: [n_nodes, 2] - 关键点坐标（归一化或像素坐标）
        edges: [E, 2] - 边连接关系
        visibility: [n_nodes] - 可见性（可选）
        normalized: 关键点是否已归一化
        img_width: 图像宽度（如果normalized=True）
        img_height: 图像高度（如果normalized=True）

    返回:
        vis_img: 绘制后的图像
    """
    vis_img = frame_img.copy()
    h, w = vis_img.shape[:2]

    # 如果关键点是归一化的，转换为像素坐标
    if normalized:
        kp_pixel = keypoints.copy()
        kp_pixel[:, 0] = kp_pixel[:, 0] * w
        kp_pixel[:, 1] = kp_pixel[:, 1] * h
    else:
        kp_pixel = keypoints.copy()

    # 绘制骨架连接（边）
    for edge in edges:
        u, v = int(edge[0]), int(edge[1])
        if u < len(kp_pixel) and v < len(kp_pixel):
            # 检查可见性
            if visibility is not None:
                if not visibility[u] or not visibility[v]:
                    continue

            pt1 = (int(kp_pixel[u, 0]), int(kp_pixel[u, 1]))
            pt2 = (int(kp_pixel[v, 0]), int(kp_pixel[v, 1]))

            # 绘制连接线
            cv2.line(vis_img, pt1, pt2, (0, 255, 0), 2)

    # 绘制关键点
    for i, kp in enumerate(kp_pixel):
        if visibility is not None and not visibility[i]:
            continue

        x, y = int(kp[0]), int(kp[1])

        # 绘制关键点（圆圈）
        color = (0, 0, 255) if visibility is None or visibility[i] else (128, 128, 128)
        cv2.circle(vis_img, (x, y), 5, color, -1)

        # 可选：显示关键点索引
        # cv2.putText(vis_img, str(i), (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    return vis_img


def check_frame_abnormal(
        keypoints: np.ndarray,
        img_width: int,
        img_height: int,
        threshold_ratio: float = 0.01,
) -> tuple:
    """
    检查单帧是否有异常关键点

    参数:
        keypoints: [n_nodes, 2] - 关键点坐标
        img_width: 图像宽度
        img_height: 图像高度
        visibility: [n_nodes] - 可见性（可选）
        threshold_ratio: 异常值阈值比例
        max_abnormal_ratio: 允许的最大异常点比例（超过此比例认为帧异常）

    返回:
        is_abnormal: 是否异常
        abnormal_count: 异常点数量
        reason: 异常原因描述
    """
    # 检测坐标范围
    # print('keypoints',keypoints.shape)
    x_min, x_max = keypoints[:, 0].min(), keypoints[:, 0].max()
    y_min, y_max = keypoints[:, 1].min(), keypoints[:, 1].max()



    # 处理像素坐标
    abnormal_count = 0
    x_threshold = img_width * threshold_ratio
    y_threshold = img_height * threshold_ratio

    for i, kp in enumerate(keypoints):
        x, y = kp[0], kp[1]

        # print('x',x)
        # print('y',y)

        # 检查是否在合理范围内
        x_valid = 0 <= x <= img_width
        y_valid = 0 <= y <= img_height

        # 检查是否为异常小的值（可能是归一化坐标被误认为是像素坐标）
        x_too_small = 0 < x < x_threshold
        y_too_small = 0 < y < y_threshold

        # print('x_valid',x_valid)
        # print('y_valid',y_valid)
        #
        # print('x_threshold',x_threshold)
        # print('y_threshold',y_threshold)
        #
        # print('x_too_small',x_too_small)
        # print('y_too_small',y_too_small)


        # 如果坐标超出图像范围或异常小，认为是异常
        if not (x_valid and y_valid) or (x_too_small and y_too_small):
            abnormal_count += 1

    # abnormal_ratio = abnormal_count / len(keypoints) if len(keypoints) > 0 else 0
    # print('abnormal_ratio',abnormal_ratio)

    if abnormal_count > 0:
        return True, abnormal_count, f"异常点比例过高 ({abnormal_count}/{len(keypoints)}), 坐标范围 x:[{x_min:.1f}, {x_max:.1f}], y:[{y_min:.1f}, {y_max:.1f}]"

    return False, abnormal_count, None

def generate_video_with_skeleton(
        video_class: str,
        video_name: str,
        processed_data_dir: str = 'data/penn_action_processed',
        original_frames_dir: str = 'data/Penn_Action/frames',
        output_video_path: str = None,
        fps: int = 30,
        normalized: bool = True
):
    """
    生成带有骨架关键点的视频

    参数:
        video_class: 视频类别（如 'baseball_pitch'）
        video_name: 视频名称（如 'video_0001'）
        processed_data_dir: 处理后的数据目录
        original_frames_dir: 原始图像帧目录
        output_video_path: 输出视频路径（如果为None，自动生成）
        fps: 视频帧率
        normalized: 关键点是否已归一化

    返回:
        output_video_path: 输出视频路径
    """
    import glob

    # 1. 加载处理后的数据
    data_dir = os.path.join(processed_data_dir, video_class, video_name)
    if not os.path.exists(data_dir):
        print(f"错误: 找不到数据目录 {data_dir}")
        return None

    # 加载metadata获取视频ID
    metadata_path = os.path.join(data_dir, 'metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        video_id = metadata.get('video_id', video_name.replace('video_', ''))
    else:
        video_id = video_name.replace('video_', '')

    # 加载所有frame数据
    frame_files = sorted(glob.glob(os.path.join(data_dir, 'frame_*.joblib')))
    if not frame_files:
        print(f"错误: 在 {data_dir} 中找不到frame文件")
        return None

    # 加载第一个frame获取边连接信息
    first_batch = joblib.load(frame_files[0])
    edges = np.column_stack([first_batch['sources'], first_batch['destinations']])
    n_nodes = first_batch['n_nodes']

    # 2. 加载原始图像
    original_frames_path = os.path.join(original_frames_dir, video_id)
    if not os.path.exists(original_frames_path):
        print(f"警告: 找不到原始图像目录 {original_frames_path}，尝试使用数据目录中的信息")
        # 如果找不到原始图像，可以创建一个黑色背景
        use_original_frames = False
    else:
        use_original_frames = True
        original_img_files = sorted(glob.glob(os.path.join(original_frames_path, '*.jpg')))

    # 3. 确定输出路径
    if output_video_path is None:
        output_dir = os.path.join('results', 'penn_action_videos', video_class)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_video_path = os.path.join(output_dir, f'{video_name}_skeleton.mp4')
    else:
        output_parent = Path(output_video_path).parent
        if str(output_parent) not in ("", "."):
            output_parent.mkdir(parents=True, exist_ok=True)

    # 4. 创建视频写入器
    # 先读取第一帧确定视频尺寸
    if use_original_frames and original_img_files:
        first_img = cv2.imread(original_img_files[0])
        if first_img is not None:
            h, w = first_img.shape[:2]
        else:
            h, w = 480, 640  # 默认尺寸
    else:
        h, w = 480, 640  # 默认尺寸

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    if not out.isOpened():
        print(f"错误: 无法创建视频文件 {output_video_path}")
        return None

    # 5. 处理每一帧
    print(f"正在生成视频: {video_class}/{video_name}")
    print(f"总帧数: {len(frame_files)}")

    for idx, frame_file in enumerate(frame_files):
        # 加载关键点数据
        batch = joblib.load(frame_file)
        keypoints = get_keypoints_from_batch(batch)  # [n_nodes, 2]
        frame_idx = batch.get('frame_index', idx)

        # 获取可见性（如果有）
        visibility = None  # Penn Action数据中可能没有单独的可见性字段

        # 加载原始图像
        if use_original_frames and original_img_files and frame_idx < len(original_img_files):
            frame_img = cv2.imread(original_img_files[frame_idx])
            if frame_img is None:
                # 如果图像加载失败，创建黑色背景
                frame_img = np.zeros((h, w, 3), dtype=np.uint8)
        else:
            # 创建黑色背景
            frame_img = np.zeros((h, w, 3), dtype=np.uint8)
            # 在背景上添加文本信息
            cv2.putText(frame_img, f'Frame {frame_idx}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # 绘制关键点和骨架
        vis_frame = visualize_keypoints_on_frame(
            frame_img, keypoints, edges, visibility,
            normalized=normalized, img_width=w, img_height=h
        )

        # 添加帧信息文本
        info_text = f'Frame: {frame_idx}/{len(frame_files) - 1} | Class: {video_class}'
        cv2.putText(vis_frame, info_text, (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 写入视频
        out.write(vis_frame)

        if (idx + 1) % 50 == 0:
            print(f"  处理进度: {idx + 1}/{len(frame_files)}")

    # 6. 释放资源
    out.release()
    print(f"视频已保存到: {output_video_path}")
    return output_video_path
if __name__ == "__main__":
    # 处理所有视频
    process_all_penn_action(
        penn_root='data/Penn_Action',
        out_root='data/penn_action_processed',
        use_normalized=True,
        bidirectional_edges=True,
        classes_list=None  # None 表示处理所有类别
    )


    # video_class='pullup' #baseball_pitch
    # video_idx='1149'
    # video_name=f'video_{video_idx}'
    # generate_video_with_skeleton(
    #     video_class=video_class,
    #     video_name=video_name,
    #     processed_data_dir='data/penn_action_processed',
    #     original_frames_dir='data/Penn_Action/frames',
    #     output_video_path=f'results/{video_class}_{video_idx}_skeleton.mp4',
    #     fps=30,
    #     normalized=True
    # )
