import cv2
import mediapipe as mp
import os
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
import json
import glob
import joblib
from pathlib import Path
from typing import List, Dict, Any, Optional


COCO_KEYPOINT_NAMES = [
    'nose',
    'left_eye',
    'right_eye',
    'left_ear',
    'right_ear',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hip',
    'right_hip',
    'left_knee',
    'right_knee',
    'left_ankle',
    'right_ankle'
]


COCO_SKELETON = [
    [0, 1], [0, 2], [1, 3], [2, 4],
    [5, 6],
    [5, 7], [7, 9],
    [6, 8], [8, 10],
    [5, 11], [6, 12],
    [11, 12],
    [11, 13], [13, 15],
    [12, 14], [14, 16],
]


def _make_edges(bidir=True):
    e = []
    for u, v in COCO_SKELETON:
        e.append([u, v])
        if bidir:
            e.append([v, u])
    return np.asarray(e, dtype=np.int64)


def _load_json(p):
    with open(p, 'r') as f:
        return json.load(f)


def _ts(frame_idx: int, fps: Optional[int]) -> float:
    return float(frame_idx) if not fps or fps <= 0 else float(frame_idx) / float(fps)


def _feat_xy(frame_json: Dict[str, Any], use_normalized=True) -> np.ndarray:
    n = 17
    feat = np.zeros((n, 2), dtype=np.float32)
    key = 'keypoints_normalized' if use_normalized else 'keypoints_pixel'
    kp = frame_json.get(key, None)
    if kp is None:
        return feat
    if isinstance(kp, list) and len(kp) == n and isinstance(kp[0], dict):
        for it in kp:
            i = int(it['index'])
            feat[i] = [float(it['x']), float(it['y'])]
        return feat
    arr = np.asarray(kp, dtype=np.float32)
    if arr.shape[0] == n and arr.shape[1] >= 2:
        feat[:, 0] = arr[:, 0]
        feat[:, 1] = arr[:, 1]
    return feat


def visualize_keypoints_on_video(video_path, frame_rate=30, output_path=None, display=True):
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    pose = mp_pose.Pose(
        min_detection_confidence=0.2,
        min_tracking_confidence=0.2,
        model_complexity=2
    )

    cap = cv2.VideoCapture(video_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    keypoints_list = []
    frame_count = 0

    print(f"开始处理视频: {video_path}")
    print(f"视频分辨率: {width}x{height}, FPS: {fps}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_rate == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            annotated_frame = frame.copy()

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    annotated_frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )

                keypoints = []
                for landmark in results.pose_landmarks.landmark:
                    keypoints.append((landmark.x, landmark.y, landmark.z))
                keypoints_list.append(keypoints)

                cv2.putText(annotated_frame, f'Frame: {frame_count}, Keypoints: {len(keypoints)}',
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(annotated_frame, f'Frame: {frame_count}, No keypoints detected',
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if display:
                cv2.imshow('Keypoints Visualization', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if output_path:
                out.write(annotated_frame)
        else:
            if output_path:
                out.write(frame)

        frame_count += 1

    cap.release()
    if output_path:
        out.release()
    if display:
        cv2.destroyAllWindows()

    print(f"处理完成！共处理 {len(keypoints_list)} 帧，总帧数: {frame_count}")
    return keypoints_list


def extract_keypoints_from_video(video_path, frame_rate=30):
    cap = cv2.VideoCapture(video_path)
    keypoints_list = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_rate == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                keypoints = []
                for landmark in results.pose_landmarks.landmark:
                    keypoints.append((landmark.x, landmark.y, landmark.z))
                keypoints_list.append(keypoints)

        frame_count += 1

    cap.release()
    return keypoints_list


def visualize_keypoints_yolopose(video_path, model_path='yolov8n-pose.pt', frame_rate=30,
                                 output_path=None, display=True, conf_threshold=0.25):
    import cv2
    import numpy as np
    from ultralytics import YOLO

    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    keypoints_pixel_list = []
    keypoints_normalized_list = []
    frame_count = 0

    print(f"开始处理视频: {video_path}")
    print(f"视频分辨率: {width}x{height}, FPS: {fps}, 总帧数: {total_frames}")
    print(f"使用模型: {model_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_rate == 0:
            results = model(frame, conf=conf_threshold, verbose=False)
            annotated_frame = frame.copy()
            annotated_frame = results[0].plot()

            if results[0].keypoints is not None:
                keypoints_data = results[0].keypoints.data

                if len(keypoints_data) > 0:
                    person_keypoints = keypoints_data[0].cpu().numpy().astype(np.float32)
                    keypoints_pixel_list.append(person_keypoints.copy())

                    normalized_keypoints = person_keypoints.copy()
                    normalized_keypoints[:, 0] = person_keypoints[:, 0] / width
                    normalized_keypoints[:, 1] = person_keypoints[:, 1] / height
                    keypoints_normalized_list.append(normalized_keypoints)

                    cv2.putText(annotated_frame,
                                f'Frame: {frame_count}, People: {len(keypoints_data)}, Res: {width}x{height}',
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    keypoints_pixel_list.append(None)
                    keypoints_normalized_list.append(None)
                    cv2.putText(annotated_frame, f'Frame: {frame_count}, No keypoints',
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                keypoints_pixel_list.append(None)
                keypoints_normalized_list.append(None)
                cv2.putText(annotated_frame, f'Frame: {frame_count}, No detection',
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if display:
                cv2.imshow('YOLOPose Keypoints Visualization', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if output_path:
                out.write(annotated_frame)
        else:
            if output_path:
                out.write(frame)

        frame_count += 1

    cap.release()
    if output_path:
        out.release()
    if display:
        cv2.destroyAllWindows()

    valid_frames = len([k for k in keypoints_pixel_list if k is not None])
    print(f"\n处理完成！")
    print(f"  总帧数: {frame_count}")
    print(f"  处理帧数: {len(keypoints_pixel_list)}")
    print(f"  有效关键点帧数: {valid_frames}")

    return {
        'keypoints_pixel': keypoints_pixel_list,
        'keypoints_normalized': keypoints_normalized_list,
        'video_info': {
            'width': width,
            'height': height,
            'fps': fps,
            'total_frames': total_frames,
            'processed_frames': len(keypoints_pixel_list),
            'valid_frames': valid_frames
        }
    }


def extract_keypoints_yolopose(video_path, model_path='yolov8n-pose.pt', frame_rate=1,
                               conf_threshold=0.25):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"警告: 无法打开视频 {video_path}")
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    keypoints_pixel_list = []
    keypoints_normalized_list = []
    frame_count = 0
    frame_indices = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_rate == 0:
            results = model(frame, conf=conf_threshold, verbose=False)

            if len(results) > 0 and results[0].keypoints is not None:
                keypoints_data = results[0].keypoints.data

                if len(keypoints_data) > 0:
                    person_keypoints = keypoints_data[0].cpu().numpy().astype(np.float32)
                    keypoints_pixel_list.append(person_keypoints.copy())

                    normalized_keypoints = person_keypoints.copy()
                    normalized_keypoints[:, 0] = person_keypoints[:, 0] / width
                    normalized_keypoints[:, 1] = person_keypoints[:, 1] / height
                    keypoints_normalized_list.append(normalized_keypoints)

                    frame_indices.append(frame_count)
                else:
                    keypoints_pixel_list.append(None)
                    keypoints_normalized_list.append(None)
                    frame_indices.append(frame_count)
            else:
                keypoints_pixel_list.append(None)
                keypoints_normalized_list.append(None)
                frame_indices.append(frame_count)

        frame_count += 1

    cap.release()

    valid_frames = len([k for k in keypoints_pixel_list if k is not None])

    return {
        'keypoints_pixel': keypoints_pixel_list,
        'keypoints_normalized': keypoints_normalized_list,
        'frame_indices': frame_indices,
        'video_info': {
            'width': width,
            'height': height,
            'fps': fps,
            'total_frames': total_frames,
            'processed_frames': len(keypoints_pixel_list),
            'valid_frames': valid_frames
        }
    }


def format_keypoints_with_indices(keypoints_array, keypoint_names):
    if keypoints_array is None:
        return None

    formatted_keypoints = []
    for idx in range(len(keypoint_names)):
        keypoint_data = {
            'index': int(idx),
            'name': keypoint_names[idx],
            'x': float(keypoints_array[idx, 0]),
            'y': float(keypoints_array[idx, 1]),
            'confidence': float(keypoints_array[idx, 2])
        }
        formatted_keypoints.append(keypoint_data)

    return formatted_keypoints


def process_video_dataset(base_dir='data/hmdb51_data/hmdb51_sta',
                          classes_list=['pullup', 'climb', 'run', 'walk', 'situp'],
                          output_dir='data/hmdb51_keypoints',
                          model_path='yolov8n-pose.pt',
                          frame_rate=1,
                          conf_threshold=0.25):
    os.makedirs(output_dir, exist_ok=True)

    class_to_label = {cls: idx for idx, cls in enumerate(classes_list)}
    label_dict = {}

    total_videos = 0
    processed_videos = 0
    failed_videos = []

    print(f"开始处理数据集...")
    print(f"类别列表: {classes_list}")
    print(f"类别到标签映射: {class_to_label}")
    print(f"输出目录: {output_dir}")
    print(f"=" * 60)

    for class_name in classes_list:
        class_dir = os.path.join(base_dir, class_name)

        if not os.path.exists(class_dir):
            print(f"警告: 类别目录不存在: {class_dir}")
            continue

        video_files = [f for f in os.listdir(class_dir) if f.endswith('.avi')]
        total_videos += len(video_files)

        print(f"\n处理类别: {class_name} (标签: {class_to_label[class_name]})")
        print(f"  视频数量: {len(video_files)}")

        for video_file in tqdm(video_files, desc=f"  处理 {class_name}"):
            video_path = os.path.join(class_dir, video_file)
            video_name = os.path.splitext(video_file)[0]
            video_output_dir = os.path.join(output_dir, class_name, video_name)
            os.makedirs(video_output_dir, exist_ok=True)

            try:
                result = extract_keypoints_yolopose(
                    video_path,
                    model_path=model_path,
                    frame_rate=frame_rate,
                    conf_threshold=conf_threshold
                )

                if result is None:
                    failed_videos.append(video_path)
                    continue

                keypoints_pixel = result['keypoints_pixel']
                keypoints_normalized = result['keypoints_normalized']
                frame_indices = result['frame_indices']
                video_info = result['video_info']

                for idx, (frame_idx, kp_pixel, kp_norm) in enumerate(zip(frame_indices,
                                                                         keypoints_pixel,
                                                                         keypoints_normalized)):
                    frame_data = {
                        'frame_index': int(frame_idx),
                        'frame_number': idx,
                        'video_info': video_info
                    }

                    if kp_pixel is not None:
                        frame_data['keypoints_pixel'] = format_keypoints_with_indices(
                            kp_pixel,
                            COCO_KEYPOINT_NAMES
                        )

                        frame_data['keypoints_normalized'] = format_keypoints_with_indices(
                            kp_norm,
                            COCO_KEYPOINT_NAMES
                        )

                        frame_filename = f'frame_{frame_idx:06d}.json'
                        frame_filepath = os.path.join(video_output_dir, frame_filename)

                        with open(frame_filepath, 'w') as f:
                            json.dump(frame_data, f, indent=2)
                    else:
                        frame_data['keypoints_pixel'] = None
                        frame_data['keypoints_normalized'] = None

                        frame_filename = f'frame_{frame_idx:06d}.json'
                        frame_filepath = os.path.join(video_output_dir, frame_filename)

                        with open(frame_filepath, 'w') as f:
                            json.dump(frame_data, f, indent=2)

                video_metadata = {
                    'video_path': video_path,
                    'video_name': video_name,
                    'class': class_name,
                    'label': class_to_label[class_name],
                    'video_info': video_info,
                    'total_frames_processed': len(frame_indices),
                    'valid_frames': len([k for k in keypoints_pixel if k is not None]),
                    'keypoint_names': COCO_KEYPOINT_NAMES
                }

                metadata_filepath = os.path.join(video_output_dir, 'metadata.json')
                with open(metadata_filepath, 'w') as f:
                    json.dump(video_metadata, f, indent=2)

                relative_video_path = os.path.join(class_name, video_name)
                label_dict[relative_video_path] = {
                    'class': class_name,
                    'label': class_to_label[class_name],
                    'original_video_path': video_path,
                    'output_dir': video_output_dir
                }

                processed_videos += 1

            except Exception as e:
                print(f"\n错误: 处理视频 {video_path} 时出错: {str(e)}")
                failed_videos.append(video_path)
                continue

    label_dict_path = os.path.join(output_dir, 'label_dict.json')
    with open(label_dict_path, 'w') as f:
        json.dump(label_dict, f, indent=2)

    class_mapping_path = os.path.join(output_dir, 'class_mapping.json')
    with open(class_mapping_path, 'w') as f:
        json.dump({
            'class_to_label': class_to_label,
            'label_to_class': {v: k for k, v in class_to_label.items()},
            'keypoint_names': COCO_KEYPOINT_NAMES
        }, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"处理完成！")
    print(f"  总视频数: {total_videos}")
    print(f"  成功处理: {processed_videos}")
    print(f"  失败: {len(failed_videos)}")
    print(f"  label_dict 已保存到: {label_dict_path}")
    print(f"  类别映射已保存到: {class_mapping_path}")

    if failed_videos:
        print(f"\n失败的视频列表:")
        for video in failed_videos[:10]:
            print(f"  - {video}")
        if len(failed_videos) > 10:
            print(f"  ... 还有 {len(failed_videos) - 10} 个失败视频")

        failed_list_path = os.path.join(output_dir, 'failed_videos.txt')
        with open(failed_list_path, 'w') as f:
            for video in failed_videos:
                f.write(f"{video}\n")
        print(f"  失败列表已保存到: {failed_list_path}")

    return label_dict


def save_single_video_no_split(
    video_dir: str,
    out_root: str,
    use_normalized: bool = True,
    bidirectional_edges: bool = True,
    edge_feature_dim: int = 4
) -> Dict[str, Any]:
    assert os.path.isdir(video_dir), f'not found: {video_dir}'
    meta_path = os.path.join(video_dir, 'metadata.json')
    assert os.path.exists(meta_path)
    meta = _load_json(meta_path)
    vi = meta.get('video_info', {})
    fps = vi.get('fps', None)
    video_class = meta.get('class', 'unknown')
    video_name = meta.get('video_name', os.path.basename(video_dir))
    frame_files = sorted(glob.glob(os.path.join(video_dir, 'frame_*.json')))
    assert frame_files, f'no frames in {video_dir}'

    edges = _make_edges(bidirectional_edges)
    E = edges.shape[0]
    src_idx = edges[:, 0].astype(np.int64)
    dst_idx = edges[:, 1].astype(np.int64)

    save_dir = os.path.join(out_root, video_class, video_name)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    n_nodes = 17
    saved = 0
    node_features_static = np.eye(n_nodes, dtype=np.float32)

    for rank, fp in enumerate(frame_files):
        fj = _load_json(fp)
        fidx = int(fj.get('frame_index', -1))
        t = _ts(fidx, fps)
        nfeat_xy = _feat_xy(fj, use_normalized=use_normalized)

        eidx = (rank * E) + np.arange(E, dtype=np.int64)
        x_src = nfeat_xy[src_idx, 0]
        y_src = nfeat_xy[src_idx, 1]
        x_dst = nfeat_xy[dst_idx, 0]
        y_dst = nfeat_xy[dst_idx, 1]
        efeat = np.stack([x_src, y_src, x_dst, y_dst], axis=1).astype(np.float32)

        batch = {
            'n_nodes': n_nodes,
            'sources': src_idx.copy(),
            'destinations': dst_idx.copy(),
            'edge_idxs': eidx,
            'timestamps': np.full((E,), t, dtype=np.float64),
            'node_features': node_features_static.copy(),
            'edge_features': efeat,
            'frame_index': fidx,
            'video_name': video_name,
            'class': video_class,
        }
        joblib.dump(batch, os.path.join(save_dir, f'frame_{fidx:06d}.joblib'))
        saved += 1

    summary = {
        'video_dir': video_dir,
        'output_dir': save_dir,
        'class': video_class,
        'video_name': video_name,
        'n_frames': saved,
        'E_per_frame': E,
        'edge_feature_dim': 4,
        'node_feature_dim': n_nodes,
        'bidirectional_edges': bidirectional_edges,
        'use_normalized_xy': use_normalized,
        'fps': fps
    }
    joblib.dump(summary, os.path.join(save_dir, 'summary.joblib'))
    with open(os.path.join(save_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f'保存: {video_class}/{video_name} 帧数={saved} 输出={save_dir}')
    return summary


def load_single_video_batches(root_dir: str, cls_name: str, video_name: str):
    video_dir = os.path.join(root_dir, cls_name, video_name)
    frame_files = sorted(glob.glob(os.path.join(video_dir, 'frame_*.joblib')))
    return [joblib.load(fp) for fp in frame_files]


def save_all_videos_no_split(
    in_root: str = 'data/hmdb51_keypoints',
    out_root: str = 'data/video_daily_data_tgn',
    classes_list: list = None,
    use_normalized: bool = True,
    bidirectional_edges: bool = True
):
    if classes_list is None:
        classes_list = [d for d in os.listdir(in_root) if os.path.isdir(os.path.join(in_root, d))]

    processed = 0
    failed = []

    for cls_name in classes_list:
        cls_dir = os.path.join(in_root, cls_name)
        if not os.path.isdir(cls_dir):
            print(f'跳过（非目录）: {cls_dir}')
            continue

        for video_name in sorted(os.listdir(cls_dir)):
            video_dir = os.path.join(cls_dir, video_name)
            if not os.path.isdir(video_dir):
                continue

            meta_path = os.path.join(video_dir, 'metadata.json')
            if not os.path.exists(meta_path):
                continue

            try:
                save_single_video_no_split(
                    video_dir=video_dir,
                    out_root=out_root,
                    use_normalized=use_normalized,
                    bidirectional_edges=bidirectional_edges,
                    edge_feature_dim=4
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


def get_keypoints_from_batch(batch):
    edge_features = batch['edge_features']
    sources = batch['sources']
    destinations = batch['destinations']
    n_nodes = batch['n_nodes']

    keypoints = np.zeros((n_nodes, 2), dtype=np.float32)
    visited = np.zeros(n_nodes, dtype=bool)

    for i, (s, d) in enumerate(zip(sources, destinations)):
        if not visited[s]:
            keypoints[s, 0] = edge_features[i, 0]
            keypoints[s, 1] = edge_features[i, 1]
            visited[s] = True

        if not visited[d]:
            keypoints[d, 0] = edge_features[i, 2]
            keypoints[d, 1] = edge_features[i, 3]
            visited[d] = True

        if np.all(visited):
            break

    return keypoints


def visualize_keypoints_on_frame(frame_img, keypoints, edges, visibility=None,
                                 normalized=True, img_width=640, img_height=480):
    vis_img = frame_img.copy()
    h, w = vis_img.shape[:2]

    if normalized:
        kp_pixel = keypoints.copy()
        kp_pixel[:, 0] = kp_pixel[:, 0] * w
        kp_pixel[:, 1] = kp_pixel[:, 1] * h
    else:
        kp_pixel = keypoints.copy()

    for edge in edges:
        u, v = int(edge[0]), int(edge[1])
        if u < len(kp_pixel) and v < len(kp_pixel):
            if visibility is not None:
                if not visibility[u] or not visibility[v]:
                    continue

            pt1 = (int(kp_pixel[u, 0]), int(kp_pixel[u, 1]))
            pt2 = (int(kp_pixel[v, 0]), int(kp_pixel[v, 1]))
            cv2.line(vis_img, pt1, pt2, (0, 255, 0), 2)

    for i, kp in enumerate(kp_pixel):
        if visibility is not None and not visibility[i]:
            continue

        x, y = int(kp[0]), int(kp[1])
        color = (0, 0, 255) if visibility is None or visibility[i] else (128, 128, 128)
        cv2.circle(vis_img, (x, y), 5, color, -1)

    return vis_img


def generate_video_with_skeleton(
        video_class: str,
        video_name: str,
        processed_data_dir: str = 'data/video_daily_data_tgn',
        original_video_dir: str = 'data/hmdb51_data/hmdb51_sta',
        output_video_path: str = None,
        fps: int = 30,
        normalized: bool = True
):
    data_dir = os.path.join(processed_data_dir, video_class, video_name)
    if not os.path.exists(data_dir):
        print(f"错误: 找不到数据目录 {data_dir}")
        return None

    frame_files = sorted(glob.glob(os.path.join(data_dir, 'frame_*.joblib')))
    if not frame_files:
        print(f"错误: 在 {data_dir} 中找不到frame文件")
        return None

    first_batch = joblib.load(frame_files[0])
    edges = np.column_stack([first_batch['sources'], first_batch['destinations']])
    n_nodes = first_batch['n_nodes']

    original_video_path = os.path.join(original_video_dir, video_class, f'{video_name}.avi')
    if not os.path.exists(original_video_path):
        print(f"警告: 找不到原始视频文件 {original_video_path}，尝试使用黑色背景")
        use_original_video = False
    else:
        use_original_video = True
        cap = cv2.VideoCapture(original_video_path)

    if output_video_path is None:
        output_dir = os.path.join('results', 'hmdb51_videos', video_class)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_video_path = os.path.join(output_dir, f'{video_name}_skeleton.mp4')
    else:
        output_parent = Path(output_video_path).parent
        if str(output_parent) not in ("", "."):
            output_parent.mkdir(parents=True, exist_ok=True)

    if use_original_video:
        ret, first_img = cap.read()
        if ret and first_img is not None:
            h, w = first_img.shape[:2]
        else:
            h, w = 480, 640
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    else:
        h, w = 480, 640

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    if not out.isOpened():
        print(f"错误: 无法创建视频文件 {output_video_path}")
        if use_original_video:
            cap.release()
        return None

    print(f"正在生成视频: {video_class}/{video_name}")
    print(f"总帧数: {len(frame_files)}")
    print(f"节点数: {n_nodes}")

    for idx, frame_file in enumerate(frame_files):
        batch = joblib.load(frame_file)
        keypoints = get_keypoints_from_batch(batch)
        frame_idx = batch.get('frame_index', idx)
        visibility = None

        if use_original_video:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame_img = cap.read()
            if not ret or frame_img is None:
                frame_img = np.zeros((h, w, 3), dtype=np.uint8)
        else:
            frame_img = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.putText(frame_img, f'Frame {frame_idx}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        vis_frame = visualize_keypoints_on_frame(
            frame_img, keypoints, edges, visibility,
            normalized=normalized, img_width=w, img_height=h
        )

        info_text = f'Frame: {frame_idx}/{len(frame_files) - 1} | Class: {video_class}'
        cv2.putText(vis_frame, info_text, (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        out.write(vis_frame)

        if (idx + 1) % 50 == 0:
            print(f"  处理进度: {idx + 1}/{len(frame_files)}")

    out.release()
    if use_original_video:
        cap.release()
    print(f"视频已保存到: {output_video_path}")
    return output_video_path


if __name__ == "__main__":
    BASE_DIR = 'data/hmdb51_data/hmdb51_sta'
    CLASSES_LIST = ['pullup', 'climb', 'run', 'walk', 'situp']
    OUTPUT_DIR = 'data/hmdb51_keypoints'
    MODEL_PATH = 'yolov8n-pose.pt'
    FRAME_RATE = 1
    CONF_THRESHOLD = 0.5

    label_dict = process_video_dataset(
        base_dir=BASE_DIR,
        classes_list=CLASSES_LIST,
        output_dir=OUTPUT_DIR,
        model_path=MODEL_PATH,
        frame_rate=FRAME_RATE,
        conf_threshold=CONF_THRESHOLD
    )

    print(f"\n数据集处理完成！")
    print(f"label_dict 包含 {len(label_dict)} 个视频")

    result = save_all_videos_no_split(
        in_root='data/hmdb51_keypoints',
        out_root='data/hmdb51_processed_data',
        classes_list=['pullup', 'climb', 'run', 'walk', 'situp'],
        use_normalized=True,
        bidirectional_edges=True,
    )
    print(result)

    ROOT = 'data/hmdb51_processed_data'
    CLASSES = ['pullup', 'climb', 'run', 'walk', 'situp']
    all_data = load_all_videos_no_split(ROOT, CLASSES)
    print(f'视频数: {len(all_data)}')

    # generate_video_with_skeleton(
    #     'pullup',
    #     '10_Pull_Ups_pullup_f_nm_np1_fr_goo_0',
    #     processed_data_dir='data/hmdb51_processed_data',
    #     original_video_dir='data/hmdb51_data/hmdb51_sta',
    #     output_video_path=None,
    #     fps=30,
    #     normalized=True
    # )
