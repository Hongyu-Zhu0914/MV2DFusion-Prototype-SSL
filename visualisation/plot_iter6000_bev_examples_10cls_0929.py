"""Black-background BEV examples from the aligned iteration-6000 export."""

import argparse
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch


CLASSES = (
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
)


def corners(box):
    x, y, _, width, length, _, yaw = box[:7]
    rectangle = np.array([[width / 2, length / 2], [width / 2, -length / 2],
                          [-width / 2, -length / 2], [-width / 2, length / 2],
                          [width / 2, length / 2]])
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)],
                         [np.sin(yaw), np.cos(yaw)]])
    return rectangle @ rotation.T + np.array([x, y])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--export-dir', required=True)
    parser.add_argument('--dataset-pkl', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--indices', nargs='+', type=int, default=[22, 34, 100, 250])
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()
    root = Path(args.export_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(args.dataset_pkl, 'rb') as file:
        data = pickle.load(file)
    infos = data['infos'] if isinstance(data, dict) else data
    manifest = json.loads((root / 'feature_manifest.json').read_text())
    state = torch.load(args.checkpoint, map_location='cpu')['state_dict']
    keys = [key for key in state if key.endswith('.prototypes')]
    if len(keys) != 1:
        raise AssertionError('Expected one prototype tensor')
    prototypes = state[keys[0]].float().numpy()
    prototypes /= np.maximum(np.linalg.norm(prototypes, axis=-1, keepdims=True), 1e-8)

    saved = []
    for index in args.indices:
        if not 0 <= index < len(manifest):
            raise ValueError('Index out of range: {}'.format(index))
        info = infos[index]
        if manifest[index]['token'] != info['token']:
            raise AssertionError('Token mismatch for frame {}'.format(index))
        prefix = root / 'features' / 'sample_{}'.format(index)
        features = np.load(str(prefix) + '_features.npy')
        scores = np.load(str(prefix) + '_scores.npy')
        boxes = np.load(str(prefix) + '_boxes.npy')
        probability = 1 / (1 + np.exp(-np.clip(scores, -50, 50)))
        predicted = probability.argmax(axis=1)
        confidence = probability.max(axis=1)
        features = features / np.maximum(np.linalg.norm(features, axis=-1, keepdims=True), 1e-8)
        similarity = np.empty(len(boxes), dtype=np.float32)
        for cls in range(10):
            selected = np.where(predicted == cls)[0]
            if len(selected):
                similarity[selected] = (features[selected] @ prototypes[cls].T).max(axis=1)
        points = np.fromfile(info['lidar_path'], dtype=np.float32).reshape(-1, 5)
        visible = (np.abs(points[:, 0]) < 60) & (np.abs(points[:, 1]) < 60)
        gt_boxes = np.asarray(info['gt_boxes'], dtype=np.float32)
        gt_names = np.asarray(info['gt_names'])
        fig, axis = plt.subplots(figsize=(13, 13), dpi=160)
        fig.patch.set_facecolor('black')
        axis.set_facecolor('black')
        axis.scatter(points[visible, 0], points[visible, 1], s=0.12,
                     c='#A9A9A9', alpha=0.22, linewidths=0)
        for box, name in zip(gt_boxes, gt_names):
            if name not in CLASSES:
                continue
            outline = corners(box)
            axis.plot(outline[:, 0], outline[:, 1], color='#24FF38', linewidth=1.2)
        selected = np.where(confidence > 0.2)[0]
        for prediction_id in selected[np.argsort(confidence[selected])]:
            conf = float(confidence[prediction_id])
            outline = corners(boxes[prediction_id])
            color = (1.0, 0.78 * (1 - conf), 0.78 * (1 - conf))
            axis.plot(outline[:, 0], outline[:, 1], color=color, linewidth=1.1)
            x, y = outline[0]
            axis.text(x, y, '{} {:.2f}/{:.2f}'.format(
                CLASSES[predicted[prediction_id]], conf, similarity[prediction_id]),
                fontsize=4.3, color=color, clip_on=True)
        axis.scatter([0], [0], c='white', s=25, marker='x')
        axis.set_xlim(-60, 60)
        axis.set_ylim(-60, 60)
        axis.set_aspect('equal')
        axis.tick_params(colors='white')
        axis.set_xlabel('LiDAR x (m)', color='white')
        axis.set_ylabel('LiDAR y (m)', color='white')
        axis.set_title('iter 6000 | sample {} | GT={} | predictions conf>0.2={}\n'
                       'Green=GT, red=prediction; label=class confidence/similarity'.format(
                           index, len(gt_boxes), len(selected)), color='white')
        path = out / 'bev_sample_{:04d}_iter6000.png'.format(index)
        fig.savefig(path, facecolor='black', bbox_inches='tight')
        plt.close(fig)
        saved.append(dict(index=index, token=info['token'], gt=len(gt_boxes),
                          predictions_conf_gt_02=len(selected), image=str(path)))
    with (out / 'bev_examples_summary.json').open('w') as file:
        json.dump(saved, file, indent=2)
    print('BEV_EXAMPLES_PASS', len(saved), flush=True)


if __name__ == '__main__':
    main()
