"""Verify coverage and row-level alignment of the full prototype export."""

# Resolve imports and relative CLI paths after the publication-layout migration.
import sys as _bootstrap_sys
import os as _bootstrap_os
from pathlib import Path as _BootstrapPath
REPO_ROOT = _BootstrapPath(__file__).resolve().parents[2]
if str(REPO_ROOT) not in _bootstrap_sys.path:
    _bootstrap_sys.path.insert(0, str(REPO_ROOT))
if __name__ == '__main__':
    _bootstrap_os.chdir(str(REPO_ROOT))


import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--export-dir', required=True)
    parser.add_argument('--dataset-pkl', required=True)
    args = parser.parse_args()
    root = Path(args.export_dir)
    summary = json.loads((root / 'export_summary.json').read_text())
    manifest = json.loads((root / 'feature_manifest.json').read_text())
    with open(args.dataset_pkl, 'rb') as file:
        data = pickle.load(file)
    infos = data['infos'] if isinstance(data, dict) else data
    with (root / 'iter_6000_val_predictions.pkl').open('rb') as file:
        predictions = pickle.load(file)
    if not len(infos) == len(manifest) == len(predictions) == summary['frames'] == 3014:
        raise AssertionError('Full validation frame coverage differs')
    if hashlib.sha256('\n'.join(sorted(x['token'] for x in infos)).encode()).hexdigest() != summary['token_sha256']:
        raise AssertionError('Validation token hash differs')
    counts = np.zeros(10, dtype=np.int64)
    valid_total = 0
    normal_total = 0
    temporal_total = 0
    min_norm = float('inf')
    max_norm = 0.0
    for index, (info, row) in enumerate(zip(infos, manifest)):
        if row['index'] != index or row['token'] != info['token']:
            raise AssertionError('Frame-token alignment failed at {}'.format(index))
        prefix = root / 'features' / 'sample_{}'.format(index)
        features = np.load(str(prefix) + '_features.npy')
        scores = np.load(str(prefix) + '_scores.npy')
        boxes = np.load(str(prefix) + '_boxes.npy')
        labels = np.load(str(prefix) + '_labels.npy')
        queries = np.load(str(prefix) + '_query_indices.npy')
        token = np.load(str(prefix) + '_token.npy')
        matched = np.load(root / 'features' / 'th_2.0' /
                          'sample_{}_gt_labels.npy'.format(index))
        count = row['valid_queries']
        if (features.shape != (count, 256) or scores.shape != (count, 10) or
                boxes.shape[0] != count or labels.shape != (count,) or
                queries.shape != (count,) or matched.shape != (count,)):
            raise AssertionError('Row count mismatch at {}'.format(index))
        if token.tolist() != [info['token']]:
            raise AssertionError('Saved token differs at {}'.format(index))
        if not np.all(queries[:-1] < queries[1:]) or np.any(queries >= row['all_queries']):
            raise AssertionError('Query indices differ at {}'.format(index))
        if np.any(np.all(scores <= -39.5, axis=1)) or not all(
                np.isfinite(a).all() for a in (features, scores, boxes)):
            raise AssertionError('Invalid or nonfinite rows at {}'.format(index))
        if not np.array_equal(labels, scores.argmax(axis=1)):
            raise AssertionError('Predicted labels differ at {}'.format(index))
        if count:
            norms = np.linalg.norm(features, axis=1)
            min_norm = min(min_norm, float(norms.min()))
            max_norm = max(max_norm, float(norms.max()))
            counts += np.bincount(labels, minlength=10)
        valid_total += count
        normal_total += row['normal_queries']
        temporal_total += row['temporal_queries']
    if not (valid_total == summary['valid_queries'] and
            normal_total == summary['normal_queries'] and
            temporal_total == summary['temporal_queries'] and
            valid_total == normal_total + temporal_total):
        raise AssertionError('Manifest totals differ')
    if not 0.99 <= min_norm <= max_norm <= 1.01:
        raise AssertionError('Feature normalization differs')
    report = dict(frames=len(infos), prediction_pkl_frames=len(predictions),
                  valid_queries=valid_total, normal_queries=normal_total,
                  temporal_queries=temporal_total,
                  predicted_class_counts=counts.tolist(),
                  feature_norm_range=[min_norm, max_norm],
                  mAP=summary.get('mAP'), NDS=summary.get('NDS'),
                  all_tokens_aligned=True, all_export_rows_aligned=True)
    with (root / 'export_integrity_report.json').open('w') as file:
        json.dump(report, file, indent=2)
    print('EXPORT_INTEGRITY_PASS', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
