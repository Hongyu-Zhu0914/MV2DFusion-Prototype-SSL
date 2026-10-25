"""CPU-only comparison of existing, token-aligned validation exports.

No model inference or source-head changes. Match globally before filtering scores.
The old band-rematch CSV is a separate regression check, not the primary result.
"""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path('/root/MV2DFusion-main')
S1 = ROOT / 'work_dirs/prototype_stage1_r50_labeled6000_10cls_0923'
S2 = ROOT / 'work_dirs/prototype_stage2_r50_labeled6000_10cls_20261025'
F1 = Path('/root/autodl-tmp/data/mv2dfusion_results/iter6000_val_full_0929')
F2 = S2 / 'features/stage2_local12000_val'
OUT = S2 / 'analysis/stage1_6000_vs_stage2_12000_conf02_08_val_20261025'
CLASSES = ('car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
           'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def normalize(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def aggregate(bucket, labels, positive, negative, similarity):
    for cls in range(10):
        for j, mask in enumerate((positive, negative)):
            values = similarity[mask & (labels == cls)]
            bucket[cls, j, 0] += len(values)
            bucket[cls, j, 1] += values.sum(dtype=np.float64)


def rows(bucket):
    result = []
    for cls, name in enumerate(CLASSES):
        pn, ps = bucket[cls, 0]
        nn, ns = bucket[cls, 1]
        assert pn > 0 and nn > 0, name
        result.append(dict(class_name=name, positives=int(pn), negatives=int(nn),
                           positive_mean=float(ps/pn), negative_mean=float(ns/nn),
                           gap=float(ps/pn-ns/nn)))
    return result


def main():
    started = time.time()
    assert not (OUT / 'comparison.json').exists(), 'Completed result already exists; do not rerun.'
    OUT.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location('aligned_export', ROOT / 'export_stage2_aligned_20261025.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    m1, m2 = read(F1/'feature_manifest.json'), read(F2/'feature_manifest.json')
    lookup = {item['token']: item['index'] for item in m2}
    assert len(m1) == len(m2) == len(lookup) == 3014
    assert len({x['token'] for x in m1}) == 3014
    assert {x['token'] for x in m1} == set(lookup)
    snap_path = S1/'prototypes/prototypes_iter_6000.pth'
    snap = torch.load(snap_path, map_location='cpu')
    proto_raw = snap['prototypes'].detach().float().cpu().numpy().copy()
    assert proto_raw.shape == (10, 10, 256) and float(snap['iter']) == 6000
    del snap
    checkpoint_path = S1/'checkpoints/iter_6000.pth'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    keys = [k for k in checkpoint['state_dict'] if k.endswith('.prototypes')]
    assert len(keys) == 1, keys
    assert np.array_equal(proto_raw, checkpoint['state_dict'][keys[0]].float().numpy())
    del checkpoint
    proto = normalize(proto_raw)
    np.save(OUT/'stage1_prototypes_verified.npy', proto_raw)
    del proto_raw
    b_global = np.zeros((10, 2, 2), dtype=np.float64)
    b_rematch = np.zeros_like(b_global)
    b_stage2 = np.zeros_like(b_global)
    query_count = 0
    for n, item in enumerate(m1):
        i, token = item['index'], item['token']
        j = lookup[token]
        def a1(field):
            return np.load(F1/'features'/f'sample_{i}_{field}.npy', allow_pickle=False)
        def a2(group, field):
            return np.load(F2/group/f'sample_{j}_{field}.npy', allow_pickle=False)
        assert str(a1('token').reshape(-1)[0]) == token
        assert str(a2('gt', 'token').reshape(-1)[0]) == token
        assert str(a2('candidates', 'token').reshape(-1)[0]) == token
        features, logits, boxes, saved_labels = (a1(k) for k in ('features','scores','boxes','labels'))
        assert len(features) == len(logits) == len(boxes) == len(saved_labels) == item['valid_queries']
        assert logits.shape[1] == 10 and features.shape[1] == 256
        assert all(np.isfinite(x).all() for x in (features, logits, boxes))
        probabilities = 1 / (1 + np.exp(-np.clip(logits, -50, 50)))
        labels = probabilities.argmax(axis=1)
        confidence = probabilities.max(axis=1)
        assert np.array_equal(labels, saved_labels)
        gt_boxes, gt_labels = a2('gt','boxes'), a2('gt','labels')
        matched, nearest = module.match(boxes, labels, confidence, gt_boxes, gt_labels)
        rematched, _ = module.match(boxes, labels, confidence, gt_boxes, gt_labels, upper=.8)
        band = (confidence > .2) & (confidence < .8)
        negative = (matched < 0) & (nearest > 2) & band
        similarity = np.zeros(len(features), dtype=np.float32)
        normalized = normalize(features)
        for cls in range(10):
            selected = (labels == cls) & band
            if selected.any():
                similarity[selected] = (normalized[selected] @ proto[cls].T).max(axis=1)
        aggregate(b_global, labels, (matched >= 0) & band, negative, similarity)
        aggregate(b_rematch, labels, (rematched >= 0) & band, negative, similarity)
        # Independently reproduce the existing Stage2 primary statistics from arrays.
        labels2, confidence2, sim2, match2, nearest2 = (a2('candidates', k) for k in
            ('labels', 'confidence', 'similarity', 'matched_gt_indices', 'nearest_gt_distance'))
        assert len(labels2) == len(confidence2) == len(sim2) == len(match2) == len(nearest2)
        assert np.isfinite(sim2).all()
        band2 = (confidence2 > .2) & (confidence2 < .8)
        aggregate(b_stage2, labels2, (match2 >= 0) & band2,
                  (match2 < 0) & (nearest2 > 2) & band2, sim2)
        query_count += len(features)
        if (n+1) % 300 == 0 or n+1 == len(m1):
            print(json.dumps(dict(frames=n+1, total=3014, seconds=round(time.time()-started, 1))), flush=True)
    r1, legacy_check, r2 = rows(b_global), rows(b_rematch), rows(b_stage2)
    csv_path = S1/'visualisation/val_iter6000_0929/positive_negative/positive_vs_clear_negative_similarity_summary_10cls_iter6000_conf_0.2_to_0.8.csv'
    with open(csv_path, encoding='utf-8-sig') as stream:
        legacy = list(csv.DictReader(stream))
    for calc, old in zip(legacy_check, legacy):
        assert calc['class_name'] == old['class']
        assert calc['positives'] == int(old['positive_count']), (calc, old)
        assert calc['negatives'] == int(old['clear_negative_count']), (calc, old)
        assert abs(calc['positive_mean'] - float(old['positive_mean_similarity'])) < 2e-6
        assert abs(calc['negative_mean'] - float(old['clear_negative_mean_similarity'])) < 2e-6
    saved_stats_path = S2/'visualisation/stage2_local12000_val/similarity_statistics.json'
    saved_rows = [r for r in read(saved_stats_path)['rows'] if r['mode'] == 'conf02_08_after_global_match']
    assert len(saved_rows) == 10
    for calc, saved in zip(r2, saved_rows):
        assert calc['class_name'] == saved['class_name']
        for key in ('positives','negatives'):
            assert calc[key] == saved[key], (calc, saved)
        for key in ('positive_mean','negative_mean','gap'):
            assert abs(calc[key]-saved[key]) < 2e-6, (calc, saved)
    compared = []
    for old, new in zip(r1, r2):
        row = {'class_name': old['class_name']}
        row.update({'stage1_'+k:v for k,v in old.items() if k != 'class_name'})
        row.update({'stage2_'+k:v for k,v in new.items() if k != 'class_name'})
        row['gap_change'] = new['gap']-old['gap']
        compared.append(row)
    limitations = [
        'Existing 3014-frame validation exports, not the 3005-frame internal test.',
        'Same tokens and matching rules; historical inference runs differ, so predictions are not paired-identical.',
        'Stage1 seed not recorded in export summary; Stage2 seed 920; sweep/GPU variation may affect candidate sets.',
        'Class means are sample-weighted within each class; macro gap is the unweighted mean of ten class gaps.',
        'Similarity separation does not by itself demonstrate pseudo-label precision/recall or semi-supervised improvement.'
    ]
    result = dict(status='PASS', frames=3014, matching='global match first, then 0.2 < confidence < 0.8',
                  positive='same class; confidence-descending one-to-one; per-class BEV distance; directional yaw <=45deg',
                  clear_negative='unmatched AND nearest ANY ten-class GT BEV distance >2m',
                  distances_m=module.DIST.tolist(), directional_classes=sorted(module.DIRECTIONAL),
                  stage1_steps=6000, stage2_additional_steps=6000, stage2_cumulative_steps=12000,
                  similarity='maximum cosine to own predicted class ten prototypes; own checkpoint feature space',
                  stage1_valid_queries=query_count, stage2_valid_queries=sum(x['valid_candidates'] for x in m2),
                  checks=dict(tokens='PASS', stage1_snapshot_matches_full_checkpoint='PASS',
                              legacy_band_rematch_csv='PASS', stage2_saved_statistics='PASS'),
                  source_sha256={str(p):sha(p) for p in (Path(__file__), ROOT/'export_stage2_aligned_20261025.py',
                      F1/'feature_manifest.json', F2/'feature_manifest.json', snap_path, csv_path, saved_stats_path)},
                  macro_gap_stage1=float(np.mean([x['gap'] for x in r1])),
                  macro_gap_stage2=float(np.mean([x['gap'] for x in r2])),
                  improved_classes=sum(x['gap_change']>0 for x in compared), rows=compared,
                  stage1_legacy_rematch_check=legacy_check, limitations=limitations,
                  runtime_seconds=time.time()-started)
    with open(OUT/'comparison.csv', 'x', newline='', encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream, fieldnames=list(compared[0])); writer.writeheader(); writer.writerows(compared)
    table=['# Stage1 6000 vs Stage2累计12000：3014帧验证集', '',
           '先整体匹配，再对正负双方筛选0.2<score<0.8；差值=正均值−明确负均值。', '',
           '|类别|6000正均值|6000负均值|6000差值|12000正均值|12000负均值|12000差值|差值变化|',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in compared:
        table.append('|'+r['class_name']+'|'+'|'.join(f'{r[k]:.6f}' for k in (
            'stage1_positive_mean','stage1_negative_mean','stage1_gap',
            'stage2_positive_mean','stage2_negative_mean','stage2_gap','gap_change'))+'|')
    table.extend(['', f"十类等权平均差值：{result['macro_gap_stage1']:.6f} → {result['macro_gap_stage2']:.6f}。",
                  '', '限制：两次既有推理候选不完全一致，非同seed严格配对控制实验；不是内部测试集结果，也不等同于伪标签精确率提升。'])
    (OUT/'comparison.md').write_text('\n'.join(table)+'\n', encoding='utf-8')
    (OUT/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
