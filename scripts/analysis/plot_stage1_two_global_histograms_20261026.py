"""Two Stage1 validation count histograms; match globally before score filtering.

CPU-only, streaming existing Stage1 projected features. Stage2 export supplies
only token-aligned raw GT arrays, never its features or prototypes.
"""

# Resolve imports and relative CLI paths after the publication-layout migration.
import sys as _bootstrap_sys
import os as _bootstrap_os
from pathlib import Path as _BootstrapPath
REPO_ROOT = _BootstrapPath(__file__).resolve().parents[2]
if str(REPO_ROOT) not in _bootstrap_sys.path:
    _bootstrap_sys.path.insert(0, str(REPO_ROOT))
if __name__ == '__main__':
    _bootstrap_os.chdir(str(REPO_ROOT))

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(str(REPO_ROOT))
S1 = ROOT/'work_dirs/prototype_stage1_r50_labeled6000_10cls_0923'
F1 = Path('/root/autodl-tmp/data/mv2dfusion_results/iter6000_val_full_0929')
S2 = ROOT/'work_dirs/prototype_stage2_r50_labeled6000_10cls_20261025'
GT = S2/'features/stage2_local12000_val'
CHECK = S2/'analysis/stage1_6000_vs_stage2_12000_conf02_08_val_20261025'
OUT = S1/'visualisation/val_iter6000_0929/positive_negative'
NAMES = ('similarity_conf02_08_after_global_match_counts.png',
         'similarity_conf_gt02_after_global_match_counts.png')
PREFIX = 'confidence_two_ranges_global_match_20261026'
CLASSES = ('car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
           'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone')
BINS = np.linspace(-1., 1., 41)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def normalize(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def main():
    started = time.time()
    outputs = [OUT/n for n in NAMES] + [OUT/(PREFIX+s) for s in ('.json', '.csv')]
    assert OUT.is_dir() and all(not p.exists() for p in outputs), 'Never overwrite existing results'
    old_hashes = {p.name: sha(p) for p in OUT.iterdir() if p.is_file()}
    spec = importlib.util.spec_from_file_location('aligned_export', ROOT/'scripts/evaluation/export_stage2_aligned_20261025.py')
    matching = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(matching)
    manifest = read(F1/'feature_manifest.json')
    gt_manifest = read(GT/'feature_manifest.json')
    lookup = {x['token']: x['index'] for x in gt_manifest}
    assert len(manifest) == len(lookup) == len(gt_manifest) == 3014
    assert {x['token'] for x in manifest} == set(lookup)
    prototype_path = CHECK/'stage1_prototypes_verified.npy'
    integrity = read(CHECK/'backup_manifest.json')
    assert sha(prototype_path) == integrity[str(prototype_path.relative_to(ROOT))]['sha256']
    previous = read(CHECK/'comparison.json')
    assert previous['checks']['stage1_snapshot_matches_full_checkpoint'] == 'PASS'
    assert sha(F1/'feature_manifest.json') == previous['source_sha256'][str(F1/'feature_manifest.json')]
    assert sha(ROOT/'scripts/evaluation/export_stage2_aligned_20261025.py') == previous['source_sha256'][str(ROOT/'scripts/evaluation/export_stage2_aligned_20261025.py')]
    prototypes = normalize(np.load(prototype_path, allow_pickle=False))
    assert prototypes.shape == (10, 10, 256)
    hist = np.zeros((2, 10, 2, 40), np.int64)
    counts = np.zeros((2, 10, 2), np.int64)
    sums = np.zeros((2, 10, 2), np.float64)
    tokens = set()
    total_queries = 0
    for number, entry in enumerate(manifest):
        i, token = entry['index'], entry['token']
        j = lookup[token]
        def load(field):
            return np.load(F1/'features'/f'sample_{i}_{field}.npy', allow_pickle=False)
        def gt(field):
            return np.load(GT/'gt'/f'sample_{j}_{field}.npy', allow_pickle=False)
        assert token not in tokens and str(load('token').reshape(-1)[0]) == token
        assert str(gt('token').reshape(-1)[0]) == token
        tokens.add(token)
        features, logits, boxes, labels = (load(k) for k in ('features','scores','boxes','labels'))
        assert len(features) == len(logits) == len(boxes) == len(labels) == entry['valid_queries']
        assert features.shape[1] == 256 and logits.shape[1] == 10
        assert all(np.isfinite(x).all() for x in (features, logits, boxes))
        probabilities = 1./(1.+np.exp(-np.clip(logits, -50, 50)))
        assert np.array_equal(labels, probabilities.argmax(axis=1))
        confidence = probabilities.max(axis=1)
        gt_boxes, gt_labels = gt('boxes'), gt('labels')
        assert len(gt_boxes) == len(gt_labels)
        assert np.isfinite(gt_boxes).all()
        matched, nearest = matching.match(boxes, labels, confidence, gt_boxes, gt_labels)
        # Exactly one global match; upper bound is applied only afterwards.
        clear_negative = (matched < 0) & (nearest > 2.)
        similarity = np.zeros(len(features), np.float32)
        normalized = normalize(features)
        for cls in range(10):
            selected = (labels == cls) & (confidence > .2)
            if selected.any():
                similarity[selected] = (normalized[selected] @ prototypes[cls].T).max(axis=1)
        assert np.isfinite(similarity).all()
        assert ((similarity >= -1.00001) & (similarity <= 1.00001)).all()
        for mode, score_mask in enumerate(((confidence > .2) & (confidence < .8), confidence > .2)):
            for role, group in enumerate((matched >= 0, clear_negative)):
                for cls in range(10):
                    values = similarity[(labels == cls) & group & score_mask]
                    hist[mode, cls, role] += np.histogram(np.clip(values,-1,1), BINS)[0]
                    counts[mode, cls, role] += len(values)
                    sums[mode, cls, role] += values.sum(dtype=np.float64)
        total_queries += len(features)
        if (number+1) % 500 == 0 or number+1 == len(manifest):
            print('FRAMES', number+1, 'SECONDS', round(time.time()-started,1), flush=True)
    assert len(tokens) == 3014 and total_queries == 1070851
    assert np.array_equal(hist.sum(axis=-1), counts)
    assert (counts[1] >= counts[0]).all()
    rows = []
    references = {r['class_name']: r for r in previous['rows']}
    for mode, mode_name in enumerate(('conf02_08_after_global_match','conf_gt02_after_global_match')):
        for cls, name in enumerate(CLASSES):
            pn, nn = counts[mode,cls]
            assert pn > 0 and nn > 0
            pm, nm = sums[mode,cls]/counts[mode,cls]
            row = dict(mode=mode_name, class_name=name, positives=int(pn), negatives=int(nn),
                       positive_mean=float(pm), negative_mean=float(nm), gap=float(pm-nm))
            if mode == 0:
                ref = references[name]
                assert pn == ref['stage1_positives'] and nn == ref['stage1_negatives']
                assert abs(pm-ref['stage1_positive_mean']) < 2e-6
                assert abs(nm-ref['stage1_negative_mean']) < 2e-6
            rows.append(row)
    width = (BINS[1]-BINS[0])*.43
    centers = (BINS[:-1]+BINS[1:])/2
    for mode, name in enumerate(NAMES):
        figure, axes = plt.subplots(5,2,figsize=(17,23))
        for cls, ax in enumerate(axes.flat):
            row = rows[mode*10+cls]
            for role, color, label in ((0,'#3676b5','Strict positives'),(1,'#ef5350','Clear negatives')):
                ax.bar(centers+(role-.5)*width, hist[mode,cls,role], width=width, color=color,label=label)
            ax.axvline(.94 if cls==7 else .90,color='black',ls='--',label='Legacy reference only')
            ax.set_xlim(-1,1)
            ax.grid(axis='y',alpha=.2)
            ax.set_title('%s | positive n=%d mean=%.3f; negative n=%d mean=%.3f' %
                         (CLASSES[cls],row['positives'],row['positive_mean'],row['negatives'],row['negative_mean']),fontsize=9)
            ax.set_xlabel('Maximum cosine to 10 prototypes of predicted class')
            ax.set_ylabel('Samples / bin')
            ax.legend(fontsize=7)
        interval = '0.2 < confidence < 0.8' if mode==0 else 'confidence > 0.2'
        figure.suptitle('Stage1 iter 6000 | validation 3014 frames | '+interval+' for BOTH groups\n'
            'Global match first, then score filter; clear negatives: unmatched and nearest ANY GT >2m; ambiguous excluded',fontsize=11)
        figure.tight_layout(rect=(0,0,1,.955))
        figure.savefig(OUT/name,dpi=150,bbox_inches='tight')
        plt.close(figure)
        with Image.open(OUT/name) as image:
            image.verify()
    with (OUT/(PREFIX+'.csv')).open('x',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    assert all(sha(OUT/name)==digest for name,digest in old_hashes.items())
    report = dict(status='PASS',frames=3014,checkpoint='Stage1 original iter_6000.pth',
        candidate_group='existing valid normal and propagated queries, not final topk detections',
        valid_queries=total_queries,source_features=str(F1),gt_source=str(GT/'gt'),
        gt_alignment='exact token lookup; raw ten-class PKL GT; not official metric filtering',
        matching='global confidence>0.2 same-class greedy one-to-one; distance/yaw; then filter each plot interval',
        distances_m=matching.DIST.tolist(),directional_classes=sorted(matching.DIRECTIONAL),
        yaw_limit_degrees=45,negative='unmatched AND nearest ANY ten-class GT BEV distance >2m',
        y_axis='raw sample count per bin; no per-frame or group normalization',
        band_matches_previous_comparison=True,old_files_unchanged=True,old_file_sha256=old_hashes,
        prototype_source=str(prototype_path),prototype_sha256=sha(prototype_path),
        source_code_sha256=sha(__file__),matching_code_sha256=sha(ROOT/'scripts/evaluation/export_stage2_aligned_20261025.py'),
        files={name:dict(bytes=(OUT/name).stat().st_size,sha256=sha(OUT/name)) for name in NAMES},
        rows=rows,runtime_seconds=time.time()-started,
        limitation='Not precision/recall; ambiguous/duplicate-near-GT queries excluded. Old four images retained.')
    (OUT/(PREFIX+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print('PASS',json.dumps(dict(frames=3014,totals=counts.sum(axis=1).tolist(),files=report['files'])),flush=True)


if __name__=='__main__':
    main()
