"""Two count histograms only; reuse exported global matching, never rematch."""
import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

CLASSES = ('car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
           'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone')
BINS = np.linspace(-1, 1, 41)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main(args):
    root, original, out = Path(args.features), Path(args.original), Path(args.out)
    assert not out.exists(), 'Do not overwrite any previous results'
    summary = read_json(root / 'export_summary.json')
    manifest = read_json(root / 'feature_manifest.json')
    assert summary['status'] == 'PASS' and summary['frames'] == len(manifest) == 3005
    assert summary['split'] == 'test' and summary['temporal_mode'] == 'scene-stream'
    reference = {r['class_name']: r for r in read_json(original / 'similarity_statistics.json')['rows']
                 if r['mode'] == 'conf02_08_after_global_match'}
    hist = np.zeros((2, 10, 2, 40), dtype=np.int64)
    counts = np.zeros((2, 10, 2), dtype=np.int64)
    sums = np.zeros((2, 10, 2), dtype=np.float64)
    tokens = set()
    for k, entry in enumerate(manifest):
        def load(name):
            return np.load(str(root / 'candidates' / ('sample_%d_%s.npy' % (entry['index'], name))), allow_pickle=False)
        label, conf, sim = load('labels'), load('confidence'), load('similarity')
        matched, distance, clear = load('matched_gt_indices'), load('nearest_gt_distance'), load('clear_negative')
        token = str(load('token')[0])
        assert token == entry['token'] and token not in tokens
        tokens.add(token)
        assert all(a.shape == label.shape for a in (conf, sim, matched, distance, clear))
        assert np.isfinite(conf).all() and np.isfinite(sim).all() and not np.isnan(distance).any()
        assert ((label >= 0) & (label < 10)).all()
        assert ((conf >= 0) & (conf <= 1)).all()
        assert ((sim >= -1.00001) & (sim <= 1.00001)).all()
        negative = (matched < 0) & (distance > 2.)
        assert np.array_equal(clear, negative)
        for mode, score_mask in enumerate(((conf > .2) & (conf < .8), conf > .2)):
            masks = ((matched >= 0) & score_mask, negative & score_mask)
            for c in range(10):
                for role, mask in enumerate(masks):
                    values = sim[(label == c) & mask]
                    hist[mode, c, role] += np.histogram(np.clip(values, -1, 1), BINS)[0]
                    counts[mode, c, role] += len(values)
                    sums[mode, c, role] += values.sum(dtype=np.float64)
        if (k + 1) % 500 == 0 or k + 1 == len(manifest):
            print('COUNTED_FRAMES', k + 1, len(manifest), flush=True)
    assert np.array_equal(hist.sum(axis=-1), counts)
    assert (counts[1] >= counts[0]).all()
    rows = []
    for mode, name in enumerate(('conf02_08_after_global_match', 'conf_gt02_after_global_match')):
        for c, cls in enumerate(CLASSES):
            means = [float(sums[mode, c, j] / counts[mode, c, j]) if counts[mode, c, j] else None for j in range(2)]
            row = dict(mode=name, class_name=cls, positives=int(counts[mode, c, 0]),
                       negatives=int(counts[mode, c, 1]), positive_mean=means[0], negative_mean=means[1],
                       gap=None if None in means else means[0] - means[1])
            if mode == 0:
                assert all(row[key] == reference[cls][key] for key in ('positives', 'negatives'))
                for key in ('positive_mean', 'negative_mean'):
                    assert (row[key] is None and reference[cls][key] is None) or np.isclose(row[key], reference[cls][key], atol=1e-7)
            rows.append(row)
    out.mkdir()
    first = 'similarity_conf02_08_after_global_match_counts.png'
    shutil.copy2(original / first, out / first)
    fig, axes = plt.subplots(5, 2, figsize=(17, 23))
    width = (BINS[1] - BINS[0]) * .43
    centers = (BINS[:-1] + BINS[1:]) / 2
    for c, ax in enumerate(axes.flat):
        row = rows[10 + c]
        for role, color, label in ((0, '#3676b5', 'Strict positives'), (1, '#ef5350', 'Clear negatives')):
            ax.bar(centers + (role - .5) * width, hist[1, c, role], width=width, color=color, label=label)
        ax.axvline(.94 if c == 7 else .90, color='black', ls='--', label='Legacy reference only')
        ax.set_xlim(-1, 1)
        ax.grid(axis='y', alpha=.2)
        pm = 'NA' if row['positive_mean'] is None else '%.3f' % row['positive_mean']
        nm = 'NA' if row['negative_mean'] is None else '%.3f' % row['negative_mean']
        ax.set_title('%s | positive n=%d mean=%s; negative n=%d mean=%s' %
                     (CLASSES[c], row['positives'], pm, row['negatives'], nm), fontsize=9)
        ax.set_xlabel('Maximum cosine to 10 prototypes of predicted class')
        ax.set_ylabel('Samples / bin')
        ax.legend(fontsize=7)
    fig.suptitle('Stage2 +6000 steps | cumulative 12000 | confidence >0.2 for BOTH groups\n'
                 'Global matching unchanged; clear negatives: unmatched and nearest ANY GT >2m; ambiguous excluded', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .955))
    second = 'similarity_conf_gt02_after_global_match_counts.png'
    fig.savefig(str(out / second), dpi=150, bbox_inches='tight')
    plt.close(fig)
    for name in (first, second):
        with Image.open(str(out / name)) as image:
            image.verify()
    assert sha(out / first) == sha(original / first)
    with (out / 'statistics.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    report = dict(status='PASS', frames=len(tokens), split='internal_test_3005',
        first_image_identical_to_original=True, band_statistics_match_original=True,
        global_matching_reused=True, rematching_performed=False,
        confidence_for_both_groups=['0.2 < score < 0.8', 'score > 0.2'],
        candidate_group='valid candidates including normal and propagated queries; not final detections',
        source_features=str(root), source_export_summary_sha256=sha(root / 'export_summary.json'),
        checkpoint_sha256=summary['checkpoint_sha256'], annotation_sha256=summary['annotation_sha256'],
        rows=rows, files={name:dict(bytes=(out/name).stat().st_size, sha256=sha(out/name)) for name in (first, second)},
        limitation='Raw ten-class PKL GT; ambiguous candidates excluded; histogram counts are not official precision/recall')
    (out / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('TWO_HISTOGRAMS_PASS', json.dumps({k:report[k] for k in ('frames','files')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', required=True)
    parser.add_argument('--original', required=True)
    parser.add_argument('--out', required=True)
    main(parser.parse_args())
