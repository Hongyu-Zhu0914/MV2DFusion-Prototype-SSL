"""Isolated seeded plain/export comparison; preserves all previous results."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/root/autodl-tmp/data/mv2dfusion_results/ssl_online_r50_ema_1011/postprocess_ema_12000_1011/diagnostic_seeded_1012')
CKPT = ROOT.parent.parent / 'iter_12000.pth'

def main():
    ROOT.mkdir(exist_ok=True)
    cases = [('plain_seed1011', 'evaluate_online_ema_splits_1011.py', 1011),
             ('export_seed1011', 'export_online_ema_npy_1011.py', 1011),
             ('plain_seed1012', 'evaluate_online_ema_splits_1011.py', 1012),
             ('plain_seed1011_repeat', 'evaluate_online_ema_splits_1011.py', 1011)]
    for name, script, seed in cases:
        target = ROOT / name
        if (target / 'subset_validation_summary.json').exists():
            continue
        if target.exists() and any(target.iterdir()):
            raise RuntimeError('Incomplete diagnostic retained: ' + str(target))
        # Seed before importing/building either path, with identical arguments.
        code = ('import random,numpy as np,torch,runpy; '
                f'random.seed({seed}); np.random.seed({seed}); '
                f'torch.manual_seed({seed}); torch.cuda.manual_seed_all({seed}); '
                f'runpy.run_path({script!r},run_name="__main__")')
        args = [sys.executable, '-u', '-c', code, '--checkpoint', str(CKPT),
                '--branch', 'student', '--limit', '8', '--output-dir', str(target)]
        (ROOT / 'state.json').write_text(json.dumps({'stage': name, 'status': 'running'}))
        with (ROOT / (name + '.log')).open('w') as log:
            subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, check=True)
    import mmcv
    import numpy as np
    import torch
    outputs = [mmcv.load(str(ROOT / name / 'predictions.pkl')) for name, _, _ in cases]
    def compare(left, right):
        changed = 0
        score_max = box_max = 0.
        labels_equal = True
        for a, b in zip(left, right):
            a, b = a['pts_bbox'], b['pts_bbox']
            if len(a['scores_3d']) != len(b['scores_3d']):
                return {'same_count': False}
            ds = (a['scores_3d'] - b['scores_3d']).abs().max().item()
            db = (a['boxes_3d'].tensor - b['boxes_3d'].tensor).abs().max().item()
            eq = torch.equal(a['labels_3d'], b['labels_3d'])
            changed += int(ds > 1e-6 or db > 1e-4 or not eq)
            labels_equal &= eq
            score_max, box_max = max(score_max, ds), max(box_max, db)
        return dict(same_count=True, changed_frames=changed,
                    max_score_difference=score_max, max_box_difference=box_max,
                    labels_equal=labels_equal)
    def high_conf_geometry(left, right):
        distances, scores = [], []
        for a, b in zip(left, right):
            a, b = a['pts_bbox'], b['pts_bbox']
            for idx in (a['scores_3d'] > .2).nonzero().flatten():
                same = (b['labels_3d'] == a['labels_3d'][idx]).nonzero().flatten()
                if not len(same):
                    continue
                ds = (b['boxes_3d'].tensor[same, :3] - a['boxes_3d'].tensor[idx, :3]).norm(dim=1)
                j = same[ds.argmin()]
                distances.append(ds.min().item())
                scores.append(abs((b['scores_3d'][j] - a['scores_3d'][idx]).item()))
        return dict(predictions=len(distances), median_center_error=float(np.median(distances)),
                    max_center_error=max(distances), max_score_error=max(scores))
    report = dict(frames=8, same_seed_plain_vs_export=compare(outputs[0], outputs[1]),
                  different_seed_plain_vs_plain=compare(outputs[0], outputs[2]),
                  same_seed_plain_repeat=compare(outputs[0], outputs[3]),
                  export_geometry=high_conf_geometry(outputs[0], outputs[1]),
                  repeat_geometry=high_conf_geometry(outputs[0], outputs[3]),
                  different_seed_geometry=high_conf_geometry(outputs[0], outputs[2]))
    (ROOT / 'report_with_repeat.json').write_text(json.dumps(report, indent=2))
    (ROOT / 'state.json').write_text(json.dumps({'status': 'complete'}))
    print('DIAGNOSTIC_COMPLETE', json.dumps(report), flush=True)

if __name__ == '__main__':
    main()
