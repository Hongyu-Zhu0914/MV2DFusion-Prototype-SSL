"""Metadata-only dataset audit and post-smoke checkpoint/resume verification."""

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
from collections import Counter
import copy
import json
from pathlib import Path

import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.utils import import_modules_from_strings

from scripts.prototype.train_prototype_stage2_20261025 import ROOT, sha_file, json_write, state_hash


def data_audit(cfg, out):
    reports, token_sets, scene_sets = {}, {}, {}
    specs = dict(train=cfg.data.train.ann_file, val=cfg.data.val.ann_file,
                 test=cfg.data.test.ann_file,
                 unlabeled='/root/autodl-tmp/data/nuscenes/ssl_splits/nuscenes_ssl_train_unlabeled_official_test.pkl')
    expected = dict(train=6000, val=3014, test=3005, unlabeled=6008)
    all_paths = set()
    for name, path in specs.items():
        payload = mmcv.load(path)
        infos = payload['infos']
        tokens = {x['token'] for x in infos}
        scenes = {x['scene_token'] for x in infos}
        assert len(infos) == len(tokens) == expected[name], (name, len(infos))
        paths = set()
        counts = Counter()
        for x in infos:
            paths.add(x['lidar_path'])
            paths.update(cam['data_path'] for cam in x['cams'].values())
            paths.update(y['data_path'] for y in x['sweeps'])
            if name != 'unlabeled':
                valid = np.asarray(x.get('valid_flag', np.asarray(x['num_lidar_pts']) > 0), dtype=bool)
                counts.update(np.asarray(x['gt_names'])[valid].tolist())
        missing = [p for p in paths if not Path(p).is_file()]
        assert not missing, (name, missing[:10])
        all_paths.update(paths)
        if name == 'train':
            assert len(scenes) == 150 and all(counts[c] > 0 for c in cfg.class_names)
        token_sets[name], scene_sets[name] = tokens, scenes
        reports[name] = dict(frames=len(tokens), scenes=len(scenes), path=str(path),
                             annotation_sha256=sha_file(path), sensor_files=len(paths),
                             missing_files=0, valid_gt_counts=dict(counts))
        print('SPLIT_PATHS_PASS', name, reports[name]['frames'], len(paths), flush=True)
        del payload, infos
    for left in specs:
        for right in specs:
            if left < right:
                assert not token_sets[left] & token_sets[right], (left, right, 'token overlap')
                assert not scene_sets[left] & scene_sets[right], (left, right, 'scene overlap')
    report = dict(status='PASS', splits=reports, tokens_disjoint=True, scenes_disjoint=True,
                  unique_sensor_paths=len(all_paths), sensor_contents_not_decoded=True,
                  dataset_files_not_modified=True, formal_training_started=False)
    json_write(out / 'metadata_audit.json', report)


def checkpoint_audit(cfg, out, smoke):
    import projects.mmdet3d_plugin
    import projects.fsdv2
    from mmdet3d.models import build_model
    import_modules_from_strings(**cfg.custom_imports)
    initialization = json.loads((smoke / 'initialization.json').read_text())
    summary = json.loads((smoke / 'run_summary.json').read_text())
    raw = torch.load(str(smoke / 'checkpoints/iter_4.pth'), map_location='cpu')
    assert raw['meta']['completed_steps'] == 4 and raw['meta']['stage2_steps'] == 4
    assert raw['meta']['local_iter'] == initialization['initialization']['stage1_local_iter'] + 4
    assert len(raw['sample_order']) == len(set(raw['sample_order'])) == 6000
    assert raw['optimizer']['state'] and set(raw['rng']) == {'python', 'numpy', 'torch', 'cuda'}
    assert summary['completed_steps'] == 4 and summary['new_steps_this_process'] == 2
    assert summary['frozen_detector_unchanged'] and summary['prototype_changed']
    model = build_model(cfg.model)
    model.init_weights()
    result = model.load_state_dict(raw['state_dict'], strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    assert model.fusion_bbox_head.stage2_load_kind == 'stage2_resume'
    assert state_hash(model) == raw['meta']['contract']['frozen_sha256']
    mismatches = []
    for name, value in model.state_dict().items():
        other = raw['state_dict'][name]
        if value.is_floating_point():
            equal = torch.allclose(value, other, rtol=0, atol=0, equal_nan=True)
        else:
            equal = torch.equal(value, other)
        if not equal:
            mismatches.append(name)
    assert not mismatches, mismatches[:5]
    broken = raw['state_dict'].copy()
    broken._metadata = raw['state_dict']._metadata
    del broken['fusion_bbox_head.neg_monitor_sample_count']
    try:
        model.load_state_dict(broken, strict=False)
    except RuntimeError as error:
        assert 'Incomplete stage-2 adaptive state' in str(error)
    else:
        raise AssertionError('Incomplete adaptive state was silently accepted')
    rows = [json.loads(line) for line in (smoke / 'logs/iterations.jsonl').read_text().splitlines()]
    assert [r['step'] for r in rows] == [1, 2, 3, 4]
    assert all(r['query_audit']['masked_negatives'] == 0 for r in rows)
    assert all(np.isfinite(r['loss']) and np.isfinite(r['grad_norm']) for r in rows)
    report = dict(status='PASS', checkpoint_full_state_readable=True,
                  counters=dict(stage2=4, local=int(raw['meta']['local_iter'])),
                  frozen_detector_unchanged=True, all_model_tensors_roundtrip_exact=True,
                  optimizer_rng_temporal_saved=True, incomplete_state_rejected=True,
                  unique_frames_in_finite_order=6000, valid_query_mask_pass=True,
                  note='Exact state roundtrip, not a claim of bitwise deterministic GPU continuation.',
                  formal_training_started=False)
    json_write(out / 'checkpoint_audit.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--smoke-dir')
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = Config.fromfile(str(ROOT / 'projects/configs/nusc/prototype_stage2_labeled6000_10cls_20261025.py'))
    if args.smoke_dir:
        checkpoint_audit(cfg, out, Path(args.smoke_dir))
    else:
        data_audit(cfg, out)
    print('PREPARATION_AUDIT_PASS', flush=True)
