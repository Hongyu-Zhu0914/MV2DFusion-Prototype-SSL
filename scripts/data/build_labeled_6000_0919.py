"""Materialize the existing approved scene selection; preserve full source infos."""

# Resolve imports and relative CLI paths after the publication-layout migration.
import sys as _bootstrap_sys
import os as _bootstrap_os
from pathlib import Path as _BootstrapPath
REPO_ROOT = _BootstrapPath(__file__).resolve().parents[2]
if str(REPO_ROOT) not in _bootstrap_sys.path:
    _bootstrap_sys.path.insert(0, str(REPO_ROOT))
if __name__ == '__main__':
    _bootstrap_os.chdir(str(REPO_ROOT))

import hashlib
import json
import pickle
from collections import Counter
from pathlib import Path
import numpy as np
from nuscenes.utils.splits import create_splits_scenes


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    root = Path('/root/autodl-tmp/data/nuscenes')
    selection = root / 'scene_selection_6000_0916'
    split = root / 'ssl_splits'
    source = root / 'nuscenes2d_temporal_infos_train.pkl'
    output = split / 'nuscenes_ssl_train_labeled_150scenes_6000_0919.pkl'
    if output.exists():
        raise FileExistsError(output)
    comparison = json.loads((selection / 'selection_comparison.json').read_text())
    candidate = comparison['recommended_candidate']
    assert candidate == 'balanced_150scenes_6000samples', candidate
    scene_path = selection / (candidate + '_scene_tokens.txt')
    sample_path = selection / (candidate + '_sample_tokens.txt')
    scenes = set(scene_path.read_text().split())
    tokens = set(sample_path.read_text().split())
    assert len(scenes) == 150 and len(tokens) == 6000
    meta_scenes = json.loads((root / 'v1.0-trainval/scene.json').read_text())
    meta_samples = json.loads((root / 'v1.0-trainval/sample.json').read_text())
    official_train = set(create_splits_scenes()['train'])
    assert all(x['name'] in official_train for x in meta_scenes if x['token'] in scenes)
    expected = {x['token'] for x in meta_samples if x['scene_token'] in scenes}
    assert expected == tokens, 'Incomplete scenes in selection'
    with source.open('rb') as f:
        original = pickle.load(f)
    infos = [x for x in original['infos'] if x['token'] in tokens]
    assert len(infos) == len({x['token'] for x in infos}) == 6000
    assert {x['scene_token'] for x in infos} == scenes
    counts = Counter(x['scene_token'] for x in infos)
    assert counts == Counter(x['scene_token'] for x in meta_samples if x['scene_token'] in scenes)
    other_checks = {}
    for name in ['nuscenes_ssl_train_unlabeled_official_test.pkl',
                 'nuscenes_ssl_val_split.pkl', 'nuscenes_ssl_test_split.pkl']:
        path = split / name
        with path.open('rb') as f:
            other = pickle.load(f)
        assert not tokens.intersection(x['token'] for x in other['infos']), name
        assert not scenes.intersection(x['scene_token'] for x in other['infos']), name
        other_checks[name] = dict(frames=len(other['infos']), sha256=sha(path), overlap=0)
        del other
    paths = set()
    boxes = Counter()
    for info in infos:
        assert not info.get('is_pseudo', False)
        for key in ['gt_boxes', 'gt_names', 'valid_flag', 'bboxes2d', 'labels2d',
                    'bboxes3d_cams', 'centers2d', 'depths', 'bboxes_ignore']:
            assert key in info, (info['token'], key)
        assert np.isfinite(info['gt_boxes']).all()
        boxes.update(np.asarray(info['gt_names'])[np.asarray(info['valid_flag'], dtype=bool)].tolist())
        paths.add(info['lidar_path'])
        paths.update(cam['data_path'] for cam in info['cams'].values())
        paths.update(sweep['data_path'] for sweep in info.get('sweeps', []))
    missing = []
    for raw in paths:
        path = Path(raw)
        if not path.exists() and not (root / path).exists():
            missing.append(raw)
    assert not missing, ('Missing sensor files', len(missing), missing[:10])
    result = dict(original)
    result['infos'] = infos
    # Exclusive create: old splits and an existing output are never overwritten.
    with output.open('xb') as f:
        pickle.dump(result, f, protocol=4)
    with output.open('rb') as f:
        reread = pickle.load(f)
    assert [x['token'] for x in reread['infos']] == [x['token'] for x in infos]
    assert reread['metadata'] == original['metadata']
    report = dict(source=str(source), source_sha256=sha(source), output=str(output),
                  output_sha256=sha(output), selection=candidate,
                  selection_sha256=sha(scene_path), frames=len(infos), scenes=len(scenes),
                  logs=len({x['log_token'] for x in meta_scenes if x['token'] in scenes}),
                  valid_boxes_by_class=dict(boxes), checked_sensor_paths=len(paths),
                  missing_sensor_paths=0, other_splits=other_checks,
                  whole_scenes=True, original_info_fields_preserved=True)
    report_path = output.with_suffix('.validation.json')
    with report_path.open('x') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
