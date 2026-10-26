"""Bounded real-data checks for the selected global student augmentation."""

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
import copy
import gc
import json
from pathlib import Path
import sys
import traceback

import torch
from mmcv import Config
from scripts.ssl.train_online_ema_1011 import setup, digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    report_path = Path(args.report)
    if report_path.exists():
        raise RuntimeError('Refusing to overwrite an existing report')
    report_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = Config.fromfile(args.config)
    setup(cfg)
    from projects.mmdet3d_plugin.models.utils.student_augment_ab_v2_2026_10_14 import StudentPerturbationsAB20261014V2 as Aug
    import scripts.diagnostics.smoke_online_ema_1011 as smoke

    report = dict(status='RUNNING', config=args.config, forced_component_checks=[],
                  ordinary_augmentation_stats=[], formal_training_started=False)
    models = []
    original_build = smoke.build_detector
    original_call = Aug.__call__

    def build(*a, **kw):
        model = original_build(*a, **kw)
        models.append(model)
        return model

    def audited_call(self, data, boxes):
        assert self.scope == 'global'
        image = data['img'].clone()
        points = data['points'][0][0].clone()
        box_tensor = boxes.tensor.clone()
        labels = data['gt_labels_3d'][0][0].clone()
        calibration = data['lidar2img'].clone()
        assert image.is_cuda and points.is_cuda
        before_cfg = copy.deepcopy(self.target.cfg)
        before_global = copy.deepcopy(self.global_cfg)
        try:
            # Force both alternatives on actual GPU inputs without changing the
            # production settings or consuming the ordinary pass's RNG stream.
            with torch.random.fork_rng(devices=[image.device.index]):
                self.target.cfg['photo_probability'] = 1.0
                self.global_cfg.update(image_probability=1.0, point_probability=1.0)
                for seed, jitter in [(1019, 1.0), (1020, 0.0)]:
                    torch.manual_seed(seed)
                    self.target.cfg['jitter_probability'] = jitter
                    out, stats = original_call(self, data, boxes)
                    result = out['points'][0][0]
                    assert stats['aug_photo_views'] == 6
                    assert stats['aug_global_patches'] == 6
                    assert stats['aug_targets'] == 0
                    assert torch.equal(data['img'], image)
                    assert torch.equal(data['points'][0][0], points)
                    assert torch.equal(boxes.tensor, box_tensor)
                    assert torch.equal(out['gt_bboxes_3d'][0][0].tensor, box_tensor)
                    assert torch.equal(out['gt_labels_3d'][0][0], labels)
                    assert torch.equal(out['lidar2img'], calibration)
                    assert out['img'].data_ptr() != data['img'].data_ptr()
                    assert result.data_ptr() != data['points'][0][0].data_ptr()
                    if jitter:
                        assert len(result) == len(points)
                        assert torch.equal(result[:, 3:], points[:, 3:])
                        delta = (result[:, :3]-points[:, :3]).abs()
                        assert float(delta.max()) <= before_cfg['noise_clip'] + 1e-5
                        assert stats['aug_jitter_points'] > 0
                        stats['max_xyz_displacement_m'] = float(delta.max())
                    else:
                        assert 0 < stats['aug_dropped_points'] == len(points)-len(result)
                        assert .05-1/len(points) <= stats['aug_drop_fraction'] <= .10
                    report['forced_component_checks'].append(dict(
                        kind='global_jitter' if jitter else 'global_dropout',
                        points_before=len(points), points_after=len(result),
                        teacher_inputs_unchanged=True, labels_and_calibration_unchanged=True,
                        stats=stats))
                    print('REAL_GPU_COMPONENT_PASS', json.dumps(report['forced_component_checks'][-1]), flush=True)
                    del out, stats, result
        finally:
            self.target.cfg.clear(); self.target.cfg.update(before_cfg)
            self.global_cfg.clear(); self.global_cfg.update(before_global)
        out, stats = original_call(self, data, boxes)
        assert torch.equal(data['img'], image)
        assert torch.equal(data['points'][0][0], points)
        assert torch.equal(boxes.tensor, box_tensor)
        report['ordinary_augmentation_stats'].append(dict(stats))
        return out, stats

    try:
        smoke.build_detector = build
        Aug.__call__ = audited_call
        # Reuse the tested real/pseudo/empty, gradient, and exact EMA checks.
        sys.argv = ['scripts/diagnostics/smoke_online_ema_1011.py', '--config', args.config,
                    '--expected-labeled', '6000']
        smoke.main()
        model = models[0]
        assert int(model.ema_updates) == 2
        assert all(not p.requires_grad for p in model.teacher.parameters())
        assert all(not p.requires_grad for p in model.student.pts_backbone.parameters())
        assert digest(model.student.pts_backbone) == digest(model.teacher.pts_backbone)
        assert len(report['forced_component_checks']) == 2
        report.update(status='GPU_GLOBAL_PREFLIGHT_PASS', ema_updates=2,
                      pseudo_real_empty_checks=True, frozen_fsd_equal=True,
                      peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
    except Exception:
        report.update(status='FAILED', error=traceback.format_exc())
        raise
    finally:
        smoke.build_detector = original_build
        Aug.__call__ = original_call
        report_path.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print('GPU_GLOBAL_PREFLIGHT_PASS', flush=True)


if __name__ == '__main__':
    main()
