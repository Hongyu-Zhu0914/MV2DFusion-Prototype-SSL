"""Bounded real-input CPU checks. No detector or optimizer is constructed."""

# Resolve imports and relative CLI paths after the publication-layout migration.
import sys as _bootstrap_sys
import os as _bootstrap_os
from pathlib import Path as _BootstrapPath
REPO_ROOT = _BootstrapPath(__file__).resolve().parents[2]
if str(REPO_ROOT) not in _bootstrap_sys.path:
    _bootstrap_sys.path.insert(0, str(REPO_ROOT))
if __name__ == '__main__':
    _bootstrap_os.chdir(str(REPO_ROOT))

import copy
import json
from pathlib import Path
import torch
from mmcv import Config
from mmcv.parallel import DataContainer, collate
from mmdet3d.datasets import build_dataset
from scripts.ssl.train_online_ema_1011 import setup
from projects.mmdet3d_plugin.models.utils.student_augment_2026_10_14 import StudentPerturbations20261014


def unpack(value):
    # Collate containers hold one CPU device group. Metadata can itself contain
    # non-collated DataContainers; leave their contents intact like MMCV scatter.
    if isinstance(value,DataContainer): return value.data[0]
    if isinstance(value,dict): return {k:unpack(v) for k,v in value.items()}
    if isinstance(value,list): return [unpack(v) for v in value]
    if isinstance(value,tuple): return tuple(unpack(v) for v in value)
    return value


def main():
    cfg=Config.fromfile('projects/configs/nusc/ssl_online_r50_ema_aug_2026_10_14.py');setup(cfg)
    dataset=build_dataset(cfg.data.train)
    assert (dataset.supervised_length,dataset.pseudo_length)==(6000,6008)
    architecture=Config.fromfile(cfg.model.detector_config)
    reports=[]
    for name,index in [('labeled',0),('unlabeled',dataset.supervised_length)]:
        torch.manual_seed(1014)
        data=unpack(collate([dataset[index]],samples_per_gpu=1))
        boxes=data['gt_bboxes_3d'][0][0]
        original_img=data['img'].clone(); original_pts=data['points'][0][0].clone()
        matrices=data['lidar2img'].clone()
        # Force branches for an interface test only. Production parameters untouched.
        settings=dict(cfg.model.student_augmentation)
        settings.update(photo_probability=1.,object_probability=1.)
        outputs=[]
        for modality in (0.,1.):
            settings['image_modality_probability']=modality
            out,stats=StudentPerturbations20261014(settings,architecture.point_cloud_range)(data,boxes)
            assert torch.equal(data['img'],original_img) and torch.equal(data['points'][0][0],original_pts)
            assert torch.equal(out['lidar2img'],matrices)
            assert out['img'].shape==data['img'].shape and torch.isfinite(out['img']).all()
            assert out['points'][0][0].shape[1]==original_pts.shape[1]
            assert torch.equal(out['gt_bboxes_3d'][0][0].tensor,boxes.tensor)
            outputs.append(stats)
        reports.append(dict(domain=name,img_shape=list(data['img'].shape),
                            points_before=len(original_pts),boxes=len(boxes),checks=outputs))
        del data,out,original_img,original_pts
    report=dict(status='REAL_INPUT_CPU_PASS',frames=2,reports=reports,
                note='Unlabeled PKL has empty GT; real teacher pseudo prediction needs GPU.')
    p=Path('/root/autodl-tmp/data/mv2dfusion_results/augmentation_2026-10-14/real_input_checks_2026-10-14.json')
    p.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
