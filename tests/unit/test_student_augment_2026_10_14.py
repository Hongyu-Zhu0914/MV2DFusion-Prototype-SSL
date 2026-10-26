"""CPU deterministic invariants; no real detector training or weight updates."""

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
import importlib.util
import json
from pathlib import Path
import torch


class Boxes:
    def __init__(self, tensor):
        self.tensor = tensor
    def __len__(self):
        return len(self.tensor)
    @property
    def gravity_center(self):
        center = self.tensor[:, :3].clone()
        center[:, 2] += self.tensor[:, 5]/2
        return center
    @property
    def corners(self):
        signs = self.tensor.new_tensor([[-1,-1,-1],[-1,-1,1],[-1,1,-1],[-1,1,1],
                                       [1,-1,-1],[1,-1,1],[1,1,-1],[1,1,1]])
        local = signs[None]*self.tensor[:, None, 3:6]/2
        c, s = torch.cos(self.tensor[:, 6]), torch.sin(self.tensor[:, 6])
        x = local[:, :, 0]*c[:, None] - local[:, :, 1]*s[:, None]
        y = local[:, :, 0]*s[:, None] + local[:, :, 1]*c[:, None]
        return torch.stack((x,y,local[:,:,2]), -1)+self.gravity_center[:,None]
    def __getitem__(self, mask):
        return Boxes(self.tensor[mask])
    def to(self, device):
        return Boxes(self.tensor.to(device))


def inputs():
    torch.manual_seed(12)
    raw = torch.rand(1, 1, 2, 3, 64, 64)*200+20
    mean = torch.tensor([100.,110.,120.]).view(1,1,1,3,1,1)
    std = torch.tensor([20.,30.,40.]).view(1,1,1,3,1,1)
    image = (raw-mean)/std
    image[...,56:,:] = 0; image[...,:,56:] = 0
    points = torch.rand(110, 5)
    points[:100,:3] = (points[:100,:3]-.5)*1.5 + torch.tensor([0.,0.,10.])
    points[100:,:3] = torch.tensor([8.,8.,10.])
    matrix = torch.tensor([[100.,0.,32.,0.],[0.,100.,32.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]])
    boxes = Boxes(torch.tensor([[0.,0.,9.,2.,2.,2.,0.,0.,0.]]))
    return dict(img=image, points=[[points]],
                lidar2img=matrix.repeat(1,1,2,1,1),
                img_metas=[[dict(is_pseudo=True,img_shape=[(56,56,3)]*2,
                    img_norm_cfg=dict(mean=[100.,110.,120.],std=[20.,30.,40.],to_rgb=True))]],
                gt_bboxes_3d=[[boxes]], gt_labels_3d=[[torch.tensor([0])]],
                prev_exists=torch.ones(1,1)), boxes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--helper', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--wrapper-tests', action='store_true')
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('student_helper', args.helper)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    Aug = module.StudentPerturbations20261014
    data, boxes = inputs()
    image, points = data['img'].clone(), data['points'][0][0].clone()
    labels, calibration = data['gt_labels_3d'][0][0].clone(), data['lidar2img'].clone()
    base = dict(photo_probability=0., object_probability=1., image_modality_probability=0.)
    assert module.project_box(boxes.corners[0], calibration[0,0,0],56,56) is not None
    behind = boxes.corners[0].clone(); behind[:,2] *= -1
    assert module.project_box(behind,calibration[0,0,0],56,56) is None
    # Membership must use rotated local axes, not an axis-aligned envelope.
    mask = module.points_in_box(torch.tensor([[0.,.9,0.],[.9,0.,0.]]),
        torch.zeros(3),torch.tensor([2.,.4,2.]),torch.tensor(torch.pi/2))
    assert mask.tolist() == [True, False]
    out, stats = Aug(dict(base,jitter_probability=1.),[-50,-50,-5,50,50,20])(data,boxes)
    assert stats['aug_jitter_points'] == 100 and len(out['points'][0][0]) == 110
    difference = out['points'][0][0][:,:3]-points[:,:3]
    assert difference.abs().max() <= .020001 and difference.abs().sum() > 0
    assert torch.equal(out['points'][0][0][100:], points[100:])
    assert torch.equal(out['points'][0][0][:,3:],points[:,3:])
    out, stats = Aug(dict(base,jitter_probability=0.,dropout_ratio=(.1,.1)))(data,boxes)
    assert stats['aug_dropped_points'] == 10 and len(out['points'][0][0]) == 100
    assert torch.equal(out['points'][0][0][-10:], points[-10:])
    # Sparse objects are protected instead of being erased by dropout.
    sparse = dict(data,points=[[points[-15:].clone()]])
    _, stats = Aug(dict(base,jitter_probability=0.))(sparse,boxes)
    assert stats['aug_dropped_points'] == 0 and stats['aug_skipped_sparse'] == 1
    operations = set()
    for seed in range(12):
        torch.manual_seed(seed)
        out, stats = Aug(dict(base,image_modality_probability=1.))(data,boxes)
        assert stats['aug_image_patches'] == 1
        assert torch.equal(out['points'][0][0],points)
        assert not torch.equal(out['img'],image)
        assert torch.equal(out['img'][...,56:,:],image[...,56:,:])
        assert torch.equal(out['img'][...,:,56:],image[...,:,56:])
        assert torch.equal(out['lidar2img'],calibration)
        for operation in ('erase', 'mosaic', 'blur'):
            if stats['aug_'+operation+'_patches']:
                operations.add(operation)
    assert operations == {'erase','mosaic','blur'}, operations
    out, stats = Aug(dict(photo_probability=1.,object_probability=0.))(data,boxes)
    assert stats['aug_photo_views'] == 2 and not torch.equal(out['img'],image)
    assert torch.equal(out['img'][...,56:,:],image[...,56:,:])
    disabled = Aug(dict(enabled=False))
    rng = torch.get_rng_state().clone()
    out, stats = disabled(data,boxes)
    assert torch.equal(torch.get_rng_state(),rng) and torch.equal(out['img'],image)
    out['img'].zero_(); out['points'][0][0].zero_()
    assert torch.equal(data['img'],image) and torch.equal(data['points'][0][0],points)
    assert torch.equal(data['gt_labels_3d'][0][0],labels)
    assert torch.equal(data['lidar2img'],calibration)
    assert torch.equal(boxes.tensor,inputs()[1].tensor)
    checks = ['input_alias_isolation','noise_bound','non_xyz_preserved','background_preserved',
              'dropout_budget','sparse_protection','projection_and_near_plane',
              'oriented_membership','image_padding','disabled_rng_unchanged','all_image_operations']
    if args.wrapper_tests:
        from scripts.ssl.train_online_ema_1011 import setup
        from mmcv import Config
        cfg = Config.fromfile('projects/configs/nusc/ssl_online_r50_ema_aug_2026_10_14.py')
        setup(cfg)
        from projects.mmdet3d_plugin.models.detectors.mv2dfusion_online_ema_aug_2026_10_14 import MV2DFusionOnlineEMAAug20261014 as Wrapper
        class Student(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.weight=torch.nn.Parameter(torch.tensor(1.)); self.calls=0
                self.use_grid_mask=True
            def forward(self,**kw):
                self.calls+=1; self.received=kw; self.gridmask_used=self.use_grid_mask
                return {'loss_fake': self.weight.square()}
        class Tiny:
            def __init__(self, threshold=.8):
                self.student=Student(); self.score_threshold=threshold
                self.num_classes=10; self.point_cloud_range=[-50,-50,-5,50,50,20]
                self.augment_labeled=False
                self.disable_student_gridmask=True
                self.student_perturbations=Aug(dict(base,jitter_probability=1.))
            def reset_frame(self,model): pass
            def teacher_predict(self,data):
                assert torch.equal(data['img'],image) and torch.equal(data['points'][0][0],points)
                return dict(boxes_3d=boxes, scores_3d=torch.tensor([.9]),labels_3d=torch.tensor([0]))
        tiny=Tiny(); losses=Wrapper.forward_train(tiny,**data)
        assert tiny._ema_step_pending and tiny.student.calls==1
        assert tiny.last_online_stats['selected']==1 and tiny.last_online_stats['aug_jitter_points']==100
        assert torch.equal(data['points'][0][0],points)
        assert not tiny.student.gridmask_used and tiny.student.use_grid_mask
        tiny=Tiny(1.); tiny.student.weight.grad=torch.ones_like(tiny.student.weight)
        losses=Wrapper.forward_train(tiny,**data)
        assert not tiny._ema_step_pending and tiny.student.calls==0 and tiny.student.weight.grad is None
        assert 'loss_empty_pseudo' in losses
        labeled=copy.deepcopy(data); labeled['img_metas'][0][0]['is_pseudo']=False
        tiny=Tiny(); Wrapper.forward_train(tiny,**labeled)
        assert tiny.student.calls==1 and not tiny.last_online_stats['aug_enabled']
        assert torch.equal(tiny.student.received['points'][0][0],points)
        assert tiny.student.gridmask_used
        from projects.mmdet3d_plugin.models.detectors.mv2dfusion_online_ema_1011 import MV2DFusionOnlineEMA1011
        assert Wrapper.update_teacher is MV2DFusionOnlineEMA1011.update_teacher
        checks += ['teacher_clean_boundary','selected_pseudo_injection','empty_skip_no_ema',
                   'labeled_branch_unchanged','original_ema_inherited','config_imports',
                   'pseudo_gridmask_suspended_and_restored']
    report = dict(status='CPU_TEST_PASS',checks=checks,
                  gpu_forward_backward_verified=False, parameters_experimentally_validated=False)
    output=Path(args.output); output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)

if __name__=='__main__': main()
