"""CPU checks for the separately configurable global and target schemes."""
import argparse
import json
from pathlib import Path
import torch
from mmcv import Config
from projects.mmdet3d_plugin.models.utils.student_augment_2026_10_14 import (
    StudentPerturbations20261014,
)
from projects.mmdet3d_plugin.models.utils.student_augment_ab_v2_2026_10_14 import (
    StudentPerturbationsAB20261014V2,
)


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
        x = local[:,:,0]*c[:,None]-local[:,:,1]*s[:,None]
        y = local[:,:,0]*s[:,None]+local[:,:,1]*c[:,None]
        return torch.stack((x,y,local[:,:,2]),-1)+self.gravity_center[:,None]

    def __getitem__(self, item):
        return Boxes(self.tensor[item])


def sample():
    torch.manual_seed(20261014)
    raw = torch.rand(1,1,2,3,64,64)*200+20
    mean = torch.tensor([100.,110.,120.]).view(1,1,1,3,1,1)
    std = torch.tensor([20.,30.,40.]).view(1,1,1,3,1,1)
    image = (raw-mean)/std
    image[...,56:,:] = 0
    image[...,:,56:] = 0
    points = torch.rand(110,5)
    points[:100,:3] = (points[:100,:3]-.5)*1.5+torch.tensor([0.,0.,10.])
    points[100:,:3] = torch.tensor([8.,8.,10.])  # background, outside the pseudo box
    matrix = torch.tensor([[100.,0.,32.,0.],[0.,100.,32.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]])
    boxes = Boxes(torch.tensor([[0.,0.,9.,2.,2.,2.,0.,0.,0.]]))
    data = dict(img=image, points=[[points]], lidar2img=matrix.repeat(1,1,2,1,1),
                img_metas=[[dict(is_pseudo=True,img_shape=[(56,56,3)]*2,
                    img_norm_cfg=dict(mean=[100.,110.,120.],std=[20.,30.,40.],to_rgb=True))]])
    return data, boxes


def equal_tensors(a, b):
    if torch.is_tensor(a):
        return torch.equal(a,b)
    if isinstance(a, (list,tuple)):
        return len(a)==len(b) and all(equal_tensors(x,y) for x,y in zip(a,b))
    return a==b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', default='ssl_aug_ab_v2_2026-10-14/cpu_test_ab_v2_2026-10-14.json')
    args = parser.parse_args()
    data, boxes = sample()
    target_cfg = dict(enabled=True, photo_probability=1., photo_views=2,
                      object_probability=1., image_modality_probability=0.,
                      jitter_probability=1.)
    target = StudentPerturbations20261014(target_cfg,[-50,-50,-5,50,50,20])
    ab_target = StudentPerturbationsAB20261014V2(
        target_cfg,[-50,-50,-5,50,50,20],scope='target')
    torch.manual_seed(29); expected, expected_stats = target(data,boxes)
    torch.manual_seed(29); actual, actual_stats = ab_target(data,boxes)
    assert equal_tensors(expected['img'],actual['img'])
    assert equal_tensors(expected['points'],actual['points'])
    assert {k:v for k,v in expected_stats.items()} == {
        k:v for k,v in actual_stats.items() if k!='aug_scope_global'}

    # The global scheme must work without boxes and perturb out-of-box points.
    global_cfg = dict(enabled=True, photo_probability=1., photo_views=2,
                      object_probability=1., image_modality_probability=0.,
                      jitter_probability=1.)
    settings = dict(image_probability=1.,image_views=6,photo_views=6,
                    image_patch_area=(.10,.10),point_probability=1.)
    global_aug = StudentPerturbationsAB20261014V2(
        global_cfg,[-50,-50,-5,50,50,20],scope='global',global_cfg=settings)
    torch.manual_seed(11); global_with_box, stats = global_aug(data,boxes)
    torch.manual_seed(11); global_without_box, _ = global_aug(data,Boxes(boxes.tensor[:0]))
    assert stats['aug_global_patches']==2  # synthetic sample has two camera views
    assert stats['aug_global_jitter_points']==110
    assert torch.equal(global_with_box['img'],global_without_box['img'])
    assert torch.equal(global_with_box['points'][0][0],global_without_box['points'][0][0])
    assert not torch.equal(global_with_box['points'][0][0][100:,:3],data['points'][0][0][100:,:3])
    assert torch.equal(global_with_box['points'][0][0][:,3:],data['points'][0][0][:,3:])
    assert torch.equal(global_with_box['img'][...,56:,:],data['img'][...,56:,:])
    assert torch.equal(global_with_box['img'][...,:,56:],data['img'][...,:,56:])
    assert torch.equal(data['points'][0][0],sample()[0]['points'][0][0])

    # In global mode, full-frame patches may land outside all pseudo boxes.
    patch_cfg = dict(global_cfg, photo_probability=0.)
    patch_only = StudentPerturbationsAB20261014V2(
        patch_cfg,[-50,-50,-5,50,50,20],scope='global',global_cfg=settings)
    changed_outside_box = False
    for seed in range(20):
        torch.manual_seed(seed)
        patched, patch_stats = patch_only(data,boxes)
        diff = (patched['img']-data['img']).abs().sum(dim=3)[0,0]
        # The projected pseudo box is centered; inspect frame border/background.
        if (diff[:, :10, :].sum()+diff[:, -10:, :].sum()+
                diff[:, :, :10].sum()+diff[:, :, -10:].sum()).item()>0:
            changed_outside_box = True
            break
    assert changed_outside_box

    # A global dropout is independent of box membership and bounded by the configured ratio.
    drop_cfg = dict(global_cfg, jitter_probability=0., dropout_ratio=(.10,.10),
                    min_remaining_points=16)
    dropper = StudentPerturbationsAB20261014V2(
        drop_cfg,[-50,-50,-5,50,50,20],scope='global',global_cfg=settings)
    torch.manual_seed(5); dropped, drop_stats = dropper(data,boxes)
    assert drop_stats['aug_global_dropped_points']==11
    assert len(dropped['points'][0][0])==99

    configs = [
        Config.fromfile('projects/configs/nusc/ssl_online_r50_ema_aug_global_v2_2026_10_14.py'),
        Config.fromfile('projects/configs/nusc/ssl_online_r50_ema_aug_target_v2_2026_10_14.py'),
    ]
    assert [cfg.model.augmentation_scope for cfg in configs]==['global','target']
    assert configs[0].work_dir != configs[1].work_dir
    from scripts.ssl.train_online_ema_1011 import setup
    from mmdet.models import DETECTORS
    for cfg in configs:
        setup(cfg)
        assert DETECTORS.get(cfg.model.type) is not None
    report = dict(status='AB_CPU_TEST_PASS', checks=[
        'target_compat', 'global_box_independence', 'full_frame_patch',
        'whole_cloud_jitter', 'whole_cloud_dropout', 'padding_and_nonxyz_preserved',
        'config_isolation', 'wrapper_registration'],
        gpu_forward_backward_verified=False, formal_training_started=False,
        metrics_validated=False)
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__=='__main__':
    main()
