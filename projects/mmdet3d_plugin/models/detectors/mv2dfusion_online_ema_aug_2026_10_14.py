"""Opt-in augmentation wrapper; old EMA and fixed-teacher implementations unchanged."""
import copy
import torch
from mmcv.parallel import DataContainer
from mmdet.models import DETECTORS
from .mv2dfusion_online_ema_1011 import MV2DFusionOnlineEMA1011
from .mv2dfusion_online_ssl_0919 import select_pseudo
from ..utils.student_augment_2026_10_14 import StudentPerturbations20261014


@DETECTORS.register_module()
class MV2DFusionOnlineEMAAug20261014(MV2DFusionOnlineEMA1011):
    def __init__(self, student_augmentation=None, augment_labeled=False,
                 disable_student_gridmask=True, **kwargs):
        super().__init__(**kwargs)
        self.augment_labeled = bool(augment_labeled)
        self.student_perturbations = StudentPerturbations20261014(
            student_augmentation, self.point_cloud_range)
        self.disable_student_gridmask = bool(disable_student_gridmask)

    def forward_train(self, **data):
        # Do not perturb before parent teacher_predict; instead preserve its flow
        # here and insert the student-copy boundary immediately before the student.
        self._ema_step_pending = False
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            raise NotImplementedError('Augmentation wrapper is single GPU only')
        if data['img'].ndim != 6 or tuple(data['img'].shape[:2]) != (1, 1):
            raise ValueError('Requires B=1, temporal queue=1')
        self.reset_frame(self.student)
        if 'prev_exists' in data:
            data['prev_exists'] = torch.zeros_like(data['prev_exists'])
        is_pseudo = bool(data['img_metas'][0][0].get('is_pseudo', False))
        self.last_online_stats = dict(is_pseudo=is_pseudo, candidates=0, selected=0,
                                     aug_enabled=False)
        if is_pseudo:
            # Clean shared view is used before any student-only perturbation.
            pred = self.teacher_predict(data)
            keep = select_pseudo(pred, self.score_threshold, self.num_classes, self.point_cloud_range)
            count = int(keep.sum())
            self.last_online_stats.update(candidates=len(keep), selected=count)
            if not count:
                for parameter in self.student.parameters():
                    parameter.grad = None
                return {'loss_empty_pseudo': torch.zeros((), device=data['img'].device,
                                                        requires_grad=True)}
            device = data['img'].device
            boxes = pred['boxes_3d'][keep].to(device)
            labels = pred['labels_3d'][keep].to(device=device, dtype=torch.long)
            data['gt_bboxes_3d'] = [[boxes]]
            data['gt_labels_3d'] = [[labels]]
            data['img_metas'] = copy.deepcopy(data['img_metas'])
            data['img_metas'][0][0]['gt_bboxes_3d'] = DataContainer(boxes, cpu_only=True)
            data['img_metas'][0][0]['gt_labels_3d'] = DataContainer(labels)
        if is_pseudo or self.augment_labeled:
            boxes = data['gt_bboxes_3d'][0][0]
            data, stats = self.student_perturbations(data, boxes)
            self.last_online_stats.update(stats)
        # Otherwise keep the original labeled supervision and input unchanged.
        original_gridmask = self.student.use_grid_mask
        if is_pseudo and self.disable_student_gridmask:
            self.student.use_grid_mask = False
        try:
            losses = self.student(return_loss=True, **data)
        finally:
            self.student.use_grid_mask = original_gridmask
        self._ema_step_pending = 'loss_empty_pseudo' not in losses
        return losses
