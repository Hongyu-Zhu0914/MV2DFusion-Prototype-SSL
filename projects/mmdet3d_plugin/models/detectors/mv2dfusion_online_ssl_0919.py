"""Opt-in, single-frame online teacher/student scaffold.

Teacher predictions and student targets share geometric augmentation. This is
not yet a weak/strong-view or temporal-consistency implementation. Teacher stays
fixed; prototype filtering requires a separately validated, aligned adapter.
"""
import copy
import torch
from mmcv import Config
from mmcv.parallel import DataContainer
from mmcv.runner import load_checkpoint
from mmdet.models import DETECTORS, build_detector
from mmdet.models.detectors.base import BaseDetector
from .mv2dfusion_offline_ssl_0915 import MV2DFusionOfflineSSL0915  # registration


def select_pseudo(pred, threshold, num_classes, point_cloud_range):
    """Strict confidence threshold; reject invalid labels, geometry and NaNs."""
    boxes, scores, labels = pred['boxes_3d'], pred['scores_3d'], pred['labels_3d']
    if not (len(boxes) == len(scores) == len(labels)):
        raise ValueError('Teacher box/score/label alignment mismatch')
    xyz = boxes.gravity_center
    limits = xyz.new_tensor(point_cloud_range)
    mask = (scores > threshold) & torch.isfinite(scores)
    mask &= torch.isfinite(boxes.tensor).all(dim=-1)
    mask &= (labels >= 0) & (labels < num_classes)
    mask &= (boxes.tensor[:, 3:6] > 0).all(dim=-1)
    mask &= ((xyz >= limits[:3]) & (xyz <= limits[3:])).all(dim=-1)
    return mask


@DETECTORS.register_module()
class MV2DFusionOnlineSSL0919(BaseDetector):
    def __init__(self, detector_config, initial_checkpoint, score_threshold=0.8,
                 pseudo_loss_weight=0.25, teacher_update='fixed',
                 prototype_filter=False, train_cfg=None, test_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        if teacher_update != 'fixed':
            raise NotImplementedError('Teacher EMA is intentionally not enabled pending design approval')
        if prototype_filter:
            raise NotImplementedError('Attach a validated per-box prototype similarity adapter first')
        if not 0 <= score_threshold <= 1:
            raise ValueError('score_threshold must be in [0, 1]')
        cfg = Config.fromfile(detector_config)
        architecture = copy.deepcopy(cfg.model)
        self.teacher = build_detector(copy.deepcopy(architecture))
        architecture['type'] = 'MV2DFusionOfflineSSL0915'
        architecture['pseudo_loss_weight'] = pseudo_loss_weight
        self.student = build_detector(architecture)
        self.initial_checkpoint = initial_checkpoint
        self.score_threshold = float(score_threshold)
        self.point_cloud_range = cfg.point_cloud_range
        self.num_classes = len(cfg.class_names)
        self.teacher_update = teacher_update
        self.last_online_stats = {}
        self._weights_initialized = False
        self.teacher.requires_grad_(False)
        self.teacher.eval()

    def init_weights(self):
        if self._weights_initialized:
            return
        # Do not run pretrained submodule initializers over the full baseline.
        load_checkpoint(self.teacher, self.initial_checkpoint, map_location='cpu', strict=True)
        load_checkpoint(self.student, self.initial_checkpoint, map_location='cpu', strict=True)
        # FSD normally applies this flag in init_weights. We load a complete
        # checkpoint directly, so preserve its freezing policy explicitly.
        if getattr(self.student.pts_backbone, 'freeze', False):
            self.student.pts_backbone.requires_grad_(False)
        self.teacher.requires_grad_(False)
        self.teacher.eval()
        self._weights_initialized = True

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        return self

    @staticmethod
    def reset_frame(model):
        model.fusion_bbox_head.reset_memory()
        model.prev_scene_token = None
        model.test_clip_id = 1

    def teacher_predict(self, data):
        # Only sensor/calibration inputs cross the teacher boundary. No GT.
        fields = ('img', 'lidar2img', 'intrinsics', 'extrinsics', 'timestamp',
                  'img_timestamp', 'ego_pose', 'ego_pose_inv', 'prev_exists')
        inputs = {key: data[key][:, 0] for key in fields if key in data}
        inputs['points'] = data['points'][0][0].unsqueeze(0)
        meta = {key: copy.deepcopy(value) for key, value in data['img_metas'][0][0].items()
                if not key.startswith('gt_') and key != 'ann_info'}
        for key in ('lidar2img', 'intrinsics', 'extrinsics'):
            meta[key] = inputs[key][0].detach().cpu().numpy()
        self.reset_frame(self.teacher)
        self.teacher.eval()
        with torch.no_grad():
            prediction = self.teacher.simple_test([meta], **inputs)[0]['pts_bbox']
        self.reset_frame(self.teacher)
        return prediction

    def forward_train(self, **data):
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            raise NotImplementedError('Online v1 is single GPU only')
        if data['img'].ndim != 6 or tuple(data['img'].shape[:2]) != (1, 1):
            raise ValueError('Online v1 supports batch=1, temporal queue=1 only')
        self.reset_frame(self.student)
        if 'prev_exists' in data:
            data['prev_exists'] = torch.zeros_like(data['prev_exists'])
        is_pseudo = bool(data['img_metas'][0][0].get('is_pseudo', False))
        self.last_online_stats = dict(is_pseudo=is_pseudo, candidates=0, selected=0)
        if is_pseudo:
            pred = self.teacher_predict(data)
            keep = select_pseudo(pred, self.score_threshold, self.num_classes, self.point_cloud_range)
            count = int(keep.sum())
            self.last_online_stats.update(candidates=len(keep), selected=count)
            if count == 0:
                # Distributed training is not supported by this first version.
                # Clear any stale gradient tensors retained by MMCV zero_grad;
                # otherwise AdamW momentum/decay can update an empty frame.
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
        return self.student(return_loss=True, **data)

    def forward(self, return_loss=True, **data):
        if return_loss:
            return self.forward_train(**data)
        return self.student(return_loss=False, **data)

    def extract_feat(self, *args, **kwargs):
        return self.student.extract_feat(*args, **kwargs)

    def simple_test(self, *args, **kwargs):
        return self.student.simple_test(*args, **kwargs)

    def aug_test(self, *args, **kwargs):
        return self.student.aug_test(*args, **kwargs)
