"""Opt-in EMA wrapper; historical fixed-teacher code is unchanged."""
import torch
from mmcv.runner import HOOKS, Hook
from mmdet.models import DETECTORS
from projects.mmdet3d_plugin.models.detectors.mv2dfusion_online_ssl_0919 import MV2DFusionOnlineSSL0919


@DETECTORS.register_module()
class MV2DFusionOnlineEMA1011(MV2DFusionOnlineSSL0919):
    def __init__(self, ema_decay=0.999, teacher_update='ema', **kwargs):
        if teacher_update != 'ema' or not 0 <= ema_decay < 1:
            raise ValueError('Expected EMA teacher and 0 <= decay < 1')
        super().__init__(teacher_update='fixed', **kwargs)
        self.teacher_update = 'ema'
        self.ema_decay = float(ema_decay)
        self.register_buffer('ema_updates', torch.zeros((), dtype=torch.long))
        self._ema_step_pending = False

    def forward_train(self, **data):
        self._ema_step_pending = False
        losses = super().forward_train(**data)
        self._ema_step_pending = 'loss_empty_pseudo' not in losses
        return losses

    @torch.no_grad()
    def update_teacher(self):
        if not self._ema_step_pending:
            return False
        teacher = dict(self.teacher.named_parameters())
        student = dict(self.student.named_parameters())
        if teacher.keys() != student.keys():
            raise RuntimeError('Teacher/student parameter names differ')
        for name, target in teacher.items():
            source = student[name]
            if target.shape != source.shape:
                raise RuntimeError('EMA shape mismatch: ' + name)
            if source.requires_grad:
                target.mul_(self.ema_decay).add_(source, alpha=1-self.ema_decay)
            else:
                # Preserve frozen weights exactly (avoid floating-point EMA drift).
                target.copy_(source)
        # Buffers (including BN statistics and integer counters) are copied,
        # never blindly averaged. Temporal memory is plain runtime state and reset.
        teacher_buffers = dict(self.teacher.named_buffers())
        student_buffers = dict(self.student.named_buffers())
        if teacher_buffers.keys() != student_buffers.keys():
            raise RuntimeError('Teacher/student buffer names differ')
        for name, target in teacher_buffers.items():
            target.copy_(student_buffers[name])
        self.reset_frame(self.teacher)
        self.teacher.eval()
        self.ema_updates.add_(1)
        self._ema_step_pending = False
        return True


@HOOKS.register_module()
class OnlineEMAHook1011(Hook):
    """Register at priority 45: FP32 optimizer 40, checkpoint 50."""
    def before_run(self, runner):
        names = [type(h).__name__ for h in runner._hooks]
        optimizer = names.index('OptimizerHook')
        ema = names.index(type(self).__name__)
        if optimizer >= ema:
            raise RuntimeError('EMA must execute after optimizer')
        if 'CheckpointHook' in names and ema >= names.index('CheckpointHook'):
            raise RuntimeError('Checkpoint must include current EMA update')

    def after_train_iter(self, runner):
        model = runner.model.module
        model.update_teacher()
        runner.log_buffer.update(dict(ema_updates=int(model.ema_updates),
                                     **model.last_online_stats), 1)
