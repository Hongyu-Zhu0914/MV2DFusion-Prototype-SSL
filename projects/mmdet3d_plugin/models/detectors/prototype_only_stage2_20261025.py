"""Stage-2 initialization from a full learned Stage-1 model, not an R50 reset."""
from collections import OrderedDict
from pathlib import Path

import torch
from mmdet.models import DETECTORS
from .prototype_only_detector_0920 import PrototypeOnlyDetector0920


PROTO_STATE_PREFIXES = (
    'fusion_bbox_head.prototypes', 'fusion_bbox_head.proto_',
    'fusion_bbox_head.feat_norm.', 'fusion_bbox_head.local_iter',
    'fusion_bbox_head.neg_', 'fusion_bbox_head.stage2_',
)


def is_stage2_state(name):
    return name.startswith(PROTO_STATE_PREFIXES)


@DETECTORS.register_module()
class PrototypeOnlyStage2Detector20261025(PrototypeOnlyDetector0920):
    def __init__(self, stage1_checkpoint, **kwargs):
        kwargs.pop('baseline_checkpoint', None)
        self.stage1_checkpoint = str(stage1_checkpoint)
        self.stage2_initialization = None
        super().__init__(baseline_checkpoint=self.stage1_checkpoint, **kwargs)

    def init_weights(self):
        if self.stage2_initialization is not None:
            return
        path = Path(self.stage1_checkpoint)
        if not path.is_file():
            raise FileNotFoundError(path)
        raw = torch.load(str(path), map_location='cpu')
        source = raw['state_dict']
        if not any(k.startswith('fusion_bbox_head.proto_proj.') for k in source):
            raise ValueError('A complete, learned Stage-1 checkpoint is required')
        if any(k.startswith('fusion_bbox_head.stage2_') for k in source):
            raise ValueError('Use --resume-checkpoint for a Stage-2 checkpoint')
        # Preserve spconv version metadata; never rebuild a plain metadata-free dict.
        state = OrderedDict((k, v) for k, v in source.items())
        if hasattr(source, '_metadata'):
            state._metadata = source._metadata.copy()
        result = self.load_state_dict(state, strict=True)
        head = self.fusion_bbox_head
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(result)
        if head.stage2_load_kind != 'stage1_initialization' or int(head.stage2_steps) != 0:
            raise RuntimeError('Unexpected Stage-1 to Stage-2 initialization')
        head._check_stage1_ready()
        self.stage2_initialization = dict(
            checkpoint=str(path.resolve()), strict_load=True,
            stage1_local_iter=int(head.local_iter), stage2_steps=0,
            old_checkpoint_meta_iter=raw.get('meta', {}).get('iter'),
            load_kind=head.stage2_load_kind,
            update_counts=head.proto_update_count.sum(1).tolist(),
        )
        self._apply_freeze()
