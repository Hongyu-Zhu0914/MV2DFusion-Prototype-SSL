"""Freeze the detector and BN state, learning only projection + feature norm."""
import torch
from collections import OrderedDict
from mmdet.models import DETECTORS
from .mv2dfusion import MV2DFusion

PROTO_PREFIXES = ('fusion_bbox_head.prototypes', 'fusion_bbox_head.proto_',
                  'fusion_bbox_head.feat_norm.', 'fusion_bbox_head.local_iter')


def is_proto_state(name):
    return name.startswith(PROTO_PREFIXES)


@DETECTORS.register_module()
class PrototypeOnlyDetector0920(MV2DFusion):
    def __init__(self, baseline_checkpoint='weights/r50.pth', **kwargs):
        super().__init__(**kwargs)
        self.baseline_checkpoint = baseline_checkpoint
        self.baseline_load_report = None
        self._apply_freeze()

    def _apply_freeze(self):
        for name, param in self.named_parameters():
            param.requires_grad_(name.startswith(('fusion_bbox_head.proto_proj.',
                                                 'fusion_bbox_head.feat_norm.')))
        for module in self.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()

    def init_weights(self):
        if self.baseline_load_report is not None:
            return
        raw = torch.load(self.baseline_checkpoint, map_location='cpu')
        source = raw.get('state_dict', raw)
        state = OrderedDict((k[7:] if k.startswith('module.') else k, v)
                            for k, v in source.items())
        # spconv uses module version metadata to interpret kernel layout.
        # Dropping it can trigger an incorrect legacy-weight permutation.
        if hasattr(source, '_metadata'):
            state._metadata = OrderedDict((k[7:] if k.startswith('module.') else k, v)
                                          for k, v in source._metadata.items())
        if any(is_proto_state(k) for k in state):
            raise ValueError('Expected detector-only baseline, not a previous prototype checkpoint')
        result = self.load_state_dict(state, strict=False)
        unexpected_missing = [k for k in result.missing_keys if not is_proto_state(k)]
        if result.unexpected_keys or unexpected_missing:
            raise RuntimeError((result.unexpected_keys, unexpected_missing))
        head = self.fusion_bbox_head
        assert int(head.local_iter) == 0 and int(head.proto_update_count.sum()) == 0
        assert int(head.proto_feature_queue_count.sum()) == 0
        self.baseline_load_report = dict(missing_new_keys=result.missing_keys,
                                         unexpected_keys=result.unexpected_keys,
                                         local_iter=0, inherited_old_prototypes=False)
        self._apply_freeze()

    def train(self, mode=True):
        super().train(mode)
        self._apply_freeze()
        return self
