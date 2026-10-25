"""Cheap CPU tests for EMA math, buffer copying, skip and sampling resume."""
import copy
import torch
from torch import nn
from projects.mmdet3d_plugin.models.detectors.mv2dfusion_online_ema_1011 import MV2DFusionOnlineEMA1011
from train_online_ema_1011 import AlternatingSceneSampler


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.teacher = nn.Linear(2, 1)
        self.student = copy.deepcopy(self.teacher)
        self.teacher.requires_grad_(False)
        self.teacher.register_buffer('counter', torch.tensor(1))
        self.student.register_buffer('counter', torch.tensor(7))
        self.ema_decay = 0.9
        self.register_buffer('ema_updates', torch.tensor(0))
        self._ema_step_pending = True
    def reset_frame(self, model):
        pass


def main():
    m = Tiny()
    before = m.teacher.weight.clone()
    with torch.no_grad(): m.student.weight.add_(1)
    assert MV2DFusionOnlineEMA1011.update_teacher(m)
    assert torch.allclose(m.teacher.weight, before+0.1)
    assert int(m.teacher.counter) == 7 and int(m.ema_updates) == 1
    saved = copy.deepcopy(m.state_dict())
    assert not MV2DFusionOnlineEMA1011.update_teacher(m)
    assert all(torch.equal(v,m.state_dict()[k]) for k,v in saved.items())
    restored = Tiny(); restored.load_state_dict(saved, strict=True)
    assert int(restored.ema_updates) == 1
    class Dataset:
        supervised_length = 4
        data_infos = [dict(scene_token=str(i//2), timestamp=i) for i in range(8)]
        def __len__(self): return 8
    d = Dataset()
    order = list(AlternatingSceneSampler(d, 8))
    assert len(set(order)) == 8
    assert all((idx >= 4) == bool(step%2) for step,idx in enumerate(order))
    assert list(AlternatingSceneSampler(d, 8, start=3)) == order[3:]
    print('EMA_UNIT_PASS: math, integer buffer copy, skip, state restore, sampler resume', flush=True)


if __name__ == '__main__': main()
