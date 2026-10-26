"""Single-GPU entry point: SyncBN context without a DDP reducer.

Uses the repository training runner unchanged. World-size=1 permits SyncBN,
while MMDataParallel tolerates different used parameters on real/pseudo steps.
Example (only when training is explicitly desired):
 PYTHONPATH=. python train_online_ssl_0919.py projects/configs/nusc/ssl_online_r50_fixed_teacher_0919.py --no-validate
The checked-in scaffold config is bounded to TWO iterations, not a full run.
"""

# Resolve imports and relative CLI paths after the publication-layout migration.
import sys as _bootstrap_sys
import os as _bootstrap_os
from pathlib import Path as _BootstrapPath
REPO_ROOT = _BootstrapPath(__file__).resolve().parents[2]
if str(REPO_ROOT) not in _bootstrap_sys.path:
    _bootstrap_sys.path.insert(0, str(REPO_ROOT))
if __name__ == '__main__':
    _bootstrap_os.chdir(str(REPO_ROOT))

import os
import runpy
import sys
import tempfile
import torch
from mmcv import Config


def main():
    if len(sys.argv) < 2:
        raise SystemExit('Pass the online config path, followed by tools/train.py arguments')
    if '--launcher' in sys.argv or int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise SystemExit('This entry point is single GPU only; do not use torchrun/DDP')
    cfg = Config.fromfile(sys.argv[1])
    if cfg.model.type != 'MV2DFusionOnlineSSL0919':
        raise SystemExit('Only the online SSL wrapper is supported')
    torch.cuda.set_device(0)
    with tempfile.TemporaryDirectory(prefix='mv2d_online_') as directory:
        torch.distributed.init_process_group('nccl', init_method='file://' + directory + '/rendezvous',
                                             rank=0, world_size=1)
        try:
            sys.argv[0] = 'tools/train.py'
            sys.argv.extend(['--launcher', 'none'])
            runpy.run_path('tools/train.py', run_name='__main__')
        finally:
            torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
