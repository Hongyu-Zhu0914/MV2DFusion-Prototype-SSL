"""Launch the authorized 12000-attempt run once, with final student validation."""
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/root/MV2DFusion-main')
OUT = Path('/root/autodl-tmp/data/mv2dfusion_results/ssl_aug_global_2026-10-19')
CONFIG = ROOT/'projects/configs/nusc/ssl_online_r50_ema_aug_global_2026_10_19.py'
EXPECTED_CONFIG_SHA256 = '216d6ad1453bcf06cbb7e6928f3b800a9b8a8b8a1aa4f0a36537b85b32d4a246'


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def main():
    if hashlib.sha256(CONFIG.read_bytes()).hexdigest() != EXPECTED_CONFIG_SHA256:
        raise RuntimeError('Config changed since approved score-0.2 preflight')
    if any(OUT.glob('iter_*.pth')) or (OUT/'train_console.log').exists():
        raise RuntimeError('Existing formal run; refusing a duplicate launch')
    assert (OUT/'preflight_score02/exit_code').read_text().strip() == '0'
    (OUT/'formal_launch_lock').mkdir()  # Atomic guard; never automatically removed.
    command = [sys.executable, '-u', str(ROOT/'train_online_ema_1011.py'),
               '--config', str(CONFIG), '--work-dir', str(OUT)]
    state = dict(status='starting', started_at_beijing=now(), command=command,
                 config_sha256=EXPECTED_CONFIG_SHA256, score_threshold=0.2,
                 max_attempts=12000, initialization='weights/r50.pth',
                 final_student_val_frames=3014, screen='ssl_global_score02_12000_1019')
    (OUT/'launch_manifest_2026-10-19.json').write_text(json.dumps(state, indent=2)+'\n')
    with (OUT/'train_console.log').open('x') as log:
        result = subprocess.run(command, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
                                env=dict(os.environ, PYTHONPATH=str(ROOT), OMP_NUM_THREADS='1'))
    (OUT/'train_exit_code').write_text(str(result.returncode)+'\n')
    state.update(status='complete' if result.returncode == 0 else 'failed',
                 finished_at_beijing=now(), exit_code=result.returncode)
    (OUT/'launch_manifest_2026-10-19.json').write_text(json.dumps(state, indent=2)+'\n')
    return result.returncode


if __name__ == '__main__':
    sys.exit(main())
