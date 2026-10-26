"""Idempotent smoke -> val -> internal-test export; never train or shut down."""

# Resolve imports and relative CLI paths after the publication-layout migration.
import sys as _bootstrap_sys
import os as _bootstrap_os
from pathlib import Path as _BootstrapPath
REPO_ROOT = _BootstrapPath(__file__).resolve().parents[2]
if str(REPO_ROOT) not in _bootstrap_sys.path:
    _bootstrap_sys.path.insert(0, str(REPO_ROOT))
if __name__ == '__main__':
    _bootstrap_os.chdir(str(REPO_ROOT))

import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(str(REPO_ROOT))
RUN = ROOT / 'work_dirs/prototype_stage2_r50_labeled6000_10cls_20261025'
CFG = ROOT / 'projects/configs/nusc/prototype_stage2_labeled6000_10cls_20261025.py'
CKPT = RUN / 'checkpoints/iter_6000.pth'
SCRIPT = ROOT / 'scripts/evaluation/export_stage2_aligned_20261025.py'
STATUS = RUN / 'analysis/postprocess_exports_status_20261025.json'


def status(payload):
    payload['time_epoch'] = time.time()
    tmp = STATUS.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(STATUS)


def main():
    os.chdir(str(ROOT))
    assert (RUN / 'train_exit_code').read_text().strip() == '0'
    audit = json.loads((RUN / 'analysis/final_checkpoint_audit_20261025.json').read_text())
    assert audit['status'] == 'PASS' and audit['completed_steps'] == 6000
    lock = RUN / 'analysis/postprocess_exports_20261025.lock'
    fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    done = []
    phase = None
    try:
        phases = [
            ('smoke_2', 'val', 2, RUN / 'preflight/export_smoke_2_20261025/evaluation',
             RUN / 'preflight/export_smoke_2_20261025/features'),
            ('custom_val_3014', 'val', 3014, RUN / 'evaluation/stage2_local12000_val',
             RUN / 'features/stage2_local12000_val'),
            ('internal_test_3005', 'test', 3005, RUN / 'evaluation/stage2_local12000_test',
             RUN / 'features/stage2_local12000_test'),
        ]
        for phase, split, frames, output, features in phases:
            summary = output / 'export_summary.json'
            if summary.exists():
                result = json.loads(summary.read_text())
                assert result['status'] == 'PASS' and result['frames'] == frames
                assert result['checkpoint_sha256'] == audit['checkpoint_sha256']
                done.append(phase)
                continue
            status(dict(status='RUNNING', current_phase=phase, completed_phases=done))
            command = [sys.executable, '-u', str(SCRIPT), '--config', str(CFG), '--checkpoint', str(CKPT),
                       '--split', split, '--output-dir', str(output), '--feature-dir', str(features),
                       '--seed', '920', '--temporal-mode', 'scene-stream']
            if phase == 'smoke_2':
                command += ['--limit', '2', '--skip-metrics']
            log = RUN / 'logs' / ('postprocess_%s_20261025.log' % phase)
            with log.open('x') as stream:
                result_code = subprocess.call(command, stdout=stream, stderr=subprocess.STDOUT)
            (RUN / 'logs' / ('postprocess_%s_exit_code_20261025' % phase)).write_text(str(result_code))
            if result_code:
                raise RuntimeError('%s exited %s; see %s' % (phase, result_code, log))
            result = json.loads(summary.read_text())
            assert result['status'] == 'PASS' and result['frames'] == frames
            assert result['checkpoint_sha256'] == audit['checkpoint_sha256']
            done.append(phase)
        status(dict(status='EXPORTS_COMPLETE', current_phase=None, completed_phases=done,
                    visualisation_and_local_backup_pending=True, shutdown_not_performed=True))
    except Exception:
        status(dict(status='FAILED', current_phase=phase, completed_phases=done,
                    error=traceback.format_exc(), shutdown_not_performed=True))
        raise
    finally:
        lock.unlink()


if __name__ == '__main__':
    main()
