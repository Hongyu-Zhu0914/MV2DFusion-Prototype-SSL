"""Run only metadata checks and four smoke steps, including a process restart."""

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
import traceback

ROOT = REPO_ROOT
RUN = ROOT / 'work_dirs/prototype_stage2_r50_labeled6000_10cls_20261025'
OUT = RUN / 'preflight/preparation_20261025'
SMOKE = OUT / 'entry_smoke'
OUT.mkdir(parents=True, exist_ok=True)
os.chdir(str(ROOT))
env = os.environ.copy()
env.update(PYTHONPATH=str(ROOT), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
phases = [
    ('metadata', ['scripts/diagnostics/verify_stage2_preparation_20261025.py', '--output-dir', str(OUT)]),
    ('smoke_first2', ['scripts/prototype/train_prototype_stage2_20261025.py', '--work-dir', str(SMOKE), '--stop-after', '2']),
    ('smoke_resume4', ['scripts/prototype/train_prototype_stage2_20261025.py', '--work-dir', str(SMOKE), '--stop-after', '4',
                       '--resume-checkpoint', str(SMOKE / 'checkpoints/iter_2.pth')]),
    ('checkpoint_roundtrip', ['scripts/diagnostics/verify_stage2_preparation_20261025.py', '--output-dir', str(OUT), '--smoke-dir', str(SMOKE)]),
]
status = dict(status='RUNNING', formal_training_started=False, completed_phases=[])
try:
    for name, arguments in phases:
        if (OUT / (name + '.success')).exists():
            status['completed_phases'].append(name)
            continue
        status['current_phase'] = name
        (OUT / 'pipeline_status.json').write_text(json.dumps(status, indent=2))
        command = [sys.executable] + arguments
        print('PREFLIGHT_START', name, flush=True)
        with (OUT / (name + '.log')).open('a') as stream:
            result = subprocess.run(command, cwd=str(ROOT), env=env, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError('%s failed exit=%s; see %s.log' % (name, result.returncode, name))
        (OUT / (name + '.success')).write_text('exit_code=0\n')
        status['completed_phases'].append(name)
    for filename in ['metadata_audit.json', 'checkpoint_audit.json']:
        assert json.loads((OUT / filename).read_text())['status'] == 'PASS'
    status.update(status='PASS', ready_for_formal_training=True,
                  formal_training_started=False, configured_new_steps=6000)
    (OUT / 'pipeline_status.json').write_text(json.dumps(status, indent=2))
    (OUT / 'pipeline_exit_code').write_text('0\n')
    print('STAGE2_PREPARATION_PASS', flush=True)
except Exception as exc:
    status.update(status='FAILED', error=repr(exc), ready_for_formal_training=False)
    (OUT / 'pipeline_status.json').write_text(json.dumps(status, indent=2))
    (OUT / 'pipeline_exit_code').write_text('1\n')
    traceback.print_exc()
    sys.exit(1)
