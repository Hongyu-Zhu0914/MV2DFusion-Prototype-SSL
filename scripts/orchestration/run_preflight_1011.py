"""Bounded checks only. Never launches the full schedule or powers off."""

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

ROOT = Path(str(REPO_ROOT))
OUT = ROOT/'work_dirs/ssl_online_r50_ema_1011/preflight_final_v2'


def run(name, arguments):
    with open(OUT/(name+'.log'), 'w') as log:
        result = subprocess.run([sys.executable]+arguments, cwd=str(ROOT), stdout=log,
                                stderr=subprocess.STDOUT, env=dict(os.environ, OMP_NUM_THREADS='1', PYTHONPATH=str(ROOT)))
    print(name, 'exit', result.returncode, flush=True)
    if result.returncode:
        raise RuntimeError(name+' failed; see '+str(OUT/(name+'.log')))


def main():
    eval_only = '--eval-only' in sys.argv
    continuing = '--continue' in sys.argv or eval_only
    if OUT.exists() and not continuing:
        raise RuntimeError('Preflight directory exists; do not overwrite')
    OUT.mkdir(parents=True, exist_ok=continuing)
    train_dir = OUT/'runner'
    if eval_only:
        assert (train_dir/'iter_4.pth').exists()
    elif continuing:
        assert (train_dir/'iter_2.pth').exists() and not (train_dir/'iter_4.pth').exists()
        assert not (OUT/'resume_to4_retry.log').exists()
    else:
        run('unit', ['tests/unit/test_ema_unit_1011.py'])
        run('adamw_2step', ['scripts/ssl/train_online_ema_1011.py', '--max-iters', '2',
                           '--work-dir', str(train_dir), '--skip-final-eval'])
    if not eval_only:
        run('resume_to4_retry' if continuing else 'resume_to4', ['scripts/ssl/train_online_ema_1011.py', '--max-iters', '4',
                          '--work-dir', str(train_dir), '--resume', str(train_dir/'iter_2.pth'), '--skip-final-eval'])
    import torch
    checkpoint = torch.load(str(train_dir/'iter_4.pth'), map_location='cpu')
    assert checkpoint['meta']['ssl_completed_iters'] == 4
    assert int(checkpoint['state_dict']['ema_updates']) == 4
    assert 'ssl_rng' in checkpoint['meta']
    assert checkpoint['optimizer']['state']
    assert any(k.startswith('teacher.') for k in checkpoint['state_dict'])
    assert any(k.startswith('student.') for k in checkpoint['state_dict'])
    state = checkpoint['state_dict']
    for name,value in state.items():
        if name.startswith('student.pts_backbone.'):
            assert torch.equal(value,state['teacher.'+name[len('student.'):]])
    del checkpoint, state
    import gc
    gc.collect()
    rows = [json.loads(line) for line in (train_dir/'integrity.jsonl').read_text().splitlines()]
    assert [r['iter'] for r in rows] == [2,4]
    assert all(r['frozen_unchanged'] for r in rows)
    for branch in ['student', 'teacher']:
        run('eval_'+branch+'_v3', ['scripts/evaluation/validate_online_ema_1011.py', '--checkpoint', str(train_dir/'iter_4.pth'),
                            '--branch', branch, '--limit', '2', '--output-dir', str(OUT/('eval_'+branch+'_v3'))])
    run('eval_baseline', ['scripts/evaluation/validate_online_ema_1011.py', '--baseline', '--limit', '2',
                         '--output-dir', str(OUT/'eval_baseline')])
    reports = {branch: json.loads((OUT/('eval_'+branch+('_v3' if branch != 'baseline' else ''))/'subset_validation_summary.json').read_text())
               for branch in ['student','teacher','baseline']}
    assert len({r['token_sha256'] for r in reports.values()}) == 1
    (OUT/'preflight_summary.json').write_text(json.dumps(dict(
        status='PASS', formal_training_started=False, gpu_optimizer_steps=4,
        resume_iter=4, ema_updates=4, frozen_unchanged=True,
        evaluation_smoke_frames=2, evaluation_reports=reports,
        limitation='2-frame evaluation validates plumbing, not model performance; no full 3014-frame evaluation yet'), indent=2)+'\n')
    print('PREFLIGHT_ALL_PASS', flush=True)


if __name__ == '__main__': main()
