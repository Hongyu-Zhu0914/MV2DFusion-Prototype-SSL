"""Four bounded AdamW iterations, recovery, and two-frame inference checks."""
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path('/root/MV2DFusion-main')
OUT = Path('/root/autodl-tmp/data/mv2dfusion_results/ssl_aug_global_2026-10-19/preflight')
CONFIG = 'projects/configs/nusc/ssl_online_r50_ema_aug_global_2026_10_19.py'


def run(name, arguments):
    path = OUT/(name+'.log')
    if path.exists():
        raise RuntimeError('Refusing to overwrite '+str(path))
    (OUT/'pipeline_state.json').write_text(json.dumps(dict(stage=name, status='running'))+'\n')
    with path.open('w') as log:
        result = subprocess.run([sys.executable]+arguments, cwd=str(ROOT),
                                stdout=log, stderr=subprocess.STDOUT,
                                env=dict(os.environ, PYTHONPATH=str(ROOT), OMP_NUM_THREADS='1'))
    if result.returncode:
        raise RuntimeError(name+' failed: '+str(result.returncode)+'; see '+str(path))
    print('STAGE_PASS', name, flush=True)


def main():
    import torch
    math_report = json.loads((OUT/'gpu_report.json').read_text())
    assert math_report['status'] == 'GPU_GLOBAL_PREFLIGHT_PASS'
    assert (OUT/'gpu_math_exit_code').read_text().strip() == '0'
    train_dir = OUT/'runner'
    assert not train_dir.exists(), 'Use a new output directory; do not overwrite smoke weights'
    run('adamw_2step', ['train_online_ema_1011.py', '--config', CONFIG,
                        '--max-iters', '2', '--work-dir', str(train_dir), '--skip-final-eval'])
    run('resume_to4', ['train_online_ema_1011.py', '--config', CONFIG,
                      '--max-iters', '4', '--work-dir', str(train_dir),
                      '--resume', str(train_dir/'iter_2.pth'), '--skip-final-eval'])
    checkpoint = torch.load(str(train_dir/'iter_4.pth'), map_location='cpu')
    assert checkpoint['meta']['ssl_completed_iters'] == 4
    assert checkpoint['optimizer']['state']
    assert 'ssl_rng' in checkpoint['meta']
    assert hasattr(checkpoint['state_dict'], '_metadata')
    updates = int(checkpoint['state_dict']['ema_updates'])
    assert 2 <= updates <= 4
    for name, value in checkpoint['state_dict'].items():
        if name.startswith('student.pts_backbone.'):
            assert torch.equal(value, checkpoint['state_dict']['teacher.'+name[len('student.'):]])
    del checkpoint
    import gc
    gc.collect()
    rows = [json.loads(line) for line in (train_dir/'integrity.jsonl').read_text().splitlines()]
    assert [r['iter'] for r in rows] == [2, 4]
    assert all(r['frozen_unchanged'] for r in rows)
    for branch in ['student', 'teacher']:
        run('eval_'+branch, ['validate_online_ema_1011.py', '--config', CONFIG,
                            '--checkpoint', str(train_dir/'iter_4.pth'), '--branch', branch,
                            '--limit', '2', '--output-dir', str(OUT/('eval_'+branch))])
    summaries = {b: json.loads((OUT/('eval_'+b)/'subset_validation_summary.json').read_text())
                 for b in ['student', 'teacher']}
    assert len({v['token_sha256'] for v in summaries.values()}) == 1
    assert all(v['frames'] == 2 and v['smoke_only'] for v in summaries.values())
    summary = dict(status='GLOBAL_PREFLIGHT_ALL_PASS', formal_training_started=False,
                   completed_smoke_attempts=4, ema_updates=updates,
                   valid_pseudo_updates=updates-2, frozen_unchanged=True,
                   optimizer_rng_metadata_restored=True, audits=rows,
                   math_and_components=math_report, evaluation=summaries,
                   limitation='Bounded checks only; 2-frame metrics are not performance evidence.')
    (OUT/'preflight_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (OUT/'pipeline_state.json').write_text(json.dumps(dict(status='complete'))+'\n')
    print('GLOBAL_PREFLIGHT_ALL_PASS', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        (OUT/'pipeline_state.json').write_text(json.dumps(dict(status='failed', error=traceback.format_exc()))+'\n')
        raise
