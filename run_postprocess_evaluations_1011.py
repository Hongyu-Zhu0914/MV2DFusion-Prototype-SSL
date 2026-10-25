"""Idempotent serial evaluation stages, using a separate GPU process per stage."""
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime

ROOT = Path('/root/autodl-tmp/data/mv2dfusion_results/ssl_online_r50_ema_1011/postprocess_ema_12000_1011')
CKPT = ROOT.parent / 'iter_12000.pth'

def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    state = ROOT / 'evaluation_pipeline_state.json'
    stages = [('teacher_val', 'val', 'teacher', False),
              ('r50_val', 'val', 'student', True),
              ('student_internal_test', 'internal-test', 'student', False),
              ('teacher_internal_test', 'internal-test', 'teacher', False)]
    for name, split, branch, baseline in stages:
        output = ROOT / name
        summary_path = output / 'subset_validation_summary.json'
        expected = 3014 if split == 'val' else 3005
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            assert summary['frames'] == expected and not summary['smoke_only']
            continue
        if output.exists() and any(output.iterdir()):
            raise RuntimeError('Incomplete output retained; inspect before retry: '+str(output))
        state.write_text(json.dumps(dict(stage=name, status='running', time=datetime.now().isoformat()), indent=2))
        args = [sys.executable, '-u', 'evaluate_online_ema_splits_1011.py',
                '--split', split, '--branch', branch, '--output-dir', str(output)]
        args += ['--baseline'] if baseline else ['--checkpoint', str(CKPT)]
        with (ROOT / (name+'.log')).open('w') as stream:
            result = subprocess.run(args, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            state.write_text(json.dumps(dict(stage=name, status='failed', exit_code=result.returncode), indent=2))
            raise RuntimeError('Evaluation failed: '+name)
        summary = json.loads(summary_path.read_text())
        assert summary['frames'] == expected and not summary['smoke_only']
    state.write_text(json.dumps(dict(status='evaluations_complete', stages=[x[0] for x in stages]), indent=2))
    print('POSTPROCESS_EVALUATIONS_PASS', flush=True)

if __name__ == '__main__':
    main()
