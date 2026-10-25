"""Queue exports behind the already-running serial evaluation pipeline."""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/root/autodl-tmp/data/mv2dfusion_results/ssl_online_r50_ema_1011/postprocess_ema_12000_1011')

def main():
    state_path = ROOT/'npy_pipeline_state.json'
    while True:
        state_path.write_text(json.dumps(dict(status='waiting_for_evaluations')))
        source = json.loads((ROOT/'evaluation_pipeline_state.json').read_text())
        if source['status'] == 'failed':
            raise RuntimeError('Previous evaluation failed; no exports started')
        if source['status'] == 'evaluations_complete':
            break
        time.sleep(30)
    stages = [('smoke_student_npy', 'student', 2), ('student_val_npy', 'student', None),
              ('teacher_val_npy', 'teacher', None)]
    for name, branch, limit in stages:
        output = ROOT/name
        done = output/'subset_validation_summary.json'
        if done.exists() and (output/'npy_export_summary.json').exists():
            continue
        if output.exists() and any(output.iterdir()):
            raise RuntimeError('Incomplete export retained: '+str(output))
        state_path.write_text(json.dumps(dict(status='running', stage=name)))
        args = [sys.executable, '-u', 'export_online_ema_npy_1011.py', '--checkpoint',
                str(ROOT.parent/'iter_12000.pth'), '--branch', branch, '--output-dir', str(output)]
        if limit:
            args += ['--limit', str(limit)]
        with (ROOT/(name+'.log')).open('w') as log:
            result = subprocess.run(args, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            state_path.write_text(json.dumps(dict(status='failed', stage=name, exit_code=result.returncode)))
            raise RuntimeError('Export failed: '+name)
        metrics = json.loads(done.read_text())
        arrays = json.loads((output/'npy_export_summary.json').read_text())
        assert metrics['frames'] == arrays['frames'] == (limit or 3014)
        assert metrics['token_sha256'] == arrays['token_sha256']
        assert arrays['npy_files'] == arrays['frames']*10
        if not limit:
            source = ROOT.parent/'validation_final_student/subset_validation_summary.json' if branch == 'student' else ROOT/'teacher_val/subset_validation_summary.json'
            reference = json.loads(source.read_text())
            assert metrics['token_sha256'] == reference['token_sha256']
            for key in ('mAP', 'NDS'):
                if abs(metrics[key]-reference[key]) > 1e-4:
                    raise AssertionError('Export evaluation differs from original: '+key)
    state_path.write_text(json.dumps(dict(status='exports_complete', stages=[s[0] for s in stages])))
    print('NPY_PIPELINE_PASS', flush=True)

if __name__ == '__main__':
    main()
