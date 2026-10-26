"""Run only the newly requested R50 internal-test evaluation; preserve old results."""

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

ROOT = Path('/root/autodl-tmp/data/mv2dfusion_results/ssl_online_r50_ema_1011/postprocess_ema_12000_1011')
OUTPUT = ROOT / 'r50_internal_test_1012'

def main():
    summary_path = OUTPUT / 'subset_validation_summary.json'
    exit_path = ROOT / 'r50_internal_test_1012_exit_code'
    state_path = ROOT / 'r50_internal_test_1012_state.json'
    if OUTPUT.exists() or exit_path.exists():
        raise RuntimeError('Existing output retained; inspect before retry')
    state_path.write_text(json.dumps({'status': 'running', 'output': str(OUTPUT)}))
    args = [sys.executable, '-u', 'scripts/evaluation/evaluate_online_ema_splits_1011.py',
            '--baseline', '--split', 'internal-test', '--output-dir', str(OUTPUT)]
    with (ROOT / 'r50_internal_test_1012.log').open('w') as log:
        result = subprocess.run(args, cwd=str(REPO_ROOT),
                                stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError('Evaluation process failed: {}'.format(result.returncode))
    summary = json.loads(summary_path.read_text())
    teacher = json.loads((ROOT / 'teacher_internal_test/subset_validation_summary.json').read_text())
    student = json.loads((ROOT / 'student_internal_test/subset_validation_summary.json').read_text())
    assert summary['frames'] == 3005 and summary['baseline'] and not summary['smoke_only']
    assert summary['token_sha256'] == teacher['token_sha256'] == student['token_sha256']
    assert summary['temporal_mode'] == teacher['temporal_mode'] == student['temporal_mode']
    state_path.write_text(json.dumps({'status': 'complete', 'summary': summary}, indent=2))
    exit_path.write_text('0\n')
    print('R50_INTERNAL_TEST_PASS', json.dumps(summary), flush=True)

if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        (ROOT / 'r50_internal_test_1012_state.json').write_text(json.dumps(
            {'status': 'failed', 'error': str(error)}, indent=2))
        (ROOT / 'r50_internal_test_1012_exit_code').write_text('1\n')
        raise
