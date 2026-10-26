"""CPU-only plotting queued behind successful NPY export. No shutdown here."""

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
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/root/autodl-tmp/data/mv2dfusion_results/ssl_online_r50_ema_1011/postprocess_ema_12000_1011')

def main():
    state=ROOT/'visualisation_pipeline_state.json'
    while True:
        state.write_text(json.dumps(dict(status='waiting_for_npy')))
        exit_path=ROOT/'npy_pipeline_exit_code'
        if exit_path.exists() and int(exit_path.read_text())!=0:
            raise RuntimeError('NPY pipeline failed; no figures generated')
        status=json.loads((ROOT/'npy_pipeline_state.json').read_text())
        if status['status']=='failed': raise RuntimeError('NPY pipeline failed')
        if status['status']=='exports_complete': break
        time.sleep(30)
    for branch in ('student','teacher'):
        export=ROOT/(branch+'_val_npy'); output=ROOT/(branch+'_visualisations')
        if (output/'visualisation_summary.json').exists(): continue
        if output.exists() and any(output.iterdir()): raise RuntimeError('Incomplete figures retained: '+str(output))
        state.write_text(json.dumps(dict(status='running',branch=branch)))
        with (ROOT/(branch+'_visualisations.log')).open('w') as log:
            proc=subprocess.run([sys.executable,'-u','scripts/visualization/plot_online_ema_npy_1011.py','--export-dir',str(export),'--out-dir',str(output)],stdout=log,stderr=subprocess.STDOUT)
        if proc.returncode:
            state.write_text(json.dumps(dict(status='failed',branch=branch,exit_code=proc.returncode)))
            raise RuntimeError('Plotting failed: '+branch)
        result=json.loads((output/'visualisation_summary.json').read_text())
        assert result['frames']==3014 and result['png_files']>=15
    state.write_text(json.dumps(dict(status='visualisations_complete')))
    print('VISUALISATION_PIPELINE_PASS',flush=True)

if __name__=='__main__': main()
