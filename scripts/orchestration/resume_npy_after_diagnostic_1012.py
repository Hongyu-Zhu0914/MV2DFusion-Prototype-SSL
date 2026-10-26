"""Resume after verified stochastic-input diagnosis; never overwrite exports."""

# Resolve imports and relative CLI paths after the publication-layout migration.
import sys as _bootstrap_sys
import os as _bootstrap_os
from pathlib import Path as _BootstrapPath
REPO_ROOT = _BootstrapPath(__file__).resolve().parents[2]
if str(REPO_ROOT) not in _bootstrap_sys.path:
    _bootstrap_sys.path.insert(0, str(REPO_ROOT))
if __name__ == '__main__':
    _bootstrap_os.chdir(str(REPO_ROOT))

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path('/root/autodl-tmp/data/mv2dfusion_results/ssl_online_r50_ema_1011/postprocess_ema_12000_1011')

def validate(directory):
    summary = json.loads((directory/'npy_export_summary.json').read_text())
    metrics = json.loads((directory/'subset_validation_summary.json').read_text())
    rows = json.loads((directory/'npy_manifest.json').read_text())
    assert summary['frames'] == metrics['frames'] == len(rows) == 3014
    assert len({r['token'] for r in rows}) == 3014
    digest = hashlib.sha256('\n'.join(sorted(r['token'] for r in rows)).encode()).hexdigest()
    assert digest == metrics['token_sha256'] == summary['token_sha256']
    assert len(list((directory/'npy').glob('*.npy'))) == 30140
    count = 0
    keys = ('features','boxes','scores','labels','query_indices','prediction_indices',
            'class_logits','gt_boxes','gt_labels','token')
    for r in rows:
        prefix = directory/'npy'/('sample_%04d' % r['frame'])
        a = {k: np.load(str(prefix)+'_'+k+'.npy',allow_pickle=False) for k in keys}
        n = len(a['scores'])
        assert a['features'].shape == (n,256)
        assert a['class_logits'].shape == (n,10)
        assert all(len(a[k]) == n for k in keys[:7])
        assert all(np.isfinite(v).all() for k,v in a.items() if k!='token')
        assert a['token'][0] == r['token'] and n == r['exported_predictions']
        assert len(a['gt_boxes']) == len(a['gt_labels'])
        assert (a['labels']>=0).all() and (a['labels']<10).all()
        assert (a['query_indices']>=0).all() and (a['query_indices']<r['query_count']).all()
        count += n
    assert count == summary['exported_predictions']
    assert np.isfinite([metrics['mAP'],metrics['NDS']]).all()
    return metrics, summary

def main():
    state = ROOT/'npy_pipeline_state.json'
    try:
        diagnostic = json.loads((ROOT/'diagnostic_seeded_1012/report_with_repeat.json').read_text())
        # This is a small-sample diagnostic gate, not proof of identical full runs.
        assert diagnostic['same_seed_plain_vs_export']['same_count']
        assert diagnostic['same_seed_plain_repeat']['same_count']
        assert diagnostic['export_geometry']['max_center_error'] < .01
        assert diagnostic['export_geometry']['max_score_error'] < .002
        assert diagnostic['repeat_geometry']['max_center_error'] < .01
        audits = {}
        for branch in ('student','teacher'):
            directory = ROOT/(branch+'_val_npy')
            if not (directory/'subset_validation_summary.json').exists():
                if directory.exists() and any(directory.iterdir()):
                    raise RuntimeError('Incomplete export retained: '+str(directory))
                state.write_text(json.dumps({'status':'running','stage':branch+'_val_npy'}))
                code = ('import random,numpy as np,torch,runpy; random.seed(1011); '
                        'np.random.seed(1011); torch.manual_seed(1011); torch.cuda.manual_seed_all(1011); '
                        'runpy.run_path("export_online_ema_npy_1011.py",run_name="__main__")')
                command = [sys.executable,'-u','-c',code,'--checkpoint',str(ROOT.parent/'iter_12000.pth'),
                           '--branch',branch,'--output-dir',str(directory)]
                with (ROOT/(branch+'_val_npy_resumed_1012.log')).open('w') as log:
                    subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
            metrics, arrays = validate(directory)
            reference_path = ROOT.parent/'validation_final_student/subset_validation_summary.json' if branch=='student' else ROOT/'teacher_val/subset_validation_summary.json'
            reference = json.loads(reference_path.read_text())
            assert metrics['token_sha256'] == reference['token_sha256']
            audits[branch] = dict(frames=3014,npy_files=30140,exported_predictions=arrays['exported_predictions'],
                                 mAP=metrics['mAP'],NDS=metrics['NDS'],
                                 reference_mAP_delta=metrics['mAP']-reference['mAP'],
                                 reference_NDS_delta=metrics['NDS']-reference['NDS'],
                                 reference_exactly_reproduced=False,
                                 alignment_verified=True)
        report = dict(branches=audits,diagnostic=diagnostic,
                      limitation='Prior independent evaluations used unseeded random sweep selection. '
                      'Metrics are recorded per run, not asserted identical. All exported rows matched their own actual decoded outputs. '
                      'Eight-frame seeded control supports numerical non-determinism and random input differences; no full deterministic re-evaluation was performed.')
        (ROOT/'npy_integrity_diagnostic_1012.json').write_text(json.dumps(report,indent=2))
        state.write_text(json.dumps({'status':'exports_complete','stages':['student_val_npy','teacher_val_npy'],
                                     'audit':'npy_integrity_diagnostic_1012.json'}))
        print('NPY_RESUME_VALIDATION_PASS',json.dumps(audits),flush=True)
    except Exception as error:
        state.write_text(json.dumps({'status':'failed','error':str(error)}))
        raise

if __name__ == '__main__':
    main()
