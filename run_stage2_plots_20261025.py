"""Idempotent CPU audit/plot queue. No model edits, inference, training or shutdown."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'work_dirs/prototype_stage2_r50_labeled6000_10cls_20261025'
STATE=RUN/'analysis/plots_status_20261025.json'


def read(path): return json.loads(path.read_text(encoding='utf-8'))


def status(state,**kwargs):
    data=dict(status=state,time_epoch=time.time(),**kwargs)
    temporary=STATE.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(STATE)
    print('PLOT_QUEUE',json.dumps(data),flush=True)


def execute(command,log):
    if log.exists(): raise RuntimeError('Refusing existing command log: '+str(log))
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    with log.open('x') as stream:
        result=subprocess.run(command,cwd=str(ROOT),stdout=stream,stderr=subprocess.STDOUT,env=env)
    if result.returncode: raise RuntimeError('Failed rc=%s; preserve log %s'%(result.returncode,log))


def main():
    with (RUN/'analysis/plots_queue_20261025.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        if STATE.exists() and read(STATE)['status']=='COMPLETE':
            print('PLOTS_ALREADY_COMPLETE',flush=True); return
        status('WAITING_EXPORTS')
        started=time.time()
        while True:
            export_state=read(RUN/'analysis/postprocess_exports_status_20261025.json')
            if export_state['status'] in ('FAILED','blocked_or_failed'):
                raise RuntimeError('Upstream export failed; no automatic retry')
            done=RUN/'analysis/postprocess_exports_exit_code_20261025'
            if done.exists():
                assert done.read_text().strip()=='0','Upstream export nonzero exit'
                break
            if time.time()-started>12*3600: raise TimeoutError('Export wait exceeded 12 hours')
            time.sleep(15)
        completed=[]
        for split,n in (('val',3014),('test',3005)):
            features=RUN/'features'/('stage2_local12000_'+split)
            summary=read(features/'export_summary.json'); assert summary['status']=='PASS' and summary['frames']==n
            audit=RUN/'analysis'/('independent_'+split+'_audit_20261025.json')
            status('AUDITING',split=split,completed_splits=completed)
            if audit.exists():
                old=read(audit)
                assert old['status']=='PASS' and old['frames']==n and old['checkpoint_sha256']==summary['checkpoint_sha256']
            else:
                execute([sys.executable,str(ROOT/'verify_stage2_exports_20261025.py'),
                         '--features',str(features),'--out',str(audit)],
                        RUN/'logs'/('independent_'+split+'_audit_20261025.log'))
            plots=RUN/'visualisation'/('stage2_local12000_'+split)
            status('PLOTTING',split=split,completed_splits=completed)
            if (plots/'plot_summary.json').exists():
                assert read(plots/'plot_summary.json')['status']=='PASS'
            else:
                execute([sys.executable,str(ROOT/'plot_stage2_results_20261025.py'),
                         '--run',str(RUN),'--features',str(features),
                         '--initial-checkpoint',str(ROOT/'work_dirs/prototype_stage1_r50_labeled6000_10cls_0923/checkpoints/iter_6000.pth'),
                         '--out',str(plots)],RUN/'logs'/('plots_'+split+'_20261025.log'))
            completed.append(split)
        status('COMPLETE',completed_splits=completed,local_backup_pending=True,
               notes='No shutdown executed or scheduled; full local artifact backup and final review still required')


if __name__=='__main__':
    try: main()
    except Exception as error:
        status('FAILED',error=repr(error)); raise
