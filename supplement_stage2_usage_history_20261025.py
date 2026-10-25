"""CPU-only prototype usage and bank-geometry history, without new inference."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from plot_stage2_results_20261025 import CLASSES, TITLE, array, heatmap, json_out, save


def state_value(state, name):
    keys=[k for k in state if k==name or k.endswith('.'+name)]
    assert len(keys)==1,(name,keys)
    return state[keys[0]].detach().float().cpu().numpy().copy()


def load_checkpoint_counters(path):
    raw=torch.load(path,map_location='cpu'); state=raw['state_dict']
    result={k:state_value(state,k) for k in ('proto_assignment_mass','proto_hard_assignment_count')}
    del state,raw
    return result


def bank_geometry(bank):
    assert bank.shape==(10,10,256) and np.isfinite(bank).all()
    bank=bank/np.maximum(np.linalg.norm(bank,axis=-1,keepdims=True),1e-8)
    flat=bank.reshape(100,256); sims=flat@flat.T
    cross=sims[np.repeat(np.arange(10),10)[:,None]!=np.repeat(np.arange(10),10)[None,:]]
    within=[(p@p.T)[np.triu_indices(10,1)].mean() for p in bank]
    return dict(within_mean=float(np.mean(within)),within_per_class=[float(x) for x in within],
                cross_mean=float(cross.mean()),cross_max=float(cross.max()),
                cross_fraction_above_04=float((cross>.4).mean()))


def history(run,initial,out):
    rows=[]
    for path in sorted((initial.parent.parent/'prototypes').glob('*.pth')):
        raw=torch.load(path,map_location='cpu')
        step=int(raw['iter']); assert 0<step<=6000
        rows.append(dict(stage='Stage1',local_iter=step,path=str(path),**bank_geometry(raw['prototypes'].float().numpy())))
    for path in sorted((run/'prototypes').glob('stage2_step_*.pth')):
        raw=torch.load(path,map_location='cpu')
        assert int(raw['local_iter'])==6000+int(raw['stage2_step'])
        rows.append(dict(stage='Stage2',local_iter=int(raw['local_iter']),path=str(path),
                         **bank_geometry(state_value(raw['state_dict'],'prototypes'))))
    rows.sort(key=lambda r:r['local_iter'])
    assert [r['local_iter'] for r in rows]==list(range(1000,12001,1000))
    fig,axes=plt.subplots(2,2,figsize=(14,9))
    for ax,key,label in zip(axes.flat,('within_mean','cross_mean','cross_max','cross_fraction_above_04'),
                            ('Mean within-class cosine','Mean cross-class cosine','Maximum cross-class cosine','Cross-class fraction cosine>0.4')):
        for stage,color in (('Stage1','#377eb8'),('Stage2','#e41a1c')):
            sub=[r for r in rows if r['stage']==stage]
            ax.plot([r['local_iter'] for r in sub],[r[key] for r in sub],'-o',label=stage,c=color)
        ax.axvline(6000,color='gray',ls='--'); ax.set(xlabel='Cumulative local_iter',ylabel=label)
        ax.grid(alpha=.2); ax.legend()
    fig.suptitle('Bank geometry during Stage1 1000-6000 and Stage2 +1000-6000\nNot a paired fixed-validation feature-separation experiment')
    save(fig,out/'stage1_stage2_bank_geometry_history.png')
    fig,ax=plt.subplots(figsize=(14,6))
    for c in range(10): ax.plot([r['local_iter'] for r in rows],[r['within_per_class'][c] for r in rows],label=CLASSES[c])
    ax.axvline(6000,c='black',ls='--'); ax.set(xlabel='Cumulative local_iter',ylabel='Mean within-class prototype cosine')
    ax.legend(ncol=5,fontsize=8); ax.set_title('Ten-class within-bank geometry; Stage2 boundary at 6000')
    save(fig,out/'stage1_stage2_within_class_history.png')
    json_out(out/'stage1_stage2_bank_geometry_history.json',dict(rows=rows,
        interpretation='Bank geometry only: projection parameters change, these are not fixed-sample positive/negative gaps'))


def usage(run,initial,features,out):
    summary=json.loads((features/'export_summary.json').read_text()); assert summary['status']=='PASS'
    manifest=json.loads((features/'feature_manifest.json').read_text()); nframes=len(manifest)
    prototypes=np.load(features/'prototypes.npy',allow_pickle=False)
    counts=np.zeros((3,10,10),dtype=np.int64)
    names=('strict_positive_all','strict_positive_conf02_08','clear_negative_conf02_08')
    for row in manifest:
        i=row['index']; feat=array(features,'candidates',i,'features')
        labels=array(features,'candidates',i,'labels'); conf=array(features,'candidates',i,'confidence')
        matched=array(features,'candidates',i,'matched_gt_indices'); neg=array(features,'candidates',i,'clear_negative')
        band=(conf>.2)&(conf<.8)
        masks=(matched>=0,(matched>=0)&band,neg&band)
        for c in range(10):
            ids=np.where(labels==c)[0]
            if not len(ids): continue
            nearest=(feat[ids]@prototypes[c].T).argmax(axis=1)
            for k,mask in enumerate(masks): counts[k,c]+=np.bincount(nearest[mask[ids]],minlength=10)
    occupancy=[]
    for k,name in enumerate(names):
        heatmap(counts[k],TITLE+'\nEvaluation nearest-prototype usage: '+name,
                out/('nearest_prototype_usage_'+name+'.png'),CLASSES,column_labels=['P%d'%p for p in range(10)])
        for c in range(10):
            total=int(counts[k,c].sum()); p=counts[k,c]/total if total else np.zeros(10)
            p=p[p>0]
            occupancy.append(dict(group=name,class_name=CLASSES[c],samples=total,
                 active_prototypes=int((counts[k,c]>0).sum()),
                 entropy_effective_prototypes=float(np.exp(-(p*np.log(p)).sum())) if total else 0.0))
    first=load_checkpoint_counters(initial); last=load_checkpoint_counters(run/'checkpoints/iter_6000.pth')
    training={}
    for key in first:
        increment=last[key]-first[key]
        # Fail rather than mislabel a reset lifetime counter as a Stage2 increment.
        assert (increment>=-1e-4).all(),('Counter reset requires manual interpretation',key)
        increment=np.maximum(increment,0)
        training[key]=increment.tolist()
        heatmap(increment,TITLE+'\nTraining Sinkhorn counter increments: '+key,
                out/('training_increment_'+key+'.png'),CLASSES,column_labels=['P%d'%p for p in range(10)])
    np.save(out/'nearest_prototype_usage_counts.npy',counts,allow_pickle=False)
    json_out(out/'prototype_usage_statistics.json',dict(frames=nframes,groups=names,occupancy=occupancy,
       training_counter_increments=training,
       caveats=['Evaluation nearest-prototype choices are not training Sinkhorn assignments',
                'Training counters subtract the exact Stage1 initializer; they include this Stage2 only',
                'Positive/negative roles use all-score greedy matching before confidence-band restriction']))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--run',required=True)
    parser.add_argument('--initial-checkpoint',required=True); parser.add_argument('--features',required=True)
    parser.add_argument('--out',required=True)
    args=parser.parse_args(); run=Path(args.run); out=Path(args.out)
    if out.exists() and any(out.iterdir()): raise RuntimeError('Refusing nonempty supplementary output')
    out.mkdir(parents=True,exist_ok=True)
    history(run,Path(args.initial_checkpoint),out)
    usage(run,Path(args.initial_checkpoint),Path(args.features),out)
    images=list(out.glob('*.png')); assert len(images)==7
    json_out(out/'supplement_summary.json',dict(status='PASS',png_count=7,source_features=args.features,
         source_checkpoint=str(run/'checkpoints/iter_6000.pth')))
    print('STAGE2_SUPPLEMENT_PASS',7,flush=True)
