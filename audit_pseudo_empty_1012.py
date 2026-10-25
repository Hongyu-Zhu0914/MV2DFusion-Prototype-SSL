"""Inference-only audit of filters and augmented/neutral unlabeled inputs."""
import copy
import gc
import json
import random
import time
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint, load_state_dict
from mmdet.models import build_detector
from mmdet3d.datasets import build_dataset
from train_online_ema_1011 import setup, AlternatingSceneSampler
from projects.mmdet3d_plugin.models.detectors.mv2dfusion_online_ssl_0919 import MV2DFusionOnlineSSL0919, select_pseudo

ROOT=Path('/root/autodl-tmp/data/mv2dfusion_results/ssl_online_r50_ema_1011')
OUT=ROOT/'diagnosis_empty_pseudo_1012'

def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/'report.json').exists():
        print('EXISTING_REPORT',flush=True); return
    state=ROOT/'postprocess_ema_12000_1011/npy_pipeline_state.json'
    while True:
        s=json.loads(state.read_text())
        if s['status']=='failed': raise RuntimeError('NPY pipeline failed; audit not started')
        if s['status']=='exports_complete': break
        (OUT/'state.json').write_text(json.dumps({'status':'waiting_for_gpu_export'}))
        time.sleep(30)
    cfg=Config.fromfile('projects/configs/nusc/ssl_online_r50_ema_1011.py'); setup(cfg)
    datasets={'augmented':build_dataset(copy.deepcopy(cfg.data.train))}
    neutral=copy.deepcopy(cfg.data.train)
    neutral.pipeline=[x for x in neutral.pipeline if x['type'] not in ('BEVGlobalRotScaleTrans','BEVRandomFlip3D','PointShuffle')]
    for stage in neutral.pipeline:
        if stage['type']=='ResizeCropFlipRotImage': stage['training']=False
    datasets['neutral_geometry']=build_dataset(neutral)
    sampler=AlternatingSceneSampler(datasets['augmented'],12000,seed=cfg.ssl_seed)
    # Same scenes across models: early/middle/late sampling positions, not known
    # individually empty frames (historical logs do not contain per-frame tokens).
    positions=[0,250,500,1000,2000,3000,4000,5500]
    indices=[sampler.orders[1][p] for p in positions]
    reports=[]
    for branch in ('r50','teacher','student'):
        (OUT/'state.json').write_text(json.dumps({'status':'running','branch':branch}))
        architecture=copy.deepcopy(Config.fromfile(cfg.model.detector_config).model)
        model=build_detector(architecture)
        if branch=='r50':
            load_checkpoint(model,cfg.model.initial_checkpoint,map_location='cpu',strict=True)
        else:
            checkpoint=torch.load(str(ROOT/'iter_12000.pth'),map_location='cpu')
            full=checkpoint['state_dict']; prefix=branch+'.'
            params=OrderedDict((k[len(prefix):],v) for k,v in full.items() if k.startswith(prefix))
            params._metadata=OrderedDict(('' if k==branch else k[len(prefix):],v)
                                         for k,v in full._metadata.items() if k==branch or k.startswith(prefix))
            load_state_dict(model,params,strict=True)
            del checkpoint,full,params
        model.cuda().eval()
        proxy=SimpleNamespace(teacher=model,reset_frame=MV2DFusionOnlineSSL0919.reset_frame)
        for view,dataset in datasets.items():
            for position,index in zip(positions,indices):
                seed=1012+position
                random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                sample=dataset[index]
                data=scatter(collate([sample],samples_per_gpu=1),[0])[0]
                data['prev_exists']=torch.zeros_like(data['prev_exists'])
                # Exact training teacher inference adapter, no forward_train,
                # backward, optimizer, EMA update, or checkpoint write.
                with torch.no_grad(): pred=MV2DFusionOnlineSSL0919.teacher_predict(proxy,data)
                boxes,scores,labels=pred['boxes_3d'],pred['scores_3d'],pred['labels_3d']
                masks={}
                mask=(scores>.8)&torch.isfinite(scores); masks['score']=int(mask.sum())
                mask=mask&torch.isfinite(boxes.tensor).all(-1); masks['finite_box']=int(mask.sum())
                mask=mask&((labels>=0)&(labels<10)); masks['valid_label']=int(mask.sum())
                mask=mask&(boxes.tensor[:,3:6]>0).all(-1); masks['positive_size']=int(mask.sum())
                xyz=boxes.gravity_center; limits=xyz.new_tensor(Config.fromfile(cfg.model.detector_config).point_cloud_range)
                mask=mask&((xyz>=limits[:3])&(xyz<=limits[3:])).all(-1); masks['in_range']=int(mask.sum())
                exact=select_pseudo(pred,.8,10,limits)
                assert torch.equal(mask,exact)
                row=dict(branch=branch,view=view,position=position,index=index,
                         token=dataset.data_infos[index]['token'],scene=dataset.data_infos[index]['scene_token'],
                         candidates=len(scores),max_score=float(scores.max()),
                         threshold_counts={str(t):int((scores>t).sum()) for t in (.2,.4,.6,.8)},
                         filter_counts=masks,selected_labels=labels[mask].cpu().tolist(),
                         teacher_eval=not model.training)
                reports.append(row)
                (OUT/'partial.json').write_text(json.dumps(reports,indent=2))
                print('AUDIT_FRAME',json.dumps(row),flush=True)
                del data,sample,pred,boxes,scores,labels,mask,exact
        del model,proxy
        gc.collect(); torch.cuda.empty_cache()
    report=dict(rows=reports,limitations=['8 sampled frames, not the complete unlabeled split.',
                  'Augmented view uses fixed diagnostic seeds, not historical per-step augmentation.',
                  'Neutral geometry is a controlled training-loader view, not the official evaluation pipeline.',
                  'No GT is available on this official test split; counts do not measure correctness.'])
    (OUT/'report.json').write_text(json.dumps(report,indent=2))
    (OUT/'state.json').write_text(json.dumps({'status':'complete'}))
    print('PSEUDO_AUDIT_PASS',flush=True)

if __name__=='__main__': main()
