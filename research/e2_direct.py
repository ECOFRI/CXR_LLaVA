import csv,json,os,sys,time,resource,traceback
from pathlib import Path
import numpy as np
from research.common import ROOT,RESEARCH,BASELINE,sha,pseudonym,now,new_run,finish_run,write_json,write_csv
from research.e_audit import parse,statistics
L=['Atelectasis','Cardiomegaly','Consolidation','Edema','Pleural Effusion']
def main():
 start=time.perf_counter();run,m=new_run('E2_direct_contrastive');print(run,flush=True)
 m.update(env=sys.prefix,command=sys.argv,dataset='CheXpert',split='locked_test',precision='FP32',estimated_gpu_hours=0,actual_gpu_hours=0,pipeline_type='E2_direct_contrastive',metric_policy='ranking_primary_fixed0.5_exploratory')
 try:
  assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
  repo=ROOT.parent/'CLEAR';ck=next(Path('/home/student4/.cache/huggingface/hub/models--peterhan91--CLEAR/snapshots').glob('*/best_model.pt'));expected='da1617a7d47970667a7a158eabfa4c4f3798d5d03b7f98caec150d8ee6ab6a03';assert sha(ck)==expected
  cohort=json.loads((BASELINE/'cohort.json').read_text());assert len(cohort)==518 and len({r['study'] for r in cohort})==500
  oldrun=RESEARCH/'runs/20261013T203617_E0A_CLEAR_CPU_10_50';old=json.loads((oldrun/'run_manifest.json').read_text());assert old['input_cohort_sha256']==sha(BASELINE/'cohort.json') and old['checkpoint_sha256']==expected
  sources={str(repo/p):sha(repo/p) for p in ['README.md','examples/quickstart.py','examples/concepts/exp_zeroshot.py','src/clear/model.py','src/clear/hub.py','src/clear/simple_tokenizer.py','src/clear/zero_shot.py']}
  protocol=dict(frozen_at=now(),pipeline_type='E2_direct_contrastive',labels=L,positive='{}',negative='no {}',label_case='lowercase',source='official exp_zeroshot.py lines357-358; README/quickstart normalization and paired softmax; zero_shot.run_softmax_eval',score='softmax([cos(image,pos),cos(image,neg)])[0], no logit scale',threshold='>=0.5 fixed exploratory uncalibrated; no tuning',primary='AUROC/AUPRC',bootstrap='1000 patient clusters seed42 percentile95%; 500 patients/500 studies',cohort_sha256=sha(BASELINE/'cohort.json'),checkpoint_sha256=expected,source_sha256=sources,cpu_threads=4,ram_cap_bytes=8*1024**3,estimate='468 new forwards ~550seconds CPU;50cached;0GPUhours;1.52MiB globalcache',concept_status='BLOCKED_CONCEPT_ASSETS_OR_RESOURCE; local snapshot only best_model.pt; no downloads')
  (run/'E2_frozen_protocol.yaml').write_text(json.dumps(protocol,indent=2)+'\n');ph=sha(run/'E2_frozen_protocol.yaml');m.update(protocol_sha256=ph,checkpoint_sha256=expected,checkpoint=ck.parent.name,runner_sha256=sha(__file__),source_sha256=sources);write_json(run/'run_manifest.json',m);print('FROZEN',ph,flush=True)
  pred=np.full((518,5),np.nan);align=[]
  for i,r in enumerate(cohort):
   assert sha(r['image_path'])==r['image_sha256'];p=BASELINE/'cases'/f'{r["index"]:05d}.json';c=json.loads(p.read_text());assert c['relative_path']==r['relative_path'] and c['study']==r['study'] and c['image_sha256']==r['image_sha256'] and c['status']=='success'
   for j,l in enumerate(L):
    v=parse(c['responses'][l]['text']);assert v==c['responses'][l]['parsed']==c['parsed_labels'].get(l)
    if v is not None:pred[i,j]=v
   align.append(dict(image_id=pseudonym(r['relative_path']),study_id=pseudonym(r['study']),patient_id=pseudonym(str(Path(r['study']).parent)),source_sha256=r['image_sha256'],baseline_sha256=sha(p),source_path=r['image_path']))
  write_csv(run/'alignment_manifest.csv',align)
  import torch
  import torch.nn.functional as F
  from PIL import Image
  assert torch.version.cuda is None;torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(42);torch.use_deterministic_algorithms(True);sys.path.insert(0,str(repo/'src'));torch.hub.set_dir(str(RESEARCH/'cache/torch_hub'))
  from clear import load_pretrained,tokenize
  t=time.perf_counter();model,pre=load_pretrained(checkpoint_path=ck,device='cpu',local_files_only=True);model=model.float().eval().requires_grad_(False);loadsec=time.perf_counter()-t;assert repr(pre)==old['preprocess'] and sha(repo/'src/clear/hub.py')==old['clear_source_sha256']
  prompts=[s for l in L for s in (l.lower(),'no '+l.lower())];tokens=tokenize(prompts);assert tokens.shape==(10,77)
  with torch.inference_mode():text=F.normalize(model.encode_text(tokens),dim=-1);repeat=F.normalize(model.encode_text(tokens),dim=-1)
  assert text.shape==(10,768) and torch.isfinite(text).all() and float((text-repeat).abs().max())<=1e-6
  torch.save(dict(prompts=prompts,tokens=tokens,text_features=text),run/'text_features.pt');cache={r['image_id']:r for r in csv.DictReader((oldrun/'feature_manifest.csv').open())};features=[];scores=[];timings=[];gates=[]
  with Image.open(cohort[0]['image_path']) as im:x=pre(im).unsqueeze(0)
  with torch.inference_mode():
   for _ in range(3):model.encode_image(x)
  for i,r in enumerate(cohort):
   ident=align[i]['image_id'];reused=ident in cache;t=time.perf_counter()
   if reused:
    cr=cache[ident];assert sha(cr['cache_path'])==cr['cache_sha256'];p=torch.load(cr['cache_path'],map_location='cpu',weights_only=True);assert p['image_id']==ident;f=p['features']['projected_global']
   else:
    with Image.open(r['image_path']) as im:x=pre(im).unsqueeze(0)
    t=time.perf_counter()
    with torch.inference_mode():f=model.encode_image(x)
   sec=time.perf_counter()-t;assert f.shape==(1,768) and torch.isfinite(f).all() and float(f.norm())>0
   with torch.inference_mode():s=(F.normalize(f,dim=-1)@text.T).reshape(5,2).softmax(-1)[:,0]
   assert torch.isfinite(s).all() and bool(((s>0)&(s<1)).all());features.append(f.numpy().copy());scores.append(s.numpy().copy());timings.append(dict(image_id=ident,reused_E0A=reused,forward_latency_sec=None if reused else sec))
   if i+1 in (1,10):
    with Image.open(r['image_path']) as im:xx=pre(im).unsqueeze(0)
    with torch.inference_mode():ff=model.encode_image(xx);ss=(F.normalize(ff,dim=-1)@text.T).reshape(5,2).softmax(-1)[:,0]
    delta=float((ff-f).abs().max());sd=float((ss-s).abs().max());assert delta<=1e-6 and sd<=1e-6;gates.append(dict(n=i+1,status='PASS',feature_delta=delta,score_delta=sd));write_json(run/'E0B_gates.json',gates);print('E0B',i+1,'PASS',flush=True)
   assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<8*1024**3
   if (i+1)%50==0:print('E2',i+1,flush=True)
  scores=np.array(scores);np.savez(run/'projected_globals.npz',image_ids=np.array([r['image_id'] for r in align]),projected_global=np.concatenate(features));write_csv(run/'inference_timing.csv',timings)
  rows=[]
  for i,a in enumerate(align):
   for j,l in enumerate(L):rows.append(dict(image_id=a['image_id'],study_id=a['study_id'],patient_id=a['patient_id'],source_sha256=a['source_sha256'],finding=l,clear_score=float(scores[i,j]),clear_prediction=int(scores[i,j]>=.5),cxr_prediction=None if np.isnan(pred[i,j]) else int(pred[i,j]),gt=cohort[i]['labels'][l],pipeline_type='E2_direct_contrastive'))
  write_csv(run/'e2_clear_scores.csv',rows);assert sha(run/'E2_frozen_protocol.yaml')==ph and sha(ck)==expected
  from sklearn.metrics import roc_auc_score,average_precision_score,roc_curve,precision_recall_curve
  import matplotlib;matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  gt=np.array([[r['labels'][l] for l in L] for r in cohort]);groups={}
  for i,a in enumerate(align):groups.setdefault(a['patient_id'],[]).append(i)
  assert len(groups)==500;group=list(groups.values());rng=np.random.default_rng(42);boot=[np.concatenate([group[k] for k in rng.integers(0,500,500)]) for _ in range(1000)];metrics=[];fig,ax=plt.subplots(2,5,figsize=(20,8))
  for j,l in enumerate(L):
   y=gt[:,j];s=scores[:,j];av=[roc_auc_score(y[b],s[b]) for b in boot if len(np.unique(y[b]))==2];pv=[average_precision_score(y[b],s[b]) for b in boot if len(np.unique(y[b]))==2];auc=roc_auc_score(y,s);ap=average_precision_score(y,s)
   for name,p in [('CLEAR_fixed0.5_exploratory',(s>=.5).astype(float)),('CXR_joint_definite_parsed_only',pred[:,j]),('CXR_sensitivity_missing_negative',np.nan_to_num(pred[:,j],nan=0)),('CLEAR_on_CXR_parsed_subset',np.where(np.isfinite(pred[:,j]),(s>=.5).astype(float),np.nan))]:
    st=statistics(y,p);fs=[statistics(y[b],p[b])['f1'] for b in boot];ci=np.nanquantile(fs,[.025,.975]);full=name=='CLEAR_fixed0.5_exploratory';metrics.append(dict(finding=l,policy=name,**st,f1_ci_low=ci[0],f1_ci_high=ci[1],prediction_coverage=float(np.isfinite(p).mean()),auroc=auc if full else None,auprc=ap if full else None,auroc_ci_low=np.quantile(av,.025) if full else None,auroc_ci_high=np.quantile(av,.975) if full else None,auprc_ci_low=np.quantile(pv,.025) if full else None,auprc_ci_high=np.quantile(pv,.975) if full else None))
   fpr,tpr,_=roc_curve(y,s);pr,re,_=precision_recall_curve(y,s);ax[0,j].plot(fpr,tpr);ax[0,j].plot([0,1],[0,1],'--');ax[0,j].set_title(f'{l}\nAUC {auc:.3f} [{np.quantile(av,.025):.3f},{np.quantile(av,.975):.3f}]');ax[1,j].plot(re,pr);ax[1,j].set_title(f'AP {ap:.3f} [{np.quantile(pv,.025):.3f},{np.quantile(pv,.975):.3f}]');ax[0,j].set_xlabel('FPR');ax[0,j].set_ylabel('TPR');ax[1,j].set_xlabel('Recall');ax[1,j].set_ylabel('Precision')
  fig.suptitle('E2 direct contrastive only; locked CheXpert;500 patient clusters;uncalibrated');fig.tight_layout();fig.savefig(run/'e2_roc_pr.png',dpi=140);plt.close(fig);write_csv(run/'e2_matched_metrics.csv',metrics)
  from PIL import ImageDraw,ImageFont
  selected=[]
  for j,l in enumerate(L):
   inds=[i for i in range(518) if np.isfinite(pred[i,j]) and pred[i,j]!=int(scores[i,j]>=.5)];selected.extend((i,j) for i in inds[:2])
  canvas=Image.new('RGB',(1500,550*((len(selected)+2)//3)),'white');draw=ImageDraw.Draw(canvas);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15);evidence=[]
  for k,(i,j) in enumerate(selected):
   a=align[i];im=Image.open(cohort[i]['image_path']).convert('L');w,h=im.size;im=im.crop((int(w*.1),int(h*.1),int(w*.9),int(h*.9)));arr=np.array(im);arr[arr>235]=0;im=Image.fromarray(arr).convert('RGB');im.thumbnail((480,410));x=k%3*500;y=k//3*550;canvas.paste(im,(x+(500-im.width)//2,y));draw.multiline_text((x+8,y+415),f"{a['image_id']} / {L[j]}\nGT={gt[i,j]} CXR={int(pred[i,j])} CLEAR={int(scores[i,j]>=.5)} score={scores[i,j]:.4f}\nTechnical disagreement; no clinical adjudication\nPrivacy-masked crop; originals linked locally",font=font,fill='black');evidence.append(dict(image_id=a['image_id'],finding=L[j],source_path=cohort[i]['image_path'],source_sha256=a['source_sha256'],gt=int(gt[i,j]),cxr=int(pred[i,j]),clear_score=float(scores[i,j]),note='local only; privacy margins/highlights masked; no expert review'))
  canvas.save(run/'e2_disagreements_contact_sheet.png');write_csv(run/'image_evidence.csv',evidence)
  lat=[r['forward_latency_sec'] for r in timings if r['forward_latency_sec'] is not None];summary=dict(n_images=518,n_studies=500,n_patients=500,cached_images_reused=50,new_forwards=468,warmups=3,E0B_gates=gates,mean_forward_sec=float(np.mean(lat)),p95_forward_sec=float(np.quantile(lat,.95)),wall_sec=time.perf_counter()-start,load_sec=loadsec,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,torch=torch.__version__,cuda_initialized=torch.cuda.is_initialized(),text_shape=list(text.shape),global_shape=[518,768],concept_status='BLOCKED_CONCEPT_ASSETS_OR_RESOURCE',macro_AUROC=float(np.mean([r['auroc'] for r in metrics if r['policy']=='CLEAR_fixed0.5_exploratory'])),macro_AUPRC=float(np.mean([r['auprc'] for r in metrics if r['policy']=='CLEAR_fixed0.5_exploratory'])),macro_F1={p:float(np.mean([r['f1'] for r in metrics if r['policy']==p])) for p in sorted({r['policy'] for r in metrics})})
  write_json(run/'metrics.json',summary);write_json(run/'environment_manifest.json',dict(executable=sys.executable,prefix=sys.prefix,python=sys.version,torch=torch.__version__,source_sha256=sources));command=f"CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 {sys.executable} -m research.e2_direct";(run/'reproduce.sh').write_text('#!/bin/bash\nset -euo pipefail\ncd '+str(ROOT)+'\n'+command+'\n')
  report=ROOT/'reports'/f'{m["start_time"][:10]}_E2_direct_contrastive.md';lines=['# E2 direct contrastive — locked CheXpert518','',f'Run `{run}`; frozen protocol SHA256 `{ph}`.','', 'Official fixed lowercase positive/no prompts; normalized paired cosine softmax, unscaled. Ranking is primary; fixed0.5 F1 exploratory uncalibrated. Not full concept CLEAR. No test tuning. CXR discrete/missing scores have no invented AUROC.','', '|Finding|AUROC (95% cluster CI)|AUPRC (95% cluster CI)|CLEAR F1|CXR parsed F1|','|---|---|---|---|---|']
  for l in L:
   c=next(r for r in metrics if r['finding']==l and r['policy']=='CLEAR_fixed0.5_exploratory');b=next(r for r in metrics if r['finding']==l and r['policy']=='CXR_joint_definite_parsed_only');lines.append(f"|{l}|{c['auroc']:.6f} ({c['auroc_ci_low']:.6f}–{c['auroc_ci_high']:.6f})|{c['auprc']:.6f} ({c['auprc_ci_low']:.6f}–{c['auprc_ci_high']:.6f})|{c['f1']:.6f}|{b['f1']:.6f}|")
  lines+=['','Compute:','```json',json.dumps(summary,indent=2),'```','',f'Command `{command}`.','',f'Evidence `{run}/e2_roc_pr.png`, `e2_disagreements_contact_sheet.png`, `alignment_manifest.csv`, `image_evidence.csv`. Contact sheets mask margins/highlights; original pixels linked locally. Technical label disagreements, no medical expert validation.','', 'GO: direct contrastive executed. Full concepts BLOCKED_CONCEPT_ASSETS_OR_RESOURCE; bank/768D/SFR4096D assets absent locally; no downloads. No independent encoder-quality or CPU/GPU efficiency conclusion. Encoder pretraining membership uncertain. CLEAR full518 F1 and CXR parsed-only differ denominators; CSV also provides CLEAR on exactly CXR parsed mask and baseline missing-negative sensitivity.'];report.write_text('\n'.join(lines)+'\n');smoke=ROOT/'reports'/f'{m["start_time"][:10]}_E0B_text_concept_smoke.md';smoke.write_text('# E0B text/concept smoke\n\nOfficial tokenize/encode_text PASS; finite normalized10×768, tokenizer10×77, text repeat difference0. Gates:\n'+json.dumps(gates,indent=2)+'\nFull concepts absent; no label-prompt substitution for concept bank.\n')
  m.update(metric_summary=summary,outputs=[str(report),str(smoke)],next_action='Review direct-only E2; fullconcept requiresasset/resourceapproval');finish_run(run,m);print(json.dumps(summary),flush=True)
 except Exception as exc:
  (run/'failure.log').write_text(traceback.format_exc());m['blocker']=str(exc);finish_run(run,m,'FAILED');raise
if __name__=='__main__':main()
