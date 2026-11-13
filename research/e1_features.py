"""Frozen NIH full-cohort CLS/patchmean extraction, resumable, no model edits."""
import argparse,csv,json,os,resource,sys,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from research.common import ROOT,RESEARCH,sha,write_csv,write_json,now,command

def run(args):
    run=Path(args.run_dir); run.mkdir(parents=True,exist_ok=True)
    rows=list(csv.DictReader((RESEARCH/'inventory/e1_nih/cohort_full.csv').open()))
    protocol=RESEARCH/'protocols/E1_nih_exploratory_protocol.yaml'
    assert protocol.exists()
    torch.manual_seed(42);torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    wall=time.perf_counter(); cls=np.zeros((len(rows),1024 if args.encoder=='cxr' else 768),np.float32);patch=cls.copy(); complete=np.zeros(len(rows),bool)
    cache=run/'features.npz'; meta=run/'feature_manifest.csv'
    records=list(csv.DictReader(meta.open())) if meta.exists() else []
    prior=json.loads((run/'run_manifest.json').read_text()) if (run/'run_manifest.json').exists() else None
    if cache.exists():
        old=np.load(cache);cls[:]=old['cls'];patch[:]=old['patchmean'];complete[:]=old['complete'];assert list(old['image_ids'])==[r['image_id'] for r in rows]
        assert prior is not None, 'Cache without provenance manifest'
        previous={r['image_id']:r for r in records}
        assert len(previous)==int(complete.sum()), 'Manifest/cache completion mismatch'
        for i,r in enumerate(rows):
            if complete[i]:assert sha(r['image_path'])==previous[r['image_id']]['source_sha256'], 'Reused source changed'
    manifest=dict(commit=command(['git','rev-parse','HEAD'])['stdout'].strip(),status='RUNNING',encoder=args.encoder,start_time=now(),protocol_sha256=sha(protocol),cohort_sha256=sha(RESEARCH/'inventory/e1_nih/cohort_full.csv'),runner_sha256=sha(__file__),command=sys.argv,environment=sys.prefix,torch=torch.__version__,precision='FP32',batch=1,estimated_gpu_hours=0.85 if args.encoder=='cxr' else 0,ram_budget_gib=8,feature_fields=['native_cls','native_patch_mean'],warmup_forwards=3,timing_scope='native backbone only; excludes preprocessing/IO',resume_completed=int(complete.sum()))
    if args.encoder=='cxr':
        assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0';assert torch.cuda.is_available(); assert torch.cuda.mem_get_info()[0]>4*1024**3
        from research.encoder_harness import load_vision
        tower,projector,processor,cfg,digests=load_vision(ROOT/'model_cache/cxr-llava-v2-original');del projector
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        tower=tower.cuda().float().eval().requires_grad_(False); capture={}
        handle=tower.ln_post.register_forward_hook(lambda m,a,o:capture.update(native=o))
        manifest.update(model_config_sha256=sha(ROOT/'model_cache/cxr-llava-v2-original/config.json'),checkpoint_revision='b2224786bb90d54b1e1291171866706cfbb44e2b',checkpoint_tensor_sha256=digests,source_sha256=sha(ROOT/'CXR_LLAVA_HF/VisualTransformer.py'),preprocessing=processor.to_dict())
        def prepare(path):
            with Image.open(path) as im:arr=np.asarray(im.convert('L'))[...,None]
            return processor(arr,return_tensors='pt',input_data_format='channels_last')['pixel_values'].cuda().float()
        def extract(x):
            g,p=tower(x);n=capture['native'];assert n.shape==(1,1025,1024) and p.shape==(1,1024,1024);assert torch.equal(n[:,1:],p)
            return n[:,0],p.mean(1)
        def sync():torch.cuda.synchronize()
    else:
        assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';assert torch.version.cuda is None
        sys.path.insert(0,str(ROOT.parent/'CLEAR/src'));torch.hub.set_dir(str(RESEARCH/'cache/torch_hub'))
        from clear import load_pretrained
        checkpoint=list(Path('/home/student4/.cache/huggingface/hub/models--peterhan91--CLEAR/snapshots').glob('*/best_model.pt'));assert len(checkpoint)==1;checkpoint=checkpoint[0]
        expected='da1617a7d47970667a7a158eabfa4c4f3798d5d03b7f98caec150d8ee6ab6a03';assert sha(checkpoint)==expected
        model,preprocess=load_pretrained(checkpoint_path=checkpoint,device='cpu',local_files_only=True,strict=True);model=model.float().eval().requires_grad_(False)
        manifest.update(clear_commit=command(['git','rev-parse','HEAD'],cwd=ROOT.parent/'CLEAR')['stdout'].strip(),dinov2_commit=command(['git','rev-parse','HEAD'],cwd=RESEARCH/'upstream/dinov2')['stdout'].strip(),dinov2_attention_sha256=sha(RESEARCH/'upstream/dinov2/dinov2/layers/attention.py'),checkpoint_sha256=expected,source_sha256=sha(ROOT.parent/'CLEAR/src/clear/hub.py'),preprocessing=repr(preprocess))
        def prepare(path):
            with Image.open(path) as im:return preprocess(im).unsqueeze(0).float()
        def extract(x):
            f=model.visual.backbone.forward_features(x); assert f['x_norm_clstoken'].shape==(1,768) and f['x_norm_patchtokens'].shape==(1,1024,768) and f['x_norm_regtokens'].shape==(1,4,768)
            return f['x_norm_clstoken'],f['x_norm_patchtokens'].mean(1)
        def sync():pass
    if prior is not None:
        for key in ['encoder','environment','torch','runner_sha256','protocol_sha256','cohort_sha256','source_sha256','preprocessing','precision','feature_fields','checkpoint_sha256' if args.encoder=='clear' else 'checkpoint_tensor_sha256']:
            assert prior[key]==manifest[key], f'Resume provenance mismatch: {key}'
    first=next((i for i in range(len(rows)) if not complete[i]),None)
    if first is None:
        assert prior['status']=='COMPLETE', 'All complete but previous finalization absent'
        print('Verified complete cache; manifest preserved',flush=True);return
    write_json(run/'run_manifest.json',manifest)
    x=prepare(rows[first]['image_path'])
    with torch.inference_mode():
        for _ in range(3):extract(x)
    sync();work=time.perf_counter();timings=[]
    def persist():
        tmp=run/'features.tmp.npz';np.savez(tmp,cls=cls,patchmean=patch,complete=complete,image_ids=np.asarray([r['image_id'] for r in rows]));tmp.replace(cache)
        write_csv(meta,records);write_json(run/'progress.json',dict(completed=int(complete.sum()),total=len(rows),timestamp=now()))
    try:
        for i,r in enumerate(rows):
            if complete[i]:continue
            source=sha(r['image_path']);x=prepare(r['image_path']);sync();t=time.perf_counter()
            with torch.inference_mode():a,b=extract(x)
            sync();lat=time.perf_counter()-t;timings.append(lat)
            assert torch.isfinite(a).all() and torch.isfinite(b).all();delta=None
            if i in (0,9,49,len(rows)-1):
                with torch.inference_mode():aa,bb=extract(x)
                sync();delta=max(float((a-aa).abs().max()),float((b-bb).abs().max()));assert delta<=1e-6
            cls[i]=a.cpu().numpy()[0];patch[i]=b.cpu().numpy()[0];complete[i]=True
            rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;assert rss<8*1024**3
            records.append(dict(image_id=r['image_id'],patient_id=r['patient_id'],split=r['split'],source_sha256=source,forward_latency_sec=lat,determinism_max_abs_diff=delta,cls_shape=str(list(a.shape)),patchmean_shape=str(list(b.shape))))
            if i+1 in (10,50) or (i+1)%100==0 or i==len(rows)-1:
                persist();print(args.encoder,i+1,'mean_forward',np.mean(timings),'RSS_GiB',rss/1024**3,flush=True)
                if args.encoder=='cxr':
                    estimate=((time.perf_counter()-work)/(len(timings))*len(rows)+(work-wall))/3600
                    if estimate>2:raise RuntimeError(f'Projected job {estimate:.3f}GPUh exceeds2GPUh; checkpoint retained')
            if args.encoder=='cxr' and time.perf_counter()-wall>7200:raise RuntimeError('2GPUh hardstop')
        persist()
        if args.encoder=='clear':assert sha(checkpoint)==manifest['checkpoint_sha256'] and not torch.cuda.is_initialized()
        else:
            import hashlib
            for key,value in tower.state_dict().items():
                assert hashlib.sha256(value.cpu().numpy().tobytes()).hexdigest()==manifest['checkpoint_tensor_sha256']['vision_tower.'+key], 'Vision tensor changed'
        manifest['checkpoint_integrity']='PASS'
        assert sha(ROOT/'CXR_LLAVA_HF/VisualTransformer.py' if args.encoder=='cxr' else ROOT.parent/'CLEAR/src/clear/hub.py')==manifest['source_sha256'], 'Original source changed'
        manifest.update(status='COMPLETE',completed=int(complete.sum()),end_time=now(),wall_sec=time.perf_counter()-wall,extraction_wall_sec=time.perf_counter()-work,mean_forward_sec=float(np.mean(timings)),p95_forward_sec=float(np.percentile(timings,95)),peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,cache_sha256=sha(cache),cache_bytes=cache.stat().st_size,actual_gpu_hours=(time.perf_counter()-wall)/3600 if args.encoder=='cxr' else 0)
        if args.encoder=='cxr':manifest.update(gpu_peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,gpu_peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3)
        write_json(run/'run_manifest.json',manifest)
    except Exception as e:
        persist();manifest.update(status='FAILED',blocker=str(e),end_time=now(),completed=int(complete.sum()));write_json(run/'run_manifest.json',manifest);raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--encoder',choices=['cxr','clear'],required=True);p.add_argument('--run-dir',required=True);run(p.parse_args())
