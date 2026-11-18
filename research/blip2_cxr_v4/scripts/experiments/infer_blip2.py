"""Offline, single-device BLIP-2 X02 smoke inference; execution is launcher-controlled."""
import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

VR=Path(__file__).resolve().parents[2]
ROOT=VR.parents[1]


def sample_cases(config):
    path=ROOT/config['cohort_csv']
    with path.open(newline='',encoding='utf-8-sig') as stream:
        rows=list(csv.DictReader(stream))
    frontal=[(i,r) for i,r in enumerate(rows) if r['view_label'].startswith('frontal')]
    if len(frontal)<config['sample_n']:raise ValueError('Insufficient frontal cohort')
    return random.Random(config['sample_seed']).sample(frontal,config['sample_n'])


def valid_generated_report(text):
    return isinstance(text,str) and bool(text.strip())


def write_status(out,stage,status,**details):
    data={'stage':stage,'status':status,**details}
    (out/(stage+'_status.json')).write_text(json.dumps(data,indent=2)+'\n')
    (out/'status.json').write_text(json.dumps(data,indent=2)+'\n')


def probe_fp16(torch, device):
    functional=torch.nn.functional
    result={'status':'BLOCKED','operations':{},'matmul_relative_error':None,
            'matmul_relative_error_limit':1e-2}
    try:
        with torch.no_grad():
            left=torch.arange(1,65,dtype=torch.float32,device=device).reshape(8,8)/64
            right=torch.arange(64,0,-1,dtype=torch.float32,device=device).reshape(8,8)/64
            actual=left.half()@right.half()
            expected=left@right
            relative=((actual.float()-expected).norm()/expected.norm()).item()
            result['matmul_relative_error']=relative
            result['operations']['matmul']=bool(torch.isfinite(actual).all().item() and relative<1e-2)
            batched=torch.bmm(left.half().unsqueeze(0),right.half().unsqueeze(0))
            result['operations']['bmm']=bool(torch.isfinite(batched).all().item())
            normalized=functional.layer_norm(left.half(),(8,))
            result['operations']['layer_norm']=bool(torch.isfinite(normalized).all().item())
            probabilities=functional.softmax(left.half(),dim=-1)
            result['operations']['softmax']=bool(torch.isfinite(probabilities).all().item())
            activated=functional.gelu(left.half())
            result['operations']['gelu']=bool(torch.isfinite(activated).all().item())
            pixels=torch.arange(1,65,dtype=torch.float32,device=device).reshape(1,1,8,8).half()/64
            kernel=torch.ones((1,1,3,3),dtype=torch.float16,device=device)/9
            convolved=functional.conv2d(pixels,kernel)
            result['operations']['conv2d']=bool(torch.isfinite(convolved).all().item())
            indices=torch.tensor([0,2,4],dtype=torch.long,device=device)
            weights=torch.arange(1,33,dtype=torch.float32,device=device).reshape(8,4).half()/32
            embedded=functional.embedding(indices,weights)
            result['operations']['embedding']=bool(torch.isfinite(embedded).all().item())
            torch.cuda.synchronize(device)
    except (torch.cuda.OutOfMemoryError,RuntimeError) as exc:
        result['error_type']=type(exc).__name__
        return result
    result['status']='PASS' if len(result['operations'])==7 and all(result['operations'].values()) else 'BLOCKED'
    return result


def run(config,args,cases,out):
    import torch
    from PIL import Image
    from transformers import Blip2ForConditionalGeneration, Blip2Processor
    stage='stage1' if args.start_index==0 else 'stage2'
    if os.environ.get('HF_HUB_OFFLINE')!='1' or os.environ.get('TRANSFORMERS_OFFLINE')!='1':
        raise RuntimeError('HF offline environment required')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(args.gpu):
        raise RuntimeError('Exactly the requested GPU must be visible')
    try:
        available=torch.cuda.is_available() and torch.cuda.device_count()==1
        capability=torch.cuda.get_device_capability(0) if available else None
    except RuntimeError:
        write_status(out,stage,'BLOCKED',reason='CUDA capability probe failed');return 2
    if not available:
        write_status(out,stage,'BLOCKED',reason='Single CUDA device unavailable');return 2
    major,minor=capability
    built_arches=torch.cuda.get_arch_list()
    if built_arches and ('sm_'+str(major)+str(minor)) not in built_arches:
        write_status(out,stage,'BLOCKED',reason='Torch build lacks selected GPU architecture',
                     capability=[major,minor],built_arches=built_arches,fp16_probe={'status':'NOT_RUN'});return 2
    probe=probe_fp16(torch,'cuda:0')
    if probe['status']!='PASS':
        write_status(out,stage,'BLOCKED',reason='FP16 runtime probe failed',capability=[major,minor],
                     built_arches=built_arches,fp16_probe=probe);return 2
    checkpoint=ROOT/config['checkpoint_dir']
    if not checkpoint.is_dir():
        write_status(out,stage,'BLOCKED',reason='Local checkpoint missing',fp16_probe=probe);return 2
    try:
        processor=Blip2Processor.from_pretrained(str(checkpoint),local_files_only=True,use_fast=False)
        model=Blip2ForConditionalGeneration.from_pretrained(str(checkpoint),local_files_only=True,
                                                             torch_dtype=torch.float16,low_cpu_mem_usage=True)
        model=model.to('cuda:0').eval()
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        reason='OOM/CUDA error' if 'cuda' in str(exc).lower() or 'out of memory' in str(exc).lower() else 'Runtime error'
        write_status(out,stage,'BLOCKED',reason=reason,completed_cases=0,fp16_probe=probe)
        return 2
    output=out/'predictions.jsonl'
    if output.exists() and args.start_index==0 and not args.resume:
        raise FileExistsError('Predictions exist; use --resume with reviewed state')
    mode='a' if args.start_index>0 else 'w'
    selected=cases[args.start_index:args.start_index+args.max_cases]
    n_valid=0
    for cohort_index,row in selected:
        started=time.monotonic()
        try:
            with Image.open(ROOT/row['image_path']) as image:
                inputs=processor(images=image.convert('RGB'),text=config['prompt'],return_tensors='pt')
            inputs={k:(v.to(device='cuda:0',dtype=model.dtype) if k=='pixel_values'
                       else v.to('cuda:0')) for k,v in inputs.items()}
            torch.cuda.reset_peak_memory_stats(0)
            with torch.no_grad():
                tokens=model.generate(**inputs,max_new_tokens=config['max_new_tokens'],do_sample=False)
            text=processor.batch_decode(tokens,skip_special_tokens=True)[0].strip()
            peak=torch.cuda.max_memory_allocated(0)
            valid=valid_generated_report(text)
            record={'cohort_index':cohort_index,'case_key_sha256':hashlib.sha256(row['report_id'].encode()).hexdigest(),
                    'status':'success' if valid else 'invalid_empty','generated_report':text,
                    'prompt_version':config['prompt_version'],
                    'latency_sec':time.monotonic()-started,'peak_allocated_bytes':peak}
        except (torch.cuda.OutOfMemoryError,RuntimeError) as exc:
            reason='OOM/CUDA error' if 'cuda' in str(exc).lower() or 'out of memory' in str(exc).lower() else 'Runtime error'
            write_status(out,stage,'BLOCKED',reason=reason,completed_cases=selected.index((cohort_index,row)),fp16_probe=probe)
            return 2
        with output.open(mode,encoding='utf-8') as stream:
            stream.write(json.dumps(record,ensure_ascii=False)+'\n')
        mode='a'
        if not valid:
            write_status(out,stage,'FAIL',reason='empty_generated_report',completed_cases=n_valid+1,
                         n_valid=n_valid,n_invalid=1,fp16_probe=probe)
            return 1
        n_valid+=1
    write_status(out,stage,'PASS',completed_cases=len(selected),gpu=args.gpu,
                 n_valid=n_valid,n_invalid=0,
                 max_memory_allocated_bytes=torch.cuda.max_memory_allocated(0),fp16_probe=probe)
    return 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=VR/'configs/experiments/X02_smoke.yaml')
    p.add_argument('--output-dir',type=Path,default=VR/'runs/X02_smoke_dry_run')
    p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--start-index',type=int,default=0)
    p.add_argument('--max-cases',type=int,default=1)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--resume',action='store_true')
    a=p.parse_args();config=json.loads(a.config.read_text())
    if config['batch_size']!=1 or config['dtype']!='float16' or not config['local_files_only']:
        raise ValueError('X02 requires offline batch-one FP16')
    if (a.start_index,a.max_cases) not in ((0,1),(1,9)):
        raise ValueError('Only stage1 0:1 and stage2 1:10 are allowed')
    cases=sample_cases(config)
    paths_ok=sum((ROOT/r['image_path']).is_file() for _,r in cases)
    if a.dry_run:
        print(json.dumps({'status':'DRY_RUN','cohort_n':len(cases),'image_paths_found':paths_ok,
                          'checkpoint_present':(ROOT/config['checkpoint_dir']).is_dir(),
                          'stage_start':a.start_index,'stage_n':a.max_cases,'gpu_job_started':False}))
        return
    if paths_ok!=len(cases):raise FileNotFoundError('Sample image path missing')
    out=a.output_dir.resolve()
    if VR not in out.parents:raise ValueError('Output must be under V4 root')
    out.mkdir(parents=True,exist_ok=True)
    code=run(config,a,cases,out)
    sys.exit(code)

if __name__=='__main__':main()
