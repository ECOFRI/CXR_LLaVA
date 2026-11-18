"""Resource recorder around E1 jobs, preserving the frozen feature/probe runners."""
import argparse,json,os,resource,sys,time
from pathlib import Path
from research.common import now,sha,write_json

def run(a):
    start=now(); wall=time.perf_counter(); result=0
    target=Path(a.run_dir if a.mode=='features' else a.output)
    try:
        if a.mode=='features':
            from research.e1_features import run as execute
        else:
            from research.e1_probe import run as execute
        execute(a)
    except BaseException:
        result=1;raise
    finally:
        r=resource.getrusage(resource.RUSAGE_SELF)
        recorded=dict(process_pid=os.getpid(),exit_code=result,start_time=start,end_time=now(),process_wall_sec=time.perf_counter()-wall,cpu_user_sec=r.ru_utime,cpu_system_sec=r.ru_stime,peak_RSS_bytes=r.ru_maxrss*1024,wrapper_sha256=sha(__file__),command=sys.argv)
        target.mkdir(parents=True,exist_ok=True);write_json(target/'process_resources.json',recorded)
        manifest=target/'run_manifest.json'
        if manifest.exists():
            m=json.loads(manifest.read_text());m.update(process_pid=os.getpid(),exit_code=result,cpu_user_sec=r.ru_utime,cpu_system_sec=r.ru_stime,process_resource_record=str(target/'process_resources.json'))
            if a.mode=='probe':m['start_time']=start
            write_json(manifest,m)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['features','probe'],required=True);p.add_argument('--encoder',choices=['cxr','clear']);p.add_argument('--run-dir');p.add_argument('--cxr');p.add_argument('--clear');p.add_argument('--output');run(p.parse_args())
