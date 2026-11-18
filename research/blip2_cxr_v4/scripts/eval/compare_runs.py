"""Evaluate saved E4 arms on a strict, paired Indiana study cohort."""
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

VR = Path(__file__).resolve().parents[2]
ROOT = VR.parents[1]
sys.path.insert(0, str(VR / 'src'))
from vlm_medical.eval.text_metrics import case_statistics, corpus_scores
from vlm_medical.eval.clinical_metrics import confusion, enrich, label, summarize
from paired_bootstrap import draws, interval

FINDINGS = ('Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion', 'Pneumothorax')
STATE = {'Positive': '1', 'Negative': '0', 'Uncertain': '-1', 'Not mentioned': ''}


def csvrows(path):
    with path.open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def strict_cohort(rows, baseline):
    grouped = {}
    for row in rows:
        grouped.setdefault(row['arm'], []).append(row)
    if baseline not in grouped:
        raise ValueError('Baseline arm missing')
    expected = {r['image_id'] for r in grouped[baseline]}
    if len(expected) != len(grouped[baseline]):
        raise ValueError('Duplicate study in baseline')
    for arm, records in grouped.items():
        keys = [r['image_id'] for r in records]
        if len(set(keys)) != len(keys) or set(keys) != expected:
            raise ValueError(f'Study cohort mismatch: {arm}')
    return {arm: {r['image_id']: r for r in records} for arm, records in grouped.items()}, sorted(expected)


def reference_map(cohort, source):
    candidates = {r['report_id']: r for r in source}
    out = {}
    for row in cohort:
        r = candidates.get(row['report_id'])
        if not r:
            raise ValueError('Reference key mismatch')
        text = r['ground_truth_report']
        if hashlib.sha256(text.encode()).hexdigest() != row['reference_report_sha256']:
            raise ValueError('Reference hash mismatch')
        out[row['image_id']] = text
    return out


def clinical(ids, arm, refs, labels, policy):
    byclass = {}
    for f in FINDINGS:
        pairs = [(label(STATE[refs[i][f]], policy), label(STATE[labels[(i, arm, f)]], policy)) for i in ids]
        byclass[f] = enrich(confusion(pairs), len(ids))
        byclass[f]['omissions'] = sum(a == 1 and b != 1 for a,b in pairs)
        byclass[f]['hallucinations'] = sum(b == 1 and refs[i][f] in ('Negative', 'Not mentioned') for i,(a,b) in zip(ids,pairs))
    return {**summarize(byclass), 'per_pathology': byclass}


def evaluate(config):
    paths = {k: ROOT / config[k] for k in ('reports','cohort','labels','reference_source')}
    reports = [json.loads(line) for line in paths['reports'].open()]
    grouped, ids = strict_cohort(reports, config['baseline_arm'])
    cohort = csvrows(paths['cohort'])
    if len(cohort) != len(ids) or len({r['report_id'] for r in cohort}) != len(cohort) or {r['image_id'] for r in cohort} != set(ids):
        raise ValueError('Cohort file mismatch')
    refs = {r['image_id']: r for r in cohort}
    reftext = reference_map(cohort, csvrows(paths['reference_source']))
    rawlabels = csvrows(paths['labels'])
    labels = {(r['image_id'],r['arm'],r['finding']):r['label_state'] for r in rawlabels}
    if len(labels) != len(rawlabels) or len(labels) != len(ids)*len(grouped)*len(FINDINGS):
        raise ValueError('Label grid incomplete or duplicated')
    if any(r['status'] != 'success' or not r['generated_report'].strip() for r in reports):
        raise ValueError('Unsuccessful or empty saved report')
    stats = {a:{i:case_statistics(reftext[i],records[i]['generated_report']) for i in ids} for a,records in grouped.items()}
    baseline = config['baseline_arm']
    point = {}
    for arm in grouped:
        point[arm] = {'text':corpus_scores([stats[arm][i] for i in ids]),
                      'clinical':{p:clinical(ids,arm,refs,labels,p) for p in config['policies']},
                      'unchanged_vs_baseline':sum(grouped[arm][i]['generated_report'].encode()==grouped[baseline][i]['generated_report'].encode() for i in ids)}
    samples = draws(ids,config['bootstrap_reps'],config['seed'])
    metrics = ['bleu1','bleu2','bleu3','bleu4','rouge_l']
    for arm in grouped:
        text_values = {m:[] for m in metrics}
        delta_values = {m:[] for m in metrics}
        clinical_values = {p:{m:[] for m in ('micro_f1','macro_f1')} for p in config['policies']}
        clinical_delta = {p:{m:[] for m in ('micro_f1','macro_f1')} for p in config['policies']}
        for sample in samples:
            score = corpus_scores([stats[arm][i] for i in sample])
            base_score = corpus_scores([stats[baseline][i] for i in sample])
            for m in metrics:
                text_values[m].append(score[m]);delta_values[m].append(score[m]-base_score[m])
            for p in config['policies']:
                c = clinical(sample,arm,refs,labels,p)
                b = clinical(sample,baseline,refs,labels,p)
                for m in ('micro_f1','macro_f1'):
                    clinical_values[p][m].append(c[m])
                    clinical_delta[p][m].append(c[m]-b[m] if c[m] is not None and b[m] is not None else None)
        point[arm]['text_ci95']={m:interval(text_values[m]) for m in metrics}
        point[arm]['delta_vs_baseline']={m:{'value':point[arm]['text'][m]-point[baseline]['text'][m], 'ci95':interval(delta_values[m])} for m in metrics}
        for p in config['policies']:
            point[arm]['clinical'][p]['ci95']={m:interval(clinical_values[p][m]) for m in ('micro_f1','macro_f1')}
            point[arm]['clinical'][p]['delta_vs_baseline']={m:{'value':point[arm]['clinical'][p][m]-point[baseline]['clinical'][p][m] if point[arm]['clinical'][p][m] is not None and point[baseline]['clinical'][p][m] is not None else None,'ci95':interval(clinical_delta[p][m])} for m in ('micro_f1','macro_f1')}
    return {'status':'PASS','classification':'exploratory; 24 studies; descriptive unless a paired delta CI excludes zero','n_studies':len(ids),'n_reports':len(reports),'n_arms':len(grouped),'baseline':baseline,'tokenizer':config['tokenizer'],'bootstrap':{'unit':'study','seed':config['seed'],'reps':config['bootstrap_reps']},'meteor':{'value':None,'reason':'pending P2.5'},'arms':point}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=VR/'configs/eval/e4_paired.yaml')
    parser.add_argument('--output-dir',type=Path,default=VR/'runs/P2_e4_paired')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--meteor-only',action='store_true',help='Fill METEOR in existing paired metrics')
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    if config['tokenizer']!='lowercase_ascii_alnum_regex_v1':raise ValueError('Unsupported tokenizer')
    if args.meteor_only and not args.dry_run:
        import nltk
        from vlm_medical.eval.meteor_metric import score_pairs, sha256
        paths={k:ROOT/config[k] for k in ('reports','cohort','reference_source')}
        reports=[json.loads(line) for line in paths['reports'].open()]
        grouped,ids=strict_cohort(reports,config['baseline_arm'])
        cohort=csvrows(paths['cohort'])
        refs=reference_map(cohort,csvrows(paths['reference_source']))
        values={a:dict(zip(ids,score_pairs([(refs[i],records[i]['generated_report']) for i in ids]))) for a,records in grouped.items()}
        samples=draws(ids,config['bootstrap_reps'],config['seed'])
        path=args.output_dir/'metrics.json'
        result=json.loads(path.read_text())
        if result['n_studies']!=len(ids) or result['n_arms']!=len(grouped):raise ValueError('Existing paired cohort mismatch')
        baseline=config['baseline_arm']
        for arm in grouped:
            mean=sum(values[arm].values())/len(ids)
            base=sum(values[baseline].values())/len(ids)
            ci=interval([sum(values[arm][i] for i in sample)/len(sample) for sample in samples])
            delta_ci=interval([sum(values[arm][i]-values[baseline][i] for i in sample)/len(sample) for sample in samples])
            result['arms'][arm]['meteor']={'value':mean,'ci95':ci,'delta_vs_baseline':{'value':mean-base,'ci95':delta_ci}}
        resource=VR/'.cache/nltk/corpora/wordnet.zip'
        result['meteor']={'status':'PASS','scorer':'NLTK 3.6.7 vendored scorer; installed NLTK '+nltk.__version__,
                          'wordnet_sha256':sha256(resource),'tokenizer':config['tokenizer'],'bootstrap_unit':'study'}
        path.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({'status':'PASS','n_studies':len(ids),'baseline_meteor':result['arms'][baseline]['meteor']['value']}))
        return
    if args.dry_run:
        print(json.dumps({'status':'DRY_RUN','reports':config['reports'],'reps':config['bootstrap_reps']}));return
    out=args.output_dir.resolve()
    if out.exists() and (out/'metrics.json').exists() and not args.resume:raise FileExistsError(out/'metrics.json')
    out.mkdir(parents=True,exist_ok=True)
    result=evaluate(config)
    (out/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'status.json').write_text(json.dumps({'status':'PASS','n_studies':result['n_studies'],'n_reports':result['n_reports']})+'\n')
    print(json.dumps({'status':'PASS','n_studies':result['n_studies'],'n_reports':result['n_reports'],'n_arms':result['n_arms']}))

if __name__=='__main__':main()
