"""Offline 18-section descriptive report; no model loading."""
import csv,json,subprocess
from pathlib import Path
from research.common import ROOT,sha,write_json
r=ROOT/'research/runs/20261013T214811_E2_direct_contrastive';s=json.loads((r/'metrics.json').read_text());m=json.loads((r/'run_manifest.json').read_text());rows=list(csv.DictReader((r/'e2_matched_metrics.csv').open()));env=json.loads((r/'environment_manifest.json').read_text());pair=list(csv.DictReader((r/'paired_f1_differences.csv').open()));ver={p.split('==')[0]:p.split('==')[1] for p in (r/'pip_freeze.txt').read_text().splitlines() if '==' in p};report=ROOT/'reports/2026-10-13_E0B_E2_direct_contrastive_analysis.md'
text=f'''# Experiment Analysis Log — E0B and E2 direct contrastive

## 1. Experiment Information

Experiment ID `{m['run_id']}`; date2026-10-13 Asia/Ho_Chi_Minh. Execution agent Codex; project CXR_LLaVA_Improvement; repository https://github.com/hanhpm/CXR_LLaVA_Improvement; branchmaster; projectcommit `{m['commit']}`. Scripts `research/e2_direct.py`, `research/e2_finalize.py`, `research/e2_analysis_report.py`. Status **COMPLETED** for E0B1→10 and E2 directcontrastive518; fullconcept pathway **BLOCKED_CONCEPT_ASSETS_OR_RESOURCE**.

## 2. Objective

Establish official CLEAR text/paired-prompt API and evaluate one prespecified directimage/text pipeline on locked CheXpert518 against existing frozen CXR Yes/No outputs. No train/test tuning, LLM training or new dependencies.

## 3. Survey / Technical Motivation

Official CLEAR repositorycommit `{env['CLEAR_commit']}`, DINOcommit `{env['DINO_commit']}`. README and `examples/quickstart.py` specify normalizedimage/text vectors and unscaledpairedcosine softmax; `examples/concepts/exp_zeroshot.py` lines357–358 specify lowercasepositive and `no {{}}` negative prompts; `src/clear/zero_shot.py::run_softmax_eval` fixes positivefirstpairedsoftmax semantics. Sourcefile hashes in E2_frozen_protocol.yaml and environment_manifest.json.

CLEAR Nature Biomedical Engineering DOI10.1038/s41551-026-01741-4 uses a completeconceptprojection pipeline that this experiment does not reproduce. Hypothesis: fixedreleasedimage/textpipeline may yield usable ranking on commonfindings. This is distinct from isolatedencoderquality H1 and conceptutility H2.

## 4. Environment

|Component|Verified value|
|---|---|
|Host|ictserver6 (savedinventory reused)|
|Conda prefix|{env['prefix']}|
|Python|{env['python'].split()[0]}|
|PyTorch|{s['torch']}|
|torchvision|{ver.get('torchvision','see pip_freeze.txt')}|
|tokenizers|{ver.get('tokenizers','not required by CLEAR CLIP tokenizer')}|
|transformers|{ver.get('transformers','not required')}|
|huggingface-hub|{ver.get('huggingface-hub','see pip_freeze.txt')}|
|CUDA/GPU|CPUbuild; CUDAinitializedfalse;0GPUhours|
|Threads|4intraop,1interop|
|Precision|FP32|

Environment manifest and exactpipfreeze saved. Currentpipfreeze exactly matches priorCPUenvironmentsetup file. LiveMemAvailable41.2GiB; jobcap8GiB. SharedNFSfree2.178TB; accountquotaunverified. No fullsysteminventory repeated.

## 5. Dependency Setup

No dependency installation or modification. Reused `cxr-clear-cpu`. HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1; strictcachedcheckpoint loading. xFormersunavailable warnings are expectedCPUfallback; actualforward passed.

## 6. Code / Notebook Changes

New isolatedresearchscripts only; originalCLEARrepo gitstatusclean, checkpointimmutable, originalCXRcode/env/results preserved. Harness initially indexed a missing `parsed_labels` key for an unparsedbaseline response; changed isolatedharness to `.get(l)`, exactly the existingauditorpolicy. Firstattemptfailed before anytestscores. Offlinefinalizer only regeneratedreviewfigures and pairedCIs; never reranencoder.

## 7. Experiment Configuration

|Parameter|Frozen value|
|---|---|
|Input|CheXpert518frontal,500studies,500patients|
|Labels|Atelectasis,Cardiomegaly,Consolidation,Edema,Pleural Effusion|
|Preprocessing|Official448×448RGBsquarebicubic;mean101.48761/255,std83.43944/255|
|Positive/negative text|lowercasefinding / no lowercasefinding|
|Score|unscaledpairedcosinesoftmax positivefirst; notcalibratedprobability|
|Classification|score>=0.5; fixeduntunedexploratory|
|Batch|1|
|Seed|42|
|Bootstrap|1000patientclusters,percentile95%,seed42|
|Cache|50E0Aprojectedglobalsreused;468new; globalonly518×768|
|Gate|E0B1→10 successful beforecontinuedcohort|

Protocolwritten+hashed BEFORE anyscores: `{m['protocol_sha256']}`. CohortSHA `{json.loads((r/'E2_frozen_protocol.yaml').read_text())['cohort_sha256']}`. GTbinarysemantics use alreadyauditedfrozencohort; noGTchanges.

## 8. Code Executed

```python
text_features = normalize(model.encode_text(tokenize(prompts)), dim=-1)
image_features = normalize(model.encode_image(native_input), dim=-1)
logits = (image_features @ text_features.T).reshape(5, 2)
positive_score = logits.softmax(-1)[:, 0]
```

Reproduciblefullcommand in `{r}/reproduce.sh`; offlinefigure/pairedCIcommand in `offline_reproduce.sh`. A reproduction is an explicit newrun, not an automaticallyscheduledrepeat.

## 9. Output

E0Btokenizer10×77 and textfeatures10×768 finite. Imageprojection518×768 finite; endpointofficialAPI/cache differences0; pairedscoredifferences0. 2590scores saved; 518distinctsourceIDs alignedwith518originalbaselinecasehashes. Fullcheckpoint strictload passed and SHA `{m['checkpoint_sha256']}` unchanged.

## 10. Output Analysis

Exactsourcehash and identitymatches passed; sameofficialpreprocessing and checkpointverifiedbeforecachereuse. E0B1/10 officialforwardcomparedagainstcachedprojection withdelta0. CPUmemorybelowcap, noGPUinitialization. Fullconceptassetsnotpresent in approvedlocalrepo/cache/HFmodelcache; no largeassetsdownloaded. Therefore directpipelineGO, fullconceptreproductionblocked.

## 11. Quantitative Results

Primaryranking is CLEARonly; savedCXRbinary/missingresponses are not comparablecontinuousscores. AP below means sklearnaverage_precision_score, nottrapezoidalPRintegration. EachCI bootstraps500patientclusters.

|Finding|Positive/Negative|CLEAR AUROC (95%CI)|CLEAR AP (95%CI)|CLEAR fixed0.5 F1|CXR parsed F1|CXR coverage|
|---|---|---|---|---|---|---|
'''
for l in ['Atelectasis','Cardiomegaly','Consolidation','Edema','Pleural Effusion']:
 c=next(x for x in rows if x['finding']==l and x['policy']=='CLEAR_fixed0.5_exploratory');b=next(x for x in rows if x['finding']==l and x['policy']=='CXR_joint_definite_parsed_only');text+=f"|{l}|{c['gt_positive_total']}/{c['gt_negative_total']}|{float(c['auroc']):.6f} ({float(c['auroc_ci_low']):.6f}–{float(c['auroc_ci_high']):.6f})|{float(c['auprc']):.6f} ({float(c['auprc_ci_low']):.6f}–{float(c['auprc_ci_high']):.6f})|{float(c['f1']):.6f}|{float(b['f1']):.6f}|{b['covered']}/518|\n"
text+=f'''
MacroAUROC **{s['macro_AUROC']:.6f}**; macroAP **{s['macro_AUPRC']:.6f}**. CLEARfull518 macroF1 **{s['macro_F1']['CLEAR_fixed0.5_exploratory']:.6f}**; CLEARonCXRparsedmask **{s['macro_F1']['CLEAR_on_CXR_parsed_subset']:.6f}** vsCXRparsed **{s['macro_F1']['CXR_joint_definite_parsed_only']:.6f}**. CXRmissingnegative sensitivity **{s['macro_F1']['CXR_sensitivity_missing_negative']:.6f}**. CLEARcoverage518/518perclass.

|Resources|Measured|
|---|---|
|Warmedforwardmean / p95|{s['mean_forward_sec']:.6f}s / {s['p95_forward_sec']:.6f}s|
|Processwall|{s['wall_sec']:.3f}s, includinghashvalidation/model/evaluation;offlinefigurefinalization separate|
|Modelload|{s['load_sec']:.3f}s|
|PeakCPU RSS|{s['peak_RSS_bytes']}B ({s['peak_RSS_bytes']/1024**3:.6f}GiB)|
|GPUhours|0|
|Globalcache|1,624,984B includingIDs|

Meanforward excludespreprocess/cacheIO/warmup;only468newimageforwards aretimed,50reusedvectors have no fabricatedlatency.

## 12. Qualitative Results

Tenuniqueimagecases,twoperfinding, selecteddeterministicallyfromdisagreements (illustrative,notrepresentative). `{r}/e2_disagreements_contact_sheet.png` and image_evidence.csv linkGT,bothoutputs,sourcehash/localoriginalpath.

**Visualizationonly:**10%margin crop preservesinteriorintensities andaspectratio, withsourcecropbox recorded. Thecropping is not modelpreprocessing and is not a privacyguarantee; noexternalupload. Visibleposture/deviceannotations do not constituteclinicalverification. Noexpertreview of correct/missing/hallucinatedpathology performed. ROC/PRfigure includesperclassclusterCIs.

## 13. Comparison with Baseline

This compares pipeline outputs: CXR-LLaVAautoregressiveYes/No vs CLEARreleasedimage/textcosinepair. It does notisolatevisualencoder or recreatefullCLEARconceptsemantics. ClassificationCI onmatchedCXRparsedmask:

|Finding|CLEARminusCXR F1 samevalidmask|95%pairedclusterCI|Covered|
|---|---|---|---|
'''
for x in pair:text+=f"|{x['finding']}|{float(x['clear_minus_cxr_F1_same_parsed_mask']):.6f}|{float(x['ci95_low']):.6f}–{float(x['ci95_high']):.6f}|{x['covered']}|\n"
text+='''
Fixed0.5recall reaches1.0 forConsolidation/PleuralEffusion withprecision0.1098/0.3376; classificationtradeoff is substantial. ConsolidationpairedCI spanszero andAPCI is wide (29positives). Do notoptimizeonthislockedtest. NoCPU/GPUlatencyranking. Exactencoderpretrainingimageexposure remainsuncertain.

## 14. Problems Encountered

Initialpre-scoreharness KeyErrorforabsentparsedlabel correctedtoestablishedmissingpolicy innewscript; no testscoresproduced byfailedrun. Fullconceptbank/SFRembeddingassetsabsent: unresolvedasset/resourceblocker, not a justificationtorelabeldirectcosineasfullCLEAR. CPUonly andaccountquotaunverified remainexplicitlimitations. xFormerswarnings have nofailedforward.

## 15. Conclusion

E0Bofficialtextgate1→10 and onefixedE2directcontrastive518imagepass completed. CLEARranking macroAUROC0.892814,macroAP0.620352; sameparsedmask macroF1 0.521036 vsCXR0.248720 underfixedexploratory0.5. This is pipelineevidence, notclinicalvalidation orisolatedencoderH1proof. FullconceptCLEARnotrunbecauseassetsabsent;H2remainsuntested. Preservefrozenresultsandreviewmethodologybeforeanydownstreammodelselection.

## 16. Next Experiment

ReviewE1independently underNIHoverlaplimitations; reviewE2fixedprotocol andsourceevidence. Optionalfullconceptwork requiresverifiedassets,rowalignment,semantictextencoder,andexplicitmemory/timeestimate first. Useindependentdev forfuturethresholds/prompts, neverCheXpert518. Noautomaticadditionaltestinferencelaunched.

## 17. Reproducibility Checklist

Repository/branch/projectandupstreamcommits recorded; sourceandcheckpointSHAs recorded; exactenvironment/pipfreeze recorded; cohortandall518source/casehashes recorded; seed/prompt/threshold/nativepreprocess/clusterCI recorded; rawexecutionlogs,continuousscores,matchedmetricpolicies,figuresandsourceevidence saved. SuccessfulrunartifactSHAs verified. Postflight preservationcheck rehashesall518originalimagesandbaselinecases; noenvironment/code/checkpointmutations. Localartifactpaths containrestrictedsourcekeys; publicrelease wouldrequireseparatereview.

## 18. Short Lab Update

E0BtextAPI1→10passedwithzeroendpointdifference. E2directcontrastivecompleted518/518usingfrozenofficialprompts,CPUonlyand50reusedglobals; macroAUROC0.892814,AP0.620352,same-maskF10.521036vsCXR0.248720. Wall549.458s,peakRSS1.395744GiB. TenuniqueillustrativelabeldisagreementsandclusterCIslinkedlocally; noprompt/thresholdtuning, clinicalrevieworfullconceptreproduction. Fullconceptassetsremainblocked;nextreviewE1/E2limitationsbeforefurtherexperiments.
'''
report.write_text(text);m['outputs'].append(str(report));m['analysis_report_sha256']=sha(report);m['output_sha256']={str(f.relative_to(ROOT)):sha(f) for f in r.rglob('*') if f.is_file() and f.name!='run_manifest.json'};write_json(r/'run_manifest.json',m);print(report)
