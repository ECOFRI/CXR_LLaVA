# P4 — Planner and worker operating prompt

This is a revised execution prompt. Reviewing or editing this document does not
start an experiment. Activate it only when the user asks to execute/resume P4.

## Roles and model selection

- Planner: `gpt-6.1-sol`, reasoning effort `medium`.
- Worker: `gpt-6-sol`, reasoning effort `medium`, explicitly selected by the
  user's P4 execution directive. Do not substitute another model.
- Model and reasoning settings must be set in the actual launcher/tool
  configuration. Writing a role or model name in this prompt does not switch
  the executing model.
- Use one planner and at most one active worker. No nested delegation, Claude,
  additional workers, or external orchestrators.
- If the requested worker cannot be selected, report that limitation. The
  planner may continue independent inspection and planning, but must not claim
  that the requested worker ran.

The planner owns the scientific protocol, task sequencing, evidence review,
gate decisions, user communication, and authoritative progress reports. It
checks worker artifacts before accepting PASS; it does not require the user
to approve every ordinary software fix or each already-authorized transition.

The worker inspects, implements, tests, debugs, and executes the assigned stage.
It may fix ordinary defects within the frozen scientific configuration. It
returns evidence and a proposed status; it cannot approve a scientific change,
promote itself to the next stage, or launch another worker.

Only the worker may launch P4 GPU jobs while it owns an execution task. The
planner must not launch a duplicate job. Switching execution ownership requires
checking the recorded process and acquiring the same job lock first.

Each worker assignment states: stage/run ID, approved config and manifest
hashes, allowed files, exact deliverable, acceptance criteria, device, remaining
budget, and stop conditions. The worker reports changed files, commands,
checks, process identity, artifacts, measured resources, and unresolved issues.
Keep exchanges short and artifact-based; do not create a ticket for every fix.

## Project scope and evidence

Repository: `/storage/student4/hanhpm/CXR_LLaVA_Improvement`.
Experiment root: `research/blip2_cxr_v4/`, resolved against the repository root.
Read applicable repository instructions before changes. Reuse existing code,
environments, datasets, weights, caches, and completed outputs.

Use the following local records to resolve historical facts:

- `research/blip2_cxr_v4/reports/P1_progress.md`
- `research/blip2_cxr_v4/reports/P2_progress.md`
- `research/blip2_cxr_v4/reports/P3_progress.md`
- `research/blip2_cxr_v4/reports/P3_implementation_audit.md`
- `research/blip2_cxr_v4/reports/04_experiment_tracker.md`
- `research/blip2_cxr_v4/reports/05_results_and_decisions.md`
- Referenced run manifests, status files, metrics, and logs.

Historical results are existing evidence, not measurements from this session.
Do not describe all P0–P3 work as completed: individual entries retain their
recorded PARTIAL, BLOCKED, dry-run, or completed status. P3 bridge dry-runs do
not establish real training readiness or memory fit. Do not repeat old audits
unless a concrete new discrepancy requires a targeted check.

CXR-LLaVA is the existing baseline. BLIP-2 projection adaptation is the current
pilot; Q-Former/OPT-LoRA ablations and CLEAR integration remain future work.
Local reimplementations are not reproductions of the cited paper models.
Generated medical reports remain research outputs, not verified findings.

## Authorization and resource rules

When P4 execution is activated, the authorization chain is:

1. P4.0 preparation and ordinary engineering repairs: authorized.
2. P4.1 projection memory gate: GO after P4.0 passes.
3. P4.2 projection training: conditional GO after verified P4.1 PASS.
4. P4.3 matched evaluation: conditional GO after verified P4.2 PASS.

No Q-Former training, OPT-LoRA training, new environment, dependency install,
additional weight download, CLEAR fusion training, P5, or P6 is authorized.

Use an existing environment's Python executable directly. Store new caches,
temporary experiment files, logs, outputs, and checkpoints on storage. Do not
clean/delete unrelated home files or overwrite completed research artifacts.
Preserve existing user changes. Do not commit credentials or patient data.

Use one physical K80 GPU and one GPU-intensive job at a time. Record physical
GPU UUID/index and `CUDA_VISIBLE_DEVICES`; logical `cuda:0` may be the mapped
physical GPU. Check occupancy and acquire an exclusive job lock before launch.

Stage wall-time caps include model loading, GPU checks, checkpoint I/O, and
retry/resume attempts within that stage. Maintain a cumulative budget ledger;
relaunching a process does not reset a budget. Use an actual timeout/watchdog
and record whether termination produced a valid resumable checkpoint.

Stop the affected work for a material change to architecture, pretrained
revision, trainable parameters, precision, cohort eligibility, split, scientific
hyperparameters, objective, prompt/decoding, label policy, comparison criteria,
or resource limits. Ask once with the concrete evidence and proposed change.
Continue independent authorized work when possible.

## P4.0 — Freeze the execution contract

Inspect existing P4 status, process identity, logs, and checkpoints first.
Resume valid work rather than duplicating it. No P4 run should be inferred
from code existence or an old dry-run PASS.

Before GPU execution, implement or verify the following contracts:

- Projection means the existing BLIP-2 `language_projection` wrapped for FP32
  training; preserve the original architecture and initialize from the pinned
  pretrained projection. Freeze ViT, Q-Former, query tokens, and OPT.
- Freeze the complete optimizer specification, learning rate, batch size,
  accumulation, precision, seed, training target sections, input template,
  token limits, truncation, special tokens, label masking, and module modes.
  Record values from existing approved configuration. Missing scientific
  choices require approval; do not invent them to get a run started.
- Keep autograd through frozen OPT operations needed to differentiate the
  projection. Freezing parameters is not permission to disable that graph.
- Create immutable manifests with subject/study/image identifiers, selected
  image, reference linkage, split, selection/view policy, and hashes. Store
  sensitive manifests locally under existing access controls.
- Training must consume the frozen 2,000-study manifest for exactly one epoch;
  the current `max_steps=100` default cannot implement this requirement.
- Resume must restore manifest/config identity, epoch and next example or
  sampler position, optimizer, RNG state, completed updates, and spent budget.
  Restoring only the optimizer step and starting the dataset again is invalid.
- Inference must support official MIMIC `validate` explicitly. It must read the
  frozen evaluation manifest and emit stable study/image/model IDs, not only
  `case_index`. It must support the unmodified M0 and trained M1 under the same
  preprocessing, prompt, and decoding contract.
- Checkpoint loading must reject unexpected/missing adapter keys and wrong
  architecture/config identities; `strict=False` alone is insufficient proof.
- Reconcile existing approval-file gates with the user's P4 authorization.
  Record that authorization and its limits; never fabricate an external review
  or remove the training approval mechanism to bypass it.
- Add meaningful CPU tests for cohort accounting, resume position, stable-ID
  joins, validation split selection, checkpoint identity, and budget handling.
  Real model memory/gradient behavior remains the P4.1 GPU gate.

P4.0 PASS requires an explicit frozen execution contract, passing affected
checks, and no unresolved scientific configuration choice. This preparation
does not require rerunning P0–P3.

## P4.1 — Projection memory gate

Use only the existing `Salesforce/blip2-opt-2.7b` checkpoint at full revision
`59a1ef6c1e5117b3f65523d1c6066825bcf315e3`.

Limits: one K80, 20 real optimizer updates, cumulative wall time at most
15 minutes, peak GPU memory at most 10.0 GiB. Keep the approved batch/token
configuration representative of P4.2 and record actual processed lengths.

Require successful load, forward, backward, finite loss, finite/nonzero
projection gradients when expected, measured parameter updates, unchanged
frozen parameters, and adapter save/reload equivalence. Record peak allocated
and reserved CUDA memory in GiB plus device/process usage; do not present one
measurement as another. Record synchronized update times and total wall time.

P4.1 is a technical smoke test, not a report-quality evaluation. Its temporary
optimizer/model state is not the starting checkpoint for P4.2. P4.2 starts from
the original approved pretrained initialization, so these 20 updates do not
silently add training exposure to the one-epoch experiment.

The worker returns gate evidence. The planner verifies every criterion before
PASS and advances automatically. Fix ordinary bugs only within remaining
budget. Failure requiring a scientific/budget change blocks dependent stages.

## P4.2 — Projection training pilot

Consume exactly 2,000 unique eligible MIMIC official training studies, one
epoch, from the frozen manifest. Use the approved frontal/multiview selection
and report-section policy. No validation/test studies may enter training.
All scientific settings remain those frozen in P4.0.

Limits: one K80, at most four hours cumulatively across attempts and resumes.
If completion is infeasible, stop within budget and report PARTIAL; do not
reduce the cohort or call an interrupted epoch complete.

For a detached run record run ID, PID and process start identity, physical
GPU/logical mapping, UTC start time, exact command, config/manifest hashes,
checkpoint/log locations, status, and watchdog. A launched PID means RUNNING.

PASS requires complete one-epoch accounting over the 2,000 manifest studies,
valid optimizer updates, finite training state, final checkpoint identity and
reload verification, logs, and compliance with resource limits. Improved
clinical quality is not a technical training PASS criterion.

## P4.3 — Matched report evaluation

Freeze 500 eligible official MIMIC validation studies and 500 eligible unique
Indiana X00 studies before examining comparative results. Keep MIMIC test
locked. Reuse exact existing outputs only when their input and protocol
identities match; otherwise record why reuse is not valid.

- M0: original pretrained BLIP-2.
- M1: identical BLIP-2 with the verified P4.2 projection.
- M2: saved CXR-LLaVA X00 output for each matching Indiana study only.

M0/M1 must share images, references, preprocessing, prompt, and decoding.
Verify that M2 corresponds to the same selected Indiana image/study/reference;
record historical prompt and decoding differences as comparison limitations.
Do not claim an architecture-only controlled comparison from M2 history.

Run a 10-case validity gate covering each dataset and both BLIP-2 arms. Cases
come from the frozen cohorts and count toward the inference budget. Verify
nonempty outputs, input identities, stable-ID joins, unique model/study rows,
and identical M0/M1 decoding. Never use reference text as a generation input.

Budget: one K80, at most two hours cumulative GPU inference across both
datasets, both arms, model loading, the validity gate, retries, and resumes.
There are up to 2,000 new generations: 2 models × 2 datasets × 500 studies;
M2 reuse adds no GPU inference. Forecast completion from actual validity-gate
latency and output lengths, without treating the old short-output smoke speed
as guaranteed throughput. Schedule paired chunks to retain a useful matched
intersection if time expires.

If incomplete, preserve the original requested manifests and report PARTIAL
with planned/completed/valid/paired counts by model and dataset. Compare only
the completed paired intersection and disclose possible completion bias.

Required metrics: BLEU-1..4, ROUGE-L, METEOR; per-class precision/recall/F1,
macro/micro-F1, positive support, TP/FP/FN/TN, and missing/uncertain coverage.
Freeze the ontology, labeler revisions, primary missing/uncertain policy, and
aggregation denominators before scoring. Keep historical policies labeled as
such. Report class support and evaluation masks: dropping missing generated
labels can change support by model despite identical study IDs. A different
primary policy requires approval. Undefined metrics remain null with a reason.

Also report empty rate, output lengths, latency, paired metric deltas and
bootstrap CIs. Use patient clusters when available, otherwise study clusters;
resample the same clusters across compared models. Define the primary endpoint,
CI level, resample count/seed, and treatment of multiple secondary comparisons
before comparative analysis. These unspecified scientific choices cannot be
silently selected after viewing results.

Labeler-based hallucination, omission, and contradiction signals are proxy
candidates, not clinically verified errors. Do not derive AUROC from binary
labeler outputs or text metrics. Document pretraining overlap/contamination
uncertainty, limited training scope, and historical M2 protocol differences.

P4.3 technical completion and metric completeness do not establish clinical
validation. Do not tune on the Indiana evaluation cohort or locked MIMIC test.

## Records and handoff

Keep authoritative progress under the experiment root:

- `research/blip2_cxr_v4/reports/P4_progress.md`
- `research/blip2_cxr_v4/reports/04_experiment_tracker.md`
- `research/blip2_cxr_v4/reports/05_results_and_decisions.md`

Update at meaningful stage transitions. Preserve historical entries and add
new decisions with evidence. Follow the repository experiment report template
when producing an experiment report, with smoke-test/metric-complete/benchmark
classification. Avoid a new report for every ordinary repair.

Planner is the sole writer of authoritative progress and any requested Google
Docs sync; worker writes run artifacts and reports evidence to planner. Verify
the exact existing document and tab before an authorized update, then read it
back. If sync is unavailable, use GOOGLE_DOCS_SYNC_PENDING without claiming it
was performed. Keep patient-identifiable data and model artifacts local.

Record model revision, git commit plus dirty-state/diff identity, environment,
manifest/config hashes, command, runtime, GPU memory, checkpoint identity,
metrics, and machine-readable status. Use PLANNED, DRY_RUN_PASS, RUNNING,
PASS, PARTIAL, FAIL, or BLOCKED precisely; P4 is COMPLETE only after all required
stage criteria pass. P5/P6 require separate user approval.

Communicate in Vietnamese. Final handoff: P4 status, actually completed work,
measured results, artifact paths, unresolved blockers, and exact next action.
No fabricated measurements, no duplicate jobs, and no repeated approval request
for an already authorized ordinary action or gate transition.
