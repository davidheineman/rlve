# RLVE: running experiments and accumulated findings

This is the single maintained experiment guide for `rlve/`. Update it when a
result changes the recipe or interpretation; keep per-run details in manifests,
JSON/CSV results, and logs rather than adding another specialized README.
Findings below summarize local records through November 5, 2026. A reported
queue state or partial result is historical: inspect the experiment's current
manifest and completion markers before acting.

## Working conventions

- Run substantial new code, or a minimal representative version, and debug failures.
- Prefer PyTorch, explicit parameters, dataclasses and OmegaConf. Avoid `*args`,
  `**kwargs`, and new argparse interfaces when practical; existing CLIs still use argparse.
- Use `uv` where possible, simple `pyproject.toml` environments, `rich` terminal
  output and `tqdm` for long operations. Avoid large or file-level docstrings.
- Keep the root README limited to setup and running instructions.
- Do not add a new Markdown report for each change. Put durable findings here
  and use README updates sparingly.
- Use Matplotlib styling, compact subplots, minimal axis labels and K-formatted
  ticks for thousands. Omit unnecessary subtitles and bottom text.
- Save figures as PNG at **300 dpi**; produce PDF/SVG only when requested.
  Put new plotting scripts in `.plotting/` and fetched/derived data in `.cache/`
  under the relevant figure directory. Older scripts may need their output
  settings adjusted to follow these conventions.

## Start a new experiment

1. Specify the exact initializer repository, revision and ancestry: raw Base,
   released reasoning model, SFT model, or trained RL model. Pin HF/native
   identities and tokenizer assets. `Qwen3-0.6B` and `Qwen3-0.6B-Base` are
   different starting policies; the same distinction applies at 1.7B.
2. Choose the closest existing launcher below. Read its effective merged
   YAML/CLI, not only its default config. Use a fresh experiment directory;
   preserve source snapshots and hashes for each submitted run.
3. Freeze train, monitoring, stopping and final evaluation inputs separately.
   Verify prompt identity, leakage, reference answers, serialization, reward
   contracts, sampled-token limits, and actual teacher routing before scaling.
4. Declare prompts/update, samples/prompt, generated responses, objective passes,
   actual Adam updates and evaluation cost. Equal steps or response counts do
   not imply equal GPU/token costs. Count teacher work in MOPD comparisons.
5. Run applicable CPU checks, then a small real GPU integration check: sample,
   score, backward, update, save/reload and export. Check actual nonzero updates
   and post-update probability changes. CPU parsing alone does not establish GPU readiness.
6. Use `--check-only`, `--prepare-only` or `--dry-run` where supported before
   submission. Some preparation controllers also submit jobs; inspect their CLI.
   Launch within the user's requested scope, and preserve unrelated running work.

## Environments and paths

Run commands from `rlve/`, with `PYTHONPATH=.:tools` where tools import sibling
packages. This checkout uses Slime/Megatron/SGLang for dense training and a
separate Miles/Tinker gateway for shared-base LoRA; their runtime images differ.
The root `pyproject.toml` is minimal and is not a complete GPU environment.

| Resource | Location / convention |
|---|---|
| Shared launcher settings | `scripts/slurm/common.sh` |
| Dense image | `$RLVE_SCRATCH/images/slime-v0.5.0rc0-cu126.sif` |
| LoRA image | Miles image pinned by `tools/submit_data_scaling_lora.py`; original tag `dev-202610231907` |
| Scratch | `$RLVE_SCRATCH`, default resolved `/nlp/scr/$USER`; old records also use `/juice5b/scr5b/dhei` |
| HF / native models | `$RLVE_SCRATCH/models/NAME`, `NAME_torch_dist` |
| RLVE outputs | `$RLVE_RUNS_DIR`, default `$RLVE_SCRATCH/runs` |
| Modular data/runtime | `/nlp/scr/$USER/modular-mopd-rl/{data,v2}` |
| Existing host RL Python | `/nlp/scr/$USER/tinker-mopd/venv-rl/bin/python` |
| Existing plotting Python | Sibling `nano-omni/.venv-midtraining/bin/python` |

Use the pinned training container for GPU work. `source scripts/slurm/common.sh`
provides `rlve_apptainer` and CPU-only `rlve_apptainer_cpu`. Keep model/cache
paths on bound writable scratch. Install additional host analysis dependencies
with `uv`; do not replace the container's Torch/CUDA stack with a blanket sync.
Use `WANDB_API_KEY` from the environment, not Slime's legacy `--wandb-key` path.

Dense setup and a single-GPU smoke run:

```bash
bash scripts/setup/prepare.sh
RLVE_ENVIRONMENTS=Multiplication \
  RLVE_EXTRA_ARGS="--num-rollout 4 --eval-interval 2" \
  sbatch --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=00:40:00 \
  scripts/slurm/8way-50step.sbatch
```

Setup can download a large image and allocate a conversion GPU. Its default
model is **Qwen2.5-1.5B-Instruct**. Change `RLVE_MODEL_NAME`, `RLVE_MODEL_REPO`
and `RLVE_MODEL_ARGS_SCRIPT` together for another model. Other common overrides
are `RLVE_CONFIG`, `RLVE_EXTRA_ARGS` (appended last), `RLVE_WANDB_TEAM`,
`RLVE_WANDB_PROJECT`, `RLVE_WANDB_GROUP` and `RLVE_RUN_TIMEOUT`.

HF-to-native conversion must produce `latest_checkpointed_iteration.txt`;
failed conversion can otherwise allow random initialization. For new initializer
pipelines also verify tensor values, shapes, finite values and tokenizer identity.

## Choose the experiment family

These are entrypoints, not instructions to relaunch old experiments. Inspect
`--help` and use a fresh output path; some tools depend on existing frozen
datasets, models or earlier experiment roots.

| Experiment | Preparation / launcher |
|---|---|
| Short single-environment RLVE | `scripts/setup/prepare.sh`; `scripts/slurm/8way-50step.sbatch` |
| Qwen 32×32 transfer | `scripts/setup/prepare_pairwise32_eval.sh`; `scripts/slurm/1way-300step-pairwise32.sbatch` |
| R1 transfer / fixed-rollout-policy control | `scripts/slurm/submit-r1-pairwise-ood.sh`; `submit-r1-pairwise-ood-fixed-policy.sh` in the same directory; both support `--check-only` |
| Frozen-pool mixtures | `tools/prepare_rlve_mixing.py`; `scripts/slurm/1way-300step-mixing.sbatch`; R1 revision: `scripts/slurm/submit-r1-mixing-v3.sh --check-only` |
| Original R1 teacher/MOPD/RL sweeps | `scripts/slurm/submit-mopd-v3-sweeps.sh --check-only` |
| Fresh environment scaling N16/N32/N64 | `tools/prepare_data_scaling.py`; `scripts/slurm/submit-data-scaling-v1.sh --check-only` |
| N128 scaling extension | `tools/prepare_data_scaling_n128.py`; `tools/submit_data_scaling_n128.py --check-only` with its parent experiment |
| Synchronous round-two scaling | `tools/submit_data_scaling_multiround_resilient.py --check-only` with parent and generation length |
| Dense asynchronous branching | `tools/prepare_async_branching.py --check-only`; `tools/submit_async_branching.py --size 16 --check-only` |
| Shared-base LoRA teachers / async LoRA | `tools/submit_data_scaling_lora.py --check-only`; `tools/submit_async_lora.py --check-only` |
| Specialist transfer discovery | `tools/submit_mopd_clustering.py --size 64 --check-only` |
| Exact composed-adapter search | `tools/submit_composed_clustering.py --help` |
| Clustered asynchronous MOPD | `tools/submit_clustering_mopd.py --prepare-only`; then `--check-only --experiment PATH` |
| Four-task synchronous MOPD versus JointRL | `tools/prepare_transfer_regimes.py --prepare`; then `--check --sweep PATH` |
| Modular task data / Base RL sweeps | `python -m modular_rl.prepare`; `python -m modular_rl.v2.prepare`; `python -m modular_rl.v2.base_sweep --dry-run --experiment FRESH_PATH` |
| Independent Base code study | `python -m modular_rl.v2.code_study --base V2_ROOT --review FRESH_REVIEW --include-prefix`; then `base_sweep` using its cells and selection |
| Midtraining initializer RL / MOPD suite | `python -m modular_rl.suite prepare`; `python -m modular_rl.mopd_suite prepare`; inspect required roots with `--help` |
| Dolci SFT and follow-ups | `modular_rl/dolci/`: `prepare.py`, `watch.py`, `fresh_experiment.py`, `math_large_experiment.py`, `mopd_sftinit.py` |
| Later 16K-response modular RL | `python -m modular_rl.dolci.rl16k prepare --root FRESH_PATH`; separate `preflight`, `submit`, `watch` actions |
| String program trees | `python -m program_trees.experiment`; `python -m program_trees.verify`; `scripts/slurm/submit-program-trees.sh --check-only` |
| PCFG v3 / further tree tasks | `python -m program_trees.pcfg_v3 verify`; `scripts/slurm/submit-program-trees-pcfg-v3.sh --check-only`; `program_trees.next_experiments` |
| SPADE games | `tools/prepare_spade_games.py`, `tools/probe_spade_signal.py`; `scripts/slurm/spade-single-200step.sbatch` |

## Protocols that must remain distinct

### RLVE scaling and branching

- Original data scaling uses fresh R1-Distill-Qwen-1.5B training on nested
  16/32/64-task panels, later extended to 128. Selection seed is `20261023`;
  manifests live under `data/DATA-SCALING-V1/` and the N128 extension.
  Train on balanced difficulty 0/1 pairs, without adaptive curriculum.
- Dense teachers: 1 prompt × 8 responses, at most 4K responses; MOPD students:
  8 prompts × 1 response, at most 16K; joint RL: 8 prompts × 16 responses,
  at most 64K. Teachers use one GPU; original heroes use four.
  Evaluate initially/every 50 updates/at cap; 98% ID pass@1 controls stopping.
- The original scaling ID panel has **32 total prompts**, so at N64/N128 it
  does not cover every training task. The shared unseen panel has 32 tasks,
  disjoint from training. Use eight evaluation samples/prompt. Later clustered
  N64 experiments add all-task ID coverage; do not silently compare the panels.
- Original RLVE/tree answer forcing allows **1,024 initial + 128 extra sampled
  tokens**. The 2K round-two repeat changes training to 2,048 + 128 while
  preserving the 1,024-token initial evaluation limit.
- Synchronous scaling rounds use 4K student responses each. Round-two teachers
  below 98% retrain from the short round-one student's final weights; already
  successful teachers are retained. This recipe starts fresh optimizers.
- Async branches take causally available student parents bounded by both
  logical event time and the teacher lane's cumulative rollout count. Teachers
  get fresh Adam; the student preserves Adam/scheduler/RNG across merges.
  A branch stops at 4K responses or 98%; student consumption stops at 1K per
  teacher version or 90% of that teacher's terminal accuracy, in fair 400-response
  quanta. A zero-accuracy teacher has an exact zero recovery target.
- Dense elastic async defaults to a **16-physical-GPU cap including the two-GPU
  student**. Simulated teacher devices and physical allocation are separate.
  Shared-base async LoRA normally uses two trainers/two samplers, rank 32,
  alpha 64, attention/MLP targets, LR 2e-5; it is a distinct method from dense LR 2e-6.
- Replay replaces 10/30/50% of eight-trajectory student batches; logical budget
  includes replay. Keep fresh/replayed counts separately. FIFO buffers retain
  400 fresh teacher-scored student traces/environment, exclude the current task,
  and never reinsert replay. Cached masks/behavior/teacher scores persist with
  committed checkpoints; signed clipped off-policy replay is not fresh OPD or SFT.
- Single-question LoRA gates measure fixed-question learning; randomized ID/unseen
  metrics stay separate. Some finite-support tasks have documented teacher-gate
  overlap exceptions. Do not describe these runs as random-question training.

### Modular math, code, IF and chat

- Standard corrected RL: one prompt × eight responses, 8K total responses,
  LR 2e-6, one objective pass, fixed-token GRPO, and exact uniform-reward
  update skipping. Later suites use 64 held-out prompts/task × eight samples,
  every 50 prompt iterations. Earlier v2 uses 32 prompts; original smoke uses 16.
- The modular **2,048 sampled-token cap includes answer forcing**. Inserted
  prefixes are context with zero loss/importance mask, not sampled tokens.
  Typical reserves are math 256, code 512, chat 128; IF has no forcing.
  Keep each run's actual reserve, thinking prefix, stop IDs and decoding pinned.
- MATH uses a stratified custom split from the official training data:
  6,746 train / 750 holdout after reference exclusions. The official 5K test
  split is separate. Numeric GSM-Symbolic smoke results are a different task.
- The independent MBPP Base code study uses 681 train / 32 monitor / 256
  reserved problems, split by question groups. Later 64-question development
  panels reuse some earlier study questions; they are not new confirmation data.
  New `mbpp-contracts-20261101` prompts repair public API specifications;
  do not relabel historical scores as if they used those contracts.
- Code evaluates strict all-tests success even when training on partial credit.
  Never expose solution bodies or test assertions in policy prompts. Audit
  return types, constants, ordering and helper classes as well as reference execution.
- IF trains on fraction of constraints passed and evaluates strict all-constraints
  success. The inherited language checker can accept detection exceptions;
  retain an explicit audit when interpreting those cases.
- Chat trains on raw Skywork 1B scores using separate reward GPU(s). Local
  modular pass@k means beating the frozen **rejected** Tulu answer; it is a
  reward-model win rate. The sibling Tinker comparison uses a **chosen** answer
  and is not directly interchangeable. Preference answers never enter policy inputs.
- MOPD routes each student-generated response to its task's frozen teacher;
  sampled reverse-KL advantages use teacher/student token scores. Environment
  rewards are diagnostics, not its training advantages. Balanced batches contain
  two prompts/task × one response. Teacher-bank names do not specify the student initializer.
- The older modular MOPD comparison used 4K student responses. Later runs use
  8K and 32,768, with different initializers and EOS handling. The baseline-SFT
  4,096-update study with eval every 32 updates generates **264,192 evaluation
  responses**; evaluation can dominate cost. `mopd_sftinit.py` itself defaults
  to interval 256, so inspect the prepared plan for the eval32 variant.
- Later RL16K runs use one prompt with 32 samples for math or 16 for other tasks,
  16K training responses, fixed 2K loss denominator and evaluation every 512
  responses. These are deliberate group-size/budget changes from the 8K controls.

## Correctness and optimization lessons

- **Check answer framing first.** RLVE training expects `\boxed{}` via
  `ChatTemplate_BoxedAnswer`; some Gym APIs default to `<answer>`. An unframed
  instruct answer can give every response the same failure reward and no GRPO signal.
  Tree tasks require their output-string framing instead of generic math framing.
- **Round-trip frozen problems through JSON and re-score references.** Integer
  dictionary keys can become strings (e.g. BucketSorting), breaking evaluation
  while live-object training succeeds. Seed separation alone also does not ensure
  text separation in tiny difficulty-zero question spaces.
- **Uniform groups must have exact zero advantages.** Identical float32 fractional
  rewards can produce false residual advantages after normalization. Check
  original global rewards before distributed partitioning and collectively
  apply/skip updates. Report actual Adam clocks separately from collected groups.
- **A scalar policy loss near zero can have nonzero gradients.** Verify changes
  after the update. Conversely Adam momentum can move a policy on a zero-gradient
  step; skipping Adam and its scheduler is necessary when exact no-update is intended.
- **One on-policy objective pass starts with PPO ratios of one.** Zero clip fraction
  is expected and does not bound cumulative policy drift. Tightening PPO epsilon
  alone will not anchor these updates to the initial SFT policy.
- **Loss normalization is an experimental choice.** Per-response means overweight
  short responses at the token level; fixed-cap normalization removes that length
  factor but alone did not stabilize long runs. Keep `calculate_per_token_loss`
  disabled when the custom reducer already supplies normalization. Normalize
  reduced TIS/OIS diagnostics by `train/normalization_mass` before interpreting ratios.
- **For GRPO reference anchoring use explicit K3 loss:** `--use-kl-loss
  --kl-loss-type k3 --kl-loss-coef VALUE`, with reward-side `kl_coef` zero.
  The GRPO return branch ignores reward-side `kl_coef`; direct linear sampled
  `kl` loss has a reference-independent gradient. Treat the coefficient as a trial parameter.
- **MaxRL changes scaling, not the all-fail problem.** The centered estimator is
  `(r - mean(r)) / (mean(r) + 1e-6)`, without extra whitening. With one prompt
  per update it rescales the whole gradient; clipping/Adam can attenuate the
  intended hard-prompt weighting. Compare matched multi-prompt batches and
  objective passes before attributing changes to exploration.
- **Keep completion quality beside answer accuracy.** Track correct-and-finished,
  truncation, forcing, post-answer tails and repetition. A correct box can be
  followed by incoherent text until the cap; more tokens can prolong a loop.
- **Audit code transport and extraction.** Strip only terminal Qwen markers from
  bare code as well as fenced code. Preserve marker strings inside Python literals.
  The optional syntax-only forced-code fallback selects a complete earlier
  implementation only if the forced continuation is malformed; never select by test success.
- **Align teacher/student EOS labels in MOPD.** Raw Base commonly ends with
  151643 (`endoftext`), Dolci SFT with 151645 (`im_end`). Accept both as stop IDs
  and set the appropriate per-bank label transport when conventions differ.
  Transport is a vocabulary permutation, not duplicated probability mass.
  A baseline-SFT student reverses which bank needs alignment. Old replay scores
  cannot be reused under a changed EOS mapping.
- **Match SFT and RL prefixes.** Short Dolci worked solutions use an empty
  thinking scaffold; `MODULAR_RL_ENABLE_THINKING=0` prefills it. Open thinking,
  temperature changes and stop-token changes can alter initial accuracy without
  any learning. Compare against an initial policy with identical decoding/hardware.

## What the experiments have shown

All percentages below come from saved local reports, usually one training seed.
Monitoring panels used to tune recipes are development data. Prompt-bootstrap
intervals condition on the trained policy and sampled responses; they do not
measure training-seed variation.

### Transfer and mixture selection

- Short fixed-difficulty versus curriculum sweeps showed similar mean ID gains
  (~15 percentage points) and approximately zero unseen-task gain. They used
  different environment cohorts, so this is descriptive evidence that disabling
  curriculum did not remove the transfer gap.
- Learned mixture surfaces have weak predictive validation. The 73-mixture R1
  study had leave-one-mixture-out aggregate R² **−0.034**. Best observed mix 21
  scored 35.9% versus uniform 29.3%, but that is an in-sample selection.
  Earlier matched-window proposed/natural hero scores were Qwen 18.5/17.7%
  and R1 24.8/26.6%; the fitted proposal did not reliably win across models.
- Dynamic reward filtering can hang forever when no mixed group is available
  and changes accepted mixture proportions. Bound retries, report fallback and
  all generated work, and inspect completed training-step progress.
- Teacher reuse can reduce sampled rollouts while adding optimizer passes.
  Small-group fast-teacher trials motivated p1r8, but p1r2 often had no signal.
  Compare fixed-threshold accuracy against allocated GPU time on matched hardware.

### SFT, code and math

- The corrected code comparison demonstrates a **sample-efficiency** benefit:

  | Initializer | Initial pass@1 | Responses to first measured 35% | Final pass@1 at 8K |
  |---|---:|---:|---:|
  | Raw Base | 2.15% | 4.8K | 40.23% |
  | Baseline Dolci SFT | 16.80% | 3.6K | 41.60% |
  | Code-heavy SFT | 24.61% | 1.2K | 40.23% |

  Only 351–369 of 1,000 prompt groups produced Adam updates. The endpoints
  do not establish a 40% architectural ceiling. Sparse tests, unclear MBPP
  contracts, a small training pool and one seed limit generalization claims.
- An earlier code grader penalized correct bare code ending in a Qwen transport
  marker. Offline rescoring changed code-SFT initial accuracy from 12.30% to
  24.22%. Fresh corrected-reward runs are the meaningful learning comparison;
  rescoring old answers cannot reconstruct a corrected training trajectory.
- Chat-heavy SFT was initially treated as an IF specialist, but contained only
  0.95% Precise-IF input tokens versus baseline 4.01%. Later IF20/IF50 are
  actual IF treatments. The natural complement won the original C4 choice
  (1.016925 versus 1.037514 BPB), not every downstream task.
- The original 1B code-heavy SFT repeated its coding pool ~5.4 times; a 5B
  repeat would be ~27 passes. The original math-heavy mix repeated its pool
  ~3.1 times and was dominated by Persona MATH. More tokens are not more unique data.
- Fresh code preparation uses correctness-filtered Python traces, deduplicated
  by normalized question across sources, and 8K contexts. Filter long complete
  conversations rather than cutting reasoning to fit; audit source capacity.
- Short-math refresh produced only **7.76M unique input tokens**, not a 1B
  corpus. Repeating it ~129 times to fill 1B was rejected. Its `final/` corpus
  supersedes the original pilot candidate. A four-step warmup failed the
  integration gate; full-epoch checks did yield real RL updates.
- Refreshed large-math SFT uses complete short worked solutions, 80/20 math/general
  input tokens and at most two exposures per retained block/question. Exclude
  custom MATH holdouts and official test questions explicitly. Synthetic answer
  agreement and a source's "decontam" name are insufficient quality/leakage checks.
  The launched corpus used available committed reviews, not a completed full GPU screen.
- The refreshed 1B math SFT started at **50.39%**, reached **54.69%** after
  8K RL responses, and retained 73.44% pass@8. Its paired pass@1 gain interval
  was approximately [−0.39,+8.79] points. Truncation was negligible; the small
  gain did not establish broader hard-problem coverage.
- Four-pass single-prompt MaxRL/MAI follow-ups degraded badly. Single-pass
  eight-prompt GRPO showed a modest gain; MaxRL did not show a net pass@1 gain
  in that snapshot. Batch size, reuse and loss reduction changed together, so
  no isolated cause was established. Larger unused MATH panels and repeat seeds
  are more informative than selecting another peak on 64 development questions.
- Historical released-reasoning-model diagnostics did confirm short-run math
  gains on reserved 256-question panels: 0.6B 52.34→59.08%, 1.7B
  70.75→75.88%. Longer replications regressed. These are **not Base-model results**.
  Separate raw-Base math controls showed promising monitoring gains
  (0.6B 18.36→44.53%, 1.7B 27.73→59.77%); reserved confirmation was pending
  in those reports. Base code prefixing improved early parseability and mixed
  reward density, but did not by itself establish superior final accuracy.

### Modular MOPD

- Original raw-Base students trained against SFT-bank teachers nearly stopped
  terminating: ~511/512 code/IF responses hit the cap. GPU probes showed SFT
  teachers strongly penalizing the Base EOS label. EOS alignment substantially
  repaired termination; syntax-only code fallback also exposed code lost by
  forcing/extraction. Counterfactual rescoring is not a repaired training result.
- Correct routing and exact prompt-token audits passed. Under the newer common
  baseline-SFT student setup, the task math teacher retained **55.9%** accuracy
  in an independent matching-protocol export check, while the student was
  around **34%** at ~768 updates. Math gradients improved the sampled objective;
  most of the gap was reliability on questions both policies could sometimes solve.
  Sparse teacher-only coverage was an additional gap, not the whole explanation.
- IF initialization helped aggregate transfer, but repetitive tails and specific
  constraint failures remained. On some repetitive prefixes the teacher also
  assigned almost unit probability to continuing the loop. Matching sampled
  teacher scores therefore need not teach a terminating solution.
- Those ~768–832-update observations preceded the 4,096-update endpoint. They
  do not establish the final outcome or prove cross-task interference. Useful
  tests are matched task-only versus balanced MOPD, matched task exposure, and
  verifier-correct teacher demonstrations. Teacher-generated traces require an
  explicit demonstration objective, not relabeling them as student on-policy samples.

### LoRA clustering and compute comparisons

- LoRA LR 2e-6 underlearned the matched dense teacher bank: final mean 39.28%
  versus dense 57.01%. At 350 matched updates, LR 2e-5 reached 57.25%, versus
  dense 54.30% and LoRA-v1 35.42%. This supports the rate trial, not optimality.
  That v2 attempt timed out; its canceled recovery must not be treated as completed.
- N64 balanced pair discovery selected k=32. Disjoint confirmation gave base
  17.43%, specialists 47.20%, selected leave-own-adapter-out 28.32%, random LOO
  21.94%, outside-matched 20.95%, and selected-full 41.76%. Selected LOO minus
  random was **+6.38 points**, 95% environment-bootstrap [+2.98,+9.97]. This
  supports cross-task transfer in those controls, not every individual group.
- Pairwise transfer averages are not composed-adapter performance. Older merges
  used rank-32 compression; the later composition search uses exact concatenated
  factors with equal mean-delta scaling. Own-task inclusion changes the question.
  At k=8, composition confirmation gains were high/random/low
  +15.31/+10.35/+8.45 points, including each target's own adapter.
- These discovery rankings did **not** guarantee MOPD rankings. Completed N64
  k=8 student ID scores were high/random/low **45.70/47.07/48.24%** under
  the same eight simulated devices and 16K clock deadline. N32 optimized k=8
  beat random (55.47 versus 45.70%), while optimized k=2 lost (39.06 versus
  45.31%). These are exploratory single-seed method comparisons.
- The synchronous four-task transfer-regime comparison also did not show a
  universal MOPD advantage: e.g. worst-1 final ID was JointRL 72.01% versus
  MOPD 55.08%, best-1 97.27 versus 93.75%. Other groups were closer; task
  difficulty and opposing directional effects matter. Use the independent final
  panel and full teacher-plus-student cost, not discovery labels alone.
- Rollout speedup is not GPU-hour speedup. In the completed high-group horizon
  comparison at 90% validation, joint RL cost 6.12 GPU-hours versus teacher-inclusive
  MOPD 35.92. The archived low-group result was incomplete; that snapshot cannot
  establish a relationship between teacher slope and compute gain.
- Program-tree high-depth/breadth runs often had all-zero groups and nearly
  universal truncation. Executable labels were checked; serial-versus-parallel
  composition mistakes and sparse exact-string reward were real issues.
- SPADE partial terminal reward can rise without full wins. One game yielded
  ~0.72 reward for timeout behavior and had a seed-invariant winning script.
  Inspect reward code, full solve rate, within-seed variance and game errors.
  A one-seed/four-play screen cannot establish learnability or game quality.

## Cluster runtime and storage lessons

- Default to portable code across A100, H100, H200, L40S, 3090, A5000, A6000,
  RTX 6000 Ada and Titan hardware. Provide kernel/precision/memory fallbacks
  where practical and validate the allocated device; document any measured
  limitation of an existing pinned runtime instead of assuming it applies to
  all hardware. A launcher preflight should produce a few real training-step
  logs so startup failures are visible.
- Default submissions use `jag-standard,miso,sphinx,miso-lo,sphinx-lo,jag-lo,sc-loprio`.
  The account usually has access to about 24–32 low-priority GPUs: expose enough
  independent useful work to use that capacity, avoiding unnecessary dependencies,
  hardware restrictions or low concurrency caps. Actual allocations depend on Slurm.
  Choose one-, four- or eight-GPU allocations according to expected queue delay
  plus runtime, preserving the logical training batch and declared experiment budget.
  A user-requested narrower queue selection takes precedence. Reference hardware:

  | Partitions | GPU types |
  |---|---|
  | `sc-loprio` | Mixed, including A100, H100, H200, L40S |
  | `sc-freegpu` | 1070, P100, V100, Titan |
  | `jag-urgent,jag-important,jag-hi,jag-standard,jag-lo` | 3090, A5000, A6000, RTX 6000 Ada, Titan |
  | `miso,miso-lo,miso-interactive` | H200 |
  | `sphinx,sphinx-lo` | A100, H100, H200 |

- SC uses Apptainer rather than Pyxis. Inspect the current Slurm resources and
  constraints with `sbatch --test-only`, `sinfo` and `scontrol`; do not copy old
  queue forecasts or node exclusions as permanent cluster facts. Existing jobs
  commonly use account `miso` and combinations of `jag-standard,miso,sphinx,
  miso-lo,sphinx-lo,jag-lo,sc-loprio`. Many newer runs restrict low-priority queues.
- Pending jobs are not allocated GPUs. Request only real input dependencies;
  a direct RL control does not depend on MOPD teachers. Preserve the declared
  concurrency/resource policy of each experiment rather than imposing one global policy.
- Dense 1.5B colocation on 48GB cards needed optimizer CPU offload, its precision-aware
  options, ~0.15 SGLang static memory fraction and a 2,048-token packing budget.
  Longer concurrent generations sometimes require more KV capacity, so this is
  not a universal memory profile. Validate long prompts and both model wake-ups.
- Host RAM peaks during checkpoint serialization can exceed steady-state usage:
  1.7B modular policy runs needed 128GiB, dedicated-reward chat 160GiB; eight
  colocated offloaded 1.5B runs needed 400–450GiB total in earlier sweeps.
- Static single-response microbatches plus periodic synchronized CUDA-cache
  release fixed some accumulation-memory failures while preserving logical batches.
  Allocator settings and packing are runtime-specific; retain their provenance.
- Shared-base Miles multi-LoRA requires separate training/sampling GPUs and no
  optimizer CPU offload. The pinned older gateway needed FlashAttention 2 on
  Ampere/Ada; FA4 backward failed there. Check numerical backward on actual hardware.
- Slurm GPU indices can differ from container indices. Respect the allocated
  numeric mask for Miles and verify visible UUIDs; avoid host-index assumptions.
- Use `scripts/slurm/port_slot.sh` shared per-node locks. Slurm's private `/tmp`
  does not isolate network ports, so locks solely under `/tmp` can collide.
  Use short unique Ray paths and clean up only owned processes, not global `ray stop`.
- Never edit a running shell script/source snapshot in place. It caused EOF and
  stale-handle failures after successful training. Slurm also spools submitted
  batch files; editing their on-disk copies does not update pending jobs.
- Preserve full transactional model/Adam/scheduler/RNG/prompt-source state for
  exact continuation. Ignore uncommitted saves, archive uncommitted traces on
  rollback, retain stable W&B identity and deduplicate replayed metric indices.
  SGLang restarts do not promise bitwise GPU sampling RNG continuation.
- Final-weight-only and adapter-only exports support inference/fine-tuning,
  not exact optimizer continuation. Preemption may restart from the base with
  a new W&B attempt. Older scaling runs intentionally used final-only storage;
  do not inherit that limitation by accident in a new long experiment.
- Concurrent dense optimizer checkpoints exhausted scratch repeatedly (~21–24GiB
  each versus ~3.4GiB final 1.5B weights). Budget overlapping saves, teacher banks
  and concurrent experiments against combined capacity. Respect launchers' storage gates.
- Before deleting native checkpoints, validate terminal-update evidence and HF
  tensor/export receipts. A failed shell/export can coexist with a valid final
  trained checkpoint; recover that model rather than retraining it. Preserve
  active recovery states and model weights. Archive gzip evidence only with round-trip hashes.
- NFS memory-mapped sampler files can leave `.nfs*` placeholders. Retire obsolete
  versions after publishing replacements and retry collection; preserve live
  readers. Flex samplers must pass adapter/log-probability identity checks.
- LoRA speed-v2 (one trainer/three samplers) was slower in early measurements;
  v3 restores two trainers and allows extra generation jobs. More samplers are
  a hypothesis for speed, not a measured end-to-end guarantee.
- Monitor completed-step progress, not just HTTP health or `RUNNING`. Async
  watchdogs commonly allow 45 minutes startup / 30 minutes no progress, with
  bounded retries and cancellation respected. Stop the monitor before deliberate
  coordinator maintenance. Consult that controller's STOP/lease mechanism first.

## Metrics, evidence and plots

- Use `david/rollouts` for actual training-response counts. In older logs,
  baseline `david/step=0` precedes training; post-update `step=9` means ten
  completed updates. For fixed batches, post-update responses are
  `(step + 1) × batch_size`. W&B `_step` counts log events, not training updates.
- Standard accuracy keys are `david/eval/{val_envs,test_envs}/pass@1` and
  `pass@8`; also retain per-environment metrics, `train/*`, `rollout/*`, `perf/*`,
  generation/truncation diagnostics and actual optimizer counts. Treat shared-mode
  W&B history and local records as evidence; summaries may be stale while writers run.
- With n sampled answers and c successes, pass@k is
  `1 - C(n-c,k)/C(n,k)`. Pass@1 averages outcomes; pass@n is any-success per
  question. Do not infer pass@16 from eight samples or treat 64×8 as 512 independent questions.
- Bootstrap paired questions for within-task comparisons and paired environments
  for task-transfer summaries. Prompt folds, random-assignment ranges and
  environment bootstraps measure different uncertainty. Keep selection and final
  confirmation disjoint and account for multiple candidate comparisons.
- Pairwise matrices use evaluation targets as rows and training sources as
  columns. Exclude self-transfer when claiming cross-task effects. Row-normalized
  or smoothed deltas are relative display quantities, not raw directional gains.
- Fetch exact histories, preserve selected attempts/source hashes in `.cache/`,
  and deduplicate repeated checkpoints by the latest coherent committed attempt.
  Select attempts by completion/progress, not highest accuracy. Curves end at the
  last real evaluation. Carried-forward scores need distinct styling.
- Count all teacher and student training for standalone MOPD cost comparisons;
  shared teacher banks are counted once for total physical sweep cost. Record
  discarded attempts separately. Allocated GPU-hours include evaluation/export
  work in GPU jobs; timing estimates are not utilization or MFU measurements.
- The async simulated clock counts per-device rollout work/idle causality;
  it excludes evaluation, communication, sequence-length variation and discovery
  cost. Plot physical elapsed time separately and disclose queue/preemption treatment.
- New comparisons should use raw unsmoothed curves and declare initial weights,
  panel, decoding, response/teacher cost and run status in cached metadata.
  Keep these details out of the figure subtitle unless essential.

Offline regeneration from the workspace root (legacy renderers may use older
DPI/export settings; adjust them when producing a new requested figure):

```bash
nano-omni/.venv-midtraining/bin/python \
  rlve/plot/mopd-modular-v3/.plotting/make_plots.py
nano-omni/.venv-midtraining/bin/python \
  rlve/plot/mopd-clustering-v2/.plotting/make_plots.py
```

Both support `--refresh`; refreshing reads experiment records and may query W&B.
Keep output data/provenance even when retiring a narrative report.

## Verification and locating old evidence

Use the existing environment with the relevant dependencies and run only the
checks appropriate to your change. Examples from `rlve/`:

```bash
PYTHONPATH=.:tools uv run --no-project --python .venv/bin/python python -m unittest \
  slime.tests.test_checkpoint_state slime.tests.test_eval_metrics \
  slime.tests.test_async_branching slime.tests.test_transfer_regimes
PYTHONPATH=.:tools uv run --no-project \
  --python /nlp/scr/$USER/tinker-mopd/venv-rl/bin/python python -m pytest -q \
  modular_rl/v2/test_losses.py modular_rl/v2/test_code_reward.py \
  modular_rl/v2/test_snapshot.py
```

LoRA SDK/kernel tests need the pinned Miles image. Native FP32 and matched
packing/GPU BF16 gradient replays passed for the audited initial 0.6B batch;
unmatched HF BF16 references did not. Neither result certifies a complete
distributed training trajectory.

| Evidence family | Durable pointers |
|---|---|
| RLVE panels / run definitions | `data/` manifests, `scripts/slurm/sweeps/`; per-run `plan.json`, `jobs.tsv`, input hashes |
| Modular frozen data and original studies | `/nlp/scr/dhei/modular-mopd-rl/`; `v2/reviews/` |
| Original Dolci / corrected code and IF | `/nlp/scr/dhei/dolci-mix-20261029`, `dolci-followup-20261031`, `dolci-maxrl-fresh-20261101` |
| Refreshed math | `/nlp/scr/dhei/math-refresh-20261101/final`, `math-large-20261101` |
| Newer modular MOPD | `/nlp/scr/dhei/dolci-mopd-sftinit-4096-eval32-20261103`; local plot `.cache/` snapshots |
| LoRA discovery | `$RLVE_RUNS_DIR/mopd-clustering-n64-20261031-165345`, `composed-clustering-n64-20261102-150018` |
| Structured comparisons | `plot/mopd-clustering/{n64,mopd,transfer-regimes}/` results JSON/CSV; `plot/mopd-scaling/setting-*/` snapshots |
| Current W&B project families | `rl-data-scaling-*`, `rl-clustering-*`, `modular-mopd-rl-v1`, `modular-mopd-r1-v1`; use actual IDs in manifests/caches |

The 140 pre-consolidation Markdown files are preserved verbatim in
`.cache/documentation/2026-11-05/original-markdown.tar.gz`, with path/SHA-256
inventory in its sibling `manifest.json`. This is a local ignored backup;
retain/copy it if moving the workspace. To inspect one old report without
restoring the documentation sprawl:

```bash
tar -xOf .cache/documentation/2026-11-05/original-markdown.tar.gz \
  modular_rl/v2/QWEN06B_2K_DEEP_DIVE.md
```

Dataset cards, generated question examples, third-party documentation and license
provenance remain at their original paths; their specialized content is not run guidance.
