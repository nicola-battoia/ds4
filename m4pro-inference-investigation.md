# M4 Pro 24 GB inference investigation — 2026-09-05

## Objective and safeguards

Improve actual DeepSeek V4 Flash generation throughput on this machine while
preserving the target model's outputs. Use one large model process at a time,
the existing benchmark/test tools, and serial matched runs. Record hypotheses
before changing implementation and record failed experiments as well as wins.

## Starting point and upstream

- Branch: `codex/m4pro-24gb-deepseek-flash`.
- Existing five-file changes were reviewed, committed as `dc1f31c`, and pushed
  to `origin/codex/m4pro-24gb-deepseek-flash` before new implementation work.
- Fetched Antirez upstream on 2026-09-05: `9ab7053`, 142 commits not yet merged.
- Existing profile: 32K context, 1024 prefill chunk, 896 cached experts, 18
  parallel tensor reads, IQ2XXS/Q2K Flash 0731 GGUF (86,720,111,488 bytes).
- Historical numbers in `deepseek-flash-m4-pro-24gb.md` are prior-session
  evidence, not today's baseline. DSpark with SSD streaming was rejected.

## Iteration 0 — integrate and measure

Plan before edits:

1. Merge upstream, resolving conflicts to retain the local M4 profile and
   upstream architecture changes. Build and run appropriate regression tests.
2. Read the SSD cache, I/O queue, Metal scheduling, and benchmark code vertically.
   Review primary upstream and Apple sources for relevant optimizations.
3. Establish serial greedy CLI and repository benchmark baselines with complete
   output, decode timing, cache/I/O diagnostics, and memory observations.
4. Choose the largest measured bottleneck. Implement a small exact-output
   optimization, then alternate control/candidate runs with identical prompts,
   lengths, cache capacities and model bytes. Keep a rollback for diagnosis
   where useful; do not retain unproven semantic variants.
5. Validate any promising result on useful prompts and filled context; record
   limits and commands so the result is reproducible.

## Evidence

Raw local logs will live in `misc/m4pro-investigation-20260905/` (ignored).
Durable findings, reproduction commands, and compact result tables belong here.

### Integration and baseline setup

- Upstream merge completed as `766f853`; `make -j8` passed without warnings.
  README follows upstream's new concise layout and links to the local profile.
- Hardware confirmed: 24 GiB, 10 performance + 4 efficiency CPU cores; no
  competing inference process. This is a live desktop, so compare alternating
  runs and report process memory rather than assuming an idle machine.
- First instrumented CLI workload: Python longest contiguous increasing run,
  greedy, 192 output tokens, existing wrapper defaults. Record selected routes
  and streaming timing. Route tracing flushes frequently and is excluded from
  final headline throughput.
- The resident same-engine decode A/B harness shares its global expert cache
  between sessions. It cannot fairly compare streaming throughput. Plan:
  extend `ds4-bench` with an optional full decode-logit capture, then use fresh
  serial processes in ABBA order for independent cache state and exact output
  comparison. Capture overhead must remain outside measured decode time.

## Candidate experiments, pending baseline evidence

1. Disable redundant immediate `F_RDADVISE` hints with the existing diagnostic
   switch while retaining parallel `pread` and compare measured setup/wait time.
2. Reduce exhaustive cache-victim scans without changing hotness/LRU ordering.
   Existing scans visit 30,720 potential entries for an 896-entry cache.
3. Simulate cache policy on actual selected routes before choosing changes.
   Current policy counts selections and halves frequency every 16 tokens.

Telemetry caveat: asynchronous `pread_ms` includes overlapped computation;
it is not pure SSD service time and must not be added to GPU execution time.

### First diagnostic finding and next experiment

The first live run is substantially slower than the historical 3–4 t/s.
A two-second stack sample during generation found most main-thread samples
waiting for Metal completion/router readiness, not blocked in expert pread.
The system compressed other processes and began swapping during startup.
Hypothesis: cached copies of streamed expert pages compete with repeatedly
used mmap-backed dense weights, causing GPU page faults. This is provisional,
not a measured optimization result.

Plan: first test the existing no-readahead switch. Then add a diagnostic-only
`DS4_METAL_STREAMING_EXPERT_NOCACHE=1` that sets Apple's `F_NOCACHE` on the
read-only model descriptor for Metal SSD streaming. Pair it with no-readahead
to test explicit reads without populating the filesystem cache; default
behavior stays identical. This changes transfer/cache behavior, not weights
or computation. Keep only if full-logit and real throughput tests justify it.

Upstream documentation correction: `docs/SPECULATIVE_DECODING.md` claimed
streaming support, but `git show upstream/main:ds4.c` still unconditionally
rejects `ssd_streaming && mtp_path` at lines 63325–63331. Corrected that guide
to match the implementation; the M4 wrapper's rejection remains appropriate.

## Iteration 1 — transfer policy screen (completed 2026-09-06)

The initial 192-token CLI baseline completed at 0.26 generation t/s, with
6.85 GB peak footprint and zero process swaps. The 256-token/24-decode screen
then produced the following results (32K allocated, 896 slots, 1K chunk):

| Order | Transfer policy | Prefill t/s | Generation t/s |
| ---: | --- | ---: | ---: |
| 1 | Default | 3.30 | 0.28 |
| 2 | No readahead | 3.56 | 0.28 |
| 3 | No readahead + F_NOCACHE | 3.49 | 0.27 |
| 4 | No readahead + F_NOCACHE | 8.99 | 4.78 |
| 5 | No readahead | 14.43 | 5.51 |
| 6 | Default | 11.37 | 5.52 |

All six complete decode-logit captures are byte-identical (25 rows, 129,280
logits/row; SHA-256 `d066dd2b916dd6eae7c4981795cdb6c28b20f490641842ae3bbbde469dccc816`).
The large speed transition also occurred for the unchanged default: it is a
system/cache-state effect, **not evidence that F_NOCACHE improves speed**.
The warm default and no-readahead decode rates are indistinguishable. Keep
F_NOCACHE diagnostic only while investigating; do not select it by default.

The 192-token profile spent 1.04 seconds in victim scans versus 92.26 seconds
waiting for resident GPU work in split layers. Removing scans cannot explain
the initial slowdown. The warm screen reduced that GPU wait to 22.7 ms.

## Iteration 2 — stable decode baseline and scheduling

Plan before edits: use 96-token serial runs once the machine is in the faster
state; screen the minimum miss count for splitting resident/missing experts
(current 3; candidates 1, 2, and disabled) with a diagnostic override. The
final accumulation order stays unchanged. Check full logits, then repeat any
candidate with captures disabled on both prose and useful code prompts.
Run a small byte-verified Metal I/O microbenchmark separately to decide whether
a larger transfer implementation is worth integrating. No concurrent models.

The next morning's first control again runs slowly. Deferred the remaining
split screen after this control to prioritize a more direct hypothesis:
Metal explicitly skips residency sets for *all* SSD-streaming model views,
including the bounded static non-routed map. Plan a diagnostic that requests
residency for the active mapped spans only when mapped bytes + live graph +
expert cache fit within 95% of the device working-set recommendation. It does
not copy weights or request residency for the entire GGUF. Compare default
and diagnostic logits/timing, then test with and without the queue residency
binding to separate persistent residency from submission-time accounting.

Apple documents that residency sets control when and how long allocations
remain GPU-accessible, while competing apps can delay residency requests:
<https://developer.apple.com/documentation/metal/mtlresidencyset>.

The standalone Metal I/O screen read 972 MiB and verified all bytes and slab
guards. For 1/2/6 expert batches, pread medians were 0.832/1.425/3.928 ms;
Metal I/O was 0.973/1.544/3.860 ms. Those cached-read timings do not demonstrate
a useful transfer win, so do not integrate a new transport at this point.

Complete route simulation also rejected simple decay tuning: current policy
predicts 106.37 misses/token; half-life 8 predicts 107.31, LRU 116.23, and
half-life 32 predicts 107.77. Per-layer LFU predicts 102.66 but adds policy
complexity for a small potential gain. The offline oracle is a theoretical
bound (about 62), not a realizable measured inference speed.

### Residency diagnostic: promising startup, not yet an isolated gain

The next 96-token control sustained 0.25 t/s and 951,983 process page faults.
Three 24-token residency probes achieved 5.63, 5.45 and 5.58 t/s with identical
logits. The first probe recorded 11,120 page faults and 35.8 ms of split GPU
wait versus 54.82 seconds in the longer control. Request-only residency was
as fast as adding the set to the queue.

However, the rollback immediately afterward also achieved 5.69 t/s. Thus the
experiment has not separated active-span residency from warmed system state.
Do not claim a 22x implementation speedup. Keep it diagnostic pending a
controlled long-context/cold-start comparison. Resume the split screen using
the same residency setting for all variants to isolate scheduling effects.

### Split-threshold screen

With the same residency diagnostic for every run, a 256-token prefix and 96
greedy decode steps produced 5.80 t/s (threshold 3), 6.03 (2), 6.19 (1), and
5.68 (splitting disabled). All 97-row captures share SHA-256
`bf56ec6641fc16123f78126f9ea0a2fa90e68f93aab587272f7027344ae55064`.
The final threshold-3 control is pending. Candidate threshold 1 is promising:
it overlaps even one SSD miss with the resident experts, preserving the
single final down/sum reduction and every output bit in this test.

Next plan: remove residency from the comparison, use a 2K filled prefix and
64 generated tokens, and compare threshold 3, threshold 1, and threshold 1
without immediate readahead in forward/reverse order. The extra read-policy
case tests interaction with the earlier start of resident work. After that,
test a complete generated Python function and conversational use, run focused
Metal regressions, and retain only measured improvements.

The final threshold-3 control returned 5.88 t/s with the same complete logit
capture. Threshold 1 is about 6% faster than the mean of the two controls,
so proceed to the longer-context comparison.

## Vertical implementation findings

- The wrapper selects the local IQ2XXS gate/up + Q2_K down GGUF, Metal graph
  execution, SSD streaming, a 32K allocated context, a 1K prefill chunk and
  896 expert slots. The full 80.76-GiB model remains mmap-backed; routed
  cache misses are explicitly read into allocated Metal buffers.
- `ds4_ssd.c` handles budget/locked-memory utilities. The active cache, slab
  allocator and persistent pread pool are in `ds4_metal.m`; model-span
  selection and per-layer graph scheduling are in `ds4.c`.
- `metal_graph_encode_decode_layer_phase()` schedules attention, routing and
  shared-expert work. Its async selected-load service waits for the GPU's
  selected-ID event and starts exact demand reads while shared work executes.
- `ds4_gpu_stream_expert_cache_begin_selected_load()` reserves protected
  reusable slots, issues read hints and dispatches up to 18 tensor reads.
  A cache miss loads all three slices of an expert, not a different expert or
  an approximation. Pending command sequence numbers protect in-flight slots.
- `ds4_gpu_routed_moe_one_tensor()` classifies selected slots as resident or
  missing. The split path submits resident gate/up computation, waits for
  missing bytes, then computes the remaining gate/up slots. Both paths issue
  the final down/sum once across all six slots, preserving accumulation order.
  The threshold experiment changes only when that established path is used.
- The cache globally retains frequently selected experts with a 16-token
  half-life and LRU tie breaking. The local hybrid prefill's 18-token tail
  resets frequency history and prepares decode locality after layer-major
  prefill. Merely increasing the expert cache can crowd out dense weights.
- The selected routed footprint is 1.70 GiB per token before cache hits.
  On the recorded code trace, roughly 0.70 GiB/token still needs demand loads.
  SSD transfer and its overlap with GPU execution therefore dominate the
  opportunity; the approximately 5 ms/token victim scan is a secondary cost.
- `ds4-bench` uses a fixed greedy, non-EOS probe; the CLI uses normal stopping
  behavior. Report both. Diagnostic logit-copy time is excluded from benchmark
  timing, but final throughput acceptance must also run without captures.
- DSpark would require another 5.6 GiB support GGUF, and the actual upstream
  engine rejects it with SSD streaming. The new upstream M5 TensorOps, TP,
  CUDA and GLM improvements do not remove this machine's streaming constraint.

## Iteration 3 — interactive display state (critical confound)

`pmset -g log` shows the display turned on at 11:55:04 and off at 12:06:35
on September 6. The first residency probe started at 12:06:37. That precise
coincidence invalidates attributing the large 0.25-to-5.63 t/s jump to the
residency change. Earlier "warm" numbers must also be treated as potentially
display-state dependent. The smaller scheduling A/B comparisons all ran in
the same display-off state and still show a reproducible improvement:

| 2K prefix, 64 decode tokens | First run | Reverse-order run |
| --- | ---: | ---: |
| Existing split threshold 3 | 4.97 t/s | 4.88 t/s |
| Split threshold 1 | 5.22 t/s | 5.19 t/s |
| Threshold 1, no readahead | 5.09 t/s | 4.64 t/s |

All six 65-row full-logit captures are identical (SHA-256
`1ed061450904d11bde0f0a37c932a3e58512db3b05e018b82510374b2dcf2dbd`).
Retain normal readahead. Threshold 1 improves this screen by about 5.7%.

Plan before further changes: hold the display awake temporarily with
`caffeinate -diu` for serial interactive-profile tests. Compare the existing
896-slot profile, the residency diagnostic at 896, then smaller 768/640/512
caches at the same graph size. This tests whether foreground display resources
push the existing profile into a paging regime. Do not change system power
settings or close the user's applications. Then validate the best practical
profile at filled context with useful CLI output and exact logits.

### Awake-display results and next acceptance plan

The awake-display controls returned 5.56 and 5.48 t/s at 256 input / 16
generated tokens. The residency probe returned 5.60; 768, 640 and 512 slots
returned 5.88, 5.73 and 5.72 respectively. All 17-row logit captures match
SHA-256 `5e842e98971d3814989531cda32898e5d1e7d7020bc0e84b418e1e3a393e6521`.
Display state alone therefore does not explain the original slow regime.
The small cache comparison is too short to select a new default.

Plan before editing: remove the unproven residency and F_NOCACHE diagnostics.
Keep only the split-threshold diagnostic for an awake-display, 4K-input /
128-output comparison, in forward/reverse order: 896 slots with threshold 3,
896 with threshold 1, and 768 with threshold 1. Capture exact logits and
process memory; then perform capture-free acceptance at a filled 16K context
and on complete CLI coding output. Any final scheduling default will be
limited to the tested small-memory M4 Pro profile. Other Metal hardware keeps
its existing scheduling policy.

## Iteration 4 — read-pool completion barrier

The first longer 4K split-threshold candidate is slower (4.49 versus 5.21
t/s), while two 768-slot candidates return 5.23 and 5.26. The reverse 896
candidate and control are still running. Do not select threshold 1 solely
from the earlier short-context wins.

Source inspection found a separate avoidable dependency: the pread pool wakes
all 18 threads for every batch, and completion waits for every designated
worker to acknowledge the batch, even when another worker has already drained
all its tasks. With three reads, the useful I/O can finish before a scheduled
worker ever runs. The consumer needs completed reads, not acknowledgements
from workers that performed no read.

Plan before editing: add a diagnostic alternative that counts unfinished
read tasks and releases the consumer as soon as the last read completes.
Keep the mutex publication boundary and add a generation check before taking
another task, so late workers cannot mix batches. The existing worker barrier
remains the control during the experiment. First stress the actual pool with
small byte-checked reads, changing task counts and immediately reusing task
arrays across thousands of batches, including shutdown/reinitialization.
Then compare full-model logits and serial throughput. Compile and run these
tests only after the ongoing 4K screen finishes to avoid CPU contention.

The completed awake 4K / 128-token screen returned 5.21 and 4.94 t/s for
896/threshold-3, 4.49 and 5.17 for 896/threshold-1, and 5.23 and 5.26 for
768/threshold-1. Every 129-row capture matches SHA-256
`f8d784fef819601e0e90cd3a9f477820149a68cbcdfd9332c121915bf739198c`.
The 896-slot split change is not a reliable improvement across workloads:
reject it as a new default. Keep threshold 3 for the pool experiment. The
768-slot result remains a possible profile tuning, requiring a separate
matched comparison because both cache and split policy changed.

The actual-pool stress test passed 5,000 worker-barrier and 5,000 task-barrier
batches, with changing task/worker counts, byte comparisons, guard checks,
immediate array reuse and ten shutdown/reinitialization cycles. Its small-read
timings were 511.6 and 498.4 ms respectively; these are not inference timings.
The full-model task-completion probes returned 4.83 and 4.62 t/s versus a
4.90 first control; every 129-row logit capture matches
`86ac192d57db32a172a1dcf5bbfb47d4a692734a30be3376896f7906cac211e2`.
The final control is pending. The new barrier is not a promising performance
change; discard it if the last control confirms this result.

## Iteration 5 — revalidate the capacity profile

Plan before further edits: remove the rejected split and read-pool diagnostics
and restore the merged production runtime. Compare 896, 1024, 960 and 768
expert slots in forward/reverse order, using the same awake-display 2K prefix,
128 decode steps, 32K allocation and 1K prefill chunk. Larger cache capacity
could reduce the unavoidable SSD bytes per token; smaller capacity could help
retain dense pages. Use the same threshold 3 throughout to isolate capacity.

The header-only planner confirms that 1024 slots use a 6.750-GiB expert cache
and 16.414-GiB conservative total, below the 16.872-GiB guard. Do not test
larger values outside that guard. Previous 1024-slot conclusions used only
32 generated tokens and did not provide a matched current-branch comparison.
Choose a new wrapper default only after filled-context and complete CLI
acceptance show a gain. Also preserve an isolated copy of pushed commit
`dc1f31c` for a direct pre-/post-upstream comparison, without moving the branch.

The final pool control returned 4.85 t/s. The candidate mean (4.725) was
below the control mean (4.875), so both runtime experiments were removed.
Their patch and stress-test artifacts remain under the ignored investigation
directory. The working runtime is again identical to merged commit 520c4a6.
Both the restored build and isolated pre-upstream build passed without
warnings. macOS reported no recorded thermal or performance warnings; that
does not prove constant GPU clocks or eliminate desktop noise.

### Upstream regression-harness issue discovered during validation

The complete `--metal-kernels`, MXFP4 Metal, GPU session-state and GLM KDA
tests pass. The new `test_metal_moe_prefill` fails on its first M4 IQ2/Q2 case
because it checks NaN-poisoned gate/up scratch that the fused grouped SwiGLU
path deliberately never writes. Source confirmation: the fused encoder is
passed `midbuf` and weights, without gate/up destinations.

Plan before editing the test: compare the consumed mid, expert output (where
materialized) and final output, retaining NaN poisoning and exact comparisons.
Use the unfused grouped calculation for the reference run by disabling its
pair SwiGLU fusion, so the M4 test actually compares separate and fused math.
Keep the existing tiny-kernel tolerance and unowned TP-slot handling. This
changes test assumptions, not the inference kernels or output requirements.

The corrected MoE test passes all 31 M4 cases, including grouped IQ2/Q2,
MXFP4 and synthetic local ownership cases. Consumed grouped intermediates and
outputs match the unfused reference exactly; tiny cases keep their existing
tolerance. The M5-only static batch section is not exercised on this M4.

### Additional upstream research

[Issue #810](https://github.com/antirez/ds4/issues/810) reports the same GGUF
running faster with a smaller wired expert cache on a 64-GB M5 Pro. It supports
testing both directions, but its absolute rates and cache sizes do not transfer
to this 24-GB M4. [Issue #636](https://github.com/antirez/ds4/issues/636)
suggests cross-layer router lookahead and segmented LRU based on a different
model/runtime; these are hypotheses, not demonstrated DS4 improvements. Its
static-hotness concern also predates the dynamic decaying frequencies in this
branch. [The Anemll sidecar fork](https://github.com/Anemll/ds4-ssd/blob/main-alpha/docs/STREAMING_KNOBS.md)
documents another cache/file-cache interaction. Its expert-major storage would
require an additional large artifact here; the direct I/O screen has not yet
justified that migration.

The completed 2K / 128-token capacity screen found no reliable throughput
gain: 896 slots returned 5.09/5.21 t/s; 1024 returned 5.11/5.05; 960 returned
5.26/4.98; 768 returned 5.25/5.12. All eight 129-row captures match
`86ac192d57db32a172a1dcf5bbfb47d4a692734a30be3376896f7906cac211e2`.
The higher memory use is not justified. Keep 896 as the control.

## Iteration 6 — complete user-facing code output and upstream comparison

Plan: generate a complete Python function through the normal CLI, greedy with
a 384-token ceiling, without route tracing, internal timing summaries or logit
copies. Compare current/896, current/512, current/384 and pushed pre-upstream
`dc1f31c`/896 in forward/reverse order, with the same actual prompt, model,
32K allocation and 1K prefill chunk. Smaller caches test whether reclaimable
file cache can help more than extra wired copies. Capture CLI throughput,
complete stdout, wall time and process memory, then inspect and execute the
generated function against adversarial cases. This also separates gains from
the upstream merge from new local tuning.

## Iteration 7 — choose overlap from actual I/O readiness

Plan before editing: revisit the split policy using information the runtime
already has. A fixed missing-expert threshold cannot distinguish outstanding
I/O from an early load whose bytes are already ready. Add a diagnostic that
splits mixed resident/missing work only while an existing early read batch is
still pending; use the established threshold when no early batch exists.
When reads have finished, install the bytes and use one routed pass. This
avoids extra command stages when there is no I/O left to overlap, while
allowing useful overlap for one or two genuinely outstanding misses.

Read the pool state under its existing mutex; do not change the reader pool,
cache policy, selected experts or final reduction order. Limit the diagnostic
to DeepSeek Metal streaming. The CLI comparison continues using the unchanged
compiled runtime. Build this probe after that comparison, then require exact
full logits and matched throughput at both 2K and 4K before selecting it.

The complete CLI comparison returned 4.54/4.27 t/s for current/896,
4.46/4.28 for current/512, 4.41/4.20 for current/384, and 4.43/4.14 for
pre-upstream/896. All eight stdout files match SHA-256
`3e8dec161a6ae4466de424faacecd34e1c44aba5cd0ce76087a65f91d1b1c2c5`.
The inspected generated function passes seven adversarial list-input cases,
returns a new list and preserves its input. There were zero process swaps.
Peak footprints were approximately 6.89, 4.17 and 3.26 GB for 896/512/384
slots respectively. The smaller caches save memory but do not improve code
generation throughput. The roughly 3% mean upstream difference is too small
to attribute confidently in this desktop run; do not headline it as a win.

For a stricter upstream comparison, apply only the already validated optional
decode-capture harness to the isolated pre-upstream source. Its inference
implementation remains the pushed `dc1f31c` version.

The I/O-readiness comparison completed with byte-identical full logits:

| Filled context / 128 generated | Control runs | Readiness-based runs | Mean gain |
| --- | --- | --- | ---: |
| 2K | 4.82, 4.79 t/s | 5.02, 4.83 t/s | 2.5% |
| 4K | 4.83, 4.50 t/s | 4.83, 4.68 t/s | 1.9% |

Captures match the earlier 2K and 4K SHA-256 values. The small positive
difference needs capture-free validation; both groups also show time-dependent
desktop/storage variation. Plan: run normal CLI code in ABBA order, then a
filled 16K benchmark in ABBA order with internal timing and dumps disabled.
Keep 896 slots, 32K allocation and 1K chunks. Require the same completed
outputs and record memory alongside speed before choosing a release default.

The disk has approximately 90 GiB available. A full additional 72.6-GiB expert
pack is inappropriate here. A separate, bounded storage-layout microbenchmark
may use the existing 486-MiB sample to test three tensor reads versus one
contiguous expert read. It must run after inference measurements, verify every
byte and delete its sample file afterward. No full model conversion is planned.

The capture-free CLI ABBA returned 4.50/4.53 t/s for controls and 4.60/4.31
for the readiness candidate. That does not establish a coding-throughput gain.
Every output still matches the validated function. The first 16K pair returns
4.98 versus 5.16 t/s; reverse-order measurements remain pending.

## Iteration 8 — protect the reused dense working set

Source inspection confirms that only allocated expert slabs are mlocked; the
bounded mmap views used repeatedly for dense work remain pageable. Earlier
Metal residency requests were soft GPU hints and did not establish physical
RAM residency. This is a separate hypothesis for the observed slow regime
with many GPU waits and page faults.

Plan before editing: add a strict diagnostic that mlocks only the currently
bounded model views in 64-MiB chunks, with complete rollback on any failure
and munlock on view replacement/cleanup. Require an explicit expert budget
of at most 512 slots. Reserve the entire requested expert cache plus at least
2 GiB for runtime tensors before admitting views under the existing 95% Metal
working-set guard. Never lock the full GGUF. Skip non-streaming and GLM modes.
Do not alter inference arithmetic, cache replacement, global VM settings or
other applications. First test 512 slots with and without locking, with the
readiness experiment disabled, exact logits and measured process/system memory.
Build and run only after the ongoing 16K acceptance completes.
