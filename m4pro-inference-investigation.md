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

The completed 16K acceptance returned 4.98/4.92 t/s for controls and
5.16/5.05 for readiness-based splitting (mean +3.1%). All four decoded
continuations are identical. Together with the unchanged coding throughput,
this is a small, workload-dependent result, not yet a release default. Keep
the readiness diagnostic disabled for the dense-lock comparison.

The bounded expert-layout microbenchmark verified all bytes and buffer guards
with a 486-MiB immediately unlinked sample. Original three-slice versus packed
one-slice median read times were 0.970/1.386 ms (one expert), 1.411/2.428 ms
(two), and 3.370/6.713 ms (six). This does not justify a sidecar conversion;
fewer reads also reduce the available I/O parallelism. The sample is removed.

The first dense-lock run preserved all 129 full-logit rows but returned
4.99 versus 5.01 t/s, with prefill dropping from 50.65 to 30.60 t/s. The
diagnostic also locks changing layer views during prefill, creating avoidable
work before the final 8.20-GiB dense view. Its reverse runs remain pending;
do not retain this implementation without an actual decode benefit.

## Iteration 9 — scheduling urgency of inference I/O

Source inspection shows the SSD selected-load service and persistent pread
workers use the inherited/default pthread QoS. Upstream already assigns an
explicit QoS to its unrelated TP gate service. Apple's
[Apple silicon performance guide](https://developer.apple.com/documentation/apple-silicon/tuning-your-code-s-performance-for-apple-silicon/)
recommends classifying work so the scheduler can choose cores appropriately;
its [pthread QoS guide](https://developer.apple.com/library/archive/documentation/Performance/Conceptual/power_efficiency_guidelines_osx/PrioritizeWorkAtTheTaskLevel.html)
documents the supported per-thread API.

Plan before editing: after the dense comparison, remove rejected diagnostic
code and test explicit user-initiated QoS for only the DeepSeek SSD selected
load service and persistent read workers. Leave the main thread, other apps,
global scheduler settings, cache geometry and all math unchanged. Use a
diagnostic flag initially, exact logits in serial 2K ABBA, then normal CLI
and longer-context acceptance if it helps. This is a scheduling hypothesis,
not an assumption that the current workers run on efficiency cores.

The dense-lock ABBA finished at 5.01/4.72 t/s for controls and 4.99/4.98
for locked runs. All four captures match the 2K reference exactly, with zero
process swaps and no system-swap increase. The candidate's mean +2.5% is
driven by the slower last control; it adds roughly 27 seconds of prefill
locking work and does not establish a useful default. Remove both dense-lock
and readiness diagnostics from the runtime before testing QoS; their patch
is retained with the ignored experiment artifacts.

## Iteration 10 — start exact hash-routed reads before attention

The early hash-routed layers know their six expert identities directly from
the input token, but currently initiate their reads after encoding attention
and the router. The existing helper already reads that exact table on the CPU;
moving its first call earlier needs no prediction or new routing algorithm.

Plan before editing: add a diagnostic call to that helper at entry to the
full single-device Flash IQ2 SSD decode layer. Keep the existing later call:
the pending-load matcher deduplicates the same batch and the override remains
the same six IDs. This gives reads the CPU's attention-encoding interval as
additional lead time, without an extra GPU submission or any arithmetic
change. Exclude other phases, resident/quality/multi-device paths, and profiling.
Check full logits and fresh-process ABBA throughput after the QoS screen;
retain only if complete output tests confirm a useful gain.

The QoS screen completed at 5.17/5.14 t/s for controls and 5.13/4.92 for
explicit user-initiated workers. All four full-logit captures match exactly,
with zero process swaps. Reject the scheduling change (mean -2.5%) and
remove it before testing the hash-read timing change. The same existing
reader pool and its original QoS remain the control.

## Iteration 11 — skip empty cache layers during victim selection

The replacement loops scan all 120 x 256 possible cache entries, although
Flash has only 43 layers and the cache already maintains exact per-layer
entry counts. This permits a smaller optimization than introducing a new
replacement policy or data structure.

Plan before editing: in the single/batch reusable-buffer scans, skip layers
whose existing entry count is zero. Preserve traversal order, hotness and
age comparisons, protection, and in-flight retry behavior for every possible
victim. Initially put it behind one diagnostic flag; compare exact logits
and full-model throughput with hash timing disabled. Use existing timing
counters to confirm fewer scans. Also run the existing encoder timeline
diagnostic on the baseline to identify the remaining GPU and scheduling costs;
its extra encoder boundaries disqualify its timings as headline throughput.

The earlier-hash-read ABBA returned 4.89/5.18 t/s for controls and
5.09/5.02 for candidates: mean +0.4%, with exact full logits in every run.
The first pair's apparent +4% did not survive reverse-order testing. Reject
this change; the current compiled diagnostic remains disabled while collecting
the baseline timeline. Remove its source before the cache-scan experiment.

## Iteration 12 — overlap readahead inside the existing reader pool

The baseline encoder timeline preserved all 33 full-logit rows. Its 32 decode
steps took 7.223 seconds, with 2.254 seconds of measured GPU encoder execution
(31%). The separate streaming counters, covering the 18-token prefill tail
plus generation, recorded 3.500 seconds in serial readahead calls versus
0.132 seconds scanning cache entries. Mean load preparation was 1.890 ms;
the asynchronous pread interval averaged 2.010 ms. These intervals overlap,
so they must not be added or treated as pure SSD service time.

Plan before editing: retain the same advisory ranges, but test issuing each
range's `F_RDADVISE` inside the persistent worker immediately before its
`pread`. The selected-load service can then publish the batch without waiting
for every hint in sequence. Keep direct reads, byte counts, cache placement,
completion barriers and GPU order unchanged. The synchronous fallback uses
the same read helper. Protect optional hint timing counters because hints can
now run concurrently, and describe their sum as worker time rather than wall
time. A diagnostic flag initially restricts this change to DeepSeek SSD.
Compare against the default and the existing no-readahead option, with empty
layer skipping disabled, full logits and longer decode probes before acceptance.

The empty-layer scan ABBA returned 5.14/5.04 t/s for controls and 5.20/4.73
for candidates, with exact logits. Candidate prefill varied from 43.43 to
20.82 t/s as system swap grew and page faults increased; zero process swaps
did not capture that system pressure. This does not establish a generation
benefit. Remove the diagnostic and keep the original victim scan for the
readahead comparison, so only one timing policy changes.

The actual private reader pool passed 5,000 serial-hint, 5,000 worker-hint
and 5,000 no-hint batches using a 64-KiB fixture. All bytes, inactive slots
and guards were exact through changing task/worker counts, immediate task
reuse and pool restarts. Both hint modes recorded exactly 47,496 hints;
the no-hint mode recorded zero. Small cached reads were not faster with worker
hints (585 versus 522 ms), so only full-model results can justify the change.

## Iteration 13 — fuse streamed IQ2 activation in registers

The measured largest GPU kernel is the masked address-table IQ2 gate/up pair.
Its current implementation writes gate/up rows and immediately rereads them
to apply SwiGLU, serially on one SIMD lane. The resident ID-based kernel
already reduces the same dot products into registers and distributes the four
row activations across four lanes.

Plan before editing: extend the shared IQ2 pair helper with a compile-time
fused finish, and use it only from the masked address-table entry point. Keep
the same quantization loop, per-row SIMD reductions, gate/up diagnostic writes,
clamp and weight multiplication. The ordinary helper instantiation remains
unchanged. Prepare this in an ignored source copy and select it through the
repository's existing `DS4_METAL_MOE_SOURCE` override; no new runtime mode is
needed. Require the repository Metal kernel tests and full-model byte-exact
logits before longer alternating throughput and complete CLI acceptance.
Run only after the readahead series finishes.

The six-run hint comparison completed with byte-exact full logits in every
run. Generation was 5.15/5.36 t/s for the original serial hints, 4.79/4.66
for worker hints, and 4.67/4.47 without hints. Relative to the control mean,
worker hints lost 10.1% and no hints lost 13.0%. Keep the original hint order;
remove the worker diagnostic before the register-fusion experiment. The
large hint timing counter did not identify work that could simply be removed
or moved without changing effective SSD service.

The register candidate compiles and passes the generic Metal/MoE regression
suites, but the first full-model capture differs, including the prefill
frontier, and eventually chooses different tokens. Stop its throughput series
immediately; its 4.74 t/s is not an acceptable speed result. Next isolate the
masked address kernel with identical synthetic weights, inputs, masks and
poisoned outputs in one process, comparing gate, up and mid separately. This
will distinguish indexing/reduction mistakes from changed floating-point
contraction across the removed memory boundary. Preserve release arithmetic;
do not relax correctness tolerances to accept the candidate.

The isolated shader comparison localizes the drift to activation: gate/up
are bit-exact for every tested mask and shape, while mid differs by 1–3 ULP.
Those small differences amplify over the model. Full 4096-by-2048 six-expert
kernel time is also effectively neutral (210.5 versus 210.4 microseconds in
this cached synthetic fixture). Reject this register finish; no production
shader was changed. Keep the focused fixture for subsequent arithmetic tests.

## Iteration 14 — recheck reader concurrency under current upstream

The M4-specific 18-reader default predates the merge and the present, much
faster machine state. Read-ahead already queues all selected tensor ranges;
excess synchronous readers could add filesystem and scheduling contention.
Plan before changing any defaults: use the existing reader-count override
with 18, 9, 6, 6, 9, 18 workers in separate, serial 2K/128-generation runs.
Keep the original shader, cache, hint ordering and all other settings. Require
full-logit equality, then longer generation and CLI acceptance if fewer
readers show a repeatable benefit. Do not infer a new default from the old
3.61-to-4.04 t/s measurement alone.

## Iteration 15 — reduce IQ2 lookup overhead without changing activation

The focused fixture now exposes gate/up separately from the streamed activation.
Two bounded shader candidates can therefore be screened without rerunning a
large model for every compiler variation. Plan before editing: prepare source
copies that (A) apply the IQ2 sign to the exactly representable integer grid
value before multiplying the input, or (B) read the small grid/sign tables from
constant memory directly, avoiding per-threadgroup copies and their barrier.
Restrict each copy to the existing shared IQ2 pair helper; preserve its original
lane-zero writes and the original activation entry point. Compare every bit
of gate/up/mid, inactive masks and realistic shapes, with repeated alternating
GPU timings. Only a meaningful exact kernel improvement advances to full-model
ABBA and CLI tests. Do not run GPU fixtures alongside the reader series.

## Iteration 16 — wake only readers assigned to the batch

Code inspection found that every read batch broadcasts to all pool threads,
including idle readers outside the requested count. The earlier completion-
barrier experiment retained that broadcast and therefore did not test this
cost. Plan before editing: prepare an isolated runtime copy that signals only
n_workers sleepers and lets any unclaimed worker take one of n_workers leases
for the current generation. Retain dynamic task claiming and require all leased
workers to finish before publishing completion. The generation predicate must
also admit threads that were starting or returning to sleep when signalled,
so correctness cannot depend on retaining condition-variable signals.

Use the existing mutex, generation, active/remaining counts and shutdown
broadcast, with no additional pool or thread count. Stress varying batches,
reader limits, immediate descriptor/task reuse, failed reads and pool restarts
before any model run. Compare the compiled candidate against the unchanged
binary, with the original 18-reader cap and shader. Run after reader-count
screening; source copies remain ignored until end-to-end evidence justifies
applying a release change.

The reader-count screen completed with exact 2K full-logit captures throughout:
18 readers returned 4.52/4.46 t/s; nine returned 4.54/5.10; six returned
4.96/4.97. Six improves the paired control mean by 10.6%. This is the first
clear matching forward/reverse generation result in the current machine state.
No default is changed yet. Proceed with six/18/18/six at a filled 4K context
and 512 generated tokens, comparing the entire 513-row capture. Then use the
complete CLI code fixture without tracing/capture. If those confirm a benefit,
check a filled 16K context. Defer the prepared shader and pool-wakeup candidates
so these acceptance runs isolate the existing reader-count setting.

## Iteration 17 — keep bulk prefill concurrency while limiting decode reads

The six-reader screen reduced prefill from about 47.4 to 40.0 t/s, despite
its generation gain. That matters for long prompts. Plan before editing:
combine the prepared selective-wakeup pool with a six-reader cap only in the
single-token early selected-load entry point, on the 24-GiB M4 Pro. Keep the
18-reader pool capacity and the existing bulk-prefill reader policy. Preserve
the existing explicit reader-count override. GLM is outside this experiment.

The pool should retain one broadcast when every reader has a lease, and signal
only the required number otherwise. This avoids waking unused readers during
decode while preserving the bulk path. Prepare an isolated candidate binary;
require the pool stress fixture, exact full-model logits and serial comparisons
against both original 18-reader behavior and the fixed-six setting. Apply only
if CLI generation and filled-context runs confirm the gain without moving the
cost into prompt processing.

Pool review follows the [POSIX condition-variable contract](https://pubs.opengroup.org/onlinepubs/9699919799/functions/pthread_cond_broadcast.html):
a wakeup does not itself grant work. The mutex-protected generation and
unclaimed-worker predicate bounds admission even with spurious wakeups or
threads that had not begun waiting when signalled. Completion still waits for
all admitted workers to release their leases.

Fixed-six 4K/512 BAAB completed: six returned prefill/generation
44.30/4.66 and 53.86/4.51 t/s; 18 returned 49.48/4.25 and 45.17/4.57.
All four 513-row captures match SHA-256
`14ff1378ca19bcfa0b8f1cbb2049b31d356b9f4bad67beb0b1b60fec4c7cf84f`.
Generation mean improves by 4.0%, but the reverse pair loses 1.3% and prefill
varies substantially. The earlier 10.6% is therefore a short-screen result,
not a release claim. Test the phase-specific/selective-wakeup candidate next;
do not change the global M4 default from these results alone.

The selective-wakeup pool passed 30,000 changing batches at pool limits 6,
9 and 18, including byte/guard checks, 30 restarts and EOF recovery. Bounded
fixture times were 541/607/779 ms; these are correctness stress observations,
not model throughput. The combined candidate build is warning-free.

Both remaining IQ2 lookup candidates were bit-exact in the focused fixture
but slower at production shape. Integer-sign folding took 307 versus 207 us
for six experts; constant-table lookup took 210 versus 208 us, and 142 versus
130 us for a partial mask. Reject both. No production Metal shader has changed.

The combined decode-only/selective-wakeup 2K ABBA returned 5.20/5.21 t/s
for controls and 5.30/5.31 for candidates: +1.9%, with a 0.10 t/s gain in
each direction and exact full logits throughout. Prefill means were 48.84 and
51.48 t/s; whole-run means were 67.27 and 64.86 seconds. This is a modest
lead, not the fixed-six screen's 10.6% claim.

Next isolate whether the decode cap is needed: compile the selective-wakeup
pool alone, retaining the original reader limits. Compare original / wake-only /
decode-cap / decode-cap / wake-only / original through the complete CLI code
fixture, without logit capture or profiling. Require the same complete source
output and execution cases. Advance the simplest variant with a repeatable
end-to-end advantage to longer filled-context acceptance.

## Iteration 18 — screen scan-resistant cache admission offline

The earlier cache simulation assumed mandatory admission and LRU ties. A
one-off miss can therefore displace an established entry even if it is unlikely
to recur. Plan before runtime edits: first screen the small equivalent ranking
change of evicting the most recently inserted entry among equally cold experts,
while retaining ordinary LRU ties above a low frequency threshold. This can
make recently loaded one-off experts act as temporary staging slots without
allocating a second cache. Use the existing recorded real routes, empty and
hotlist seeds, multiple cold thresholds and the same 16-token decay.

This offline model omits GPU in-flight protection and exact prefill cache
contents. A large miss reduction would justify a real implementation experiment;
small or inconsistent savings will not. Run the simulation between model series,
so CPU work does not contaminate inference measurements.

The cold-MRU route screen found no useful admission shortcut. Threshold zero
matches current LFU/LRU exactly at 106.37 misses/token; thresholds one and
two worsen it to 107.58 and 111.22. Empty/hotlist starts converge to the same
results after the tail. Reject this ranking change without runtime edits.

## Iteration 19 — disable implicit filesystem read-ahead, keep explicit hints

Apple's [file-descriptor implementation](https://github.com/apple-oss-distributions/xnu/blob/main/bsd/kern/kern_descrip.c)
sets the per-open-file `FNORDAHEAD` flag for `F_RDAHEAD=0`.
[The vnode read path](https://github.com/apple-oss-distributions/xnu/blob/main/bsd/vfs/vfs_vnops.c)
passes it as `IO_RAOFF`, and
[clustered reads](https://github.com/apple-oss-distributions/xnu/blob/main/bsd/vfs/vfs_cluster.c)
use it to disable automatic read-ahead. This is distinct from DS4's explicit
`F_RDADVISE` requests, whose removal already lost performance. Public XNU main
is explanatory source, not proof of the exact installed kernel revision.

Plan before editing: prepare a baseline runtime copy that sets `F_RDAHEAD=0`
on the engine's model descriptor only in DeepSeek Metal SSD mode. Retain
explicit hints, ordinary page caching, all read offsets/lengths, 18-reader
concurrency, cache policy and GPU code. Log whether the fcntl succeeded.
The descriptor belongs to this engine and closes with it; this changes neither
the GGUF nor a system setting. Compare full logits and alternating full-model
throughput, then CLI/filled context if promising. Do not combine it with the
worker candidate during this screen. Build/run after the current CLI series.

The six complete CLI responses are byte-identical to the validated code fixture.
Generation was 4.62/3.84 t/s for controls, 4.54/4.52 for wake-only, and
4.31/4.66 for wake plus the decode cap. Prefill was 4.68/5.03, 5.23/5.12,
and 5.10/5.27 respectively. Whole-run seconds were 53.83/59.58, 52.45/52.76,
and 54.90/51.32. The final control's large slowdown drives apparent mean wins;
neither candidate beats the first control consistently. Do not ship either
from this evidence. Keep their isolated binaries for possible follow-up and
screen automatic read-ahead separately on the original runtime.
