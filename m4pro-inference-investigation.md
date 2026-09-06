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
