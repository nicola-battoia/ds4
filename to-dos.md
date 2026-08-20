# DeepSeek V4 Flash on a 24 GiB M4 Pro

Goal: make an authentic DeepSeek V4 Flash checkpoint useful on this MacBook
Pro, with a meaningful context window and measured generation throughput at
least comparable to the working Laguna adaptation. Preserve correctness and
the existing resident Metal, SSD-streaming, CUDA, ROCm, distributed, and CPU
reference paths.

## Guardrails

- [x] Preserve the completed Laguna work on its own branch.
- [x] Create a separate Flash branch from the current Flash implementation.
- [x] Never run two huge model processes concurrently.
- [x] Keep at least 100 GiB of SSD free.
- [x] Prove memory fit before downloading another large artifact.
- [x] Record exact model identity, command, context, cache policy, throughput,
  memory footprint, swap delta, and output for every decisive model-backed run.
- [x] Do not claim physical cold-SSD performance after a full-file integrity
  read; distinguish DS4-cache-cold from macOS-file-cache-cold.

## 1. Establish the Flash baseline

- [x] Confirm branch ancestry, current Flash features, and relevant deltas from
  `ds4f-mxfp4`, `laguna-s2.1`, and earlier SSD-streaming work.
- [x] Reconfirm the exact M4 Pro device limits, host memory pressure, swap, and
  free SSD space from the new branch.
- [x] Inventory any existing Flash GGUFs, prefixes, conversion artifacts, and
  support models without duplicating large files.
- [x] Build current main on Metal and run its non-model/focused regressions.
- [x] Reproduce the current documented 64 GiB SSD-streaming plan in the memory
  accounting code before modifying it.

## 2. Quantify the true 24 GiB floor

- [x] Parse authoritative metadata and all tensor descriptors for the selected
  Flash artifact before downloading payloads.
- [x] Break down routed experts, attention, compressors, shared experts,
  embedding/output, routers, and other always-live tensors by quantization.
- [x] Calculate exact non-routed residency, maximum selected-expert slab,
  prefill staging, graph scratch, logits, and KV memory at 4K/8K/16K/32K.
- [x] Quantify the minimum legal cache that can bind all selected experts and
  the I/O volume per token under complete cache thrash.
- [x] Compare q2-imatrix, q2-q4-imatrix, MXFP4, and any newer official/current
  Flash artifacts for total size, static floor, supported kernels, and quality.
- [x] Identify whether capacity, transient graph allocation, Metal buffer
  limits, cache locality, or SSD traffic is the first hard bottleneck.

## 3. Investigate and test hypotheses vertically

- [x] Test the existing selected-expert SSD path with the smallest legal cache
  and low context using a structural checkpoint before the full model.
- [x] Evaluate a host-memory-aware automatic cache cap rather than trusting only
  Metal's device working-set recommendation on unified memory.
- [x] Determine whether Flash prefill can use a bounded chunk or token-major
  fallback without changing logits or regressing the normal fast path.
- [x] Determine whether Q8/F16 non-routed tensors can be safely mapped, lazily
  paged, streamed, or repacked to lower the resident floor.
- [x] Determine whether an existing lower-precision static-weight artifact is
  sufficient; if not, prototype the narrowest reproducible repack/conversion.
- [x] Measure cache locality and test whether a smaller frequency-based or
  layer-aware cache improves throughput per GiB.
- [x] Retain only hypotheses with real memory, correctness, and throughput
  evidence; remove dead experimental branches and flags.

## 4. Implement the selected design

- [x] Add or extend a read-only Flash memory planner with tests.
- [x] Add an explicit planner rejection guard for unsafe 24 GiB configurations.
- [x] Implement the minimal bounded-memory Metal/SSD changes.
- [x] Keep manual cache overrides intentional while making the automatic policy
  safe for this machine.
- [x] Add focused numerical tests for every new quantization/kernel/schedule
  path and session replay coverage for the CLI/agent/server graph.
- [x] Document supported and rejected backend/streaming combinations.

## 5. Authenticate and validate the result

- [x] Download or reuse the smallest viable full Flash checkpoint only after
  the projected plan fits.
- [x] Verify exact file size and SHA-256 against its publisher.
- [x] Run minimum-cache, automatic-cache, and cache/context sweep experiments.
- [x] Compare live decode with fresh replay logits.
- [x] Run a complete deterministic coding-quality prompt and execute its output.
- [x] Run a long repository prompt at the recommended context.
- [x] Record sustained prefill/generation speed, RSS, peak task footprint,
  process swaps, system-wide swap delta, cache hit rate, and routed read volume.
- [x] Rebuild CPU-only, restore Metal, and pass the relevant regression suites.

## 6. Deliverables

- [x] A copy-paste interactive terminal command for this exact machine.
- [x] A measured recommended context/cache configuration.
- [x] Source, kernels, planner, and focused tests for the adaptation.
- [x] A detailed experiment report with failed hypotheses and compromises.
- [x] A clear usable/not-usable verdict against the Laguna baseline.

## Discovery log

- 2026-08-20: The completed Laguna adaptation is preserved at
  `codex/m4pro-24gb-laguna` (`ca69dbe`) and its remote counterpart. Created
  `codex/m4pro-24gb-deepseek-flash` from `upstream/main` (`84cc882`) because
  current main contains 155 Flash-focused commits absent from the Laguna base.
  The first known Flash q2 candidate is about 80.764 GiB with an 8.197 GiB
  non-routed floor; those earlier header figures must now be reproduced against
  the current code and current upstream artifact before choosing a design.

- 2026-08-20: Verified the current publisher object directly: exact size
  `86,720,111,488` bytes, SHA-256
  `ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0`.
  Its first 8 MiB are byte-identical to the earlier prefix. The authenticated
  directory describes 1,328 tensors: 72.562 GiB routed and 8.197 GiB
  non-routed. Each of 43 routed layers is 1.688 GiB, each expert is 6.75 MiB,
  and one token selects exactly 258 expert slots / 1.701 GiB.

- 2026-08-20: Current main builds warning-free on Apple clang. Its blanket
  `make test` correctly rejects the preserved Laguna `ds4flash.gguf` symlink as
  the wrong architecture; model-independent tests are being run separately.
  `ds4f-mxfp4` has no commits absent from current main, so there is no hidden
  branch implementation to port.

- 2026-08-20: Built an 8 MiB-on-disk sparse structural clone with the authentic
  80.76 GiB logical layout. A 258-slot cache completed a 10-token prefill and
  eight-token decode at 32K context with zero process swaps. The C planner
  reports 0.61 GiB KV + 0.25 GiB buffers; Metal reported a 2.12 GiB task
  footprint and 15.55 structural decode tok/s. This proves allocation and
  selected-expert binding fit, not authentic-weight performance. SSD mode pins
  only the 0.99 GiB embedding span initially; other non-routed weights use
  reclaimable exact model views.

- 2026-08-20: The model-independent baseline is clean: the main Metal build is
  warning-free; agent, evaluator extractor, layer-pack, multi-GPU placement,
  and GPU-argument suites pass. The blanket model suite is intentionally
  deferred until `ds4flash.gguf` names Flash again, because its ignored symlink
  still points at the preserved Laguna checkpoint. A pristine `84cc882`
  baseline binary is also built under `/tmp` for matched authentic A/B runs.

- 2026-08-20: The current q2 imatrix 0731 file is the only viable published
  Flash artifact for 24 GiB. The mixed q2/q4 artifact is 97.6 GB and forces six
  larger routed layers to bypass DS4's uniform-size cache; MXFP4 is 156 GB and
  Q4 is 165 GB. Their non-routed recipe is unchanged, while their selected
  expert slabs are larger, so none lowers the static floor or SSD traffic.

- 2026-08-20: Reproduced the documented 64 GiB manual plan: a `32GB` routed
  budget first reserves 3.375 GiB for two complete prefill layers, leaving
  about 28.62 GiB / 4,342 dynamic q2 expert slots. On this M4 Pro the automatic
  plan instead has a 6.005 GiB total routed budget and 399 dynamic slots after
  the same reserve.

- 2026-08-20: The pre-existing startup estimate omitted most of the batch Metal
  workspace. Exact single-tier allocation mirroring shows that a worst-case
  4,096-row prefill at 32K needs 0.608 GiB KV plus 4.220 GiB workspace/state,
  not just the previously reported 0.252 GiB score/mask scratch. Adding all
  non-routed pages and the automatic routed budget reaches about 19.0 GiB,
  above Metal's 17.760 GiB recommendation though below 24 GiB physical RAM.
  `--prefill-chunk 2048` reduces workspace/state to 2.210 GiB (2.650 GiB with
  KV) and the conservative automatic total to 16.852 GiB while preserving the
  399-slot cache; this was the first 24 GiB-safe candidate for authentic
  testing.

- 2026-08-20: Audited direct M3/M5 dispatch gates left outside the existing
  pre-M5 port. Admitting M4 for those same exact-shape kernels passes the full
  Metal kernel suite bit-for-bit. The cleanest matched 32K structural decode
  improved from 15.55 to 19.45 tok/s after enabling the compressor-pair and
  gathered-KV paths; the broader gate set remains provisional until authentic
  per-feature A/B evidence is available.

- 2026-08-20: Downloaded the authentic q2-imatrix artifact only after the
  bounded plan fit. Exact local size is `86,720,111,488` bytes and full SHA-256
  is `ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0`,
  matching the publisher. The first run preceded the checksum pass; all later
  runs are intentionally described as DS4-cache cold/warm rather than physical
  SSD cold.

- 2026-08-20: Authentic cache sweep at 32K/2K found the actual constraint.
  Auto 399 slots used a 3.08 GiB task footprint but managed 2.52 t/s over a
  short cold run. Exact 519 reached 3.23 t/s, and 768 reached 3.61 t/s with the
  default nine SSD readers. Exact 1024 looked modest at 7.21 GiB footprint but
  caused mapped non-routed pages to churn and collapsed to about 0.27 t/s; the
  run was stopped after eight decode tokens. Unified-memory working-set
  headroom, not process RSS or Metal's single-buffer limit, is the first hard
  capacity bottleneck.

- 2026-08-20: Eighteen parallel routed-tensor reads at the safe 768-slot cache
  sustained 4.04 generation t/s over 96 tokens, with 5.54 prompt t/s, 5.52 GiB
  task footprint, zero process swaps, 59.6% aggregate cache hit rate, and
  0.689 GiB of routed reads per generated token. This is now the automatic M4
  Pro reader count; other devices keep nine and the environment override is
  preserved. Disabling the aggregate M4 direct ports gave 4.06 decode t/s but
  reduced the matched short prefill from 5.54 to 5.03 t/s, so their authentic
  decode effect is neutral while the bounded-prefill effect is positive.

- 2026-08-20: Exact-count planner support mirrors the CLI's `N` semantics and
  applies a 95% recommended-working-set admission guard. It reports the 768
  profile at 0.440 GiB KV, 2.210 GiB workspace/state, 5.062 GiB cache, and
  15.910 GiB conservative unified demand (`ok`); 1024 reaches 17.597 GiB and is
  rejected as `over-ws`, matching the observed cliff. Twelve planner tests
  pass.

- 2026-08-20: Task-level validation is mixed but useful. The model returned all
  first sixteen primes, generated a complete Python program at 3.95 t/s that
  passed 6/6 adversarial executions, and preserved a two-turn terminal chat
  whose second reply ran at 4.34 t/s. It also initially made a simple arithmetic
  error before detecting and correcting it. Four-token live decode versus fresh
  replay retained the same top token (`RMS=0.0914`, `max_abs=0.426`), making the
  IQ2 quality compromise explicit rather than hiding it.

- 2026-08-21: Short-prompt success did not generalize to filled context. With
  768 cache entries, a 512-token hybrid run reached 3.72 generation t/s, but
  2K and 4K prompts collapsed to about 0.23 t/s. Per-token profiling placed
  roughly four seconds in selected-expert loading, not Metal attention. A
  longer 64-token token-major tail still produced only 0.24 t/s and reduced
  prefill to 11.75 t/s, ruling out blind warmup length as the solution.

- 2026-08-21: Two independent conditions are required after long prefill. A
  hybrid schedule handles the prefix layer-major, resets route hotness at the
  phase boundary, then processes the final 18 tokens through the established
  token-major path. Disabling only that tail at 4K/896 produced a 4.20-second
  first token and 0.26 t/s. Keeping the tail but using 768 entries produced
  0.23 t/s at 2K. Raising the cache to the planner-safe 896 entries recovered
  2K to 2.83 t/s and 4K to 3.23 t/s.

- 2026-08-21: Shrinking the prefill chunk from 2K to 1K reduced the 32K
  conservative plan from 16.753 to 15.571 GiB while preserving the 896-entry
  cache. The 4K rate remained 3.27 generation t/s. This also made 1024 entries
  legal (16.414 GiB conservative), but a 32-token run stayed at 3.26 t/s and
  raised peak footprint to 8.81 GB, so 896/1K is the final balance.

- 2026-08-21: Final acceptance used 16K tokens of this repository's README
  with a 32K allocation, 1K chunks, 896 cached experts, 18 readers, and the
  hybrid tail. It measured 56.79 prefill t/s, 3.82 generation t/s, and a
  286.7 ms first decode step. `/usr/bin/time -l` reported 6,376,718,336 bytes
  maximum RSS, 7,921,259,992 bytes peak task footprint, and zero swaps.

- 2026-08-21: The full Metal kernel target passes, as do CPU-only and restored
  Metal warning-free builds, twelve planner tests, agent/evaluator/layer-pack/
  placement/GPU-argument suites, and the strengthened authentic long-code
  streaming regression. Canonical versus hybrid prefill chose the same top
  token with 18/20 top overlap and 0.760 RMS drift; three hybrid captures were
  byte-identical across cold, warm, and repeated cache states. A real PTY run
  reached `ds4>`, answered `READY`, returned to the prompt, and exited normally.
