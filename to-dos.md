# DS4 on a 24 GB M4 Pro

Goal: make at least one DS4-supported model useful on this MacBook Pro, with a
measured nonzero decode rate and enough context for meaningful local work. Speed
may be traded for capacity, but correctness, machine responsiveness, and honest
measurement remain mandatory.

## Guardrails

- [x] Preserve the normal resident Metal, existing SSD-streaming, CUDA, ROCm,
  distributed, and CPU-reference paths.
- [x] Do not run more than one huge model process at a time.
- [x] Keep experimental controls diagnostic and narrow; do not create a second
  permanent inference semantics.
- [x] Prove projected memory fit before downloading or starting an 81+ GB model.
- [x] Keep at least 100 GiB of SSD free after model and experiment artifacts.
  Final free space is 299 GiB.
- [x] Record every model-backed run: commit, model identity, command, context,
  resident/cache plan, peak memory, prefill speed, decode speed, and output.

## 1. Establish the machine and repository baseline

- [x] Read `AGENT.md`, architecture documentation, public API, backends, tests,
  release QA, and supporting model tools.
- [x] Confirm hardware: MacBook Pro Mac16,8, M4 Pro, 14 CPU cores, 20 GPU cores,
  24 GB unified LPDDR5, internal 1 TB Apple SSD.
- [x] Confirm software: macOS 26.2, arm64, Apple clang 21, Metal supported.
- [x] Confirm current free storage: about 349 GiB (`df`) / 374 GB decimal.
- [x] Confirm no local GGUF model is currently indexed.
- [x] Confirm clean `main` worktree and enumerate remote branches.
- [x] Measure usable Metal recommended working set: 17.760 GiB, with a
  13.320 GiB maximum single Metal buffer on this M4 Pro.
- [x] Capture current memory pressure baseline: 24 GiB physical memory and 63%
  system-wide free memory before builds/model work.
- [x] Measure sequential and representative routed-read behavior without
  consuming additional material disk space. Full-file SHA read-and-hash is
  0.50 GiB/s; selected-expert `pread` traffic and end-to-end rates are recorded
  separately because the integrity pass warmed the macOS file cache.

## 2. Quantify the unmodified memory and I/O floor

- [x] Obtain authoritative GGUF metadata/tensor directory for the smallest
  supported checkpoint without first downloading the full payload.
- [x] Break Flash q2 bytes down into embeddings/output, attention, compressors,
  shared experts, routers, routed experts, and metadata.
- [x] Calculate the minimum non-routed resident set and one-layer/one-expert
  streaming staging requirements.
- [x] Calculate KV, indexer, scratch, prefill, logits, and session memory at 2K,
  4K, 8K, 16K, and 32K contexts.
- [x] Exercise and extend memory-plan tests for a 24 GB recommended working set.
- [x] Establish an end-to-end selected-expert lower bound: the exact 10-slot
  DS4-cache-cold run reread 4.79 GiB and still decoded at 5.82 tokens/s. This is
  not labeled physically disk-cold because macOS cache was not destructively
  flushed.
- [x] Decide whether stock Flash q2 streaming can fit before implementation.

## 3. Investigate existing project approaches

- [x] Compare `origin/main`, `origin/ds4f-mxfp4`, `origin/glm5.2`, and
  `origin/laguna-s2.1` for model-loading, quantization, streaming, and memory
  policy changes relevant to 24 GB Metal.
- [x] Review history for earlier cold-decode prefill, selected-expert cache,
  mmap/residency, low-context, and memory-pressure experiments.
- [x] Evaluate available model artifacts: Flash q2, mixed q2/q4, MXFP4, DSpark,
  GLM reduced precision, and whether any support model is independently usable.
- [x] Check upstream model metadata and licenses using primary sources.

## 4. Implement the narrowest viable 24 GB path

- [x] Add a read-only model memory-plan/inspection tool that reports tensor-class
  bytes and the minimum viable Metal/SSD-streaming plan without allocating the
  full inference graph.
- [x] Make 24 GB planning failures explicit and actionable instead of relying on
  VM pressure or allocation failure.
- [x] If the current selected-expert cache can fit, add a low-memory streaming
  preset/automatic plan based on measured working-set headroom.
- [x] Implement bounded token-major Laguna SSD prefill so row-proportional
  graph scratch is allocated for one token instead of up to 16K tokens.
- [x] Remove unnecessary two-layer prefill headroom only if a bounded one-layer
  or token-major prefill schedule is implemented and proven correct.
- [x] Bound mixed Q2_K/Q3_K expert staging in a largest-class slab and add
  selected-address Q3_K kernels for Laguna's 10 active experts.
- [x] Ensure cache misses cannot pin
  unbounded mmap ranges.
- [x] Add targeted mixed-slot, top-10, tiny-cache, truncated-input, and
  low-memory context-accounting tests; preserve the existing cancellation and
  allocation cleanup paths.
- [x] Confirm the 4.074 GiB static floor fits with sufficient graph and OS
  headroom, so static streaming, repacking, and pruning are not needed for the
  first viable vertical path.

## 5. Validate and iterate

- [x] Build the normal Metal binaries warning-free.
- [x] Build CPU-only as a compile check, then restore the normal Metal build.
- [x] Run the relevant memory/placement/parser and elevated Metal tests. All
  non-model suites and the focused Laguna attention/general Metal groups pass.
  The monolithic `make test` additionally invokes unrelated long model-quality
  suites and is not used as the acceptance gate for this 44.95 GiB vertical.
- [x] Run parser unit/compile checks and the full C static analyzer. The Python
  planner has no native memory surface for ASan; analyzer findings are in
  pre-existing paths and none point into the new planner/streaming changes.
- [x] Download and SHA-256 verify the smallest viable full model after the plan
  fits: official mixed Laguna Q2_K/Q3_K, 44.946 GiB.
- [x] Run DS4-cache-cold startup and one-token smoke tests at context 64.
- [x] Sweep 10-slot, 4 GiB, 6 GiB, and 10.13 GiB caches at 64, 8K, and 32K
  while sampling memory pressure, process counters, and system swap deltas.
- [x] Run short deterministic correctness comparisons across candidate schedules.
- [x] Measure meaningful prompts at the tested 32K context, including an
  839-token repository prompt and a complete code-quality prompt.
- [x] Retain only the repeatable 4 GiB automatic-cache adaptation.

## 6. Deliverables

- [x] A reproducible command/configuration for this exact M4 Pro MacBook.
- [x] Measured peak memory, SSD footprint, startup time, prefill rate, decode
  rate, stable context, and qualitative output/correctness evidence.
- [x] Code and tests for the selected adaptation.
- [x] Documentation of compromises, unsupported combinations, and failure modes.
- [x] A clear verdict: Laguna is useful at about 3.5--4 sustained tokens/s with
  32K context; routed expert rereads are the limiting cost, not capacity.

## Discovery log

- 2026-08-19: Baseline machine inspection complete. No local GGUF is available.
  `origin/main` is clean. Candidate remote branches are `ds4f-mxfp4`, `glm5.2`,
  `laguna-s2.1`, and `responses-api`. The smallest documented complete model is
  Flash q2 at about 81 GB; the existing documented streaming recipe starts at
  64 GB RAM, so 24 GB requires proving or reducing the non-routed/runtime floor.
- 2026-08-19: Parsed 16 MiB range prefixes from the authoritative Hugging Face
  objects. Flash q2 is 80.764 GiB: 72.562 GiB routed and 8.197 GiB non-routed.
  Its one-token selected-expert set is about 1.701 GiB and two routed prefill
  layers are about 3.375 GiB. The Laguna mixed Q2/Q3 artifact is 44.946 GiB:
  40.869 GiB routed and only 4.074 GiB non-routed. Its one-token selected set is
  about 1.775 GiB and two routed layers are about 1.934 GiB.
- 2026-08-19: `origin/laguna-s2.1` is the leading prototype base. It already has
  native Metal inference and the 44.95 GiB artifact, but explicitly refuses SSD
  streaming. Porting the mature selected-expert cache interface into its Laguna
  graph should fit substantially better than Flash on this 24 GiB machine. The
  branch is 155 main commits behind, so this will be proven vertically there
  first and only then reconciled with current mainline changes.
- 2026-08-19: The M4 Pro reports a 17.760 GiB recommended Metal working set and
  a 13.320 GiB maximum buffer. Laguna's original graph tied scratch to as many
  as 16K prompt rows (roughly 6 GiB); SSD mode now allocates one-row scratch and
  uses token-major prefill. The Metal expert cache now accepts Laguna's 10-way
  routing, uses max-Q3-sized slots for both routed quantization bands, and has
  native Q3 selected-address pair/down kernels. The normal build is warning
  free, the new Metal source compiles at runtime on the M4 Pro, and focused
  Laguna attention plus general Metal numeric suites pass.
- 2026-08-19: Initial Metal-only context planning accounted for Laguna's 12
  global and 36 sliding-window KV layers and allowed a 10.132 GiB cache. This
  proved allocation feasibility, but authentic-model swap measurements later
  showed that Metal's device recommendation alone is not a safe unified-memory
  policy on the 24 GiB host; the final host-aware cap supersedes this plan.
- 2026-08-19: A synthetic file-backed Metal test now executes the exact Laguna
  Q3 selected-address path with 10 of 12 experts and compares it to resident
  execution. All 256 outputs are bit-identical (zero mismatches, zero maximum
  absolute error), and exactly 10 experts enter the SSD cache. The memory-plan
  parser has six passing unit tests, including mixed Q2/Q3 accounting and
  global/sliding KV accounting. Both CPU-only and Metal builds are warning-free.
- 2026-08-19: Stock Flash q2 is not the first viable target: its 8.197 GiB
  non-routed floor and 1.701 GiB selected set leave materially less runtime and
  cache headroom than Laguna's 4.074 GiB floor. Laguna is also 35.8 GiB smaller
  on disk, so full-model validation proceeds with Laguna first.
- 2026-08-19: A sparse 44.95 GiB structural clone built from the authentic
  Laguna header exposed two integration gaps before the real payload arrived.
  The graph still forced resident routed tensors, and the initial SSD Metal map
  covered only the embedding. Streaming state now reaches the token graph,
  batch MoE is rejected in SSD mode, and the initial map covers exactly 142
  non-routed spans totaling 4.07 GiB. The sparse end-to-end run completed a
  44-token prompt plus two decode evaluations, loaded 470 experts (1.60 GiB)
  through explicit pread, and then decoded with 10/10 cache hits in every
  routed layer. Q2 and Q3 each remain bit-identical to resident execution.
- 2026-08-19: The automatic 32K plan ran end to end with a 10.13 GiB cache
  ceiling, 1.57 GiB KV allocation, 0.75 MiB scratch, and a 15.78 GiB projected
  full working set. A 262K request was rejected before graph allocation because
  its 12.07 GiB graph plus 4.07 GiB static floor cannot leave room for a
  complete routed top-10 set. Manual caches below 10 entries are likewise
  rejected. Session-level live decode versus fresh replay compared all 100,352
  logits exactly (`max_abs=0`).
- 2026-08-19: Downloaded the authentic 48,260,803,968-byte Laguna mixed
  checkpoint and verified SHA-256
  `61fc66596597985cb9408a8530de6322d9e0d5b1d2ad4ed6503938018e0ce903`.
  The full read-and-hash took 89.43 seconds (0.50 GiB/s). The exact 10-slot
  lower bound produced deterministic output at 5.82 aggregate decode tokens/s
  while exercising complete cache thrash without any whole-model fallback.
- 2026-08-19: Real cache sweep rejected the preliminary 10.13 GiB automatic
  policy: it raised system swap by 2,973 MiB. A 6 GiB cap delivered 3.35 input
  and 3.45 decode tokens/s on an 839-token/32K run, but system swap still grew
  529 MiB. The controlled 4 GiB repeat improved those rates to 3.67 and 3.98,
  lowered peak footprint from 7.70 to 5.70 GiB, and reduced system swap by
  288 MiB. Automatic Laguna cache sizing is therefore capped at one sixth of
  physical memory: 1,059 slots/3.999 GiB and a 9.644 GiB 32K plan here.
- 2026-08-19: Final authentic-model auto-policy smoke reported the intended
  4.00 GiB cache and 9.64 GiB 32K plan, generated at 5.55 decode tokens/s,
  recorded zero process swaps, and left system swap 32 MiB lower. A sustained
  final-policy quality run then processed 76 tokens at 3.81 tokens/s and
  generated a complete 116-token answer at 4.03 tokens/s, with 3.59 GiB max
  RSS, 5.70 GiB peak footprint, zero process swap, and 168 MiB less system
  swap. Its generated Python and two assertions execute successfully. Elevated
  Laguna attention and general Metal suites pass, and 299 GiB of SSD remains
  free.
