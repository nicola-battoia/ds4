# DeepSeek V4 Flash on an M4 Pro with 24GB

## Result

The September 6 upstream revalidation keeps the 896-expert profile below.
With a filled 16K prefix from `speed-bench/promessi_sposi.txt` and 128 generated
tokens, unchanged controls measured **4.92–4.98 generation tokens/s**. A
complete CLI coding task measured **4.50–4.53 tokens/s** and its generated
function passed seven execution cases. The current investigation, exact-logit
checks and candidate results are recorded in
[the investigation journal](m4pro-inference-investigation.md).

The following original profile measurements used different prompts and shorter
decode probes; they are historical evidence, not a matched speed comparison.

Authentic DeepSeek V4 Flash is usable on the 14-core M4 Pro with 24GB of
unified memory when its routed experts are streamed from the internal SSD.
The balanced profile is a 32K allocated context, a 1K prefill chunk, an exact
896-expert cache, and 18 parallel expert-tensor reads. On 16K tokens of this
repository's README it sustained `56.79` prefill tokens/s and `3.82`
generation tokens/s, with a `287 ms` first decode step, zero process swaps,
`6.38 GB` maximum RSS, and a `7.92 GB` peak task footprint.

That is comparable to the Laguna adaptation's roughly 3.5--4 sustained
tokens/s, but Flash was validated here with a much larger 16K-token repository
prompt. A 32K buffer is allocated and memory-safe; 16K is the largest filled
context measured end to end, not a claim that all 32K tokens were filled.

This is a capacity profile for the very compressed IQ2/Q2 artifact. It is not
equivalent in quality or speed to running a larger quantization resident on a
high-memory Mac. Important answers still need verification.

## Tested machine and artifact

- MacBook Pro `Mac16,8`
- Apple M4 Pro: 14 CPU cores, Apple M4 Pro Metal device
- 24GB LPDDR5 unified memory
- Metal recommended maximum working set: `19,069,665,280` bytes
  (`17.760 GiB`)
- Metal maximum single buffer: `14,302,248,960` bytes (`13.320 GiB`)
- Internal Apple Fabric SSD
- Model:
  `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf`
- Publisher object size: `86,720,111,488` bytes (`80.764 GiB`)
- SHA-256:
  `ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0`
- Publisher:
  <https://huggingface.co/antirez/deepseek-v4-gguf/blob/main/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf>

The local file size and full-file SHA-256 were verified after the first
physical-cold-ish run. The checksum read warms macOS's file cache, so later
runs are described only as DS4-cache cold or warm, not physical-SSD cold.

## Why it fits

The artifact contains 1,328 tensors. Routed experts occupy `72.562 GiB`
(`89.9%`) and are the part DS4 can stream selectively. The non-routed tensor
floor is `8.197 GiB`:

| Tensor family | Logical bytes |
| --- | ---: |
| Attention and compressors | 5.367 GiB |
| Shared experts | 1.071 GiB |
| Token embedding | 0.986 GiB |
| Output | 0.524 GiB |
| Routers | 0.084 GiB |
| Other and norms | 0.165 GiB |

All 43 routed layers have the same `1.688 GiB` layout. One expert slot is
`6.75 MiB`; six experts per layer means one token selects 258 slots, or
`1.701 GiB` of routed weights. The 896-slot cache covers about 3.47 complete
token working sets. Earlier runs observed `0.23` generation tokens/s with
768 entries and `2.83` with 896 after 2K prefill. The September revalidation
did not reproduce a capacity threshold: matched 768/896 runs were both around
5 tokens/s. System and file-cache state can cause large slowdowns even with
unchanged settings, so those earlier numbers do not establish a cache-size
cause. The 896 profile remains the tested balance; larger caches did not
produce a reliable generation gain in the longer revalidation.

The header-only planner mirrors the complete single-tier Metal graph rather
than DS4's older startup line, which counts only a subset of graph workspace.
At 32K context and a 1K prefill chunk it reports:

| Item | 896-expert profile |
| --- | ---: |
| KV cache | 0.356 GiB |
| Metal workspace and compressor state | 1.112 GiB |
| Exact dynamic expert cache | 5.906 GiB |
| Startup plan (embedding + graph + cache) | 8.360 GiB |
| Conservative plan (all non-routed + graph + cache) | 15.571 GiB |
| 95% Metal working-set guard | 16.872 GiB |

Task footprint does not account for the entire mapped/driver working set.
The conservative plan still matters: 1024 entries with a 2K workspace require
17.597 GiB and exceed the guard. A 1K workspace puts 1024 entries below the
guard, but September's matched 128-token comparison found no reliable speed
gain over 896. The additional memory is not justified by those measurements.

## Implementation

- `gguf-tools/model_memory_plan.py` parses a complete GGUF or header prefix,
  classifies every tensor, mirrors DeepSeek4 KV/workspace geometry, models
  automatic, NGB, and exact-count cache semantics, and applies a 5% Metal
  working-set guard for untracked runtime overhead.
- `tests/test_model_memory_plan.py` covers truncated headers, mixed routed
  classes, context geometry, bounded prefill, byte-budget reserve semantics,
  exact expert counts, host caps, and the working-set edge guard.
- The M4 Pro Metal SSD path defaults to 18 parallel `pread` workers. Other
  devices retain the existing default of nine, and
  `DS4_METAL_STREAMING_EXPERT_PREAD_THREADS` remains an override.
- Long streamed prompts now use a hybrid prefill schedule: layer-major chunks
  handle the prefix efficiently, routed-cache hotness is reset at the phase
  boundary, and the final 18 tokens use the established token-major path. This
  leaves both cache residency and route frequency in a decode-like state.
  `DS4_METAL_DISABLE_STREAMING_DECODE_PREFILL_TAIL=1` is the matched rollback.
  It is limited to fast SSD-streaming Pro/Flash graph sessions; resident,
  CPU, quality/imatrix, and non-DeepSeek4 paths retain their prior schedule.
  Authentic validation in this report is Metal-only.
- Direct kernels previously gated to M3/M5 are admitted on M4 behind the
  aggregate rollback `DS4_METAL_DISABLE_M4_DIRECT_PORTS`. The complete Metal
  kernel suite is bit-exact with the M4 paths enabled. In the authentic paired
  run, decode was neutral within noise and short prefill improved from 5.03 to
  5.54 tokens/s.
- `run-deepseek-flash-m4pro-24gb.sh` captures the measured safe defaults while
  leaving ordinary DS4 arguments available for overrides.

## Authentic benchmark results

These historical rows use Metal, SSD streaming, 32K allocated context, 18
readers, greedy generation, and one model process at a time. Aggregate speed
includes the first decode step. Their short probes do not establish cache-size
causation; use the longer matched comparisons in the investigation journal.

| Filled context | Chunk | Cache | Schedule | Generated | Prefill | Generation | First decode | Result |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 512 | 2K | 768 | hybrid | 16 | 14.60 t/s | 3.72 t/s | 370 ms | Short fill works |
| 2K | 2K | 768 | hybrid | 8 | 18.36 t/s | 0.23 t/s | 3869 ms | Historical slowdown |
| 2K | 2K | 896 | hybrid | 8 | 36.40 t/s | 2.83 t/s | 403 ms | Historical larger-cache run |
| 4K | 2K | 896 | no tail | 8 | 80.82 t/s | 0.26 t/s | 4201 ms | Phase rollback fails |
| 4K | 2K | 896 | hybrid | 16 | 68.62 t/s | 3.23 t/s | 293 ms | Both fixes present |
| 8K | 2K | 896 | hybrid | 16 | 79.29 t/s | 2.90 t/s | 414 ms | Useful context |
| 16K README | 1K | 896 | hybrid | 16 | **56.79 t/s** | **3.82 t/s** | **287 ms** | Original acceptance |

The earlier short 768-entry/18-reader run generated 96 tokens at `4.04` t/s,
with a 59.6% hit rate and `0.689 GiB` of routed reads per generated token. It
was useful for early reader-count tuning but did not cover longer generation.
For the original 16K acceptance measurement, `/usr/bin/time -l`
reported `6,376,718,336` bytes maximum RSS, `7,921,259,992` bytes peak task
footprint, and zero process swaps.

System-wide swap is noisier because this was a live desktop with substantial
pre-existing swap. Earlier 768-entry tuning runs coincided with 0.14--1.15 GiB
of system-wide change. Zero process swaps do not rule out GPU page faults,
file-cache eviction or pressure on other applications. The cache is mlocked for stable
Metal latency, so close memory-heavy applications if macOS memory pressure is
already elevated.

## Correctness and quality checks

- The full Metal kernel suite passes bit-for-bit on the M4 path.
- Four-token live decode versus fresh full-prefill replay on the final cache
  profile chose the same top token. Across 129,280 logits, RMS drift was
  `0.774` and maximum absolute drift was `4.887`, inside the existing IQ2
  batch/decode variation but not bit-exact.
- A strengthened long-code regression compares canonical layer-major prefill
  with the hybrid path. Both chose token 671; top-5 overlap was 3/5, top-20
  overlap 18/20, RMS `0.760`, and maximum absolute drift `4.035`. Cold, warm,
  and repeated hybrid captures were byte-identical.
- A deterministic prime-list prompt returned all first sixteen primes.
- A generated Python longest-increasing-run program completed at `3.95 t/s`
  and passed 6/6 adversarial execution cases.
- A two-turn interactive PTY conversation preserved context; the second reply
  generated at `4.34 t/s`.
- The Q2 model initially made a simple arithmetic error, noticed it in its own
  verification, and corrected it on the next turn. This is a useful warning
  about the quality cost of fitting Flash into 24GB.
- CPU-only and restored Metal builds are warning-free. Twelve planner tests,
  agent/evaluator/layer-pack/placement/GPU-argument suites, and the complete
  Metal kernel target pass.

## Run it

```sh
./download_model.sh ds4f-q2
make -j8
./run-deepseek-flash-m4pro-24gb.sh
```

DSpark is not supported by this 24GB profile. The main model must use SSD
streaming here, while the current upstream runtime explicitly rejects
`--ssd-streaming` together with `--mtp`. An experimental M4 Pro run that lifted
that guard accepted none of 46 forced draft tokens on the upstream `c_add`
fixture. In the warm matched 32-token run, ordinary decoding reached 3.29 t/s
and DSpark reached 0.69 t/s with zero accepted drafts. The wrapper therefore
rejects `DS4_DSPARK_SUPPORT`; speculative decoding remains for machines that
can keep the main model resident.

At the `ds4>` prompt, type normally. Useful commands are `/help`, `/nothink`,
`/think`, `/read FILE`, and `/quit`. For a one-shot prompt:

```sh
./run-deepseek-flash-m4pro-24gb.sh \
  -p 'Explain the ownership rules in this C function.' \
  --tokens 256
```

To inspect the exact plan without loading model payloads:

```sh
python3 gguf-tools/model_memory_plan.py ./ds4flash.gguf \
  --host-memory-gib 24 \
  --working-set-gib 17.76 \
  --contexts 4096,8192,16384,32768 \
  --prefill-chunk 1024 \
  --expert-cache-count 896
```

The wrapper accepts `DS4_MODEL`, `DS4_CTX`, `DS4_PREFILL_CHUNK`, and
`DS4_M4PRO_CACHE_EXPERTS`. A 768-entry fallback saves `0.844 GiB`. Current
matched testing found similar throughput at 768 and 896 entries; keep the
default unless the smaller footprint is useful for your other applications.

## Rejected alternatives

- The mixed q2/q4 artifact is 97.6 GB and has six larger routed layers that do
  not fit the uniform expert cache; it increases both selected bytes and I/O.
- MXFP4 is about 156 GB and Q4 about 165 GB. Their non-routed recipe is not
  smaller, and their expert slabs are larger.
- A 4K prefill chunk needs about 4.22 GiB of workspace/state at 32K. A 2K
  chunk lowers that to 2.21 GiB, but the final 1K setting needs only 1.11 GiB
  and leaves substantially more room for the routed cache and macOS.
- Smaller caches fit more easily but have not produced a repeatable generation
  speed gain over the current 896-slot profile.
- A 64-token token-major tail did not repair an earlier slow 4K/768 run
  (`0.24` t/s) and cut prefill to `11.75` t/s. The extra prefill cost did not
  establish a useful generation benefit.
- 1024 entries with the old 2K workspace exceed the conservative memory guard.
  A 1K workspace fits that guard, but the longer matched comparison did not
  show a generation gain over 896 entries.
