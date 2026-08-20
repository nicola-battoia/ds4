# DS4 Laguna on a 24 GB M4 Pro: experiment log

Machine: MacBook Pro Mac16,8, Apple M4 Pro (10 performance + 4 efficiency CPU
cores, 20 GPU cores), 24 GiB unified memory, internal 1 TB Apple SSD. Metal's
`recommendedMaxWorkingSetSize` is 17.760 GiB and its maximum single buffer is
13.320 GiB. Experiments use branch `codex/m4pro-24gb-laguna`, based on
`448d5695d1c86401a4e9447c440feb983b73e6de`.

## Verdict

Laguna S 2.1 is usable on this machine through the implemented Metal
selected-expert SSD path. The recommended 32K configuration automatically
keeps 1,059 max-Q3 cache slots (3.999 GiB) and plans 9.644 GiB for DS4. A final
quality run processed 76 input tokens at 3.81 tokens/s and generated a complete
116-token answer at 4.03 tokens/s. The process used 3.59 GiB maximum RSS and a
5.70 GiB peak task footprint with zero process swaps. An 839-token repository
prompt also sustained 3.67 input tokens/s, and a short smoke reached 5.55 decode
tokens/s. Quality is model-appropriate: the deterministic coding run produced
a correct order-preserving Python deduplication function whose two generated
assertions execute successfully.

This does not make the whole 44.946 GiB checkpoint resident. The 4.074 GiB
non-routed signal path remains mapped, while only selected routed experts are
read into a bounded cache. Throughput is the compromise: roughly 3.5--4
tokens/s for sustained repository work instead of full-residency speed.

## Repository/branch decision

`origin/ds4f-mxfp4` is already merged into current main and does not provide a
smaller independently runnable model. `origin/glm5.2` adds a different, much
larger model family. Current main has intentionally retired Laguna, whereas
`origin/laguna-s2.1` contains its native graph, tokenizer/chat handling, and
Q2_K/Q3_K support. That branch is 155 main commits behind and 17 commits ahead,
so the adaptation is being proven vertically on a dedicated
`codex/m4pro-24gb-laguna` branch rather than hiding a large model-family
reintroduction inside an unrelated mainline merge. The streaming/cache changes
preserve the resident Metal, CUDA, ROCm, CPU reference, and existing
non-Laguna SSD paths.

## Candidate selection and memory plan

The smallest complete DeepSeek V4 Flash artifact is not the best 24 GiB
starting point. Its Q2 checkpoint is 80.764 GiB, including 8.197 GiB of
non-routed tensors that must be available on every token. The mixed Laguna
checkpoint is 44.946 GiB, with 40.869 GiB in routed experts and a 4.074 GiB
non-routed floor. Its selected routed work is also slightly smaller: 1.596 GiB
of exact expert payload per token (1.775 GiB when every cache slot is sized for
the larger Q3_K band), versus 1.701 GiB for Flash. Laguna DFlash is only a
speculative support model and cannot generate independently. The GLM and full
Laguna Q4 candidates have larger static or total footprints.

The implementation therefore streams only Laguna's routed experts while
retaining its attention, router, shared/dense FFN, embedding, norm, and output
tensors. The planner uses 80% of Metal's recommendation for static weights plus
cache, a stricter 90% ceiling after adding the context graph, and a measured
host-memory cap of one sixth of physical RAM. The last rule keeps the automatic
cache at 4 GiB on this 24 GiB machine:

| Context | KV | One-row scratch | Expert cache | Projected DS4 | Result |
| ---: | ---: | ---: | ---: | ---: | :--- |
| 8,192 | 0.445 GiB | 0.001 GiB | 3.999 GiB | 8.519 GiB | accept |
| 32,768 | 1.570 GiB | 0.001 GiB | 3.999 GiB | 9.644 GiB | accept |
| 65,536 | 3.070 GiB | 0.001 GiB | 3.999 GiB | 11.144 GiB | accept |
| 131,072 | 6.070 GiB | 0.001 GiB | 3.999 GiB | 14.144 GiB | accept |
| 262,144 | 12.070 GiB | 0.001 GiB | 0 GiB | 16.145 GiB | reject |

The last row is rejected before graph allocation because it cannot leave even
one complete top-10 expert set inside the safety ceiling. These are allocation
plans, not claims that every accepted long context has equal throughput. 32K
is the tested working recommendation; the 64K and 128K rows prove allocator
feasibility only and are not presented as sustained quality/performance results.

## Implemented path

- Laguna SSD mode now maps 142 disjoint non-routed tensor spans instead of
  making the 44.946 GiB model resident.
- Prefill is token-major with one row of scratch. The prior 16K-row graph could
  reserve roughly 6 GiB before routed-cache storage.
- The cache uses explicit bounded `pread` misses and never falls back to pinning
  a routed mmap view or whole routed layer.
- Q2_K and Q3_K routed layers share max-Q3 slots. New Q3 selected-address pair
  and down kernels accept Laguna's ten active experts.
- Context accounting follows Laguna's 12 global-attention and 36 sliding-window
  layers, rather than charging all 48 layers for full-context KV.
- Automatic sizing combines Metal's working-set limits, requested-context
  graph size, and the one-sixth host-RAM cap. Fewer than ten slots, 262K on this
  device, non-Metal Laguna streaming, DFlash+SSD, distributed SSD, and tensor
  parallel SSD combinations fail explicitly instead of silently becoming
  resident.
- Session replay uses the same SSD-aware token graph as the one-shot CLI.

## Structural checkpoint test

Before the real object completed downloading, the authentic 16 MiB GGUF prefix
was extended as an APFS sparse file to its declared 48,260,803,968-byte size.
This preserves all 59 metadata entries and 814 tensor descriptors but replaces
most tensor payloads with sparse zeroes. It is useful for proving allocation,
mapping, scheduling, cache I/O, and kernel integration. It says nothing about
model quality and its I/O rates are not representative of real SSD reads.

Command:

```sh
DS4_METAL_MEMORY_REPORT=1 \
DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY=1 \
DS4_TOKEN_TIMING=1 \
./ds4 --metal --ssd-streaming --ssd-streaming-cold \
  --ssd-streaming-cache-experts 2GB \
  -m /tmp/ds4-laguna-sparse-zero.gguf -c 64 -n 3 \
  --nothink --temp 0 -p Hi
```

Observed on 2026-08-19:

- Startup mapped 142 disjoint non-routed spans totaling 4.07 GiB. Metal did
  not request full-model residency.
- Graph allocation used a one-token prefill cap, 0.01 GiB KV, and 0.75 MiB
  scratch.
- A 44-token prompt completed through 44 token-major model passes.
- The cache retained 470 routed experts and copied 1.60 GiB through 47 explicit
  parallel `pread` load calls. It then served two decode evaluations with no
  additional reads and all 94 routed layers reporting 10/10 resident experts.
- Q2 and Q3 layers shared one max-Q3 slab; tracked routed live memory was
  1.60 GiB and anonymous runtime tensors were 12.75 MiB.
- The two true decode evaluations took 44.883 ms and 44.462 ms (22.28 and
  22.49 tokens/s). This is a cache-hit compute upper bound because the sparse
  model routes every token identically.
- `/usr/bin/time -l` reported 1,747,517,440 bytes maximum RSS, a
  2,293,401,616-byte peak task footprint, and zero swaps. Sparse zero pages make
  these task-memory figures lower than the real checkpoint will be.

The first structural attempt failed immediately because the inherited SSD path
mapped only `token_embd.weight`; Laguna's first attention range was absent. The
mapping was corrected to include the exact non-routed static set. A separate
call-chain audit found that token decode still passed `force_resident=true`;
streaming state is now carried by the Laguna graph and its batch-MoE path is
explicitly forbidden in SSD mode.

The same sparse structure also completed a 32K-context plan using the original
Metal-only ceiling: 1.57 GiB KV, 0.75 MiB scratch, 10.13 GiB maximum expert
cache, and 15.78 GiB projected DS4 working set. This proved the graph but later
real-model pressure tests showed that the ceiling was too aggressive for a
24 GiB unified-memory host; the final automatic cache is 4 GiB. A 262K request
was rejected before graph allocation because its
12.07 GiB graph and 4.07 GiB static floor cannot retain a complete top-10
expert set within the 90% safety target. An explicit nine-expert cache is also
rejected before mapping any routed tensor.

An exact 10-slot run exercised the minimum legal cache rather than only the
rejection boundary. It planned 4.12 GiB total at context 64, retained exactly
10 entries, and completed one prefill plus one decode evaluation with zero
cache hits, 940 misses, 930 evictions, and 3.19 GiB of bounded `pread` traffic.
No routed model view or whole-layer fallback was created. Its sparse-file rate
is not meaningful, but the eviction and address-table behavior proves that the
minimum top-k cache remains bounded under complete thrashing.

The session API was tested separately with two live decode steps followed by a
fresh replay of the same three-token prefix. All 100,352 logits compared
exactly (`max_abs=0`, `rms=0`). This exercises the graph path used by the agent
and server, not only one-shot CLI generation.

## Numeric kernel evidence

The focused M4 Metal suite constructs a file-backed model fragment containing
Q2_K and Q3_K routed tensors. It runs 10 of 12 selected experts in resident and
SSD-cache modes, keeps both quantization classes in the same max-Q3 slab, and
compares 256 output floats per class. Result: 0/256 mismatches and zero maximum
absolute error for both Q2 and Q3, with 20 entries retained across the two
layers. The complete Laguna attention numeric suite and general Metal kernel
suite also pass.

## Real checkpoint results

The official artifact was downloaded to
`gguf/laguna-s-2.1-RoutedQ2_K-Last27Q3_K.gguf`. Its size is exactly
48,260,803,968 bytes (44.946 GiB), and a complete local read produced SHA-256
`61fc66596597985cb9408a8530de6322d9e0d5b1d2ad4ed6503938018e0ce903`, matching
the publisher. Hashing the complete file read 44.946 GiB in 89.43 seconds, a
0.50 GiB/s sequential read-and-hash lower bound.

All rows below use that checkpoint and commit base. `pread` is logical routed
expert traffic, so it may exceed model size when experts are evicted and read
again. The required full-file integrity pass warmed macOS's file cache; these
are DS4-cache-cold tests where noted, not claims of a physically cold SSD.

| Cache / context / work | Prefill | Decode | Cache hit | Routed `pread` | Max RSS | Peak footprint | System swap delta |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Exact minimum 10 slots, 64 ctx, raw `Hi`, 3 output | 0.40 t/s | 5.82 t/s aggregate | 0% | 4.79 GiB | 0.13 GiB | 0.25 GiB | not sampled |
| Original auto 10.13 GiB, 8K, 66+96 tokens | 3.53 t/s | 3.96 t/s | 64.4% | 92.06 GiB | 9.09 GiB | 10.72 GiB | **+2,973 MiB** |
| Manual 6 GiB, 8K, 59+32 tokens | 4.43 t/s | 4.85 t/s | 52.1% | 69.23 GiB | 5.38 GiB | 6.58 GiB | -752 MiB |
| Auto 6 GiB, 32K, 71+180 tokens | 3.79 t/s | 4.08 t/s | 51.9% | 192.89 GiB | 5.36 GiB | 7.70 GiB | -1,820 MiB |
| Auto 6 GiB quality run, 8K, 62+128 tokens | 3.21 t/s | 3.76 t/s | 53.5% | 119.11 GiB | 5.37 GiB | 6.58 GiB | -232 MiB |
| Auto 6 GiB sustained, 32K, 839+16 tokens | 3.35 t/s | 3.45 t/s | 57.5% | 584.52 GiB | 5.36 GiB | 7.70 GiB | **+529 MiB** |
| Manual 4 GiB controlled repeat, 32K, 839+2 tokens | **3.67 t/s** | **3.98 t/s** | 47.8% | 704.83 GiB | **3.59 GiB** | **5.70 GiB** | -288 MiB |
| Final auto 4 GiB smoke, 32K, raw `Hi`, 2 output | 0.59 t/s | 5.55 t/s | not profiled | not profiled | 1.63 GiB | 5.69 GiB | -32 MiB |
| Final auto 4 GiB quality, 32K, 76+116 tokens | **3.81 t/s** | **4.03 t/s** | 44.0% | 172.82 GiB | **3.59 GiB** | **5.70 GiB** | -168 MiB |

Every process reported zero swaps in `/usr/bin/time -l`. System-wide swap was
already about 11--14 GiB before these runs even when `memory_pressure -Q`
reported 72--75% memory free; macOS retains old encrypted swap. Its delta is
therefore supporting pressure evidence, not a per-process attribution. The
10.13 GiB run clearly induced pressure. The 6 GiB setting was usually safe but
still grew system swap during the sustained 839-token run. The controlled 4 GiB
repeat used 1.66 GiB less peak task footprint, finished in 228.96 seconds versus
254.77 seconds, and did not grow swap. That is why the implemented default is
one sixth of host RAM rather than the largest value Metal alone permits.
The final smoke's complete startup, one-token prefill, and two-token output took
1.97 seconds wall-clock; its prefill number is intentionally not treated as a
sustained rate because it includes cold setup around one input token.

### Quality and consistency

The deterministic quality prompt asked for a concise order-preserving Python
deduplicator and assertions. The model completed this valid program:

```python
def dedupe(seq):
    seen = set()
    result = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

assert dedupe([1, 2, 2, 3, 1, 4]) == [1, 2, 3, 4]
assert dedupe(['a', 'b', 'a', 'c', 'b']) == ['a', 'b', 'c']
```

The generated code was executed and both assertions passed. A separate
session-level test generated two live steps, rebuilt a fresh session by
replaying the same prefix, and compared all 100,352 logits exactly
(`max_abs=0`, `rms=0`). The synthetic Metal test compares the streamed and
resident Q2 and Q3 selected-expert kernels: both have 0/256 mismatches and zero
maximum absolute error.

### Recommended invocation

```sh
./download_model.sh laguna-q2-q3
make -j8
./ds4 --metal --ssd-streaming \
  -m gguf/laguna-s-2.1-RoutedQ2_K-Last27Q3_K.gguf \
  -c 32768 --temp 0 -p "Explain this repository and identify one safe improvement"
```

Do not pass an explicit cache size for the normal 24 GiB configuration; auto
selects 4 GiB. `--ssd-streaming-cache-experts NGB` intentionally overrides the
host guard for experiments. Use `--ssd-streaming-cold` only when measuring a
cold DS4 cache; default popularity preload is better for interactive use.

### Verification performed

- Normal Metal and CPU-only compile checks are warning-free.
- Six memory-planner parser/accounting tests pass.
- Laguna attention numeric tests pass against double-precision references.
- The general Metal kernel group passes, including the new mixed Q2/Q3
  selected-10 SSD test.
- Non-model CLI, server, agent, eval extractor, Q4 dot, layer-pack, placement,
  GPU-argument, and download-parser tests pass. The monolithic `make test`
  target also includes unrelated long model-quality suites and was not used as
  a single acceptance gate for this 44.95 GiB vertical run.

At the end of the experiment `/System/Volumes/Data` had 299 GiB free, well
above the 100 GiB guardrail.

### Primary references

- [Official mixed Laguna Q2_K/Q3_K model card](https://huggingface.co/antirez/Laguna-S-2.1-GGUF)
- [Apple `recommendedMaxWorkingSetSize`](https://developer.apple.com/documentation/metal/mtldevice/recommendedmaxworkingsetsize)
- [Apple `maxBufferLength`](https://developer.apple.com/documentation/metal/mtldevice/maxbufferlength)
