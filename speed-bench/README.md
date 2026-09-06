## Benchmarking

Here we collect prefill and generation speed obtained with different hardware.

Run `ds4-bench` as:

```
./ds4-bench \
  -m ds4flash.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 \
  --ctx-max 65536 \
  --step-incr 2048 \
  --gen-tokens 128
```

Provide PR including your numbers if your hardware was not already tested.
Call the benchmark csv file something like `m3_max.csv` or alike, so that
it is clear what hardware was used for the benchmark.

To generate an SVG graph from a CSV file:

```
python3 speed-bench/plot_speed.py speed-bench/m3_max.csv --title "M3 Max t/s"
```

The script uses only the Python standard library. By default it writes a file
next to the CSV using the `_ts.svg` suffix, such as `speed-bench/m3_max_ts.svg`.

### Exact SSD-streaming decode comparison

Use separate, serial `ds4-bench` processes to compare expert-cache or I/O
policies. The two-session schedule harness below shares one expert cache, so
the first session warms the second session's routes and does not represent
ordinary streaming throughput.

For a 24GB M4 Pro diagnostic capture:

```
./ds4-bench -m ds4flash.gguf --metal --ssd-streaming \
  --ssd-streaming-cache-experts 896 --prefill-chunk 1024 \
  --ctx-alloc 32768 --ctx-start 1024 --ctx-max 1024 \
  --prompt-file speed-bench/promessi_sposi.txt --gen-tokens 64 \
  --dump-decode-logits baseline.dlog --csv baseline.csv
```

Repeat with the candidate implementation and `candidate.dlog`, then use
`cmp baseline.dlog candidate.dlog` to require bit-identical full-vocabulary
logits and consumed token IDs. Keep the same model, input, context, prefill,
and cache settings. A file is complete only when its process exits successfully.
For throughput measurements, omit the dump and alternate fresh-process order
in ABBA and BAAB blocks. Logit copying and writes are outside per-step timers
and subtracted from aggregate generation time, but they still perturb the
workload. macOS's file cache persists between processes; these are not
physical-SSD-cold measurements.

The binary dump has a 16-byte header: the eight ASCII bytes `DS4DLOG1`, a
native-endian `uint32_t` byte-order marker `0x01020304`, and a native-endian
`uint32_t` vocabulary size. Each following record contains three native-endian
`uint32_t` values (benchmark frontier, session position, consumed input token
ID), then exactly `vocab` native-endian IEEE-754 float32 logits. The prefill
row has input token ID `UINT32_MAX`; subsequent rows follow each decode step.
Ordinary greedy runs have `gen_tokens + 1` rows per completed frontier;
teacher-forced runs can stop earlier at EOS. DSpark is unsupported
because it does not expose every intermediate logit row. Files contain raw
model output, so keep them with the benchmark's input provenance.

### Metal decode schedule A/B

Build the balanced, same-engine Metal decode comparison with:

```
make metal-decode-schedule-bench
./speed-bench/metal_decode_schedule_bench \
  -m ds4flash.gguf \
  --include-selection
```

The harness prefills two sessions and alternates both variant order and
variant-to-session assignment. It aborts unless every full-vocabulary logit
row is bit-identical and, with `--include-selection`, both variants select the
same non-EOS token. Use `--candidate-env NAME` to measure a rollback control,
or `--help` to compare explicit split schedules.

To compare the default pre-M5 ratio-4 compressor pack/transpose fusion with the
legacy decode path, including token selection, use:

```
./speed-bench/metal_decode_schedule_bench \
  --candidate-env DS4_METAL_DISABLE_PRE_M5_COMPRESSOR_RATIO4_DECODE_PACK_FUSION \
  --include-selection \
  --tokens 1024
```

### Metal prefill variant A/B

Build the balanced prefill comparison. To compare the default resident pre-M5
MXFP4 pair tail-SIMDgroup cull against the original pair kernel, make the
rollback path the candidate:

```
make metal-prefill-variant-bench
./speed-bench/metal_prefill_variant_bench \
  --candidate-env DS4_METAL_DISABLE_PRE_M5_MXFP4_MOE_MM_ID_PAIR_TAIL_SIMDGROUP_CULL
```

To isolate the default routed-down tail-SIMDgroup cull from the retained pair
default, use its down-specific rollback as the candidate:

```
./speed-bench/metal_prefill_variant_bench \
  --candidate-env DS4_METAL_DISABLE_PRE_M5_MXFP4_MOE_MM_ID_DOWN_TAIL_SIMDGROUP_CULL
```

The harness uses one Metal engine and fresh sessions for every run. It warms
both variants with at least 32 tokens, alternates control/candidate order in
ABBA and BAAB blocks, poisons host logit buffers before copying, and aborts
unless every final full-vocabulary logit row is bit-identical. Defaults are an
8192-token prefix, an automatically sized 8193-token context, and two repeats;
use `--help` to override them.
