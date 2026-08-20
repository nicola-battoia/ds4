#!/usr/bin/env bash
set -euo pipefail

DS4_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$DS4_ROOT"

DS4_PROFILE_MODEL="${DS4_MODEL:-./ds4flash.gguf}"
DS4_PROFILE_CACHE="${DS4_M4PRO_CACHE_EXPERTS:-896}"
DS4_PROFILE_CTX="${DS4_CTX:-32768}"
DS4_PROFILE_PREFILL="${DS4_PREFILL_CHUNK:-1024}"

if [[ ! -x ./ds4 ]]; then
    printf '%s\n' 'DS4 is not built; run `make -j8` first.' >&2
    exit 1
fi
if [[ ! -f "$DS4_PROFILE_MODEL" ]]; then
    printf 'Model not found: %s\n' "$DS4_PROFILE_MODEL" >&2
    printf '%s\n' 'Download it with `./download_model.sh ds4f-q2`.' >&2
    exit 1
fi
if [[ ! "$DS4_PROFILE_CACHE" =~ ^[1-9][0-9]*$ ]]; then
    printf 'DS4_M4PRO_CACHE_EXPERTS must be a positive integer, got: %s\n' \
        "$DS4_PROFILE_CACHE" >&2
    exit 1
fi

exec ./ds4 \
    -m "$DS4_PROFILE_MODEL" \
    --metal \
    --ssd-streaming \
    --ssd-streaming-cache-experts "$DS4_PROFILE_CACHE" \
    --prefill-chunk "$DS4_PROFILE_PREFILL" \
    --ctx "$DS4_PROFILE_CTX" \
    --nothink \
    "$@"
