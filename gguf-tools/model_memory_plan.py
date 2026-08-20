#!/usr/bin/env python3
"""Inspect a GGUF tensor directory and estimate DS4 streaming residency.

The input may be a complete GGUF or a prefix fetched with an HTTP Range
request, provided that the prefix contains the complete metadata and tensor
directory.  For a prefix, pass the authoritative remote object size with
--file-size so the last tensor's physical span can also be reported.

This deliberately uses only the Python standard library.  It is a planning
tool, not an inference-path authority: model loading still performs DS4's
strict shape and tensor validation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
from dataclasses import dataclass
from typing import Any, BinaryIO


GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3

# GGUF metadata value types.
U8, I8, U16, I16, U32, I32, F32, BOOL, STRING, ARRAY, U64, I64, F64 = range(13)

# GGML tensor type -> (name, block elements, block bytes).  DS4 only needs a
# narrow subset, but listing the common current types makes failures explicit
# when a candidate artifact changes its signal-weight layout.
TENSOR_TYPES: dict[int, tuple[str, int, int]] = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34),
    9: ("Q8_1", 32, 40),
    10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66),
    17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50),
    20: ("IQ4_NL", 32, 18),
    21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82),
    23: ("IQ4_XS", 256, 136),
    24: ("I8", 1, 1),
    25: ("I16", 1, 2),
    26: ("I32", 1, 4),
    27: ("I64", 1, 8),
    28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56),
    30: ("BF16", 1, 2),
    34: ("TQ1_0", 256, 54),
    35: ("TQ2_0", 256, 66),
    39: ("MXFP4", 32, 17),
}

SCALAR_FORMATS = {
    U8: "<B",
    I8: "<b",
    U16: "<H",
    I16: "<h",
    U32: "<I",
    I32: "<i",
    F32: "<f",
    BOOL: "<?",
    U64: "<Q",
    I64: "<q",
    F64: "<d",
}


class GGUFError(RuntimeError):
    pass


class Reader:
    def __init__(self, fp: BinaryIO):
        self.fp = fp

    @property
    def offset(self) -> int:
        return self.fp.tell()

    def read_exact(self, count: int) -> bytes:
        if count < 0:
            raise GGUFError("negative read length")
        data = self.fp.read(count)
        if len(data) != count:
            raise GGUFError(
                f"truncated GGUF at byte {self.offset}: wanted {count}, got {len(data)}"
            )
        return data

    def unpack(self, fmt: str) -> Any:
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, self.read_exact(size))[0]

    def string(self, keep: bool = True) -> str | None:
        length = self.unpack("<Q")
        if length > (1 << 34):
            raise GGUFError(f"implausible string length {length}")
        raw = self.read_exact(length)
        if not keep:
            return None
        return raw.decode("utf-8", errors="replace")

    def value(self, value_type: int, keep: bool, depth: int = 0) -> Any:
        if depth > 4:
            raise GGUFError("metadata arrays are nested too deeply")
        if value_type in SCALAR_FORMATS:
            value = self.unpack(SCALAR_FORMATS[value_type])
            return value if keep else None
        if value_type == STRING:
            return self.string(keep)
        if value_type == ARRAY:
            element_type = self.unpack("<I")
            count = self.unpack("<Q")
            if count > (1 << 31):
                raise GGUFError(f"implausible metadata array length {count}")
            retain = keep and count <= 4096
            values = [] if retain else None
            for _ in range(count):
                item = self.value(element_type, retain, depth + 1)
                if retain:
                    values.append(item)
            return values
        raise GGUFError(f"unsupported GGUF metadata type {value_type}")


@dataclass
class Tensor:
    name: str
    dims: tuple[int, ...]
    type_id: int
    offset: int
    logical_bytes: int
    physical_bytes: int = 0

    @property
    def type_name(self) -> str:
        info = TENSOR_TYPES.get(self.type_id)
        return info[0] if info else f"TYPE_{self.type_id}"


@dataclass
class GGUF:
    version: int
    metadata: dict[str, Any]
    tensors: list[Tensor]
    alignment: int
    data_offset: int
    file_size: int | None


def keep_metadata_key(key: str) -> bool:
    suffixes = (
        "architecture",
        "name",
        "quantization_version",
        "alignment",
        "block_count",
        "context_length",
        "embedding_length",
        "vocab_size",
        "expert_count",
        "expert_used_count",
        "expert_feed_forward_length",
        "expert_shared_feed_forward_length",
        "leading_dense_block_count",
        "feed_forward_length",
        "head_count",
        "head_count_kv",
        "key_length",
        "value_length",
        "sliding_window",
    )
    return key.startswith("general.") or key.endswith(suffixes)


def tensor_logical_bytes(name: str, dims: tuple[int, ...], type_id: int) -> int:
    info = TENSOR_TYPES.get(type_id)
    if info is None:
        raise GGUFError(f"tensor {name!r} uses unknown GGML type {type_id}")
    _, block_elems, block_bytes = info
    elements = math.prod(dims)
    if elements % block_elems != 0:
        raise GGUFError(
            f"tensor {name!r} has {elements} elements, not divisible by "
            f"{block_elems} for {info[0]}"
        )
    return elements // block_elems * block_bytes


def parse_gguf(path: str, declared_file_size: int | None = None) -> GGUF:
    with open(path, "rb") as fp:
        reader = Reader(fp)
        if reader.read_exact(4) != GGUF_MAGIC:
            raise GGUFError("not a GGUF file")
        version = reader.unpack("<I")
        if version != GGUF_VERSION:
            raise GGUFError(f"unsupported GGUF version {version}; expected 3")
        tensor_count = reader.unpack("<Q")
        metadata_count = reader.unpack("<Q")
        if tensor_count > 10_000_000 or metadata_count > 1_000_000:
            raise GGUFError("implausible GGUF directory counts")

        metadata: dict[str, Any] = {}
        for _ in range(metadata_count):
            key = reader.string(True)
            assert key is not None
            value_type = reader.unpack("<I")
            keep = keep_metadata_key(key)
            value = reader.value(value_type, keep)
            if keep:
                metadata[key] = value

        tensors: list[Tensor] = []
        names: set[str] = set()
        for _ in range(tensor_count):
            name = reader.string(True)
            assert name is not None
            if name in names:
                raise GGUFError(f"duplicate tensor name {name!r}")
            names.add(name)
            n_dims = reader.unpack("<I")
            if n_dims == 0 or n_dims > 8:
                raise GGUFError(f"tensor {name!r} has invalid rank {n_dims}")
            dims = tuple(reader.unpack("<Q") for _ in range(n_dims))
            if any(dim == 0 for dim in dims):
                raise GGUFError(f"tensor {name!r} has a zero dimension")
            type_id = reader.unpack("<I")
            offset = reader.unpack("<Q")
            tensors.append(
                Tensor(name, dims, type_id, offset,
                       tensor_logical_bytes(name, dims, type_id))
            )

        alignment = int(metadata.get("general.alignment", 32))
        if alignment <= 0 or alignment & (alignment - 1):
            raise GGUFError(f"invalid GGUF alignment {alignment}")
        data_offset = (reader.offset + alignment - 1) & ~(alignment - 1)

    local_size = os.path.getsize(path)
    file_size = declared_file_size
    if file_size is None and local_size > data_offset:
        # A complete local file can be recognized because every logical tensor
        # lies inside it. A header prefix generally cannot.
        max_end = max((t.offset + t.logical_bytes for t in tensors), default=0)
        if data_offset + max_end <= local_size:
            file_size = local_size

    ordered = sorted(tensors, key=lambda item: item.offset)
    for index, tensor in enumerate(ordered):
        if index + 1 < len(ordered):
            next_offset = ordered[index + 1].offset
            if next_offset < tensor.offset:
                raise GGUFError("tensor offsets are not monotonic")
            tensor.physical_bytes = next_offset - tensor.offset
        elif file_size is not None:
            end = file_size - data_offset
            if end < tensor.offset:
                raise GGUFError("declared file size ends before the last tensor")
            tensor.physical_bytes = end - tensor.offset

    return GGUF(version, metadata, tensors, alignment, data_offset, file_size)


def classify_tensor(name: str) -> str:
    if any(part in name for part in
           (".ffn_gate_exps.", ".ffn_up_exps.", ".ffn_down_exps.")):
        return "routed_experts"
    if "token_embd" in name:
        return "embedding"
    if name.startswith("output.") or name == "output_norm.weight":
        return "output"
    if any(part in name for part in
           ("ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp",
            "ffn_gate_shared", "ffn_up_shared", "ffn_down_shared")):
        return "shared_experts"
    if any(part in name for part in
           ("ffn_gate_inp", "ffn_router", "exp_probs_b", "expert_bias")):
        return "routers"
    if ".attn_" in name:
        return "attention"
    if any(part in name for part in
           (".ffn_gate.", ".ffn_up.", ".ffn_down.")):
        return "dense_ffn"
    if "norm" in name:
        return "norms"
    return "other"


def layer_number(name: str) -> int | None:
    if not name.startswith("blk."):
        return None
    fields = name.split(".", 2)
    if len(fields) < 3:
        return None
    try:
        return int(fields[1])
    except ValueError:
        return None


def human_bytes(value: int) -> str:
    return f"{value / (1024 ** 3):.3f} GiB"


def build_report(model: GGUF) -> dict[str, Any]:
    groups: dict[str, dict[str, int]] = {}
    types: dict[str, dict[str, int]] = {}
    routed_layers: dict[int, int] = {}
    for tensor in model.tensors:
        group = classify_tensor(tensor.name)
        entry = groups.setdefault(group, {"tensors": 0, "logical_bytes": 0,
                                           "physical_bytes": 0})
        entry["tensors"] += 1
        entry["logical_bytes"] += tensor.logical_bytes
        entry["physical_bytes"] += tensor.physical_bytes
        type_entry = types.setdefault(tensor.type_name,
                                      {"tensors": 0, "logical_bytes": 0})
        type_entry["tensors"] += 1
        type_entry["logical_bytes"] += tensor.logical_bytes
        if group == "routed_experts":
            layer = layer_number(tensor.name)
            if layer is not None:
                routed_layers[layer] = routed_layers.get(layer, 0) + tensor.logical_bytes

    total = sum(t.logical_bytes for t in model.tensors)
    routed = groups.get("routed_experts", {}).get("logical_bytes", 0)
    non_routed = total - routed
    architecture = model.metadata.get("general.architecture", "unknown")
    expert_count = int(model.metadata.get(f"{architecture}.expert_count", 0) or 0)
    expert_used = int(model.metadata.get(f"{architecture}.expert_used_count", 0) or 0)
    layer_rows = []
    for layer, size in sorted(routed_layers.items()):
        layer_rows.append({
            "layer": layer,
            "bytes": size,
            "bytes_per_expert": size // expert_count if expert_count else 0,
        })
    dominant_per_expert = 0
    max_per_expert = 0
    exact_one_token_bytes = 0
    max_layer_bytes = 0
    if layer_rows:
        counts: dict[int, int] = {}
        for row in layer_rows:
            size = row["bytes_per_expert"]
            counts[size] = counts.get(size, 0) + 1
            max_per_expert = max(max_per_expert, size)
            exact_one_token_bytes += size * expert_used
            max_layer_bytes = max(max_layer_bytes, row["bytes"])
        dominant_per_expert = max(counts, key=lambda size: (counts[size], -size))

    per_token_slots = len(layer_rows) * expert_used
    return {
        "version": model.version,
        "file_size": model.file_size,
        "data_offset": model.data_offset,
        "alignment": model.alignment,
        "metadata": model.metadata,
        "tensor_count": len(model.tensors),
        "logical_tensor_bytes": total,
        "routed_expert_bytes": routed,
        "non_routed_bytes": non_routed,
        "routed_fraction": routed / total if total else 0.0,
        "groups": groups,
        "types": types,
        "routed_layers": layer_rows,
        "expert_count": expert_count,
        "expert_used_count": expert_used,
        "dominant_bytes_per_expert": dominant_per_expert,
        "max_bytes_per_expert": max_per_expert,
        "one_token_expert_slots": per_token_slots,
        "one_token_expert_bytes": per_token_slots * dominant_per_expert,
        "exact_one_token_expert_bytes": exact_one_token_bytes,
        "two_full_routed_layers_bytes": 2 * max_layer_bytes,
    }


def laguna_runtime_plan(
    report: dict[str, Any],
    context: int,
    working_set_bytes: int | None,
    host_memory_bytes: int | None = None,
) -> dict[str, int | bool]:
    metadata = report["metadata"]
    if metadata.get("general.architecture") != "laguna":
        raise GGUFError("runtime context planning is currently implemented for Laguna")
    if context <= 0:
        raise GGUFError(f"invalid context {context}")

    layers = int(metadata.get("laguna.block_count", 0) or 0)
    embd = int(metadata.get("laguna.embedding_length", 0) or 0)
    vocab = int(metadata.get("laguna.vocab_size", 0) or 0)
    kv_heads = int(metadata.get("laguna.attention.head_count_kv", 0) or 0)
    key_len = int(metadata.get("laguna.attention.key_length", 0) or 0)
    value_len = int(metadata.get("laguna.attention.value_length", 0) or 0)
    sliding = int(metadata.get("laguna.attention.sliding_window", 0) or 0)
    expert_count = int(report["expert_count"])
    expert_used = int(report["expert_used_count"])
    expert_ff = int(metadata.get("laguna.expert_feed_forward_length", 0) or 0)
    dense_ff = int(metadata.get("laguna.feed_forward_length", 0) or 0)
    head_counts_value = metadata.get("laguna.attention.head_count", [])
    head_counts = (
        [int(value) for value in head_counts_value]
        if isinstance(head_counts_value, list)
        else []
    )
    if not head_counts:
        head_counts = [int(head_counts_value or 0)] * layers
    if len(head_counts) != layers or not all(head_counts):
        raise GGUFError("Laguna attention head-count metadata is incomplete")
    if not all((layers, embd, vocab, kv_heads, key_len, value_len, sliding,
                expert_count, expert_used, expert_ff, dense_ff)):
        raise GGUFError("Laguna runtime metadata is incomplete")

    max_heads = max(head_counts)
    swa_layers = sum(1 for count in head_counts if count == max_heads)
    global_layers = layers - swa_layers
    kv_row_bytes = kv_heads * (key_len + value_len) * 2
    kv_bytes = (
        global_layers * context * kv_row_bytes
        + swa_layers * min(context, sliding) * kv_row_bytes
    )

    # Mirrors laguna_graph_alloc's one-row SSD schedule.
    q_dim = max_heads * key_len
    kv_dim = kv_heads * key_len
    ffn_max = max(dense_ff, expert_used * expert_ff)
    row_f32 = (
        8 * embd + 2 * q_dim + 2 * kv_dim + max_heads
        + 3 * ffn_max + expert_used * expert_ff + 2 * expert_count
    )
    row_other = 4 + expert_used * 8 + 2 * kv_dim * 2
    fixed = 8 + (embd + vocab) * 4
    scratch_bytes = row_f32 * 4 + row_other + fixed
    graph_bytes = kv_bytes + scratch_bytes

    cache_experts = 0
    cache_bytes = 0
    host_cache_cap_bytes = 0
    fits = True
    if working_set_bytes is not None:
        max_expert = int(report["max_bytes_per_expert"])
        static_bytes = int(report["non_routed_bytes"])
        model_cache_cap = max(0, int(working_set_bytes * 0.80) - static_bytes)
        graph_cache_cap = max(
            0, int(working_set_bytes * 0.90) - static_bytes - graph_bytes
        )
        cache_bytes = min(model_cache_cap, graph_cache_cap)
        if host_memory_bytes is not None:
            if host_memory_bytes <= 0:
                raise GGUFError("host memory must be positive")
            host_cache_cap_bytes = host_memory_bytes // 6
            cache_bytes = min(cache_bytes, host_cache_cap_bytes)
        cache_experts = cache_bytes // max_expert if max_expert else 0
        max_model_experts = len(report["routed_layers"]) * expert_count
        cache_experts = min(cache_experts, max_model_experts)
        cache_bytes = cache_experts * max_expert
        # The address-table kernel must bind the complete routed top-k at
        # once. Fewer slots would disable streaming and fall back to a whole
        # expert tensor, violating the bounded-memory plan.
        fits = cache_experts >= expert_used

    return {
        "context": context,
        "global_layers": global_layers,
        "swa_layers": swa_layers,
        "kv_bytes": kv_bytes,
        "scratch_bytes": scratch_bytes,
        "graph_bytes": graph_bytes,
        "cache_experts": cache_experts,
        "cache_bytes": cache_bytes,
        "host_cache_cap_bytes": host_cache_cap_bytes,
        "projected_ds4_bytes":
            int(report["non_routed_bytes"]) + graph_bytes + cache_bytes,
        "fits": fits,
    }


def print_report(report: dict[str, Any]) -> None:
    metadata = report["metadata"]
    print(f"architecture: {metadata.get('general.architecture', 'unknown')}")
    print(f"name: {metadata.get('general.name', 'unknown')}")
    print(f"tensors: {report['tensor_count']}")
    if report["file_size"] is not None:
        print(f"file size: {human_bytes(report['file_size'])}")
    print(f"tensor data offset: {report['data_offset']} bytes")
    print(f"logical tensor bytes: {human_bytes(report['logical_tensor_bytes'])}")
    print(f"routed experts: {human_bytes(report['routed_expert_bytes'])} "
          f"({report['routed_fraction'] * 100:.1f}%)")
    print(f"non-routed floor: {human_bytes(report['non_routed_bytes'])}")
    print("\nTensor groups:")
    for name, entry in sorted(report["groups"].items(),
                              key=lambda item: item[1]["logical_bytes"],
                              reverse=True):
        print(f"  {name:18s} {entry['tensors']:4d}  "
              f"{human_bytes(entry['logical_bytes'])}")
    print("\nTensor types:")
    for name, entry in sorted(report["types"].items(),
                              key=lambda item: item[1]["logical_bytes"],
                              reverse=True):
        print(f"  {name:12s} {entry['tensors']:4d}  "
              f"{human_bytes(entry['logical_bytes'])}")
    if report["routed_layers"]:
        sizes: dict[int, int] = {}
        for row in report["routed_layers"]:
            sizes[row["bytes_per_expert"]] = sizes.get(row["bytes_per_expert"], 0) + 1
        print("\nRouted layer size classes:")
        for size, count in sorted(sizes.items()):
            print(f"  {count:3d} layers x {human_bytes(size)} per expert")
        print(f"  active experts/token/layer: {report['expert_used_count']}")
        print(f"  one-token selected bytes (exact): "
              f"{human_bytes(report['exact_one_token_expert_bytes'])}")
        print(f"  conservative one-token slab bytes: "
              f"{human_bytes(report['one_token_expert_bytes'])}")
        print(f"  two-full-layer prefill headroom: "
              f"{human_bytes(report['two_full_routed_layers_bytes'])}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gguf", help="complete GGUF or header prefix")
    parser.add_argument(
        "--file-size", type=int,
        help="authoritative complete object size when GGUF is a prefix")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    parser.add_argument(
        "--working-set-gib", type=float,
        help="Metal recommended working set; adds bounded Laguna cache plans")
    parser.add_argument(
        "--host-memory-gib", type=float,
        help="physical unified memory; caps automatic Laguna cache at one sixth")
    parser.add_argument(
        "--contexts", default="2048,4096,8192,16384,32768",
        help="comma-separated Laguna context sizes to plan")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        model = parse_gguf(args.gguf, args.file_size)
        report = build_report(model)
        if args.working_set_gib is not None and args.working_set_gib <= 0:
            raise GGUFError("--working-set-gib must be positive")
        if args.host_memory_gib is not None and args.host_memory_gib <= 0:
            raise GGUFError("--host-memory-gib must be positive")
        contexts = [int(value) for value in args.contexts.split(",") if value]
        if model.metadata.get("general.architecture") == "laguna":
            working_set_bytes = (
                int(args.working_set_gib * 1024 ** 3)
                if args.working_set_gib is not None else None
            )
            host_memory_bytes = (
                int(args.host_memory_gib * 1024 ** 3)
                if args.host_memory_gib is not None else None
            )
            report["runtime_plans"] = [
                laguna_runtime_plan(
                    report, context, working_set_bytes, host_memory_bytes)
                for context in contexts
            ]
    except (OSError, GGUFError, OverflowError, ValueError) as exc:
        print(f"model-memory-plan: {exc}", file=sys.stderr)
        return 1
    if args.json:
        json.dump(report, sys.stdout, indent=2, sort_keys=True)
        print()
    else:
        print_report(report)
        if "runtime_plans" in report:
            print("\nLaguna SSD runtime plan:")
            print("  context       KV    scratch      cache  experts  DS4 total  status")
            for plan in report["runtime_plans"]:
                cache = (human_bytes(int(plan["cache_bytes"]))
                         if args.working_set_gib is not None else "n/a")
                experts = (str(plan["cache_experts"])
                           if args.working_set_gib is not None else "n/a")
                status = "n/a"
                if args.working_set_gib is not None:
                    status = "ok" if bool(plan["fits"]) else "reject"
                print(
                    f"  {plan['context']:7d}  "
                    f"{human_bytes(int(plan['kv_bytes'])):>10s}  "
                    f"{human_bytes(int(plan['scratch_bytes'])):>10s}  "
                    f"{cache:>10s}  {experts:>7s}  "
                    f"{human_bytes(int(plan['projected_ds4_bytes'])):>10s}  "
                    f"{status}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
