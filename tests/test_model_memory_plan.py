#!/usr/bin/env python3
"""Unit tests for the header-only GGUF memory planner."""

from __future__ import annotations

import importlib.util
import struct
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "model_memory_plan", ROOT / "gguf-tools" / "model_memory_plan.py"
)
assert SPEC and SPEC.loader
PLAN = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PLAN
SPEC.loader.exec_module(PLAN)


def string(value: str) -> bytes:
    raw = value.encode()
    return struct.pack("<Q", len(raw)) + raw


def metadata_value(key: str, type_id: int, value: object) -> bytes:
    out = bytearray(string(key))
    out += struct.pack("<I", type_id)
    if type_id == PLAN.STRING:
        out += string(str(value))
    elif type_id == PLAN.U32:
        out += struct.pack("<I", int(value))
    elif type_id == PLAN.ARRAY:
        values = list(value)  # type: ignore[arg-type]
        out += struct.pack("<IQ", PLAN.U32, len(values))
        out += b"".join(struct.pack("<I", int(item)) for item in values)
    else:
        raise AssertionError(f"unsupported test metadata type {type_id}")
    return bytes(out)


def synthetic_gguf() -> bytes:
    metadata = [
        ("general.architecture", PLAN.STRING, "laguna"),
        ("general.name", PLAN.STRING, "Synthetic Laguna"),
        ("general.alignment", PLAN.U32, 32),
        ("laguna.block_count", PLAN.U32, 2),
        ("laguna.context_length", PLAN.U32, 32),
        ("laguna.embedding_length", PLAN.U32, 256),
        ("laguna.vocab_size", PLAN.U32, 32),
        ("laguna.expert_count", PLAN.U32, 4),
        ("laguna.expert_used_count", PLAN.U32, 2),
        ("laguna.expert_feed_forward_length", PLAN.U32, 256),
        ("laguna.feed_forward_length", PLAN.U32, 512),
        ("laguna.attention.head_count", PLAN.ARRAY, [3, 6]),
        ("laguna.attention.head_count_kv", PLAN.U32, 2),
        ("laguna.attention.key_length", PLAN.U32, 128),
        ("laguna.attention.value_length", PLAN.U32, 128),
        ("laguna.attention.sliding_window", PLAN.U32, 4),
    ]
    tensor_specs = [("token_embd.weight", (256,), 8)]
    for layer, type_id in ((1, 10), (2, 11)):
        for projection in ("gate", "up", "down"):
            tensor_specs.append(
                (f"blk.{layer}.ffn_{projection}_exps.weight", (256, 2, 4), type_id)
            )

    header = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(tensor_specs), len(metadata)))
    for item in metadata:
        header += metadata_value(*item)

    offsets: list[int] = []
    cursor = 0
    for name, dims, type_id in tensor_specs:
        cursor = (cursor + 31) & ~31
        offsets.append(cursor)
        cursor += PLAN.tensor_logical_bytes(name, dims, type_id)
    for (name, dims, type_id), offset in zip(tensor_specs, offsets):
        header += string(name)
        header += struct.pack("<I", len(dims))
        header += b"".join(struct.pack("<Q", dim) for dim in dims)
        header += struct.pack("<IQ", type_id, offset)

    data_offset = (len(header) + 31) & ~31
    header += bytes(data_offset - len(header))
    header += bytes(cursor)
    return bytes(header)


def synthetic_deepseek4_gguf() -> bytes:
    metadata = [
        ("general.architecture", PLAN.STRING, "deepseek4"),
        ("general.name", PLAN.STRING, "Synthetic DeepSeek V4 Flash"),
        ("general.alignment", PLAN.U32, 32),
        ("deepseek4.block_count", PLAN.U32, 4),
        ("deepseek4.context_length", PLAN.U32, 1024),
        ("deepseek4.embedding_length", PLAN.U32, 256),
        ("deepseek4.vocab_size", PLAN.U32, 32),
        ("deepseek4.expert_count", PLAN.U32, 4),
        ("deepseek4.expert_used_count", PLAN.U32, 2),
        ("deepseek4.expert_feed_forward_length", PLAN.U32, 256),
        ("deepseek4.attention.head_count", PLAN.U32, 8),
        ("deepseek4.attention.head_count_kv", PLAN.U32, 1),
        ("deepseek4.attention.key_length", PLAN.U32, 64),
        ("deepseek4.attention.value_length", PLAN.U32, 64),
        ("deepseek4.attention.sliding_window", PLAN.U32, 8),
        ("deepseek4.attention.output_group_count", PLAN.U32, 2),
        ("deepseek4.attention.q_lora_rank", PLAN.U32, 32),
        ("deepseek4.attention.output_lora_rank", PLAN.U32, 16),
        ("deepseek4.attention.indexer.head_count", PLAN.U32, 4),
        ("deepseek4.attention.indexer.key_length", PLAN.U32, 16),
        ("deepseek4.attention.indexer.top_k", PLAN.U32, 8),
        ("deepseek4.hyper_connection.count", PLAN.U32, 2),
        ("deepseek4.hyper_connection.sinkhorn_iterations", PLAN.U32, 4),
        ("deepseek4.attention.compress_ratios", PLAN.ARRAY, [0, 0, 4, 128]),
    ]
    tensor_specs = [("token_embd.weight", (256,), 8)]
    for layer in range(4):
        for projection in ("gate", "up", "down"):
            tensor_specs.append(
                (f"blk.{layer}.ffn_{projection}_exps.weight", (256, 2, 4), 10)
            )

    header = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(tensor_specs), len(metadata)))
    for item in metadata:
        header += metadata_value(*item)

    offsets: list[int] = []
    cursor = 0
    for name, dims, type_id in tensor_specs:
        cursor = (cursor + 31) & ~31
        offsets.append(cursor)
        cursor += PLAN.tensor_logical_bytes(name, dims, type_id)
    for (name, dims, type_id), offset in zip(tensor_specs, offsets):
        header += string(name)
        header += struct.pack("<I", len(dims))
        header += b"".join(struct.pack("<Q", dim) for dim in dims)
        header += struct.pack("<IQ", type_id, offset)

    data_offset = (len(header) + 31) & ~31
    header += bytes(data_offset - len(header))
    header += bytes(cursor)
    return bytes(header)


class ModelMemoryPlanTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "model.gguf"
        self.payload = synthetic_gguf()
        self.path.write_bytes(self.payload)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_mixed_routed_classes_use_exact_and_max_accounting(self) -> None:
        model = PLAN.parse_gguf(str(self.path))
        report = PLAN.build_report(model)
        self.assertEqual(report["tensor_count"], 7)
        self.assertEqual(report["expert_count"], 4)
        self.assertEqual(report["expert_used_count"], 2)
        self.assertEqual(report["max_bytes_per_expert"], 660)
        self.assertEqual(report["exact_one_token_expert_bytes"], 2328)
        self.assertEqual(report["two_full_routed_layers_bytes"], 5280)
        self.assertGreater(report["non_routed_bytes"], 0)

    def test_declared_size_allows_header_prefix_planning(self) -> None:
        full = PLAN.parse_gguf(str(self.path))
        prefix_path = Path(self.tmp.name) / "prefix.gguf"
        prefix_path.write_bytes(self.payload[: full.data_offset])
        prefix = PLAN.parse_gguf(str(prefix_path), len(self.payload))
        self.assertEqual(prefix.file_size, len(self.payload))
        self.assertGreater(prefix.tensors[-1].physical_bytes, 0)

    def test_laguna_context_plan_counts_global_and_sliding_kv(self) -> None:
        report = PLAN.build_report(PLAN.parse_gguf(str(self.path)))
        runtime = PLAN.laguna_runtime_plan(report, 8, 1 << 30)
        self.assertEqual(runtime["global_layers"], 1)
        self.assertEqual(runtime["swa_layers"], 1)
        self.assertEqual(runtime["kv_bytes"], 12_288)
        self.assertGreater(runtime["scratch_bytes"], 0)
        self.assertTrue(runtime["fits"])

    def test_tiny_working_set_cannot_hold_one_selected_set(self) -> None:
        report = PLAN.build_report(PLAN.parse_gguf(str(self.path)))
        runtime = PLAN.laguna_runtime_plan(report, 8, 1024)
        self.assertEqual(runtime["cache_experts"], 0)
        self.assertFalse(runtime["fits"])

    def test_host_memory_caps_automatic_cache_to_one_sixth(self) -> None:
        report = PLAN.build_report(PLAN.parse_gguf(str(self.path)))
        runtime = PLAN.laguna_runtime_plan(
            report, 8, 1 << 30, host_memory_bytes=7920
        )
        self.assertEqual(runtime["host_cache_cap_bytes"], 1320)
        self.assertEqual(runtime["cache_experts"], 2)
        self.assertEqual(runtime["cache_bytes"], 1320)
        self.assertTrue(runtime["fits"])

    def test_truncated_directory_is_rejected(self) -> None:
        broken = Path(self.tmp.name) / "broken.gguf"
        broken.write_bytes(self.payload[:64])
        with self.assertRaises(PLAN.GGUFError):
            PLAN.parse_gguf(str(broken))

    def test_deepseek4_context_and_auto_cache_match_runtime_geometry(self) -> None:
        flash_path = Path(self.tmp.name) / "flash.gguf"
        flash_path.write_bytes(synthetic_deepseek4_gguf())
        report = PLAN.build_report(PLAN.parse_gguf(str(flash_path)))
        runtime = PLAN.deepseek4_runtime_plan(report, 32, 1 << 30)
        self.assertEqual(runtime["prefill_cap"], 32)
        self.assertEqual(runtime["raw_cap"], 256)
        self.assertEqual(runtime["raw_bytes"], 262_144)
        self.assertEqual(runtime["compressed_bytes"], 2_176)
        self.assertEqual(runtime["context_scratch_bytes"], 5_120)
        self.assertGreater(runtime["metal_workspace_bytes"], 5_120)
        self.assertGreater(runtime["graph_state_bytes"], 0)
        self.assertEqual(
            runtime["graph_bytes"],
            runtime["kv_bytes"] + runtime["scratch_bytes"],
        )
        self.assertEqual(runtime["cache_experts"], 8)
        self.assertEqual(runtime["cache_bytes"], 4_032)
        self.assertTrue(runtime["fast_prefill"])
        self.assertTrue(runtime["one_token_resident"])
        self.assertFalse(runtime["warning_free"])
        self.assertTrue(runtime["fits"])

    def test_deepseek4_rejects_budget_below_fast_prefill_floor(self) -> None:
        flash_path = Path(self.tmp.name) / "flash.gguf"
        flash_path.write_bytes(synthetic_deepseek4_gguf())
        report = PLAN.build_report(PLAN.parse_gguf(str(flash_path)))
        runtime = PLAN.deepseek4_runtime_plan(report, 32, 1_024)
        self.assertEqual(runtime["cache_experts"], 0)
        self.assertFalse(runtime["fast_prefill"])
        self.assertFalse(runtime["fits"])

    def test_deepseek4_bounded_prefill_reduces_graph_workspace(self) -> None:
        flash_path = Path(self.tmp.name) / "flash.gguf"
        flash_path.write_bytes(synthetic_deepseek4_gguf())
        report = PLAN.build_report(PLAN.parse_gguf(str(flash_path)))
        full = PLAN.deepseek4_runtime_plan(report, 1024, 1 << 30)
        bounded = PLAN.deepseek4_runtime_plan(
            report, 1024, 1 << 30, prefill_chunk=8
        )
        self.assertEqual(bounded["prefill_cap"], 8)
        self.assertLess(bounded["raw_cap"], full["raw_cap"])
        self.assertLess(
            bounded["metal_workspace_bytes"], full["metal_workspace_bytes"]
        )
        self.assertLess(bounded["graph_bytes"], full["graph_bytes"])

    def test_deepseek4_manual_budget_reserves_prefill_layers_first(self) -> None:
        flash_path = Path(self.tmp.name) / "flash.gguf"
        flash_path.write_bytes(synthetic_deepseek4_gguf())
        report = PLAN.build_report(PLAN.parse_gguf(str(flash_path)))
        per_expert = report["dominant_bytes_per_expert"]
        reserve = report["two_full_routed_layers_bytes"]
        requested = reserve + 3 * per_expert
        runtime = PLAN.deepseek4_runtime_plan(
            report,
            32,
            1 << 30,
            expert_budget_bytes=requested,
        )
        self.assertEqual(runtime["total_expert_budget_bytes"], requested)
        self.assertEqual(runtime["cache_experts"], 3)
        self.assertEqual(runtime["cache_bytes"], 3 * per_expert)

    def test_deepseek4_exact_cache_count_does_not_spend_dynamic_slots_on_reserve(self) -> None:
        flash_path = Path(self.tmp.name) / "flash.gguf"
        flash_path.write_bytes(synthetic_deepseek4_gguf())
        report = PLAN.build_report(PLAN.parse_gguf(str(flash_path)))
        per_expert = report["dominant_bytes_per_expert"]
        runtime = PLAN.deepseek4_runtime_plan(
            report,
            32,
            1 << 30,
            expert_cache_count=3,
        )
        self.assertEqual(runtime["requested_expert_cache_count"], 3)
        self.assertEqual(runtime["cache_experts"], 3)
        self.assertEqual(runtime["cache_bytes"], 3 * per_expert)
        self.assertEqual(runtime["prefill_headroom_bytes"], 0)
        self.assertGreater(runtime["possible_prefill_headroom_bytes"], 0)

    def test_deepseek4_working_set_guard_rejects_edge_fit(self) -> None:
        flash_path = Path(self.tmp.name) / "flash.gguf"
        flash_path.write_bytes(synthetic_deepseek4_gguf())
        report = PLAN.build_report(PLAN.parse_gguf(str(flash_path)))
        unconstrained = PLAN.deepseek4_runtime_plan(
            report,
            32,
            None,
            expert_cache_count=8,
        )
        conservative = unconstrained["conservative_unified_bytes"]
        runtime = PLAN.deepseek4_runtime_plan(
            report,
            32,
            conservative + conservative // 100,
            expert_cache_count=8,
        )
        self.assertLess(runtime["conservative_unified_bytes"],
                        conservative + conservative // 100)
        self.assertFalse(runtime["fits_working_set"])


if __name__ == "__main__":
    unittest.main()
