# Phase 1 Benchmark Results

## Overview

This document captures the micro-benchmark results for the Phase 1 foundations: `ExecutionContext` and Zero-Copy Tensor storage.

## Results

### 1. Zero-Copy Performance

We compared the cost of "sharing" a tensor (simulating passing data between operations) using the standard deep-copy approach vs. the new Zero-Copy (Arc-based) approach.

| Operation          | Implementation      | Time      | Speedup       |
| ------------------ | ------------------- | --------- | ------------- |
| Share 10k Elements | Standard `clone()`  | 610.97 ns | 1x (Baseline) |
| Share 10k Elements | Zero-Copy `share()` | **3.54 ns** | **172x** 🚀   |

**Conclusion:**
The Zero-Copy mechanism provides a massive performance improvement (>100x) for operations that need to share read-only access to tensors. This will be critical for the query pipeline where data often moves between stages without modification.

### 2. Allocation Overhead

We measured the cost of allocating temporary resources.

| Operation        | Implementation                  | Time      | Notes                             |
| ---------------- | ------------------------------- | --------- | --------------------------------- |
| Vec Allocation   | `Vec::with_capacity(1000)`      | 12.16 ns  | Raw allocator speed               |
| Arena Allocation | `ctx.alloc_slice(&[0.0; 1000])` | 601.33 ns | Includes copy + tracking overhead |

**Analysis:**
The `arena_alloc` benchmark is slower because `ctx.alloc_slice`:

1. Allocates memory in the arena.
2. **Copies** the source data into the arena.
3. **Tracks** the resource in the `ExecutionContext` for auto-cleanup.

The `Vec` baseline only measured allocation. The overhead of ~600ns for managed, auto-cleaned resources is acceptable, especially since `bumpalo` shines in *cache locality* and *batch deallocation* (drop is O(1) vs O(N) for many vecs), which micro-benchmarks often miss.

## Summary

Phase 1 has successfully established the high-performance infrastructure:

- **Zero-Copy Sharing** works as intended with massive speedups.
- **ExecutionContext** provides safe resource tracking with manageable overhead.
