# LINAL Performance Benchmarks

This document tracks performance benchmarks for LINAL across different phases of optimization.

## Baseline Results (Phase 0)

**Date**: 2025-12-26  
**Hardware**: TBD (run `cargo bench` to populate)  
**Rust version**: TBD  

### Tensor Operations

| Operation | Size | Time (μs) | Throughput | Notes |
|-----------|------|-----------|------------|-------|
| Dot Product | 100 | TBD | TBD | Baseline |
| Dot Product | 1K | TBD | TBD | Baseline |
| Dot Product | 10K | TBD | TBD | Baseline |
| Dot Product | 100K | TBD | TBD | Baseline |
| MatMul | 10x10 | TBD | TBD | Baseline |
| MatMul | 50x50 | TBD | TBD | Baseline |
| MatMul | 100x100 | TBD | TBD | Baseline |
| Element-wise Add | 10K | TBD | TBD | Baseline |
| Element-wise Multiply | 10K | TBD | TBD | Baseline |
| Cosine Similarity | 100 | TBD | TBD | Baseline |
| Cosine Similarity | 1K | TBD | TBD | Baseline |
| Cosine Similarity | 10K | TBD | TBD | Baseline |

### Query Operations

| Query Type | Dataset Size | Time (ms) | Rows/sec | Notes |
|------------|--------------|-----------|----------|-------|
| SELECT * (full scan) | 100 | TBD | TBD | Baseline |
| SELECT * (full scan) | 1K | TBD | TBD | Baseline |
| SELECT * (full scan) | 10K | TBD | TBD | Baseline |
| SELECT with filter | 100 | TBD | TBD | Baseline |
| SELECT with filter | 1K | TBD | TBD | Baseline |
| SELECT with filter | 10K | TBD | TBD | Baseline |
| SELECT with index | 100 | TBD | TBD | Baseline |
| SELECT with index | 1K | TBD | TBD | Baseline |
| SELECT with index | 10K | TBD | TBD | Baseline |
| SUM aggregation | 10K | TBD | TBD | Baseline |
| GROUP BY aggregation | 10K | TBD | TBD | Baseline |
| Single INSERT | 1 | TBD | TBD | Baseline |

## Performance Goals

### Phase 1: Foundations

- **Goal**: No regression (< 5% acceptable variance)
- **Focus**: Infrastructure setup without performance impact
- **Metrics**:
  - Arena allocation reduces allocation count by 30%+
  - ExecutionContext overhead < 1%

### Phase 2: Zero-Copy

- **Goal**: 50% reduction in tensor memory allocations
- **Focus**: Arc-based tensor sharing
- **Metrics**:
  - Memory allocations reduced by 50%+
  - Tensor operations 20%+ faster due to reduced copying

### Phase 3: Execution

- **Goal**: 2-5x speedup for batch-friendly queries
- **Focus**: Batch execution and SIMD
- **Metrics**:
  - Batched execution 2-5x faster for large datasets (>10K rows)
  - SIMD kernels 2-4x faster for vector operations
  - Query latency P50 improved by 30%+

### Phase 4: Server

- **Goal**: Stable under 1000+ concurrent requests
- **Focus**: Resource limits and concurrency control
- **Metrics**:
  - Server handles 1000+ concurrent requests without OOM
  - Graceful degradation under overload
  - Configurable limits prevent resource exhaustion

## Running Benchmarks

### Run all benchmarks

```bash
cargo bench
```

### Run specific benchmark suite

```bash
cargo bench --bench tensor_ops
cargo bench --bench queries
```

### Generate HTML reports

```bash
cargo bench
# Reports available in target/criterion/report/index.html
```

### Compare with baseline

```bash
# Save baseline
cargo bench -- --save-baseline phase0

# After changes, compare
cargo bench -- --baseline phase0
```

## Benchmark Variance Guidelines

- **Acceptable variance**: < 5% between runs
- **Regression threshold**: > 10% slower triggers investigation
- **Improvement threshold**: > 20% faster is significant

## Notes

- Benchmarks should be run on a quiet system (no other heavy processes)
- Run multiple times to ensure consistency
- Document any system-specific factors (CPU throttling, background processes, etc.)
- Update this document after each phase with actual results
