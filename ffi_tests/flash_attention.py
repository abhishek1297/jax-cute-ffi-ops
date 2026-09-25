import math
import sys
import time

import jax
import jax.numpy as jnp

from ffi_manager import FFIManager


FFIManager.register_target(
    target_name="flash_attention_v2",
    symbol_name="FlashAttention",
    lib_name="libflash_attention",
)


def jax_flash_attention_reference(q, k, v, sm_scale):
    scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) * sm_scale
    attn_weights = jax.nn.softmax(scores, axis=-1)
    return jnp.einsum("bhqk,bhkd->bhqd", attn_weights, v)


def flash_attention_ffi(q, k, v, sm_scale=None):
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])
    out_struct = jax.ShapeDtypeStruct(q.shape, q.dtype)
    return FFIManager.call_ffi(
        "flash_attention_v2",
        inputs=(q, k, v),
        out_shapes_dtypes=out_struct,
        sm_scale=float(sm_scale),
    )


jit_jax_ref = jax.jit(jax_flash_attention_reference)
jit_ffi_kernel = jax.jit(flash_attention_ffi, static_argnames=("sm_scale",))


def compute_tflops(
    batch: int, heads: int, seq_len: int, dim: int, time_ms: float
) -> float:
    flops = 4 * batch * heads * (seq_len**2) * dim
    tflops = (flops / (time_ms / 1000.0)) / 1e12
    return tflops


def verify_correctness(batch=2, heads=4, seq_len=128, dim=64, atol=1e-4, rtol=1e-4):
    print("=" * 80)
    print(f"Running Correctness Check [Shape: {batch}x{heads}x{seq_len}x{dim}]...")

    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)

    q = jax.random.normal(k1, (batch, heads, seq_len, dim), dtype=jnp.float32)
    k = jax.random.normal(k2, (batch, heads, seq_len, dim), dtype=jnp.float32)
    v = jax.random.normal(k3, (batch, heads, seq_len, dim), dtype=jnp.float32)
    sm_scale = float(1.0 / math.sqrt(dim))

    out_ref = jit_jax_ref(q, k, v, sm_scale).block_until_ready()
    out_ffi = jit_ffi_kernel(q, k, v, sm_scale).block_until_ready()

    max_diff = float(jnp.max(jnp.abs(out_ref - out_ffi)))
    mean_diff = float(jnp.mean(jnp.abs(out_ref - out_ffi)))

    print(f"  Max Absolute Difference : {max_diff:.6e}")
    print(f"  Mean Absolute Difference: {mean_diff:.6e}")

    if not jnp.allclose(out_ref, out_ffi, atol=atol, rtol=rtol):
        print("❌ Correctness check FAILED! Divergence exceeds threshold.")
        sys.exit(1)

    print("✅ Correctness check PASSED! Proceeding to benchmark scaling loop.")
    print("=" * 80)


def run_benchmark():
    verify_correctness()

    batch = 2
    heads = 8
    dim = 64
    seq_lengths = [512, 1024, 2048, 4096, 8192]
    warmup_iters = 5
    bench_iters = 50

    print(
        f"{'SeqLen':<10} | {'JAX Ref (ms)':<15} | {'FFI Kernel (ms)':<15} | {'Speedup':<10} | {'FFI TFLOPS':<10}"
    )
    print("=" * 80)

    key = jax.random.PRNGKey(42)

    for N in seq_lengths:
        key, k1, k2, k3 = jax.random.split(key, 4)
        q = jax.random.normal(k1, (batch, heads, N, dim), dtype=jnp.float32)
        k = jax.random.normal(k2, (batch, heads, N, dim), dtype=jnp.float32)
        v = jax.random.normal(k3, (batch, heads, N, dim), dtype=jnp.float32)
        sm_scale = float(1.0 / math.sqrt(dim))

        try:
            _ = jit_jax_ref(q, k, v, sm_scale).block_until_ready()
            run_jax = True
        except jax.errors.JaxRuntimeError:
            run_jax = False

        _ = jit_ffi_kernel(q, k, v, sm_scale).block_until_ready()

        if run_jax:
            for _ in range(warmup_iters):
                _ = jit_jax_ref(q, k, v, sm_scale).block_until_ready()

            t0 = time.perf_counter()
            for _ in range(bench_iters):
                _ = jit_jax_ref(q, k, v, sm_scale).block_until_ready()
            t1 = time.perf_counter()
            jax_time_ms = ((t1 - t0) / bench_iters) * 1000.0
            jax_str = f"{jax_time_ms:.3f}"
        else:
            jax_time_ms = float("inf")
            jax_str = "OOM"

        for _ in range(warmup_iters):
            _ = jit_ffi_kernel(q, k, v, sm_scale).block_until_ready()

        t0 = time.perf_counter()
        for _ in range(bench_iters):
            _ = jit_ffi_kernel(q, k, v, sm_scale).block_until_ready()
        t1 = time.perf_counter()
        ffi_time_ms = ((t1 - t0) / bench_iters) * 1000.0

        speedup = (jax_time_ms / ffi_time_ms) if run_jax else float("nan")
        ffi_tflops = compute_tflops(batch, heads, N, dim, ffi_time_ms)

        speedup_str = f"{speedup:.2f}x" if run_jax else "N/A"
        print(
            f"{N:<10} | {jax_str:<15} | {ffi_time_ms:<15.3f} | {speedup_str:<10} | {ffi_tflops:<10.2f}"
        )

    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()