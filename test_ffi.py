import ctypes
import os
import jax
import jax.ffi
import jax.numpy as jnp

# 1. Load compiled shared library via ctypes
lib_path = os.path.abspath("build/libjax_fused_ops.so")  # Point to your build target
lib = ctypes.cdll.LoadLibrary(lib_path)

# 2. Convert C-symbol into PyCapsule via built-in jax.ffi.pycapsule helper
capsule = jax.ffi.pycapsule(lib.kAddOne)

# 3. Register symbol target into JAX XLA FFI registry
jax.ffi.register_ffi_target(
    name="add_one",
    fn=capsule,
    platform="CUDA"
)

# 4. FFI invocation function wrapper
def add_one(x):
    out_shape = jax.ShapeDtypeStruct(x.shape, x.dtype)
    # jax.ffi.ffi_call returns a callable function, which we execute on `x`
    return jax.ffi.ffi_call("add_one", out_shape)(x)

if __name__ == "__main__":
    x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float32)
    
    # Test execution within JIT context
    jit_add_one = jax.jit(add_one)
    res = jit_add_one(x)
    
    print("Input: ", x)
    print("Output:", res)
    assert jnp.allclose(res, x + 1.0)
    print("Execution verified successfully!")