import jax
import jax.numpy as jnp

from ffi_manager import FFIManager


FFIManager.register_target(
    target_name="add_one",
    symbol_name="AddOne",
    lib_name="libadd_one",
)


def add_one(x):
    if x.dtype != jnp.float32:
        raise ValueError("Only float32 dtype supported")
    out_type = jax.ShapeDtypeStruct.like(x)
    return FFIManager.call_ffi("add_one", (x,), out_type)


if __name__ == "__main__":
    x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float32)

    res = jax.jit(add_one)(x)

    print("Input: ", x)
    print("Output:", res)
    assert jnp.allclose(res, x + 1.0)
    print("Test Passed!")
