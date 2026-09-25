import ctypes
from pathlib import Path

import jax
import jax.ffi


class FFIManager:
    """Central manager for loading shared libraries and making JAX FFI calls."""

    _loaded_libraries: dict[str, ctypes.CDLL] = {}
    _registered_targets: set = set()

    @classmethod
    def load_library(
        cls, lib_name: str, search_dir: str | Path | None = None
    ) -> ctypes.CDLL:
        """Loads a compiled shared library once and caches the handle."""
        if lib_name in cls._loaded_libraries:
            return cls._loaded_libraries[lib_name]

        if search_dir is None:
            search_dir = Path(__file__).parent

        lib_path = Path(search_dir) / f"{lib_name}.so"
        if not lib_path.exists():
            raise FileNotFoundError(f"Shared library not found at: {lib_path}")

        lib_handle = ctypes.cdll.LoadLibrary(str(lib_path))
        cls._loaded_libraries[lib_name] = lib_handle
        return lib_handle

    @classmethod
    def register_target(
        cls,
        target_name: str,
        symbol_name: str,
        lib_name: str,
        platform: str = "CUDA",
    ) -> None:
        """Registers an XLA FFI target using jax.ffi.pycapsule."""
        if target_name in cls._registered_targets:
            return

        lib = cls.load_library(lib_name, search_dir="lib")
        symbol = getattr(lib, symbol_name, None)
        if symbol is None:
            raise AttributeError(
                f"Symbol '{symbol_name}' not found in library '{lib_name}'"
            )

        jax.ffi.register_ffi_target(
            target_name,
            jax.ffi.pycapsule(symbol),
            platform=platform,
        )
        cls._registered_targets.add(target_name)

    @classmethod
    def call_ffi(
        cls,
        target_name: str,
        inputs: tuple[jax.Array, ...],
        out_shapes_dtypes: jax.ShapeDtypeStruct | tuple[jax.ShapeDtypeStruct, ...],
        **jax_ffi_kwargs,
    ) -> jax.Array | tuple[jax.Array, ...]:
        """Invokes a registered FFI custom target."""
        if target_name not in cls._registered_targets:
            raise RuntimeError(
                f"Target '{target_name}' is not registered. Call `register_target` first."
            )

        call_op = jax.ffi.ffi_call(
            target_name,
            out_shapes_dtypes,
        )
        return call_op(*inputs, **jax_ffi_kwargs)
