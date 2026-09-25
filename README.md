# JAX CuTE FFI

A custom CUDA FFI framework for JAX, utilizing modern NVIDIA CuTe abstractions and XLA Typed FFI to deliver tiled, memory-efficient kernel implementations.

## Installation

```bash
uv pip install "jax[cuda13]" # [cuda13-local]
cmake -S . -B build
cmake --build build
```

## Test Basic Run

```bash
PYTHONPATH=$PWD uv run python3 ffi_tests/add_one.py
```

## Flash Attention

```bash
PYTHONPATH=$PWD uv run python3 ffi_tests/flash_attention.py
```
