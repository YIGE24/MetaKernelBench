<!-- source: https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_struct_types.html -->

<a id="struct-like-jit-arguments"></a>
# Struct-like JIT Arguments

CuTe DSL supports several struct-like Python types as JIT function arguments.
Each provides a different trade-off between mutability, syntax convenience,
and low-level control.

On this page

- [Overview](#overview)
- [NamedTuple](#namedtuple)

  - [Basic usage](#basic-usage)
  - [Control flow on fields](#control-flow-on-fields)
  - [Creating a new NamedTuple value inside the kernel](#creating-a-new-namedtuple-value-inside-the-kernel)
- [`@native_struct`](#native-struct)
- [Choosing the right type](#choosing-the-right-type)
- [See also](#see-also)

<a id="overview"></a>
## Overview

| Type | Mutable fields? | Notes |
| --- | --- | --- |
| `typing.NamedTuple` | **No** | Tuple subclass — fields fixed at construction. Flattened field-by-field through the pytree system. |
| `@native_struct` | **Yes** | Generates an LLVM struct type. `llvm.insertvalue` replaces field values in-place. |
| `@dataclass(frozen=True)` | **No** | Frozen dataclass — treated as a read-only pytree container, similar to `NamedTuple`. |

<a id="namedtuple"></a>
## NamedTuple

A `typing.NamedTuple` whose fields are DSL scalar types (`Int32`,
`Float32`, etc.) can be passed directly to `@cute.jit` /
`cute.compile` without any boilerplate or protocol implementation.

**How it works.** NamedTuples are registered as pytree containers in the DSL
tree system. Each field is flattened individually through the existing DSL
type paths and reconstructed by calling the NamedTuple constructor on the way
into the kernel body. Field attribute access (`tup.a`, `tup.b`, …)
works exactly as in native Python.

<a id="basic-usage"></a>
### Basic usage
```python
from typing import NamedTuple
import cutlass
import cutlass.cute as cute

class Vec3(NamedTuple):
    x: cutlass.Int32
    y: cutlass.Int32
    z: cutlass.Int32

@cute.jit
def print_vec(v: Vec3):
    cute.printf("x=%d y=%d z=%d\n", v.x, v.y, v.z)

v = Vec3(x=cutlass.Int32(1), y=cutlass.Int32(2), z=cutlass.Int32(3))
cute.compile(print_vec, v)(v)
```

<a id="control-flow-on-fields"></a>
### Control flow on fields

Fields are DSL values inside the kernel, so they work in `if`/`else`
branches and `for` loops:

```python
@cute.jit
def clamp_positive(v: Vec3, out: cute.Tensor):
    """Write max(field, 0) for each component."""
    out[0] = cutlass.Int32(0) if v.x < cutlass.Int32(0) else v.x
    out[1] = cutlass.Int32(0) if v.y < cutlass.Int32(0) else v.y
    out[2] = cutlass.Int32(0) if v.z < cutlass.Int32(0) else v.z

@cute.jit
def triangular_sum(v: Vec3, out: cute.Tensor):
    """Sum 0..v.x-1 into out[0], and so on."""
    s = cutlass.Int32(0)
    for i in range(v.x):
        s = s + i
    out[0] = s
```

<a id="creating-a-new-namedtuple-value-inside-the-kernel"></a>
### Creating a new NamedTuple value inside the kernel

NamedTuple fields are **immutable** — the same constraint as native Python
tuples. Assigning `tup.x = ...` inside a kernel raises `AttributeError`.
To “update” a field, construct a replacement NamedTuple:

```python
@cute.jit
def scale(v: Vec3, factor: cutlass.Int32, out: cute.Tensor):
    # Construct a new Vec3 with all fields scaled
    scaled = Vec3(x=v.x * factor, y=v.y * factor, z=v.z * factor)
    out[0] = scaled.x
    out[1] = scaled.y
    out[2] = scaled.z
```

<a id="native-struct"></a>
## @native_struct

Use `@native_struct` when kernel logic needs to **accumulate into or
update** a struct field. Unlike NamedTuple, fields are mutable: each write
generates an `llvm.insertvalue` to replace the field in the underlying LLVM
struct.

```python
import cutlass
import cutlass.cute as cute

@cute.native_struct
class Accumulator:
    total: cutlass.Int32
    count: cutlass.Int32

@cute.jit
def accumulate(acc: Accumulator, values: cute.Tensor, n: cutlass.Int32):
    for i in range(n):
        acc.total = acc.total + values[i]
        acc.count = acc.count + cutlass.Int32(1)
```

`@native_struct` also supports:

- `zero_init=False` — initialize with `llvm.mlir.undef` instead of zero.
- `packed=True` — create a packed LLVM struct (no padding between fields).
- `Constexpr` fields — excluded from the native struct and passed as
  ordinary Python values.

<a id="choosing-the-right-type"></a>
## Choosing the right type

| Use case | Recommended type |
| --- | --- |
| Read-only config / parameters passed into a kernel | `NamedTuple` or `@dataclass(frozen=True)` |
| Accumulator or running state updated inside a kernel | `@native_struct` |
| Want Python-native immutable semantics (hashable, unpackable) | `NamedTuple` |
| Need fine-grained LLVM struct control (packing, zero-init) | `@native_struct` |

<a id="see-also"></a>
## See also

- [JIT Function Argument Generation](cutedsl_cute_dsl_general_dsl_jit_arg_generation.md) — overview of JIT function argument protocols
- [Static vs Dynamic layouts](cutedsl_cute_dsl_general_dsl_dynamic_layout.md) — passing `Layout` objects as JIT arguments
