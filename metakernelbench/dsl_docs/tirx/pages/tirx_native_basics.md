<!-- source: https://tvm.apache.org/docs/tirx/native_basics.html -->

<a id="tirx-basics-cuda-c-ptx-native-level"></a>
# TIRx Basics: CUDA C++/PTX native level
> **Note**
>
> Native-level kernel authoring for the **CUDA backend** (the `"cuda"`
> target): the thread hierarchy, memory scopes, the `T.cuda.*` / `T.ptx.*`
> intrinsics, and the compile / run / inspect loop. The complete kernels in
> these chapters (`scale`, `add`, `smem_demo`, `block_sum`, and the
> warp all-reduce) are tested end-to-end on a CUDA GPU.

<a id="what-native-level-means"></a>
## What “native level” means

A native-level TIRx kernel reads like a structured device kernel: you place
threads yourself, allocate shared/register buffers, write loops and barriers, and
call device intrinsics directly. There is no automatic scheduling — what you write
is what is emitted. This is the foundation the tile primitives
([Tile Primitives](tirx_tile_primitives.md)) are built on; everything here is what those primitives
ultimately lower to, so it is also where you go when a hardware feature does not
have a primitive yet.

<a id="the-authoring-model"></a>
## The authoring model

- `@T.prim_func` (or `@T.jit` for compile-time-specialized) kernels, written
  with `from tvm.script import tirx as T`;
- `T.device_entry()` plus *scope-id* intrinsics for thread binding;
- `T.match_buffer` parameters and `T.alloc_*` scratch buffers;
- ordinary loops, branches, and scalar math;
- `tvm.compile(mod, target=..., tir_pipeline="tirx")` to build, then call the
  result directly.

All native authoring uses these imports. The `__future__` import lets `@T.jit`
kernels reference compile-time parameters inside type annotations (see
[Defining a function](tirx_native_basics_cuda_functions.md)); it is harmless for ordinary kernels:

```
from __future__ import annotations
import tvm
from tvm.script import tirx as T
```

- [Your first kernel](tirx_native_basics_cuda_first_kernel.md)
- [Defining a function](tirx_native_basics_cuda_functions.md)
- [Parser utilities](tirx_native_basics_cuda_parser_utils.md)
- [Data types and expressions](tirx_native_basics_cuda_data_types.md)
- [Buffers and memory](tirx_native_basics_cuda_buffers.md)
- [Control flow](tirx_native_basics_cuda_control_flow.md)
- [CUDA C++/PTX intrinsics](tirx_native_basics_cuda_threads_sync.md)
- [Compiling and inspecting](tirx_native_basics_cuda_compiling.md)
- [In-kernel profiling with CudaProfiler](tirx_native_basics_cuda_profiling.md)
