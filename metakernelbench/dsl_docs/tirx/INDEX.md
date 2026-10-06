# TIRx Documentation — Offline Manual

Extracted from `https://tvm.apache.org/docs/tirx/`. 43 pages, 740 indexed entries.


## What this manual covers

- **Overview** — 1 page(s)
- **Installation** — 1 page(s)
- **TIRx Basics (CUDA C++/PTX native level)** — 10 page(s)
- **Tensor Layout** — 1 page(s)
- **Tile Primitives** — 22 page(s)
- **Compiler Internals** — 2 page(s)
- **API Reference** — 6 page(s)

API symbols by module:

- `tvm.tirx` — 540 symbols
- `tvm.backend` — 162 symbols

## What this manual does NOT cover

The TIRx pages link out to these, and the targets are **not in this manual**. Do not search for them here and do not expect to reach them — there is no network in this sandbox.

- `tvm.relax` — referenced 668 times
- `(Python standard library)` — referenced 350 times
- `tvm.ir` — referenced 146 times
- `tvm.arith` — referenced 5 times

Also absent: the TVM source tree, C++ APIs, and anything outside `tvm.tirx*` / `tvm.backend.cuda`.


The pages are the official documentation verbatim. Links out of the manual were reduced to bare fully qualified names, so an unlinked name is documented elsewhere, not missing by mistake. Figures are included under `assets/` and referenced from the page that uses them, with the original caption as alt text.


## How to look things up

Every API entry in `pages/` opens with an HTML anchor whose id is the fully qualified name, with the signature directly below it. Grep the anchors first — the signature is usually the whole answer, so you rarely need to read further:

```bash
grep -rin 'id=".*cudanamespace' pages/       # every entry named like cudanamespace
grep -rn 'id="tvm.backend.cuda.script.CUDANamespace' pages/   # one class and its methods
grep -n -A 10 'id="tvm.backend.cuda.script.CUDANamespace"' pages/<file>.md   # the entry itself
```

Never read an API page end to end — the largest is around 40 000 tokens, almost all of it symbols you did not ask about; read only the window around the hit.

If a grep over `pages/` returns nothing, the symbol is not in this manual. Check the "does NOT cover" list above rather than searching again. Treat an undocumented type in a signature as an opaque value you pass along, and a capability absent from the manual as absent from the DSL — look for the supported way to do the same thing. Searching for something that does not exist has no natural end.

## Page map


### Overview

- `pages/tirx_overview.md` — **Overview**
  - Design Philosophy · The Programming Model · What TIRx Enables · Next Steps

### Installation

- `pages/tirx_install.md` — **Installation**
  - Requirements · Install the TIRx compiler · Install the kernel library (optional)

### TIRx Basics (CUDA C++/PTX native level)

- `pages/tirx_native_basics.md` — **TIRx Basics: CUDA C++/PTX native level**
  - What “native level” means · The authoring model
- `pages/tirx_native_basics_cuda_buffers.md` — **Buffers and memory**
  - Declaring buffers · Shared memory · Registers · Tensor memory · Buffer APIs
- `pages/tirx_native_basics_cuda_compiling.md` — **Compiling and inspecting**
  - Inspecting the result · From simple to complex · Next steps
- `pages/tirx_native_basics_cuda_control_flow.md` — **Control flow**
  - if · loop · while
- `pages/tirx_native_basics_cuda_data_types.md` — **Data types and expressions**
  - Expression dtypes · dtype vs type · Pointers ( handle )
- `pages/tirx_native_basics_cuda_first_kernel.md` — **Your first kernel**
- `pages/tirx_native_basics_cuda_functions.md` — **Defining a function**
  - Declaring buffer parameters · What the parameter list accepts · Symbolic shapes · @T.prim_func vs @T.jit · Launch parameters
- `pages/tirx_native_basics_cuda_parser_utils.md` — **Parser utilities**
  - T.meta_var — inline a Python value · @T.inline — inline functions · @T.meta_class — parser-side state objects · T.constexpr
- `pages/tirx_native_basics_cuda_profiling.md` — **In-kernel profiling with CudaProfiler**
  - The kernel · Run it and read the trace · On a real kernel · The API · Groups and granularity · What each call wraps · Usage notes and caveats
- `pages/tirx_native_basics_cuda_threads_sync.md` — **CUDA C++/PTX intrinsics**
  - Calling backend intrinsics · Inlining raw CUDA

### Tensor Layout

- `pages/tirx_layout.md` — **Tensor Layout**
  - Interactive demo · TileLayout · SwizzleLayout, ComposeLayout · Design rationale

### Tile Primitives

- `pages/tirx_tile_primitives.md` — **Tile Primitives**
  - Calling convention · Primitive catalog · Dispatch config · Dispatch mechanism · Dispatch by primitive · See also
- `pages/tirx_tile_primitives_copy.md` — **copy**
- `pages/tirx_tile_primitives_copy_async.md` — **copy_async**
- `pages/tirx_tile_primitives_copy_async_dsmem.md` — **copy_async → dsmem**
  - What it accepts · Demonstration program · Algorithm · Generated TIRx IR · Generated CUDA · How inputs change the algorithm
- `pages/tirx_tile_primitives_copy_async_ldgsts.md` — **copy_async → ldgsts**
  - What it accepts · Demonstration program · Algorithm · Generated TIRx IR · Generated CUDA · How inputs change the algorithm
- `pages/tirx_tile_primitives_copy_async_tcgen05_cp.md` — **copy_async → tcgen05_cp**
  - What it accepts · Demonstration program · Algorithm · Generated TIRx IR · Generated CUDA · How inputs change the algorithm
- `pages/tirx_tile_primitives_copy_async_tcgen05_ldst.md` — **copy_async → tcgen05_ldst**
  - What it accepts · Demonstration program · Algorithm · Generated TIRx IR · Generated CUDA · How inputs change the algorithm
- `pages/tirx_tile_primitives_copy_async_tma.md` — **copy_async → tma**
  - What it accepts · Demonstration program · Algorithm · Generated TIRx IR · Generated CUDA · How inputs change the algorithm
- `pages/tirx_tile_primitives_copy_fallback.md` — **copy → fallback**
  - What it accepts · Demonstration program · Algorithm · Generated TIRx IR · Generated CUDA · How inputs change the algorithm
- `pages/tirx_tile_primitives_copy_gmem_smem.md` — **copy → gmem_smem**
  - What it accepts · Demonstration program · Algorithm · Generated TIRx IR · Generated CUDA · How inputs change the algorithm
- `pages/tirx_tile_primitives_copy_ldstmatrix.md` — **copy → ldstmatrix**
  - What it accepts · Demonstration program · Algorithm · Generated TIRx IR · Generated CUDA · How inputs change the algorithm
- `pages/tirx_tile_primitives_copy_reg.md` — **copy → reg**
  - What it accepts · Demonstration program · Algorithm · Generated TIRx IR · Generated CUDA · How inputs change the algorithm
- `pages/tirx_tile_primitives_elementwise.md` — **elementwise**
- `pages/tirx_tile_primitives_elementwise_reg.md` — **elementwise → reg**
  - What it accepts · Demonstration program · Algorithm · Generated TIRx IR · Generated CUDA · How inputs change the algorithm
- `pages/tirx_tile_primitives_elementwise_smem.md` — **elementwise → smem**
  - What it accepts · Demonstration program · Algorithm · Generated TIRx IR · Generated CUDA · How inputs change the algorithm
- `pages/tirx_tile_primitives_gemm.md` — **gemm**
  - What it accepts · Demonstration program · Algorithm · Generated TIRx IR · Generated CUDA · How inputs change the algorithm
- `pages/tirx_tile_primitives_gemm_async.md` — **gemm_async**
  - What it accepts · Demonstration program · Algorithm · Generated TIRx IR · Generated CUDA · How inputs change the algorithm
- `pages/tirx_tile_primitives_permute_layout.md` — **permute_layout**
  - What it accepts · Demonstration program · Algorithm · Generated TIRx IR · Generated CUDA · How inputs change the algorithm
- `pages/tirx_tile_primitives_reduction.md` — **reduction**
- `pages/tirx_tile_primitives_reduction_local.md` — **reduction → local**
  - What it accepts · Demonstration program · Algorithm · Generated TIRx IR · Generated CUDA · How inputs change the algorithm
- `pages/tirx_tile_primitives_reduction_shared.md` — **reduction → shared**
  - What it accepts · Demonstration program · Algorithm · Generated TIRx IR · Generated CUDA · How inputs change the algorithm
- `pages/tirx_tile_primitives_reduction_sm100_packed.md` — **reduction → sm100_packed**
  - What it accepts · Demonstration program · Algorithm · Generated TIRx IR · Generated CUDA · How inputs change the algorithm

### Compiler Internals

- `pages/tirx_arch_index.md` — **Compiler Internals**
- `pages/tirx_arch_lowering_pipeline.md` — **TIRx lowering pipeline**
  - Where it sits · The passes · Inside LowerTIRx · A worked example · Reproduce it yourself

### API Reference

- `pages/tirx_api_analysis.md` — **tvm.tirx.analysis**
- `pages/tirx_api_backend.md` — **tvm.backend.cuda**
- `pages/tirx_api_index.md` — **API Reference**
- `pages/tirx_api_stmt_functor.md` — **tvm.tirx.stmt_functor**
- `pages/tirx_api_tirx.md` — **tvm.tirx**
- `pages/tirx_api_transform.md` — **tvm.tirx.transform**
