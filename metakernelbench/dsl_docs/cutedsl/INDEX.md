# CuTe DSL Documentation — Offline Manual

Extracted from `https://docs.nvidia.com/cutlass/latest/media/docs/`. 44 pages, 835 indexed entries.


## What this manual covers

- **Overview** — 1 page(s)
- **Functionality** — 1 page(s)
- **Quick Start** — 1 page(s)
- **CuTe DSL** — 18 page(s)
- **MMA Programming Guides** — 4 page(s)
- **CuTe DSL API** — 15 page(s)
- **Deprecation Policy** — 1 page(s)
- **Limitations** — 1 page(s)
- **FAQs** — 1 page(s)
- **License** — 1 page(s)

API symbols by module:

- `cutlass.cute` — 501 symbols
- `cutlass.pipeline` — 153 symbols
- `cutlass.utils` — 149 symbols

## What this manual does NOT cover

The CuTe DSL pages link out to these, and the targets are **not in this manual**. Do not search for them here and do not expect to reach them — there is no network in this sandbox.

- `PTX documentation` — referenced 62 times

Also absent: the CUTLASS C++ manual, the CUTLASS source tree, and anything outside the Python DSL documentation.


The pages are the official documentation verbatim. Links out of the manual were reduced to bare fully qualified names, so an unlinked name is documented elsewhere, not missing by mistake. Figures are included under `assets/` and referenced from the page that uses them, with the original caption as alt text.


## How to look things up

Every API entry in `pages/` opens with an HTML anchor whose id is the fully qualified name, with the signature directly below it. Grep the anchors first — the signature is usually the whole answer, so you rarely need to read further:

```bash
grep -rin 'id=".*swizzle' pages/             # every entry named like swizzle
grep -rn 'id="cutlass.cute.Swizzle' pages/   # one class and its methods
grep -n -A 10 'id="cutlass.cute.Swizzle"' pages/<file>.md   # the entry itself
```

Never read an API page end to end — the largest is around 40 000 tokens, almost all of it symbols you did not ask about; read only the window around the hit.

If a grep over `pages/` returns nothing, the symbol is not in this manual. Check the "does NOT cover" list above rather than searching again. Treat an undocumented type in a signature as an opaque value you pass along, and a capability absent from the manual as absent from the DSL — look for the supported way to do the same thing. Searching for something that does not exist has no natural end.

## Page map


### Overview

- `pages/cutedsl_overview.md` — **Overview**
  - Core CuTe DSL Abstractions · Upcoming Milestones

### Functionality

- `pages/cutedsl_functionality.md` — **Functionality**
  - Supported MMA Operations · Notable Limitations

### Quick Start

- `pages/cutedsl_quick_start.md` — **Quick Start Guide**
  - Compatibility Requirements · Installation · Recommended Dependencies · Recommended Python environment variables for jupyter notebooks

### CuTe DSL

- `pages/cutedsl_cute_dsl.md` — **CuTe DSL**
- `pages/cutedsl_cute_dsl_general_autotuning_gemm.md` — **Guidance for Auto-Tuning**
- `pages/cutedsl_cute_dsl_general_compile_with_tvm_ffi.md` — **Compile with TVM FFI**
  - Enable Apache TVM FFI in CuTe DSL · Minimizing Host Overhead · Fake tensor for compilation · cute.Tensor adapter for TVM FFI · Working with torch Tensors · Working with Streams · Working with Tuples · Working with Variadic Tuples · Supported types · Error handling · Working with Devices · Exporting Compiled Module · Limitations
- `pages/cutedsl_cute_dsl_general_debugging.md` — **Debugging**
  - Getting Familiar with the Limitations · Source Code Correlation · Debug Mode · DSL Debugging · Kernel Functional Debugging · Conclusion
- `pages/cutedsl_cute_dsl_general_dsl_ahead_of_time_compilation.md` — **Ahead-of-Time (AOT) Compilation**
  - Overview · CuTe ABI AOT Workflow · Supported Argument Types · Object File Compatibility Issues · Relation to Apache TVM FFI AOT
- `pages/cutedsl_cute_dsl_general_dsl_code_generation.md` — **End-to-End Code Generation**
  - 1. Hybrid DSL: Python Metaprogramming, Structured GPU Code · 2. CuTe DSL Compilation Flow: Meta-Stage to Object-Stage · 3. Meta-Programming vs Runtime: Two Worlds in One Function · 4. CuTe DSL Code-Generation Modes
- `pages/cutedsl_cute_dsl_general_dsl_control_flow.md` — **Control Flow**
  - Overview · For Loops · If-Else Statements · While Loops · Summary of Control Flow behavior · Compile-Time Metaprogramming
- `pages/cutedsl_cute_dsl_general_dsl_dynamic_layout.md` — **Static vs Dynamic layouts**
  - Static Layout · Dynamic Layout · Static Layout vs. Dynamic Layout · Programming with Static and Dynamic Layout
- `pages/cutedsl_cute_dsl_general_dsl_introduction.md` — **Introduction**
  - Overview · Decorators · Calling Conventions
- `pages/cutedsl_cute_dsl_general_dsl_jit_arg_generation.md` — **JIT Function Argument Generation**
  - In a nutshell · Static argument vs. Dynamic argument · Type safety · JIT function arguments with customized types
- `pages/cutedsl_cute_dsl_general_dsl_jit_caching.md` — **JIT Caching**
  - Zero Compile and JIT Executor · Cache in CuTe DSL
- `pages/cutedsl_cute_dsl_general_dsl_jit_compilation_options.md` — **JIT Compilation Options**
  - JIT Compilation Options Overview · cute.compile Compilation Options as strings · cute.compile Compilation Options as separate Python types
- `pages/cutedsl_cute_dsl_general_dsl_struct_types.md` — **Struct-like JIT Arguments**
  - Overview · NamedTuple · @native_struct · Choosing the right type · See also
- `pages/cutedsl_cute_dsl_general_framework_integration.md` — **Integration with Frameworks**
  - Implicit Conversion · Explicit conversion using from_dlpack · Mark the Tensor’s Layout as Dynamic with mark_layout_dynamic · Mark the Tensor’s Layout as Dynamic with mark_compact_shape_dynamic · Leveraging TVM FFI for Faster PyTorch Interop · Bypass the DLPack Protocol
- `pages/cutedsl_cute_dsl_general_iket_profiling.md` — **IKET Profiling**
  - Requirements · End-to-End Quick Start · API Reference · Example Instrumentation Patterns · Profiling with the run-iket Tool · Viewing a Trace in Perfetto · Assumptions, Limitations, and Impact · Performance Overhead Guidance · How It Works · Troubleshooting
- `pages/cutedsl_cute_dsl_general_naming_conventions.md` — **CuTe DSL Naming Conventions**
  - Memory/space scopes · Per-thread/partitioned views and families · Data-movement copy paths · Operands and roles · Axis-order suffixes · Reading compound tokens · Concrete references
- `pages/cutedsl_cute_dsl_general_resources.md` — **Talks and Presentations**
  - Conference Talks
- `pages/cutedsl_cute_dsl_general_types.md` — **Types**
  - Overview · Core Numeric Types · Layout Algebra Types · Memory and Pointer Types · Structured Data Types · Type Hierarchies and Relationships · Best Practices · See Also

### MMA Programming Guides

- `pages/cutedsl_mma_docs_intro.md` — **Architecture-specific MMA Programming Guides**
- `pages/cutedsl_mma_docs_tcgen05_programming.md` — **tcgen05 MMA Programming Guide**
  - Global Memory (GMEM) to MMA data flow overview · Setting up the TiledMMA, MMA Ops · Partitioning Tensors · Making Fragments · Executing the GEMM (Main Loop) · Reading the accumulator from TMEM · Complete Workflow · Beyond Simple Dense MMAs
- `pages/cutedsl_mma_docs_wgmma_programming.md` — **Warpgroup MMA Programming Guide**
  - Global Memory (GMEM) to MMA data flow overview · Setting up the TiledMMA, MMA Ops · Partitioning Tensors · Pre and Post-Conditions for Partitioning · Making Fragments · Creating SMEM layouts for A and B · Executing the GEMM (Main Loop) · Complete Workflow
- `pages/cutedsl_mma_docs_wmma_programming.md` — **Warp-Level MMA Instructions Programming Guide**
  - Global Memory (GMEM) to MMA data flow overview · Setting up the TiledMMA, MMA Ops · Partitioning Tensors · Pre and Post-Conditions for Partitioning · Making Fragments · Executing the GEMM (Main Loop) · Complete Workflow · Beyond Simple Dense MMAs

### CuTe DSL API

- `pages/cutedsl_cute_dsl_api.md` — **CuTe DSL API**
- `pages/cutedsl_cute_dsl_api_changelog.md` — **Changelog for CuTe DSL API changes**
- `pages/cutedsl_cute_dsl_api_cute.md` — **cutlass.cute**
- `pages/cutedsl_cute_dsl_api_cute_arch.md` — **arch**
- `pages/cutedsl_cute_dsl_api_cute_nvgpu.md` — **cutlass.cute.nvgpu**
- `pages/cutedsl_cute_dsl_api_cute_nvgpu_common.md` — **Common**
- `pages/cutedsl_cute_dsl_api_cute_nvgpu_cpasync.md` — **cpasync submodule**
- `pages/cutedsl_cute_dsl_api_cute_nvgpu_tcgen05.md` — **tcgen05 submodule**
- `pages/cutedsl_cute_dsl_api_cute_nvgpu_warp.md` — **warp submodule**
- `pages/cutedsl_cute_dsl_api_cute_nvgpu_warpgroup.md` — **warpgroup submodule**
- `pages/cutedsl_cute_dsl_api_cute_runtime.md` — **Runtime**
- `pages/cutedsl_cute_dsl_api_pipeline.md` — **cutlass.pipeline**
- `pages/cutedsl_cute_dsl_api_utils.md` — **cutlass.utils**
- `pages/cutedsl_cute_dsl_api_utils_sm100.md` — **Utilities for SM100**
- `pages/cutedsl_cute_dsl_api_utils_sm90.md` — **Utilities for SM90**

### Deprecation Policy

- `pages/cutedsl_deprecation.md` — **Deprecation Policy**
  - Purpose · Deprecation Process · Communication · Soft Deprecations · Deprecated Features

### Limitations

- `pages/cutedsl_limitations.md` — **Limitations**
  - Overview · Notable unsupported features · Programming Model · Future Improvements · Design Limitations Likely to Remain

### FAQs

- `pages/cutedsl_faqs.md` — **FAQs**
  - General · Migration · Technical · License

### License

- `pages/cutedsl_license.md` — **Software License Agreement**
  - 1. License Grants · 2. License Restrictions · 3. Authorized Users · 4. Pre-Release · 5. Updates · 6. Components Under Other Licenses · 7. Ownership · 8. Feedback · 9. Termination · 10. Disclaimer of Warranties · 11. Limitations of Liability · 12. Use in Mission Critical Applications · 13. Governing Law and Jurisdiction · 14. Indemnity · 15. General
