<!-- source: https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/overview.html -->

<a id="overview"></a>
# Overview

CUTLASS 4.x bridges the gap between productivity and performance for CUDA kernel development.
By providing Python-based DSLs to the powerful CUTLASS C++ template library, it enables
faster iteration, easier prototyping, and a gentler learning curve for high-performance linear
algebra on NVIDIA GPUs.

Overall we envision CUTLASS DSLs as a family of domain-specific languages (DSLs).
With the release of 4.0, we are releasing the first of these in CuTe DSL.
This is a low level programming model that is fully consistent with CuTe C++ abstractions — exposing
core concepts such as layouts, tensors, hardware atoms, and full control over the hardware thread and data hierarchy.

<a id="why-cutlass-dsls"></a>
# Why CUTLASS DSLs?

While CUTLASS offers exceptional performance through its C++ template abstractions, the complexity
can present challenges for many developers. CUTLASS 4.x addresses this by:

- **Simplifying metaprogramming**: Metaprogramming in Python is a lot more intuitive than with C++
- **Accelerating Iteration**: Rapid prototyping with familiar Python syntax and blazing fast compile times
- **Lowering Barriers**: Reduced learning curve for GPU programming concepts and consistency between CuTe C++ and DSL
- **Maintaining Performance**: Generated code leverages optimized CUTLASS primitives

Students can learn GPU programming concepts without the complexity of C++ templates.
Researchers and performance engineers can rapidly explore algorithms, prototype, and tune
kernels before moving to production implementations.

<a id="key-concepts-and-approach"></a>
# Key Concepts and Approach

CUTLASS DSLs translate Python code into a custom intermediate representation (IR),
which is then Just-In-Time (JIT) compiled into optimized CUDA kernels using MLIR and ptxas.

<a id="core-cute-dsl-abstractions"></a>
## Core CuTe DSL Abstractions

- **Layouts** – Describe how data is organized in memory and across threads.
- **Tensors** – Combine data pointers or iterators with layout metadata.
- **Atoms** – Represent fundamental hardware operations like matrix multiply-accumulate (MMA) or memory copy.
- **Tiled Operations** – Define how atoms are applied across thread blocks and warps (e.g., `TiledMma`, `TiledCopy`).

For more on CuTe abstractions, refer to the CuTe C++ library documentation.

**Pythonic Kernel Expression**

Developers express kernel logic, data movement, and computation using familiar Python syntax and control flow.

The DSLs simplify expressing loop tiling, threading strategies, and data transformations using concise Python code.

**JIT Compilation**

Python kernels are compiled at runtime into CUDA device code using MLIR infrastructure and NVIDIA’s `ptxas` toolchain,
enabling rapid iteration and interactive debugging.

<a id="relationship-to-cutlass-c"></a>
# Relationship to CUTLASS C++

CUTLASS DSLs are not a replacement for the CUTLASS C++ library or its 2.x and 3.x APIs. Instead, it aims to be a high-productivity kernel
authoring framework that shares all concepts with CUTLASS 3.x C++ API such as CuTe, pipelines, schedulers etc.

- **Performance**: Generated kernels aim to match CUTLASS C++ kernels in performance; however, some performance gaps
  may exist due to missing optimizations that have been added over the years to CUTLASS C++ and may be missing in the DSLs examples.
- **Library**: The CUTLASS DSLs do not currently ship with a full GEMM/Conv autotuning profiler or library interface
  akin to CUTLASS C++. Instead, it focuses on generating and autotuning individual kernel instances (for example: via tile size exploration) and via native integration DL frameworks that support auto-tuning.

<a id="getting-started"></a>
# Getting Started

- [Quick Start Guide](cutedsl_quick_start.md) – Initial setup and installation.
- [CuTe DSL](cutedsl_cute_dsl.md) – Overview of the typical development and workflow using CuTe DSL.
- [CuTe DSL API](cutedsl_cute_dsl_api.md) – Refer to the full API documentation.
- [Limitations](cutedsl_limitations.md) – Understand current CuTe DSL constraints and differences from C++.
- [FAQs](cutedsl_faqs.md) – Common questions and known issues.

<a id="current-status-roadmap"></a>
# Current Status & Roadmap

CuTe DSL is in public beta and actively evolving. Interfaces and features are subject to
change as we improve the system.

<a id="upcoming-milestones"></a>
## Upcoming Milestones

- Public release targeted for **Summer 2025**
- Expanded support for additional data types and kernel types
- Usability improvements: better error messages, debugging tools, and streamlined APIs
- Broader integration of CUTLASS primitives and features

For known issues and workarounds, please consult the [Limitations](cutedsl_limitations.md) and [FAQs](cutedsl_faqs.md).

<a id="community-feedback"></a>
# Community & Feedback

We welcome contributions and feedback from the developer community!

You can:

- Submit bug reports or feature requests via our GitHub Issues page
- Join the CUTLASS community on Discord to ask questions and share ideas
- Contribute examples, tutorials, or enhancements to the DSLs
- Report unclear or missing documentation
- Propose support for additional data types or kernel variants
- Help prioritize roadmap features by upvoting GitHub issues

Thank you for helping shape the future of CUTLASS DSLs!
