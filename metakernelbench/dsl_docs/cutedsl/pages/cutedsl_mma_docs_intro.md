<!-- source: https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/mma_docs/intro.html -->

<a id="architecture-specific-mma-programming-guides"></a>
# Architecture-specific MMA Programming Guides

This section contains architecture-specific MMA programming guides.

- [Warp-Level MMA Instructions Programming Guide](cutedsl_mma_docs_wmma_programming.md)
  - [Global Memory (GMEM) to MMA data flow overview](cutedsl_mma_docs_wmma_programming.md#global-memory-gmem-to-mma-data-flow-overview)
  - [Setting up the TiledMMA, MMA Ops](cutedsl_mma_docs_wmma_programming.md#setting-up-the-tiledmma-mma-ops)
  - [Partitioning Tensors](cutedsl_mma_docs_wmma_programming.md#partitioning-tensors)
  - [Pre and Post-Conditions for Partitioning](cutedsl_mma_docs_wmma_programming.md#pre-and-post-conditions-for-partitioning)
  - [Making Fragments](cutedsl_mma_docs_wmma_programming.md#making-fragments)
  - [Executing the GEMM (Main Loop)](cutedsl_mma_docs_wmma_programming.md#executing-the-gemm-main-loop)
  - [Complete Workflow](cutedsl_mma_docs_wmma_programming.md#complete-workflow)
  - [Beyond Simple Dense MMAs](cutedsl_mma_docs_wmma_programming.md#beyond-simple-dense-mmas)
- [Warpgroup MMA Programming Guide](cutedsl_mma_docs_wgmma_programming.md)
  - [Global Memory (GMEM) to MMA data flow overview](cutedsl_mma_docs_wgmma_programming.md#global-memory-gmem-to-mma-data-flow-overview)
  - [Setting up the TiledMMA, MMA Ops](cutedsl_mma_docs_wgmma_programming.md#setting-up-the-tiledmma-mma-ops)
  - [Partitioning Tensors](cutedsl_mma_docs_wgmma_programming.md#partitioning-tensors)
  - [Pre and Post-Conditions for Partitioning](cutedsl_mma_docs_wgmma_programming.md#pre-and-post-conditions-for-partitioning)
  - [Making Fragments](cutedsl_mma_docs_wgmma_programming.md#making-fragments)
  - [Creating SMEM layouts for A and B](cutedsl_mma_docs_wgmma_programming.md#creating-smem-layouts-for-a-and-b)
  - [Executing the GEMM (Main Loop)](cutedsl_mma_docs_wgmma_programming.md#executing-the-gemm-main-loop)
  - [Complete Workflow](cutedsl_mma_docs_wgmma_programming.md#complete-workflow)
- [tcgen05 MMA Programming Guide](cutedsl_mma_docs_tcgen05_programming.md)
  - [Global Memory (GMEM) to MMA data flow overview](cutedsl_mma_docs_tcgen05_programming.md#global-memory-gmem-to-mma-data-flow-overview)
  - [Setting up the TiledMMA, MMA Ops](cutedsl_mma_docs_tcgen05_programming.md#setting-up-the-tiledmma-mma-ops)
  - [Partitioning Tensors](cutedsl_mma_docs_tcgen05_programming.md#partitioning-tensors)
  - [Making Fragments](cutedsl_mma_docs_tcgen05_programming.md#making-fragments)
  - [Executing the GEMM (Main Loop)](cutedsl_mma_docs_tcgen05_programming.md#executing-the-gemm-main-loop)
  - [Reading the accumulator from TMEM](cutedsl_mma_docs_tcgen05_programming.md#reading-the-accumulator-from-tmem)
  - [Complete Workflow](cutedsl_mma_docs_tcgen05_programming.md#complete-workflow)
  - [Beyond Simple Dense MMAs](cutedsl_mma_docs_tcgen05_programming.md#beyond-simple-dense-mmas)
