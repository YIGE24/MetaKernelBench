<!-- source: https://tvm.apache.org/docs/tirx/tile_primitives/copy.html -->

<a id="copy"></a>
# copy

Synchronous element copy `src → dst` between global, shared, and register
(`local`) memory. Four CUDA variants are registered. `gmem_smem` (global ↔
shared) and the two register variants take disjoint memory-scope pairs; `reg` and
`ldstmatrix` both cover `register ↔ shared`, where `ldstmatrix` is tried first
and **declines to** `reg` unless the layouts are ldmatrix fragments. All three are
priority 10; the scalar `fallback` (priority 0) runs only if they all decline.

| Variant | Pair | Lowering |
| --- | --- | --- |
| [copy → gmem\_smem](tirx_tile_primitives_copy_gmem_smem.md) | global ↔ shared | synthesized `[outer, threads, vec]` partition, vectorized `copy_Nb` |
| [copy → reg](tirx_tile_primitives_copy_reg.md) | register ↔ shared/global | partition induced by the register layout’s thread axes |
| [copy → ldstmatrix](tirx_tile_primitives_copy_ldstmatrix.md) | register ↔ shared | warp-collective `ldmatrix` / `stmatrix` (m8n8 fragments) |
| [copy → fallback](tirx_tile_primitives_copy_fallback.md) | any | scalar single-thread copy (priority 0, catch-all) |

Each variant has its own deep walkthrough — accepted input, algorithm, a runnable
demo program, the dispatch’s TIRx IR, and the generated CUDA:

- [copy → gmem\_smem](tirx_tile_primitives_copy_gmem_smem.md)
- [copy → reg](tirx_tile_primitives_copy_reg.md)
- [copy → ldstmatrix](tirx_tile_primitives_copy_ldstmatrix.md)
- [copy → fallback](tirx_tile_primitives_copy_fallback.md)
