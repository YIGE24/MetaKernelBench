<!-- source: https://tvm.apache.org/docs/tirx/tile_primitives/reduction.html -->

<a id="reduction"></a>
# reduction

Covers `sum`, `max`, `min` (reduce over `axes`). Three variants: `local`
and `shared` (priority 10, discriminated by operand storage scope) and
`sm100_packed` (priority 20, which pre-empts the others for the thread-scope
float32 case on Blackwell).

| Variant | Prio | Lowering |
| --- | --- | --- |
| [reduction → local](tirx_tile_primitives_reduction_local.md) | 10 | register src/dst; sequential thread reduction (+ optional warp shuffle) |
| [reduction → shared](tirx_tile_primitives_reduction_shared.md) | 10 | shared src/dst; adaptive group-size `__shfl_xor` tree |
| [reduction → sm100\_packed](tirx_tile_primitives_reduction_sm100_packed.md) | 20 | Blackwell thread-scope fp32 ≥8: packed `add.f32x2` / `max3`/`min3` |

- [reduction → local](tirx_tile_primitives_reduction_local.md)
- [reduction → shared](tirx_tile_primitives_reduction_shared.md)
- [reduction → sm100\_packed](tirx_tile_primitives_reduction_sm100_packed.md)
