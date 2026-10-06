<!-- source: https://tvm.apache.org/docs/tirx/tile_primitives/elementwise.html -->

<a id="elementwise"></a>
# elementwise

Covers `cast`, `fill`, the unary ops (`zero`, `reciprocal`, `sqrt`,
`exp`, `exp2`, `silu`), the binary ops (`add`, `sub`, `mul`,
`fdiv`), and `fma`. Every op registers **two** variants — `reg` and
`smem` — both at priority 10; the operand storage scope (all-register vs
all-shared) is the mutually-exclusive discriminator. Each op is described by an
`OpSpec` (a `parse` that builds the destination + source list, optional dtype
checks, and the scalar expression applied per element).

| Variant | Operands | Lowering |
| --- | --- | --- |
| [elementwise → reg](tirx_tile_primitives_elementwise_reg.md) | all register | partition induced by the register layout; op applied per register |
| [elementwise → smem](tirx_tile_primitives_elementwise_smem.md) | all shared | synthesized `[outer, threads, vec]` partition; op applied per (vectorized) element |

- [elementwise → reg](tirx_tile_primitives_elementwise_reg.md)
- [elementwise → smem](tirx_tile_primitives_elementwise_smem.md)
