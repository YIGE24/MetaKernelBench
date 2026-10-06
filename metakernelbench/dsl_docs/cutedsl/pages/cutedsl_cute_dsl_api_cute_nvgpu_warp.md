<!-- source: https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_warp.html -->

<a id="module-cutlass.cute.nvgpu.warp"></a>
# warp submodule

<a id="cutlass.cute.nvgpu.warp.Field"></a>
### `cutlass.cute.nvgpu.warp.Field`

```python
class cutlass.cute.nvgpu.warp.Field(value)
```

Bases: `Enum`

An enumeration for the fields of the MMA Atom that can be modified at runtime.

<a id="cutlass.cute.nvgpu.warp.Field.ACCUMULATE"></a>
#### `cutlass.cute.nvgpu.warp.Field.ACCUMULATE`

```python
ACCUMULATE = 'accum_c'
```

<a id="cutlass.cute.nvgpu.warp.Field.SFA"></a>
#### `cutlass.cute.nvgpu.warp.Field.SFA`

```python
SFA = 'sf_a'
```

<a id="cutlass.cute.nvgpu.warp.Field.SFB"></a>
#### `cutlass.cute.nvgpu.warp.Field.SFB`

```python
SFB = 'sf_b'
```

<a id="cutlass.cute.nvgpu.warp.MmaF16BF16Op"></a>
### `cutlass.cute.nvgpu.warp.MmaF16BF16Op`

```python
class cutlass.cute.nvgpu.warp.MmaF16BF16Op( ab_dtype: Type[cutlass.cute.typing.Numeric], acc_dtype: Type[cutlass.cute.typing.Numeric], shape_mnk: cutlass.cute.typing.Shape, )
```

Bases: `WarpMmaOp`

F16/BF16 warp-level MMA Operation.

See the PTX documentation.
This Operation covers the instructions using the `.f16` or `.bf16` qualifiers for the input operands.

**Supported data type combinations:**

| A Data Type | B Data Type | Acc Type | Mma-MNK |
| --- | --- | --- | --- |
| F16 | F16 | F16, F32 | (16,8,8), (16,8,16) |
| BF16 | BF16 | F32 | (16,8,8), (16,8,16) |

**Supported architectures:** sm\_80+

**Constraints:**

- Operand layout is fixed: A = row-major (K-major), B = col-major (K-major). Transpose is not supported.

**Execution Model:**

- WMMA (`mma.sync.aligned`) is a warp-collective synchronous operation. All lanes in the
  warp must execute the same MMA instruction in convergence.
- In user code, `cute.gemm(...)` should be issued as warp-uniform code.

```python
cute.gemm(mma_atom, d, a, b, c)
```

<a id="cutlass.cute.nvgpu.warp.MmaF16BF16Op.ab_dtype"></a>
#### `cutlass.cute.nvgpu.warp.MmaF16BF16Op.ab_dtype`

```python
ab_dtype: Type[cutlass.cute.typing.Numeric]
```

<a id="cutlass.cute.nvgpu.warp.MmaF16BF16Op.acc_dtype"></a>
#### `cutlass.cute.nvgpu.warp.MmaF16BF16Op.acc_dtype`

```python
acc_dtype: Type[cutlass.cute.typing.Numeric]
```

<a id="cutlass.cute.nvgpu.warp.MmaF16BF16Op.shape_mnk"></a>
#### `cutlass.cute.nvgpu.warp.MmaF16BF16Op.shape_mnk`

```python
shape_mnk: cutlass.cute.typing.Shape
```

<a id="cutlass.cute.nvgpu.warp.MmaF16BF16Op.__init__"></a>
#### `cutlass.cute.nvgpu.warp.MmaF16BF16Op.__init__`

```python
__init__( ab_dtype: Type[cutlass.cute.typing.Numeric], acc_dtype: Type[cutlass.cute.typing.Numeric], shape_mnk: cutlass.cute.typing.Shape, ) → None
```

<a id="cutlass.cute.nvgpu.warp.MmaTF32Op"></a>
### `cutlass.cute.nvgpu.warp.MmaTF32Op`

```python
class cutlass.cute.nvgpu.warp.MmaTF32Op(shape_mnk: cutlass.cute.typing.Shape)
```

Bases: `WarpMmaOp`

TF32 warp-level MMA operation.

This wraps `mma.sync.aligned.m16n8k{K}.row.col.f32.tf32.tf32.f32`.
Operands are TF32 and accumulation is F32.

<a id="cutlass.cute.nvgpu.warp.MmaTF32Op.shape_mnk"></a>
#### `cutlass.cute.nvgpu.warp.MmaTF32Op.shape_mnk`

```python
shape_mnk: cutlass.cute.typing.Shape
```

<a id="cutlass.cute.nvgpu.warp.MmaTF32Op.__init__"></a>
#### `cutlass.cute.nvgpu.warp.MmaTF32Op.__init__`

```python
__init__(shape_mnk: cutlass.cute.typing.Shape) → None
```

<a id="cutlass.cute.nvgpu.warp.MmaFP8Op"></a>
### `cutlass.cute.nvgpu.warp.MmaFP8Op`

```python
class cutlass.cute.nvgpu.warp.MmaFP8Op( ab_dtype: Type[cutlass.cute.typing.Numeric], acc_dtype: Type[cutlass.cute.typing.Numeric], shape_mnk: cutlass.cute.typing.Shape, )
```

Bases: `WarpMmaOp`

FP8 warp-level MMA Operation (SM89).

See the PTX documentation.
This Operation covers the instructions using the `.e4m3` or `.e5m2` qualifiers for the input operands.

**Supported data type combinations:**

| A Data Type | B Data Type | Acc Type | Mma-MNK |
| --- | --- | --- | --- |
| E4M3 | E4M3 | F16, F32 | (16,8,16), (16,8,32) |
| E5M2 | E5M2 | F16, F32 | (16,8,16), (16,8,32) |

**Supported architectures:** sm\_89+

**Constraints:**

- Operand layout is fixed: A = row-major (K-major), B = col-major (K-major). Transpose is not supported.

**Execution Model:**

- WMMA (`mma.sync.aligned`) is a warp-collective synchronous operation. All lanes in the
  warp must execute the same MMA instruction in convergence.
- In user code, `cute.gemm(...)` should be issued as warp-uniform code.

```python
cute.gemm(mma_atom, d, a, b, c)
```

<a id="cutlass.cute.nvgpu.warp.MmaFP8Op.ab_dtype"></a>
#### `cutlass.cute.nvgpu.warp.MmaFP8Op.ab_dtype`

```python
ab_dtype: Type[cutlass.cute.typing.Numeric]
```

<a id="cutlass.cute.nvgpu.warp.MmaFP8Op.acc_dtype"></a>
#### `cutlass.cute.nvgpu.warp.MmaFP8Op.acc_dtype`

```python
acc_dtype: Type[cutlass.cute.typing.Numeric]
```

<a id="cutlass.cute.nvgpu.warp.MmaFP8Op.shape_mnk"></a>
#### `cutlass.cute.nvgpu.warp.MmaFP8Op.shape_mnk`

```python
shape_mnk: cutlass.cute.typing.Shape
```

<a id="cutlass.cute.nvgpu.warp.MmaFP8Op.__init__"></a>
#### `cutlass.cute.nvgpu.warp.MmaFP8Op.__init__`

```python
__init__( ab_dtype: Type[cutlass.cute.typing.Numeric], acc_dtype: Type[cutlass.cute.typing.Numeric], shape_mnk: cutlass.cute.typing.Shape, ) → None
```

<a id="cutlass.cute.nvgpu.warp.MmaMXF4Op"></a>
### `cutlass.cute.nvgpu.warp.MmaMXF4Op`

```python
class cutlass.cute.nvgpu.warp.MmaMXF4Op( ab_dtype: Type[cutlass.cute.typing.Numeric], acc_dtype: Type[cutlass.cute.typing.Numeric], sf_type: Type[cutlass.cute.typing.Numeric], )
```

Bases: `MmaSM120BlockScaledOp`

MXF4 warp-level MMA Operation.

See the PTX documentation.
This Operation covers the instructions using the `.e2m1` qualifiers for the input operands.
.kind = {.kind::mxf4};
.scale\_vec\_size = {.scale\_vec::2X};
.stype = {.ue8m0};

**Supported data type combinations:**

| A Data Type | B Data Type | SF Data Type | Acc Type | Mma-MNK | SF Vec Size |
| --- | --- | --- | --- | --- | --- |
| E2M1 | E2M1 | UE8M0 | F32 | (16,8,64) | 32 |

**Supported architectures:** sm\_120a, sm\_120f, sm\_121a

**Constraints:**

- Operand layout is fixed: A = row-major (K-major), B = col-major (K-major). Transpose is not supported.

**Execution Model:**

- Block-scaled WMMA (`mma.sync.aligned` with `.block_scale`) is a warp-collective synchronous
  operation. All lanes in the warp must execute the same MMA instruction in convergence.
- In user code, `cute.gemm(...)` should be issued as warp-uniform code.

```python
cute.gemm(mma_atom, d, a, b, c)
```

<a id="cutlass.cute.nvgpu.warp.MmaMXF4Op.descriptive_name"></a>
#### `cutlass.cute.nvgpu.warp.MmaMXF4Op.descriptive_name`

```python
descriptive_name: ClassVar[str] = 'warp-level MXF4 MMA Operation'
```

<a id="cutlass.cute.nvgpu.warp.MmaMXF4Op.__init__"></a>
#### `cutlass.cute.nvgpu.warp.MmaMXF4Op.__init__`

```python
__init__( ab_dtype: Type[cutlass.cute.typing.Numeric], acc_dtype: Type[cutlass.cute.typing.Numeric], sf_type: Type[cutlass.cute.typing.Numeric], ) → None
```

<a id="cutlass.cute.nvgpu.warp.MmaMXF4NVF4Op"></a>
### `cutlass.cute.nvgpu.warp.MmaMXF4NVF4Op`

```python
class cutlass.cute.nvgpu.warp.MmaMXF4NVF4Op( ab_dtype: Type[cutlass.cute.typing.Numeric], acc_dtype: Type[cutlass.cute.typing.Numeric], sf_type: Type[cutlass.cute.typing.Numeric], )
```

Bases: `MmaSM120BlockScaledOp`

MXF4NVF4 warp-level MMA Operation.

See the PTX documentation.
This Operation covers the instructions using the `.e2m1` qualifiers for the input operands.
.kind = {.kind::mxf4nvf4};
.scale\_vec\_size = {.scale\_vec::4X};
.stype = {.ue4m3};

**Supported data type combinations:**

| A Data Type | B Data Type | SF Data Type | Acc Type | Mma-MNK | SF Vec Size |
| --- | --- | --- | --- | --- | --- |
| E2M1 | E2M1 | UE4M3 | F32 | (16,8,64) | 16 |

**Supported architectures:** sm\_120a, sm\_120f, sm\_121a

**Constraints:**

- Operand layout is fixed: A = row-major (K-major), B = col-major (K-major). Transpose is not supported.

**Execution Model:**

- Block-scaled WMMA (`mma.sync.aligned` with `.block_scale`) is a warp-collective synchronous
  operation. All lanes in the warp must execute the same MMA instruction in convergence.
- In user code, `cute.gemm(...)` should be issued as warp-uniform code.

```python
cute.gemm(mma_atom, d, a, b, c)
```

<a id="cutlass.cute.nvgpu.warp.MmaMXF4NVF4Op.descriptive_name"></a>
#### `cutlass.cute.nvgpu.warp.MmaMXF4NVF4Op.descriptive_name`

```python
descriptive_name: ClassVar[str] = 'warp-level MXF4NVF4 MMA Operation'
```

<a id="cutlass.cute.nvgpu.warp.MmaMXF4NVF4Op.__init__"></a>
#### `cutlass.cute.nvgpu.warp.MmaMXF4NVF4Op.__init__`

```python
__init__( ab_dtype: Type[cutlass.cute.typing.Numeric], acc_dtype: Type[cutlass.cute.typing.Numeric], sf_type: Type[cutlass.cute.typing.Numeric], ) → None
```

<a id="cutlass.cute.nvgpu.warp.MmaMXF8Op"></a>
### `cutlass.cute.nvgpu.warp.MmaMXF8Op`

```python
class cutlass.cute.nvgpu.warp.MmaMXF8Op( ab_dtype: Type[cutlass.cute.typing.Numeric], acc_dtype: Type[cutlass.cute.typing.Numeric], sf_type: Type[cutlass.cute.typing.Numeric], )
```

Bases: `MmaSM120BlockScaledOp`

MXF8 warp-level MMA Operation.

See the PTX documentation.
This Operation covers the instructions using the `.e4m3` / `.e5m2` qualifiers for the input operands.
.kind = {.kind::mxf8};
.scale\_vec\_size = {.scale\_vec::1X};
.stype = {.ue8m0};

<a id="cutlass.cute.nvgpu.warp.MmaMXF8Op.descriptive_name"></a>
#### `cutlass.cute.nvgpu.warp.MmaMXF8Op.descriptive_name`

```python
descriptive_name: ClassVar[str] = 'warp-level MXF8 MMA Operation'
```

<a id="cutlass.cute.nvgpu.warp.MmaMXF8Op.__init__"></a>
#### `cutlass.cute.nvgpu.warp.MmaMXF8Op.__init__`

```python
__init__( ab_dtype: Type[cutlass.cute.typing.Numeric], acc_dtype: Type[cutlass.cute.typing.Numeric], sf_type: Type[cutlass.cute.typing.Numeric], ) → None
```

<a id="cutlass.cute.nvgpu.warp.MmaMXF8F6F4Op"></a>
### `cutlass.cute.nvgpu.warp.MmaMXF8F6F4Op`

```python
class cutlass.cute.nvgpu.warp.MmaMXF8F6F4Op( a_dtype: Type[cutlass.cute.typing.Numeric], b_dtype: Type[cutlass.cute.typing.Numeric], acc_dtype: Type[cutlass.cute.typing.Numeric], sf_type: Type[cutlass.cute.typing.Numeric], )
```

Bases: `MmaOp`

SM120 MXF8F6F4 mixed-precision warp-level block-scaled MMA Operation.

Covers the PTX instructions using independent `.<a_type>.<b_type>`
qualifiers (one of e2m1.e4m3, e2m1.e5m2, e4m3.e2m1, e5m2.e2m1):

.kind = {.kind::mxf8f6f4};
.scale\_vec\_size = {.scale\_vec::1X};
.stype = {.ue8m0};

A and B operand dtypes are independent. Same-dtype FP4/FP4 and FP8/FP8
paths remain on `MmaMXF4Op` / `MmaMXF4NVF4Op` / `MmaMXF8Op`
respectively. Same-width mixed-FP8 (E4M3 + E5M2) and FP6 mixed pairs
are not supported.

<a id="cutlass.cute.nvgpu.warp.MmaMXF8F6F4Op.a_dtype"></a>
#### `cutlass.cute.nvgpu.warp.MmaMXF8F6F4Op.a_dtype`

```python
a_dtype: Type[cutlass.cute.typing.Numeric]
```

<a id="cutlass.cute.nvgpu.warp.MmaMXF8F6F4Op.b_dtype"></a>
#### `cutlass.cute.nvgpu.warp.MmaMXF8F6F4Op.b_dtype`

```python
b_dtype: Type[cutlass.cute.typing.Numeric]
```

<a id="cutlass.cute.nvgpu.warp.MmaMXF8F6F4Op.acc_dtype"></a>
#### `cutlass.cute.nvgpu.warp.MmaMXF8F6F4Op.acc_dtype`

```python
acc_dtype: Type[cutlass.cute.typing.Numeric]
```

<a id="cutlass.cute.nvgpu.warp.MmaMXF8F6F4Op.sf_type"></a>
#### `cutlass.cute.nvgpu.warp.MmaMXF8F6F4Op.sf_type`

```python
sf_type: Type[cutlass.cute.typing.Numeric]
```

<a id="cutlass.cute.nvgpu.warp.MmaMXF8F6F4Op.descriptive_name"></a>
#### `cutlass.cute.nvgpu.warp.MmaMXF8F6F4Op.descriptive_name`

```python
descriptive_name: ClassVar[str] = 'warp-level MXF8F6F4 mixed-precision MMA Operation'
```

<a id="cutlass.cute.nvgpu.warp.MmaMXF8F6F4Op.shape_mnk"></a>
#### `cutlass.cute.nvgpu.warp.MmaMXF8F6F4Op.shape_mnk`

```python
shape_mnk = (16, 8, 32)
```

<a id="cutlass.cute.nvgpu.warp.MmaMXF8F6F4Op.sf_vec_size"></a>
#### `cutlass.cute.nvgpu.warp.MmaMXF8F6F4Op.sf_vec_size`

```python
sf_vec_size = 32
```

<a id="cutlass.cute.nvgpu.warp.MmaMXF8F6F4Op.use_sf_layout_TV"></a>
#### `cutlass.cute.nvgpu.warp.MmaMXF8F6F4Op.use_sf_layout_TV`

```python
use_sf_layout_TV = False
```

<a id="cutlass.cute.nvgpu.warp.MmaMXF8F6F4Op.admissible_archs"></a>
#### `cutlass.cute.nvgpu.warp.MmaMXF8F6F4Op.admissible_archs`

```python
admissible_archs = [cutlass.base_dsl.arch.Arch.sm_120a, cutlass.base_dsl.arch.Arch.sm_120f, cutlass.base_dsl.arch.Arch.sm_121a, cutlass.base_dsl.arch.Arch.sm_121f]
```

<a id="cutlass.cute.nvgpu.warp.MmaMXF8F6F4Op.__init__"></a>
#### `cutlass.cute.nvgpu.warp.MmaMXF8F6F4Op.__init__`

```python
__init__( a_dtype: Type[cutlass.cute.typing.Numeric], b_dtype: Type[cutlass.cute.typing.Numeric], acc_dtype: Type[cutlass.cute.typing.Numeric], sf_type: Type[cutlass.cute.typing.Numeric], ) → None
```

<a id="cutlass.cute.nvgpu.warp.LdMatrix8x8x16bOp"></a>
### `cutlass.cute.nvgpu.warp.LdMatrix8x8x16bOp`

```python
class cutlass.cute.nvgpu.warp.LdMatrix8x8x16bOp( transpose: bool = False, num_matrices: int = 1, unpack_bits: cutlass.cute.typing.Optional.<class 'int'> | None = None, )
```

Bases: `BaseOp`

8x8 `ldmatrix` Operation.

See the PTX documentation.
This operation corresponds to the `.m8n8` qualifier.

<a id="cutlass.cute.nvgpu.warp.LdMatrix8x8x16bOp.__init__"></a>
#### `cutlass.cute.nvgpu.warp.LdMatrix8x8x16bOp.__init__`

```python
__init__( transpose: bool = False, num_matrices: int = 1, unpack_bits: cutlass.cute.typing.Optional.<class 'int'> | None = None, ) → None
```

<a id="cutlass.cute.nvgpu.warp.LdMatrix16x8x8bOp"></a>
### `cutlass.cute.nvgpu.warp.LdMatrix16x8x8bOp`

```python
class cutlass.cute.nvgpu.warp.LdMatrix16x8x8bOp( transpose: bool = False, num_matrices: int = 1, unpack_bits: cutlass.cute.typing.Optional.<class 'int'> | None = None, )
```

Bases: `BaseOp`

16x8 8b `ldmatrix` Operation with transpose

There is no direct PTX correspondance to this Op.
This actually lowers to ldmatrix with the `.m16n16` qualifier and
additional address and value permutations to match stmatrix.m16n8.trans.
Useful for vectorizing with Ampere-style 8x8 matrix thread-value layouts

<a id="cutlass.cute.nvgpu.warp.LdMatrix16x8x8bOp.__init__"></a>
#### `cutlass.cute.nvgpu.warp.LdMatrix16x8x8bOp.__init__`

```python
__init__( transpose: bool = False, num_matrices: int = 1, unpack_bits: cutlass.cute.typing.Optional.<class 'int'> | None = None, ) → None
```

<a id="cutlass.cute.nvgpu.warp.LdMatrix16x16x8bOp"></a>
### `cutlass.cute.nvgpu.warp.LdMatrix16x16x8bOp`

```python
class cutlass.cute.nvgpu.warp.LdMatrix16x16x8bOp( transpose: bool = False, num_matrices: int = 1, unpack_bits: cutlass.cute.typing.Optional.<class 'int'> | None = None, )
```

Bases: `BaseOp`

16x16 `ldmatrix` Operation with transpose and optional unpacking to 8b container.
Packed source container is 16x4b elements with 64b padding
or 16x6b elements with 32b padding (total 128b per 16 elements)

See the PTX documentation.
This operation corresponds to the `.m16n16` and the `.b4x16_p64`,``.b6x16\_p32``,``.b8`` qualifiers.

<a id="cutlass.cute.nvgpu.warp.LdMatrix16x16x8bOp.__init__"></a>
#### `cutlass.cute.nvgpu.warp.LdMatrix16x16x8bOp.__init__`

```python
__init__( transpose: bool = False, num_matrices: int = 1, unpack_bits: cutlass.cute.typing.Optional.<class 'int'> | None = None, ) → None
```

<a id="cutlass.cute.nvgpu.warp.StMatrix8x8x16bOp"></a>
### `cutlass.cute.nvgpu.warp.StMatrix8x8x16bOp`

```python
class cutlass.cute.nvgpu.warp.StMatrix8x8x16bOp( transpose: bool = False, num_matrices: int = 1, unpack_bits: cutlass.cute.typing.Optional.<class 'int'> | None = None, )
```

Bases: `BaseOp`

8x8 `stmatrix` Operation.

See the PTX documentation.
This operation corresponds to the `m8n8` qualifier.

<a id="cutlass.cute.nvgpu.warp.StMatrix8x8x16bOp.__init__"></a>
#### `cutlass.cute.nvgpu.warp.StMatrix8x8x16bOp.__init__`

```python
__init__( transpose: bool = False, num_matrices: int = 1, unpack_bits: cutlass.cute.typing.Optional.<class 'int'> | None = None, ) → None
```

<a id="cutlass.cute.nvgpu.warp.StMatrix16x8x8bOp"></a>
### `cutlass.cute.nvgpu.warp.StMatrix16x8x8bOp`

```python
class cutlass.cute.nvgpu.warp.StMatrix16x8x8bOp( transpose: bool = False, num_matrices: int = 1, unpack_bits: cutlass.cute.typing.Optional.<class 'int'> | None = None, )
```

Bases: `BaseOp`

16x8 `stmatrix` Operation.

See the PTX documentation.
This operation corresponds to the `m16n8` qualifier.

<a id="cutlass.cute.nvgpu.warp.StMatrix16x8x8bOp.__init__"></a>
#### `cutlass.cute.nvgpu.warp.StMatrix16x8x8bOp.__init__`

```python
__init__( transpose: bool = False, num_matrices: int = 1, unpack_bits: cutlass.cute.typing.Optional.<class 'int'> | None = None, ) → None
```
