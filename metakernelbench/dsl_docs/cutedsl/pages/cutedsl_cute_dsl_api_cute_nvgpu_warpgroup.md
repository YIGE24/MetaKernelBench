<!-- source: https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_warpgroup.html -->

<a id="module-cutlass.cute.nvgpu.warpgroup"></a>
# warpgroup submodule

<a id="cutlass.cute.nvgpu.warpgroup.OperandSource"></a>
### `cutlass.cute.nvgpu.warpgroup.OperandSource`

```python
class cutlass.cute.nvgpu.warpgroup.OperandSource(value)
```

Bases: `Enum`

An enumeration for the source memory location of the A input operand of the MMA.

<a id="cutlass.cute.nvgpu.warpgroup.Field"></a>
### `cutlass.cute.nvgpu.warpgroup.Field`

```python
class cutlass.cute.nvgpu.warpgroup.Field(value)
```

Bases: `Enum`

An enumeration for the fields of the MMA Atom that can be modified at runtime.

<a id="cutlass.cute.nvgpu.warpgroup.Field.ACCUMULATE"></a>
#### `cutlass.cute.nvgpu.warpgroup.Field.ACCUMULATE`

```python
ACCUMULATE = 'accum_c'
```

<a id="cutlass.cute.nvgpu.warpgroup.MmaF16BF16Op"></a>
### `cutlass.cute.nvgpu.warpgroup.MmaF16BF16Op`

```python
class cutlass.cute.nvgpu.warpgroup.MmaF16BF16Op( ab_dtype: Type[cutlass.cute.typing.Numeric], acc_dtype: Type[cutlass.cute.typing.Numeric], instruction_shape: cutlass.cute.typing.Shape, a_src: OperandSource, a_major_mode: OperandMajorMode | OperandMajorMode, b_major_mode: OperandMajorMode | OperandMajorMode, )
```

Bases: `MmaOp`

F16/BF16 warpgroup MMA Operation.

See the PTX documentation.
This Operation covers the instructions using the `.f16` or `.bf16` qualifiers for the input operands.

**Supported data type combinations:**

| A Data Type | B Data Type | Acc Type | Mma-K |
| --- | --- | --- | --- |
| F16 | F16 | F16, F32 | 16 |
| BF16 | BF16 | F32 | 16 |

**Supported architectures:** sm\_90a

**Constraints:**

- Mma-M = 64
- 8 <= Mma-N <= 256, step 8
- A and B support both K-major and MN-major (transpose) when A is in shared memory (descriptor).
  When A is in registers, only B can be transposed.

**Execution Model:**

- WGMMA is asynchronous and collective at warpgroup scope (4 contiguous warps).
  In user code, `cute.gemm(...)` should be issued warpgroup-uniformly.
- Before issuing `cute.gemm(...)`, call `cute.nvgpu.warpgroup.fence()` to order
  prior register writes to accumulator/A fragments with subsequent WGMMA reads.
- After issuing `cute.gemm(...)`, call `cute.nvgpu.warpgroup.commit_group()`.
  Use `cute.nvgpu.warpgroup.wait_group(N)` before consuming or reusing accumulator
  values from pending WGMMA groups.

```python
cute.nvgpu.warpgroup.fence()
cute.gemm(tiled_mma, acc, tCrA[tile_crd], tCrB[tile_crd], acc)
cute.nvgpu.warpgroup.commit_group()
cute.nvgpu.warpgroup.wait_group(1)
# ... pipeline continues ...
cute.nvgpu.warpgroup.wait_group(0)
```

<a id="cutlass.cute.nvgpu.warpgroup.MmaF16BF16Op.descriptive_name"></a>
#### `cutlass.cute.nvgpu.warpgroup.MmaF16BF16Op.descriptive_name`

```python
descriptive_name: ClassVar[str] = 'warpgroup F16/BF16 MMA Operation'
```

<a id="cutlass.cute.nvgpu.warpgroup.MmaF16BF16Op.__init__"></a>
#### `cutlass.cute.nvgpu.warpgroup.MmaF16BF16Op.__init__`

```python
__init__( ab_dtype: Type[cutlass.cute.typing.Numeric], acc_dtype: Type[cutlass.cute.typing.Numeric], instruction_shape: cutlass.cute.typing.Shape, a_src: OperandSource, a_major_mode: OperandMajorMode | OperandMajorMode, b_major_mode: OperandMajorMode | OperandMajorMode, ) → None
```

<a id="cutlass.cute.nvgpu.warpgroup.MmaF8Op"></a>
### `cutlass.cute.nvgpu.warpgroup.MmaF8Op`

```python
class cutlass.cute.nvgpu.warpgroup.MmaF8Op( a_dtype: Type[cutlass.cute.typing.Numeric], b_dtype: Type[cutlass.cute.typing.Numeric], acc_dtype: Type[cutlass.cute.typing.Numeric], instruction_shape: cutlass.cute.typing.Shape, a_src: OperandSource, a_major_mode: OperandMajorMode | OperandMajorMode, b_major_mode: OperandMajorMode | OperandMajorMode, )
```

Bases: `MmaOp`

F8 warpgroup MMA Operation.

See the PTX documentation.
This Operation covers the instructions using the `.e4m3` or `.e5m2` qualifiers for the input operands.

**Supported data type combinations:**

| A Data Type | B Data Type | Acc Type | Mma-K |
| --- | --- | --- | --- |
| E4M3, E5M2 | E4M3, E5M2 | F16, F32 | 32 |

**Supported architectures:** sm\_90a

**Constraints:**

- Mma-M = 64
- 8 <= Mma-N <= 256, step 8
- A and B data types are independent (mixed FP8 allowed)
- Transpose (MN-major) is not supported for A or B. Both operands must be K-major.

**Execution Model:**

- WGMMA is asynchronous and collective at warpgroup scope (4 contiguous warps).
  In user code, `cute.gemm(...)` should be issued warpgroup-uniformly.
- Before issuing `cute.gemm(...)`, call `cute.nvgpu.warpgroup.fence()` to order
  prior register writes to accumulator/A fragments with subsequent WGMMA reads.
- After issuing `cute.gemm(...)`, call `cute.nvgpu.warpgroup.commit_group()`.
  Use `cute.nvgpu.warpgroup.wait_group(N)` before consuming or reusing accumulator
  values from pending WGMMA groups.

```python
cute.nvgpu.warpgroup.fence()
cute.gemm(tiled_mma, acc, tCrA[tile_crd], tCrB[tile_crd], acc)
cute.nvgpu.warpgroup.commit_group()
cute.nvgpu.warpgroup.wait_group(1)
# ... pipeline continues ...
cute.nvgpu.warpgroup.wait_group(0)
```

<a id="cutlass.cute.nvgpu.warpgroup.MmaF8Op.descriptive_name"></a>
#### `cutlass.cute.nvgpu.warpgroup.MmaF8Op.descriptive_name`

```python
descriptive_name: ClassVar[str] = 'warpgroup F8 MMA Operation'
```

<a id="cutlass.cute.nvgpu.warpgroup.MmaF8Op.__init__"></a>
#### `cutlass.cute.nvgpu.warpgroup.MmaF8Op.__init__`

```python
__init__( a_dtype: Type[cutlass.cute.typing.Numeric], b_dtype: Type[cutlass.cute.typing.Numeric], acc_dtype: Type[cutlass.cute.typing.Numeric], instruction_shape: cutlass.cute.typing.Shape, a_src: OperandSource, a_major_mode: OperandMajorMode | OperandMajorMode, b_major_mode: OperandMajorMode | OperandMajorMode, ) → None
```

<a id="cutlass.cute.nvgpu.warpgroup.SmemLayoutAtomKind"></a>
### `cutlass.cute.nvgpu.warpgroup.SmemLayoutAtomKind`

```python
class cutlass.cute.nvgpu.warpgroup.SmemLayoutAtomKind(value)
```

Bases: `Enum`

Enum class for the kinds of SMEM layout atoms for SM90.

Given a swizzle kind, an SMEM layout atom is the compact layout of smallest size that can
be used to construct an SMEM layout using blocked product for operand A or B such that the
resulting layout is legal for both TMA and UMMA.

Note that there are other ways of creating legal layouts for operand A and B.

<a id="cutlass.cute.nvgpu.warpgroup.SmemLayoutAtomKind.MN_INTER"></a>
#### `cutlass.cute.nvgpu.warpgroup.SmemLayoutAtomKind.MN_INTER`

```python
MN_INTER = 1
```

<a id="cutlass.cute.nvgpu.warpgroup.SmemLayoutAtomKind.MN_SW32"></a>
#### `cutlass.cute.nvgpu.warpgroup.SmemLayoutAtomKind.MN_SW32`

```python
MN_SW32 = 2
```

<a id="cutlass.cute.nvgpu.warpgroup.SmemLayoutAtomKind.MN_SW64"></a>
#### `cutlass.cute.nvgpu.warpgroup.SmemLayoutAtomKind.MN_SW64`

```python
MN_SW64 = 3
```

<a id="cutlass.cute.nvgpu.warpgroup.SmemLayoutAtomKind.MN_SW128"></a>
#### `cutlass.cute.nvgpu.warpgroup.SmemLayoutAtomKind.MN_SW128`

```python
MN_SW128 = 4
```

<a id="cutlass.cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_INTER"></a>
#### `cutlass.cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_INTER`

```python
K_INTER = 5
```

<a id="cutlass.cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW32"></a>
#### `cutlass.cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW32`

```python
K_SW32 = 6
```

<a id="cutlass.cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW64"></a>
#### `cutlass.cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW64`

```python
K_SW64 = 7
```

<a id="cutlass.cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW128"></a>
#### `cutlass.cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW128`

```python
K_SW128 = 8
```

<a id="cutlass.cute.nvgpu.warpgroup.make_smem_layout_atom"></a>
### `cutlass.cute.nvgpu.warpgroup.make_smem_layout_atom`

```python
cutlass.cute.nvgpu.warpgroup.make_smem_layout_atom( kind: SmemLayoutAtomKind, element_type: Type[cutlass.cute.typing.Numeric], *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.ComposedLayout
```

Makes a SMEM layout Atom.

This function creates a composed layout in unit of elements consistent with the requested layout
Atom kind and element data type.

**Parameters:**

- **kind** ([*SmemLayoutAtomKind*](#cutlass.cute.nvgpu.warpgroup.SmemLayoutAtomKind)) – The kind of layout Atom
- **element\_type** (*Type*[*Numeric*]) – The element data type to construct the layout for

**Returns:**

The SMEM layout atom

**Return type:**

ComposedLayout

<a id="cutlass.cute.nvgpu.warpgroup.fence"></a>
### `cutlass.cute.nvgpu.warpgroup.fence`

```python
cutlass.cute.nvgpu.warpgroup.fence( *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

See the PTX documentation.

<a id="cutlass.cute.nvgpu.warpgroup.commit_group"></a>
### `cutlass.cute.nvgpu.warpgroup.commit_group`

```python
cutlass.cute.nvgpu.warpgroup.commit_group( *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

See the PTX documentation.

<a id="cutlass.cute.nvgpu.warpgroup.wait_group"></a>
### `cutlass.cute.nvgpu.warpgroup.wait_group`

```python
cutlass.cute.nvgpu.warpgroup.wait_group( group: Any, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

See the PTX documentation.
