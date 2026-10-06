<!-- source: https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_tcgen05.html -->

<a id="module-cutlass.cute.nvgpu.tcgen05"></a>
# tcgen05 submodule

<a id="cutlass.cute.nvgpu.tcgen05.Repetition"></a>
### `cutlass.cute.nvgpu.tcgen05.Repetition`

```python
class cutlass.cute.nvgpu.tcgen05.Repetition(value)
```

Bases: `Enum`

An enumeration for the number of repetitions of a given TMEM copy within the instruction.

<a id="cutlass.cute.nvgpu.tcgen05.Repetition.x1"></a>
#### `cutlass.cute.nvgpu.tcgen05.Repetition.x1`

```python
x1 = 1
```

<a id="cutlass.cute.nvgpu.tcgen05.Repetition.x2"></a>
#### `cutlass.cute.nvgpu.tcgen05.Repetition.x2`

```python
x2 = 2
```

<a id="cutlass.cute.nvgpu.tcgen05.Repetition.x4"></a>
#### `cutlass.cute.nvgpu.tcgen05.Repetition.x4`

```python
x4 = 4
```

<a id="cutlass.cute.nvgpu.tcgen05.Repetition.x8"></a>
#### `cutlass.cute.nvgpu.tcgen05.Repetition.x8`

```python
x8 = 8
```

<a id="cutlass.cute.nvgpu.tcgen05.Repetition.x16"></a>
#### `cutlass.cute.nvgpu.tcgen05.Repetition.x16`

```python
x16 = 16
```

<a id="cutlass.cute.nvgpu.tcgen05.Repetition.x32"></a>
#### `cutlass.cute.nvgpu.tcgen05.Repetition.x32`

```python
x32 = 32
```

<a id="cutlass.cute.nvgpu.tcgen05.Repetition.x64"></a>
#### `cutlass.cute.nvgpu.tcgen05.Repetition.x64`

```python
x64 = 64
```

<a id="cutlass.cute.nvgpu.tcgen05.Repetition.x128"></a>
#### `cutlass.cute.nvgpu.tcgen05.Repetition.x128`

```python
x128 = 128
```

<a id="cutlass.cute.nvgpu.tcgen05.TmemLoadRedOp"></a>
### `cutlass.cute.nvgpu.tcgen05.TmemLoadRedOp`

```python
class cutlass.cute.nvgpu.tcgen05.TmemLoadRedOp(value)
```

Bases: `Enum`

An enumeration for the possible reduce operations for TMEM load operations.

<a id="cutlass.cute.nvgpu.tcgen05.Pack"></a>
### `cutlass.cute.nvgpu.tcgen05.Pack`

```python
class cutlass.cute.nvgpu.tcgen05.Pack(value)
```

Bases: `Enum`

An enumeration for the possible packing patterns for TMEM to RMEM copies.

<a id="cutlass.cute.nvgpu.tcgen05.Pack.NONE"></a>
#### `cutlass.cute.nvgpu.tcgen05.Pack.NONE`

```python
NONE = 1
```

<a id="cutlass.cute.nvgpu.tcgen05.Pack.PACK_16b_IN_32b"></a>
#### `cutlass.cute.nvgpu.tcgen05.Pack.PACK_16b_IN_32b`

```python
PACK_16b_IN_32b = 2
```

<a id="cutlass.cute.nvgpu.tcgen05.Unpack"></a>
### `cutlass.cute.nvgpu.tcgen05.Unpack`

```python
class cutlass.cute.nvgpu.tcgen05.Unpack(value)
```

Bases: `Enum`

An enumeration for the possible unpacking patterns for RMEM to TMEM copies.

<a id="cutlass.cute.nvgpu.tcgen05.Unpack.NONE"></a>
#### `cutlass.cute.nvgpu.tcgen05.Unpack.NONE`

```python
NONE = 1
```

<a id="cutlass.cute.nvgpu.tcgen05.Unpack.UNPACK_32b_IN_16b"></a>
#### `cutlass.cute.nvgpu.tcgen05.Unpack.UNPACK_32b_IN_16b`

```python
UNPACK_32b_IN_16b = 2
```

<a id="cutlass.cute.nvgpu.tcgen05.Ld16x64bOp"></a>
### `cutlass.cute.nvgpu.tcgen05.Ld16x64bOp`

```python
class cutlass.cute.nvgpu.tcgen05.Ld16x64bOp( repeat: ~cutlass.cute.nvgpu.tcgen05.copy.Repetition = <Repetition.x1>, pack: ~cutlass.cute.nvgpu.tcgen05.copy.Pack = <Pack.NONE>, )
```

Bases: `_LdBase`

16x64b TMEM load Operation.

See the PTX documentation.
This Operation corresponds to the `.16x64b` qualifier.

<a id="cutlass.cute.nvgpu.tcgen05.Ld16x64bOp.__init__"></a>
#### `cutlass.cute.nvgpu.tcgen05.Ld16x64bOp.__init__`

```python
__init__( repeat: ~cutlass.cute.nvgpu.tcgen05.copy.Repetition = <Repetition.x1>, pack: ~cutlass.cute.nvgpu.tcgen05.copy.Pack = <Pack.NONE>, ) → None
```

<a id="cutlass.cute.nvgpu.tcgen05.Ld16x128bOp"></a>
### `cutlass.cute.nvgpu.tcgen05.Ld16x128bOp`

```python
class cutlass.cute.nvgpu.tcgen05.Ld16x128bOp( repeat: ~cutlass.cute.nvgpu.tcgen05.copy.Repetition = <Repetition.x1>, pack: ~cutlass.cute.nvgpu.tcgen05.copy.Pack = <Pack.NONE>, )
```

Bases: `_LdBase`

16x128b TMEM load Operation.

See the PTX documentation.
This Operation corresponds to the `.16x128b` qualifier.

<a id="cutlass.cute.nvgpu.tcgen05.Ld16x128bOp.__init__"></a>
#### `cutlass.cute.nvgpu.tcgen05.Ld16x128bOp.__init__`

```python
__init__( repeat: ~cutlass.cute.nvgpu.tcgen05.copy.Repetition = <Repetition.x1>, pack: ~cutlass.cute.nvgpu.tcgen05.copy.Pack = <Pack.NONE>, ) → None
```

<a id="cutlass.cute.nvgpu.tcgen05.Ld16x256bOp"></a>
### `cutlass.cute.nvgpu.tcgen05.Ld16x256bOp`

```python
class cutlass.cute.nvgpu.tcgen05.Ld16x256bOp( repeat: ~cutlass.cute.nvgpu.tcgen05.copy.Repetition = <Repetition.x1>, pack: ~cutlass.cute.nvgpu.tcgen05.copy.Pack = <Pack.NONE>, )
```

Bases: `_LdBase`

16x256b TMEM load Operation.

See the PTX documentation.
This Operation corresponds to the `.16x256b` qualifier.

<a id="cutlass.cute.nvgpu.tcgen05.Ld16x256bOp.__init__"></a>
#### `cutlass.cute.nvgpu.tcgen05.Ld16x256bOp.__init__`

```python
__init__( repeat: ~cutlass.cute.nvgpu.tcgen05.copy.Repetition = <Repetition.x1>, pack: ~cutlass.cute.nvgpu.tcgen05.copy.Pack = <Pack.NONE>, ) → None
```

<a id="cutlass.cute.nvgpu.tcgen05.Ld16x32bx2Op"></a>
### `cutlass.cute.nvgpu.tcgen05.Ld16x32bx2Op`

```python
class cutlass.cute.nvgpu.tcgen05.Ld16x32bx2Op( repeat: ~cutlass.cute.nvgpu.tcgen05.copy.Repetition = <Repetition.x1>, pack: ~cutlass.cute.nvgpu.tcgen05.copy.Pack = <Pack.NONE>, )
```

Bases: `_LdBase`

16x32bx2 TMEM load Operation.

See the PTX documentation.
This Operation corresponds to the `.16x32bx2` qualifier.

<a id="cutlass.cute.nvgpu.tcgen05.Ld16x32bx2Op.__init__"></a>
#### `cutlass.cute.nvgpu.tcgen05.Ld16x32bx2Op.__init__`

```python
__init__( repeat: ~cutlass.cute.nvgpu.tcgen05.copy.Repetition = <Repetition.x1>, pack: ~cutlass.cute.nvgpu.tcgen05.copy.Pack = <Pack.NONE>, ) → None
```

<a id="cutlass.cute.nvgpu.tcgen05.Ld32x32bOp"></a>
### `cutlass.cute.nvgpu.tcgen05.Ld32x32bOp`

```python
class cutlass.cute.nvgpu.tcgen05.Ld32x32bOp( repeat: ~cutlass.cute.nvgpu.tcgen05.copy.Repetition = <Repetition.x1>, pack: ~cutlass.cute.nvgpu.tcgen05.copy.Pack = <Pack.NONE>, )
```

Bases: `_LdBase`

32x32b TMEM load Operation.

See the PTX documentation.
This Operation corresponds to the `.32x32` qualifier.

<a id="cutlass.cute.nvgpu.tcgen05.Ld32x32bOp.__init__"></a>
#### `cutlass.cute.nvgpu.tcgen05.Ld32x32bOp.__init__`

```python
__init__( repeat: ~cutlass.cute.nvgpu.tcgen05.copy.Repetition = <Repetition.x1>, pack: ~cutlass.cute.nvgpu.tcgen05.copy.Pack = <Pack.NONE>, ) → None
```

<a id="cutlass.cute.nvgpu.tcgen05.St16x64bOp"></a>
### `cutlass.cute.nvgpu.tcgen05.St16x64bOp`

```python
class cutlass.cute.nvgpu.tcgen05.St16x64bOp( repeat: ~cutlass.cute.nvgpu.tcgen05.copy.Repetition, unpack: ~cutlass.cute.nvgpu.tcgen05.copy.Unpack = <Unpack.NONE>, )
```

Bases: `_StBase`

16x64b TMEM store Operation.

See the PTX documentation.
This Operation corresponds to the `.16x64` qualifier.

<a id="cutlass.cute.nvgpu.tcgen05.St16x64bOp.__init__"></a>
#### `cutlass.cute.nvgpu.tcgen05.St16x64bOp.__init__`

```python
__init__( repeat: ~cutlass.cute.nvgpu.tcgen05.copy.Repetition, unpack: ~cutlass.cute.nvgpu.tcgen05.copy.Unpack = <Unpack.NONE>, ) → None
```

<a id="cutlass.cute.nvgpu.tcgen05.St16x128bOp"></a>
### `cutlass.cute.nvgpu.tcgen05.St16x128bOp`

```python
class cutlass.cute.nvgpu.tcgen05.St16x128bOp( repeat: ~cutlass.cute.nvgpu.tcgen05.copy.Repetition, unpack: ~cutlass.cute.nvgpu.tcgen05.copy.Unpack = <Unpack.NONE>, )
```

Bases: `_StBase`

16x128b TMEM store Operation.

See the PTX documentation.
This Operation corresponds to the `.16x128` qualifier.

<a id="cutlass.cute.nvgpu.tcgen05.St16x128bOp.__init__"></a>
#### `cutlass.cute.nvgpu.tcgen05.St16x128bOp.__init__`

```python
__init__( repeat: ~cutlass.cute.nvgpu.tcgen05.copy.Repetition, unpack: ~cutlass.cute.nvgpu.tcgen05.copy.Unpack = <Unpack.NONE>, ) → None
```

<a id="cutlass.cute.nvgpu.tcgen05.St16x256bOp"></a>
### `cutlass.cute.nvgpu.tcgen05.St16x256bOp`

```python
class cutlass.cute.nvgpu.tcgen05.St16x256bOp( repeat: ~cutlass.cute.nvgpu.tcgen05.copy.Repetition, unpack: ~cutlass.cute.nvgpu.tcgen05.copy.Unpack = <Unpack.NONE>, )
```

Bases: `_StBase`

16x256b TMEM store Operation.

See the PTX documentation.
This Operation corresponds to the `.16x256` qualifier.

<a id="cutlass.cute.nvgpu.tcgen05.St16x256bOp.__init__"></a>
#### `cutlass.cute.nvgpu.tcgen05.St16x256bOp.__init__`

```python
__init__( repeat: ~cutlass.cute.nvgpu.tcgen05.copy.Repetition, unpack: ~cutlass.cute.nvgpu.tcgen05.copy.Unpack = <Unpack.NONE>, ) → None
```

<a id="cutlass.cute.nvgpu.tcgen05.St16x32bx2Op"></a>
### `cutlass.cute.nvgpu.tcgen05.St16x32bx2Op`

```python
class cutlass.cute.nvgpu.tcgen05.St16x32bx2Op( repeat: ~cutlass.cute.nvgpu.tcgen05.copy.Repetition, unpack: ~cutlass.cute.nvgpu.tcgen05.copy.Unpack = <Unpack.NONE>, )
```

Bases: `_StBase`

16x32x2b TMEM store Operation.

See the PTX documentation.
This Operation corresponds to the `.16x32x2` qualifier.

<a id="cutlass.cute.nvgpu.tcgen05.St16x32bx2Op.__init__"></a>
#### `cutlass.cute.nvgpu.tcgen05.St16x32bx2Op.__init__`

```python
__init__( repeat: ~cutlass.cute.nvgpu.tcgen05.copy.Repetition, unpack: ~cutlass.cute.nvgpu.tcgen05.copy.Unpack = <Unpack.NONE>, ) → None
```

<a id="cutlass.cute.nvgpu.tcgen05.St32x32bOp"></a>
### `cutlass.cute.nvgpu.tcgen05.St32x32bOp`

```python
class cutlass.cute.nvgpu.tcgen05.St32x32bOp( repeat: ~cutlass.cute.nvgpu.tcgen05.copy.Repetition, unpack: ~cutlass.cute.nvgpu.tcgen05.copy.Unpack = <Unpack.NONE>, )
```

Bases: `_StBase`

32x32b TMEM store Operation.

See the PTX documentation.
This Operation corresponds to the `.32x32` qualifier.

<a id="cutlass.cute.nvgpu.tcgen05.St32x32bOp.__init__"></a>
#### `cutlass.cute.nvgpu.tcgen05.St32x32bOp.__init__`

```python
__init__( repeat: ~cutlass.cute.nvgpu.tcgen05.copy.Repetition, unpack: ~cutlass.cute.nvgpu.tcgen05.copy.Unpack = <Unpack.NONE>, ) → None
```

<a id="cutlass.cute.nvgpu.tcgen05.OperandSource"></a>
### `cutlass.cute.nvgpu.tcgen05.OperandSource`

```python
class cutlass.cute.nvgpu.tcgen05.OperandSource(value)
```

Bases: `Enum`

An enumeration for the source memory location of the A input operand of the MMA.

<a id="cutlass.cute.nvgpu.tcgen05.CtaGroup"></a>
### `cutlass.cute.nvgpu.tcgen05.CtaGroup`

```python
class cutlass.cute.nvgpu.tcgen05.CtaGroup(value)
```

Bases: `Enum`

An enumeration for the `cta_group` qualifier of the MMA.

<a id="cutlass.cute.nvgpu.tcgen05.CtaGroup.ONE"></a>
#### `cutlass.cute.nvgpu.tcgen05.CtaGroup.ONE`

```python
ONE = 1
```

<a id="cutlass.cute.nvgpu.tcgen05.CtaGroup.TWO"></a>
#### `cutlass.cute.nvgpu.tcgen05.CtaGroup.TWO`

```python
TWO = 2
```

<a id="cutlass.cute.nvgpu.tcgen05.Field"></a>
### `cutlass.cute.nvgpu.tcgen05.Field`

```python
class cutlass.cute.nvgpu.tcgen05.Field(value)
```

Bases: `Enum`

An enumeration for the fields of the MMA Atom that can be modified at runtime.

<a id="cutlass.cute.nvgpu.tcgen05.Field.NEGATE_A"></a>
#### `cutlass.cute.nvgpu.tcgen05.Field.NEGATE_A`

```python
NEGATE_A = 'neg_a'
```

<a id="cutlass.cute.nvgpu.tcgen05.Field.NEGATE_B"></a>
#### `cutlass.cute.nvgpu.tcgen05.Field.NEGATE_B`

```python
NEGATE_B = 'neg_b'
```

<a id="cutlass.cute.nvgpu.tcgen05.Field.ACCUMULATE"></a>
#### `cutlass.cute.nvgpu.tcgen05.Field.ACCUMULATE`

```python
ACCUMULATE = 'accum_c'
```

<a id="cutlass.cute.nvgpu.tcgen05.Field.SFA"></a>
#### `cutlass.cute.nvgpu.tcgen05.Field.SFA`

```python
SFA = 'sf_a'
```

<a id="cutlass.cute.nvgpu.tcgen05.Field.SFB"></a>
#### `cutlass.cute.nvgpu.tcgen05.Field.SFB`

```python
SFB = 'sf_b'
```

<a id="cutlass.cute.nvgpu.tcgen05.Field.DISABLE_OUTPUT_LANE"></a>
#### `cutlass.cute.nvgpu.tcgen05.Field.DISABLE_OUTPUT_LANE`

```python
DISABLE_OUTPUT_LANE = 'disable_output_lane'
```

<a id="cutlass.cute.nvgpu.tcgen05.MmaTF32Op"></a>
### `cutlass.cute.nvgpu.tcgen05.MmaTF32Op`

```python
class cutlass.cute.nvgpu.tcgen05.MmaTF32Op( instruction_shape: cutlass.cute.typing.Shape, cta_group: CtaGroup, a_src: OperandSource, a_major_mode: OperandMajorMode | OperandMajorMode, b_major_mode: OperandMajorMode | OperandMajorMode, )
```

Bases: `MmaOp`

TF32 tcgen05 MMA Operation.

See the PTX documentation.
This Operation corresponds to the `.kind::tf32` qualifier.

**Supported data type combinations:**

| A Data Type | B Data Type | Acc Type | Mma-K |
| --- | --- | --- | --- |
| TF32 | TF32 | F32 | 8 |

**Supported architectures:** sm\_100a, sm\_100f, sm\_103a, sm\_103f, sm\_110a, sm\_110f

**Constraints:**

- CtaGroup.ONE: Mma-M in {64, 128}; 8 <= Mma-N <= 256, step 8
- CtaGroup.TWO: Mma-M in {128, 256}; 16 <= Mma-N <= 256, step 16
- A and B support both K-major and MN-major (transpose), but only with
  128B swizzling with 32B swizzle-atomicity. Transpose A requires
  a\_src=SMEM. When a\_src=TMEM, A is always K-major.

**Execution Model:**

- `cute.gemm(...)` (PTX: `tcgen05.mma`) is asynchronous. Issue granularity is
  single-thread (for `.cta_group::1`) or single-thread in a CTA pair
  (for `.cta_group::2`), per PTX issue rules.
- In user code, issue `cute.gemm(...)` as warp-uniform and do not wrap it in
  `elect_one()`.
- To observe/sequence MMA completion for dependent non-pipelined operations, call
  `cute.nvgpu.tcgen05.commit(...)` (PTX: `tcgen05.commit`) and follow the
  corresponding completion wait/synchronization path.
- For completion of tcgen05 TMEM load/store operations, use
  `tcgen05.wait::ld` / `tcgen05.wait::st` (PTX waits).
- For ordering tcgen05 operations across threads, use
  `tcgen05.fence::before_thread_sync` / `tcgen05.fence::after_thread_sync`
  (PTX fences) together with an execution-order synchronization mechanism.

```python
# CORRECT: warp-uniform tcgen05 MMA
cute.gemm(mma_atom, d, a, b, c)

# Signal completion of prior tcgen05 MMA operations
with cute.arch.elect_one():
    cute.nvgpu.tcgen05.commit(mbar_ptr, None, cta_group)
```

<a id="cutlass.cute.nvgpu.tcgen05.MmaTF32Op.descriptive_name"></a>
#### `cutlass.cute.nvgpu.tcgen05.MmaTF32Op.descriptive_name`

```python
descriptive_name: ClassVar[str] = 'tcgen05 TF32 MMA Operation'
```

<a id="cutlass.cute.nvgpu.tcgen05.MmaTF32Op.__init__"></a>
#### `cutlass.cute.nvgpu.tcgen05.MmaTF32Op.__init__`

```python
__init__( instruction_shape: cutlass.cute.typing.Shape, cta_group: CtaGroup, a_src: OperandSource, a_major_mode: OperandMajorMode | OperandMajorMode, b_major_mode: OperandMajorMode | OperandMajorMode, ) → None
```

<a id="cutlass.cute.nvgpu.tcgen05.MmaF16BF16Op"></a>
### `cutlass.cute.nvgpu.tcgen05.MmaF16BF16Op`

```python
class cutlass.cute.nvgpu.tcgen05.MmaF16BF16Op( ab_dtype: Type[cutlass.cute.typing.Numeric], acc_dtype: Type[cutlass.cute.typing.Numeric], instruction_shape: cutlass.cute.typing.Shape, cta_group: CtaGroup, a_src: OperandSource, a_major_mode: OperandMajorMode | OperandMajorMode, b_major_mode: OperandMajorMode | OperandMajorMode, )
```

Bases: `MmaOp`

F16/BF16 tcgen05 MMA Operation.

See the PTX documentation.
This Operation corresponds to the `.kind::f16` qualifier.

**Supported data type combinations:**

| A Data Type | B Data Type | Acc Type | Mma-K |
| --- | --- | --- | --- |
| F16 | F16 | F16, F32 | 16 |
| BF16 | BF16 | F32 | 16 |

**Supported architectures:** sm\_100a, sm\_100f, sm\_103a, sm\_103f, sm\_110a, sm\_110f

**Constraints:**

- CtaGroup.ONE: Mma-M in {64, 128}; 8 <= Mma-N <= 256, step 8
- CtaGroup.TWO: Mma-M in {128, 256}; 16 <= Mma-N <= 256, step 16
- A and B support both K-major and MN-major (transpose), except with
  128B swizzling with 32B swizzle-atomicity. Transpose A requires
  a\_src=SMEM. When a\_src=TMEM, A is always K-major.

**Execution Model:**

- `cute.gemm(...)` (PTX: `tcgen05.mma`) is asynchronous. Issue granularity is
  single-thread (for `.cta_group::1`) or single-thread in a CTA pair
  (for `.cta_group::2`), per PTX issue rules.
- In user code, issue `cute.gemm(...)` as warp-uniform and do not wrap it in
  `elect_one()`, as `elect_one()` insertion is handled by the compiler.
- To observe/sequence MMA completion for dependent non-pipelined operations, call
  `cute.nvgpu.tcgen05.commit(...)` (PTX: `tcgen05.commit`) and follow the
  corresponding completion wait/synchronization path.

```python
# CORRECT: warp-uniform tcgen05 MMA
cute.gemm(mma_atom, d, a, b, c)

# Signal completion of prior tcgen05 MMA operations
with cute.arch.elect_one():
    cute.nvgpu.tcgen05.commit(mbar_ptr, None, cta_group)
```

<a id="cutlass.cute.nvgpu.tcgen05.MmaF16BF16Op.descriptive_name"></a>
#### `cutlass.cute.nvgpu.tcgen05.MmaF16BF16Op.descriptive_name`

```python
descriptive_name: ClassVar[str] = 'tcgen05 F16/BF16 MMA Operation'
```

<a id="cutlass.cute.nvgpu.tcgen05.MmaF16BF16Op.__init__"></a>
#### `cutlass.cute.nvgpu.tcgen05.MmaF16BF16Op.__init__`

```python
__init__( ab_dtype: Type[cutlass.cute.typing.Numeric], acc_dtype: Type[cutlass.cute.typing.Numeric], instruction_shape: cutlass.cute.typing.Shape, cta_group: CtaGroup, a_src: OperandSource, a_major_mode: OperandMajorMode | OperandMajorMode, b_major_mode: OperandMajorMode | OperandMajorMode, ) → None
```

<a id="cutlass.cute.nvgpu.tcgen05.MmaI8Op"></a>
### `cutlass.cute.nvgpu.tcgen05.MmaI8Op`

```python
class cutlass.cute.nvgpu.tcgen05.MmaI8Op( ab_dtype: Type[cutlass.cute.typing.Numeric], instruction_shape: cutlass.cute.typing.Shape, cta_group: CtaGroup, a_src: OperandSource, a_major_mode: OperandMajorMode | OperandMajorMode, b_major_mode: OperandMajorMode | OperandMajorMode, )
```

Bases: `MmaOp`

I8 tcgen05 MMA Operation.

See the PTX documentation.
This Operation corresponds to the `.kind::i8` qualifier.

**Supported data type combinations:**

| A Data Type | B Data Type | Acc Type | Mma-K |
| --- | --- | --- | --- |
| Int8, Uint8 | Int8, Uint8 | Int32 | 32 |

**Supported architectures:** sm\_100a, sm\_100f, sm\_103a, sm\_103f, sm\_110a, sm\_110f

**Constraints:**

- CtaGroup.ONE: Mma-M in {64, 128}; Mma-N in {8, 16, 24, 32, 48, 64, 80, …, 256}
  (step 8 for Mma-N <= 32, then step 16 for Mma-N > 32; values like 40, 56 are invalid)
- CtaGroup.TWO: Mma-M in {128, 256}; 16 <= Mma-N <= 256, step 16
- A and B signedness are independent (mixed signed/unsigned allowed)
- A and B support both K-major and MN-major (transpose), except with
  128B swizzling with 32B swizzle-atomicity. Transpose A requires
  a\_src=SMEM. When a\_src=TMEM, A is always K-major.
- With B MN-major (8-bit B transpose): Mma-N step changes to 16 for CG1, 32 for CG2.

**Execution Model:**

- `cute.gemm(...)` (PTX: `tcgen05.mma`) is asynchronous. Issue granularity is
  single-thread (for `.cta_group::1`) or single-thread in a CTA pair
  (for `.cta_group::2`), per PTX issue rules.
- In user code, issue `cute.gemm(...)` as warp-uniform and do not wrap it in
  `elect_one()`, as `elect_one()` insertion is handled by the compiler.
- To observe/sequence MMA completion for dependent non-pipelined operations, call
  `cute.nvgpu.tcgen05.commit(...)` (PTX: `tcgen05.commit`) and follow the
  corresponding completion wait/synchronization path.

```python
# CORRECT: warp-uniform tcgen05 MMA
cute.gemm(mma_atom, d, a, b, c)

# Signal completion of prior tcgen05 MMA operations
with cute.arch.elect_one():
    cute.nvgpu.tcgen05.commit(mbar_ptr, None, cta_group)
```

<a id="cutlass.cute.nvgpu.tcgen05.MmaI8Op.descriptive_name"></a>
#### `cutlass.cute.nvgpu.tcgen05.MmaI8Op.descriptive_name`

```python
descriptive_name: ClassVar[str] = 'tcgen05 I8 MMA Operation'
```

<a id="cutlass.cute.nvgpu.tcgen05.MmaI8Op.__init__"></a>
#### `cutlass.cute.nvgpu.tcgen05.MmaI8Op.__init__`

```python
__init__( ab_dtype: Type[cutlass.cute.typing.Numeric], instruction_shape: cutlass.cute.typing.Shape, cta_group: CtaGroup, a_src: OperandSource, a_major_mode: OperandMajorMode | OperandMajorMode, b_major_mode: OperandMajorMode | OperandMajorMode, ) → None
```

<a id="cutlass.cute.nvgpu.tcgen05.MmaFP8Op"></a>
### `cutlass.cute.nvgpu.tcgen05.MmaFP8Op`

```python
class cutlass.cute.nvgpu.tcgen05.MmaFP8Op(**kwargs)
```

Bases: `MmaOp`

F8 tcgen05 MMA Operation.

Deprecated since version Use: [`MmaF8F6F4Op`](#cutlass.cute.nvgpu.tcgen05.MmaF8F6F4Op) instead.

See the PTX documentation.

**Supported data type combinations:**

| A Data Type | B Data Type | Acc Type | Mma-K |
| --- | --- | --- | --- |
| E4M3, E5M2 | E4M3, E5M2 | F16, F32 | 32 |

**Supported architectures:** sm\_100a, sm\_100f, sm\_103a, sm\_103f, sm\_110a, sm\_110f

**Constraints:**

- A and B data types must be the same
- CtaGroup.ONE: Mma-M in {64, 128}; 8 <= Mma-N <= 256, step 8
- CtaGroup.TWO: Mma-M in {128, 256}; 16 <= Mma-N <= 256, step 16
- With B-major=MN: Mma-N step doubles (16 for CG1, 32 for CG2)
- A and B support both K-major and MN-major (transpose), except with
  128B swizzling with 32B swizzle-atomicity. Transpose A requires
  a\_src=SMEM. When a\_src=TMEM, A is always K-major.
- With 8-bit B transpose (MN-major): N step changes to 16 for CG1, 32 for CG2.

**Execution Model:**

- `cute.gemm(...)` (PTX: `tcgen05.mma`) is asynchronous. Issue granularity is
  single-thread (for `.cta_group::1`) or single-thread in a CTA pair
  (for `.cta_group::2`), per PTX issue rules.
- In user code, issue `cute.gemm(...)` as warp-uniform and do not wrap it in
  `elect_one()`, as `elect_one()` insertion is handled by the compiler.
- To observe/sequence MMA completion for dependent non-pipelined operations, call
  `cute.nvgpu.tcgen05.commit(...)` (PTX: `tcgen05.commit`) and follow the
  corresponding completion wait/synchronization path.

```python
# CORRECT: warp-uniform tcgen05 MMA
cute.gemm(mma_atom, d, a, b, c)

# Signal completion of prior tcgen05 MMA operations
with cute.arch.elect_one():
    cute.nvgpu.tcgen05.commit(mbar_ptr, None, cta_group)
```

<a id="cutlass.cute.nvgpu.tcgen05.MmaFP8Op.descriptive_name"></a>
#### `cutlass.cute.nvgpu.tcgen05.MmaFP8Op.descriptive_name`

```python
descriptive_name: ClassVar[str] = 'tcgen05 F8 MMA Operation'
```

<a id="cutlass.cute.nvgpu.tcgen05.MmaFP8Op.__init__"></a>
#### `cutlass.cute.nvgpu.tcgen05.MmaFP8Op.__init__`

```python
__init__( ab_dtype: Type[cutlass.cute.typing.Numeric], acc_dtype: Type[cutlass.cute.typing.Numeric], instruction_shape: cutlass.cute.typing.Shape, cta_group: CtaGroup, a_src: OperandSource, a_major_mode: OperandMajorMode | OperandMajorMode, b_major_mode: OperandMajorMode | OperandMajorMode, ) → None
```

<a id="cutlass.cute.nvgpu.tcgen05.MmaF8F6F4Op"></a>
### `cutlass.cute.nvgpu.tcgen05.MmaF8F6F4Op`

```python
class cutlass.cute.nvgpu.tcgen05.MmaF8F6F4Op( a_dtype: Type[cutlass.cute.typing.Numeric], b_dtype: Type[cutlass.cute.typing.Numeric], acc_dtype: Type[cutlass.cute.typing.Numeric], instruction_shape: cutlass.cute.typing.Shape, cta_group: CtaGroup, a_src: OperandSource, a_major_mode: OperandMajorMode | OperandMajorMode, b_major_mode: OperandMajorMode | OperandMajorMode, )
```

Bases: `MmaOp`

F8F6F4 tcgen05 MMA Operation.

See the PTX documentation.

**Supported data type combinations:**

| A Data Type | B Data Type | Acc Type | Mma-K |
| --- | --- | --- | --- |
| E4M3, E5M2, E3M2, E2M3, E2M1 | E4M3, E5M2, E3M2, E2M3, E2M1 | F16, F32 | 32 |

**Supported architectures:** sm\_100a, sm\_100f, sm\_103a, sm\_103f, sm\_110a, sm\_110f

**Constraints:**

- A and B data types are independent (mixed F8/F6/F4 allowed)
- CtaGroup.ONE: Mma-M in {64, 128}; 8 <= Mma-N <= 256, step 8
- CtaGroup.TWO: Mma-M in {128, 256}; 16 <= Mma-N <= 256, step 16
- With B-major=MN: Mma-N step doubles (16 for CG1, 32 for CG2)
- A and B support both K-major and MN-major (transpose), except with
  128B swizzling with 32B swizzle-atomicity. Transpose A requires
  a\_src=SMEM. When a\_src=TMEM, A is always K-major.
- With 8-bit B transpose (MN-major): N step changes to 16 for CG1, 32 for CG2.

**Execution Model:**

- `cute.gemm(...)` (PTX: `tcgen05.mma`) is asynchronous. Issue granularity is
  single-thread (for `.cta_group::1`) or single-thread in a CTA pair
  (for `.cta_group::2`), per PTX issue rules.
- In user code, issue `cute.gemm(...)` as warp-uniform and do not wrap it in
  `elect_one()`, as `elect_one()` insertion is handled by the compiler.
- To observe/sequence MMA completion for dependent non-pipelined operations, call
  `cute.nvgpu.tcgen05.commit(...)` (PTX: `tcgen05.commit`) and follow the
  corresponding completion wait/synchronization path.

```python
# CORRECT: warp-uniform tcgen05 MMA
cute.gemm(mma_atom, d, a, b, c)

# Signal completion of prior tcgen05 MMA operations
with cute.arch.elect_one():
    cute.nvgpu.tcgen05.commit(mbar_ptr, None, cta_group)
```

<a id="cutlass.cute.nvgpu.tcgen05.MmaF8F6F4Op.descriptive_name"></a>
#### `cutlass.cute.nvgpu.tcgen05.MmaF8F6F4Op.descriptive_name`

```python
descriptive_name: ClassVar[str] = 'tcgen05 F8F6F4 MMA Operation'
```

<a id="cutlass.cute.nvgpu.tcgen05.MmaF8F6F4Op.__init__"></a>
#### `cutlass.cute.nvgpu.tcgen05.MmaF8F6F4Op.__init__`

```python
__init__( a_dtype: Type[cutlass.cute.typing.Numeric], b_dtype: Type[cutlass.cute.typing.Numeric], acc_dtype: Type[cutlass.cute.typing.Numeric], instruction_shape: cutlass.cute.typing.Shape, cta_group: CtaGroup, a_src: OperandSource, a_major_mode: OperandMajorMode | OperandMajorMode, b_major_mode: OperandMajorMode | OperandMajorMode, ) → None
```

<a id="cutlass.cute.nvgpu.tcgen05.MmaMXF8Op"></a>
### `cutlass.cute.nvgpu.tcgen05.MmaMXF8Op`

```python
class cutlass.cute.nvgpu.tcgen05.MmaMXF8Op(**kwargs)
```

Bases: `BlockScaledMmaOp`

MXF8 tcgen05 BlockScaled MMA Operation.

Deprecated since version Use: [`MmaMXF8F6F4Op`](#cutlass.cute.nvgpu.tcgen05.MmaMXF8F6F4Op) instead.

See the PTX documentation.
This Operation corresponds to the `.kind::mxf8f6f4` qualifier.

**Supported data type combinations:**

| A Data Type | B Data Type | SF Data Type | Acc Type | Mma-K | SF Vec Size |
| --- | --- | --- | --- | --- | --- |
| E4M3, E5M2 | E4M3, E5M2 | UE8M0 | F32 | 32 | 32 |

**Supported architectures:** sm\_100a, sm\_103a

**Constraints:**

- A and B data types must be the same
- CtaGroup.ONE: Mma-M = 128; 8 <= Mma-N <= 256, step 8
- CtaGroup.TWO: Mma-M in {128, 256}; 16 <= Mma-N <= 256, step 16
- A and B support both K-major and MN-major (transpose), except with
  128B swizzling with 32B swizzle-atomicity. Transpose A requires
  a\_src=SMEM. When a\_src=TMEM, A is always K-major.
- With 8-bit B transpose (MN-major): N step changes to 16 for CtaGroup.ONE, 32 for CtaGroup.TWO.

**Execution Model:**

- `cute.gemm(...)` (PTX: `tcgen05.mma`) is asynchronous. Issue granularity is
  single-thread (for `.cta_group::1`) or single-thread in a CTA pair
  (for `.cta_group::2`), per PTX issue rules.
- In user code, issue `cute.gemm(...)` as warp-uniform and do not wrap it in
  `elect_one()`, as `elect_one()` insertion is handled by the compiler.
- For block-scaled MMA, pass A and B as paired operands in `cute.gemm(...)`:
  `[a, sfa]` and `[b, sfb]`.
- To observe/sequence MMA completion for dependent non-pipelined operations, call
  `cute.nvgpu.tcgen05.commit(...)` (PTX: `tcgen05.commit`) and follow the
  corresponding completion wait/synchronization path.

```python
# CORRECT: warp-uniform tcgen05 MMA
cute.gemm(mma_atom, d, [a, sfa], [b, sfb], c)

# Signal completion of prior tcgen05 MMA operations
with cute.arch.elect_one():
    cute.nvgpu.tcgen05.commit(mbar_ptr, None, cta_group)
```

<a id="cutlass.cute.nvgpu.tcgen05.MmaMXF8Op.descriptive_name"></a>
#### `cutlass.cute.nvgpu.tcgen05.MmaMXF8Op.descriptive_name`

```python
descriptive_name: ClassVar[str] = 'tcgen05 MXF8 BlockScaled MMA Operation'
```

<a id="cutlass.cute.nvgpu.tcgen05.MmaMXF8Op.__init__"></a>
#### `cutlass.cute.nvgpu.tcgen05.MmaMXF8Op.__init__`

```python
__init__( ab_dtype: Type[cutlass.cute.typing.Numeric], instruction_shape: cutlass.cute.typing.Shape, cta_group: CtaGroup, a_src: OperandSource, a_major_mode: OperandMajorMode | OperandMajorMode, b_major_mode: OperandMajorMode | OperandMajorMode, ) → None
```

<a id="cutlass.cute.nvgpu.tcgen05.MmaMXF8F6F4Op"></a>
### `cutlass.cute.nvgpu.tcgen05.MmaMXF8F6F4Op`

```python
class cutlass.cute.nvgpu.tcgen05.MmaMXF8F6F4Op( a_dtype: Type[cutlass.cute.typing.Numeric], b_dtype: Type[cutlass.cute.typing.Numeric], instruction_shape: cutlass.cute.typing.Shape, cta_group: CtaGroup, a_src: OperandSource, a_major_mode: OperandMajorMode | OperandMajorMode, b_major_mode: OperandMajorMode | OperandMajorMode, )
```

Bases: `BlockScaledMmaOp`

MXF8F6F4 tcgen05 BlockScaled MMA Operation.

See the PTX documentation.
This Operation corresponds to the `.kind::mxf8f6f4` qualifier.

**Supported data type combinations:**

| A Data Type | B Data Type | SF Data Type | Acc Type | Mma-K | SF Vec Size |
| --- | --- | --- | --- | --- | --- |
| E4M3, E5M2, E3M2, E2M3, E2M1 | E4M3, E5M2, E3M2, E2M3, E2M1 | UE8M0 | F32 | 32 | 32 |

**Supported architectures:** sm\_100a, sm\_103a

**Constraints:**

- A and B data types are independent (mixed F8/F6/F4 allowed)
- CtaGroup.ONE: Mma-M = 128; 8 <= Mma-N <= 256, step 8
- CtaGroup.TWO: Mma-M in {128, 256}; 16 <= Mma-N <= 256, step 16
- A and B support both K-major and MN-major (transpose), except with
  128B swizzling with 32B swizzle-atomicity. Transpose A requires
  a\_src=SMEM. When a\_src=TMEM, A is always K-major.
- With 8-bit B transpose (MN-major): N step changes to 16 for CtaGroup.ONE, 32 for CtaGroup.TWO.

**Execution Model:**

- `cute.gemm(...)` (PTX: `tcgen05.mma`) is asynchronous. Issue granularity is
  single-thread (for `.cta_group::1`) or single-thread in a CTA pair
  (for `.cta_group::2`), per PTX issue rules.
- In user code, issue `cute.gemm(...)` as warp-uniform and do not wrap it in
  `elect_one()`, as `elect_one()` insertion is handled by the compiler.
- For block-scaled MMA, pass A and B as paired operands in `cute.gemm(...)`:
  `[a, sfa]` and `[b, sfb]`.
- To observe/sequence MMA completion for dependent non-pipelined operations, call
  `cute.nvgpu.tcgen05.commit(...)` (PTX: `tcgen05.commit`) and follow the
  corresponding completion wait/synchronization path.

```python
# CORRECT: warp-uniform tcgen05 MMA
cute.gemm(mma_atom, d, [a, sfa], [b, sfb], c)

# Signal completion of prior tcgen05 MMA operations
with cute.arch.elect_one():
    cute.nvgpu.tcgen05.commit(mbar_ptr, None, cta_group)
```

<a id="cutlass.cute.nvgpu.tcgen05.MmaMXF8F6F4Op.descriptive_name"></a>
#### `cutlass.cute.nvgpu.tcgen05.MmaMXF8F6F4Op.descriptive_name`

```python
descriptive_name: ClassVar[str] = 'tcgen05 MXF8F6F4 BlockScaled MMA Operation'
```

<a id="cutlass.cute.nvgpu.tcgen05.MmaMXF8F6F4Op.__init__"></a>
#### `cutlass.cute.nvgpu.tcgen05.MmaMXF8F6F4Op.__init__`

```python
__init__( a_dtype: Type[cutlass.cute.typing.Numeric], b_dtype: Type[cutlass.cute.typing.Numeric], instruction_shape: cutlass.cute.typing.Shape, cta_group: CtaGroup, a_src: OperandSource, a_major_mode: OperandMajorMode | OperandMajorMode, b_major_mode: OperandMajorMode | OperandMajorMode, ) → None
```

<a id="cutlass.cute.nvgpu.tcgen05.MmaMXF4Op"></a>
### `cutlass.cute.nvgpu.tcgen05.MmaMXF4Op`

```python
class cutlass.cute.nvgpu.tcgen05.MmaMXF4Op( instruction_shape: cutlass.cute.typing.Shape, cta_group: CtaGroup, a_src: OperandSource, )
```

Bases: `BlockScaledMmaOp`

MXF4 tcgen05 BlockScaled MMA Operation.

See the PTX documentation.
This Operation corresponds to the `.kind::mxf4` qualifier.

**Supported data type combinations:**

| A Data Type | B Data Type | SF Data Type | Acc Type | Mma-K | SF Vec Size |
| --- | --- | --- | --- | --- | --- |
| E2M1 | E2M1 | UE8M0 | F32 | 64 | 32 |

**Supported architectures:** sm\_100a, sm\_103a

**Constraints:**

- CtaGroup.ONE: Mma-M = 128; 8 <= Mma-N <= 256, step 8
- CtaGroup.TWO: Mma-M in {128, 256}; 16 <= Mma-N <= 256, step 16
- Transpose (MN-major) is not supported. Both A and B must be K-major.

**Execution Model:**

- `cute.gemm(...)` (PTX: `tcgen05.mma`) is asynchronous. Issue granularity is
  single-thread (for `.cta_group::1`) or single-thread in a CTA pair
  (for `.cta_group::2`), per PTX issue rules.
- In user code, issue `cute.gemm(...)` as warp-uniform and do not wrap it in
  `elect_one()`, as `elect_one()` insertion is handled by the compiler.
- For block-scaled MMA, pass A and B as paired operands in `cute.gemm(...)`:
  `[a, sfa]` and `[b, sfb]`.
- To observe/sequence MMA completion for dependent non-pipelined operations, call
  `cute.nvgpu.tcgen05.commit(...)` (PTX: `tcgen05.commit`) and follow the
  corresponding completion wait/synchronization path.

```python
# CORRECT: warp-uniform tcgen05 MMA
cute.gemm(mma_atom, d, [a, sfa], [b, sfb], c)

# Signal completion of prior tcgen05 MMA operations
with cute.arch.elect_one():
    cute.nvgpu.tcgen05.commit(mbar_ptr, None, cta_group)
```

<a id="cutlass.cute.nvgpu.tcgen05.MmaMXF4Op.descriptive_name"></a>
#### `cutlass.cute.nvgpu.tcgen05.MmaMXF4Op.descriptive_name`

```python
descriptive_name: ClassVar[str] = 'tcgen05 MXF4 BlockScaled MMA Operation'
```

<a id="cutlass.cute.nvgpu.tcgen05.MmaMXF4Op.__init__"></a>
#### `cutlass.cute.nvgpu.tcgen05.MmaMXF4Op.__init__`

```python
__init__( instruction_shape: cutlass.cute.typing.Shape, cta_group: CtaGroup, a_src: OperandSource, ) → None
```

<a id="cutlass.cute.nvgpu.tcgen05.MmaMXF4NVF4Op"></a>
### `cutlass.cute.nvgpu.tcgen05.MmaMXF4NVF4Op`

```python
class cutlass.cute.nvgpu.tcgen05.MmaMXF4NVF4Op( sf_dtype: Type[cutlass.cute.typing.Numeric], instruction_shape: cutlass.cute.typing.Shape, cta_group: CtaGroup, a_src: OperandSource, )
```

Bases: `BlockScaledMmaOp`

MXF4NVF4 tcgen05 BlockScaled MMA Operation.

See the PTX documentation.
This Operation corresponds to the `.kind::mxf4nvf4` qualifier.

**Supported data type combinations:**

| A Data Type | B Data Type | SF Data Type | Acc Type | Mma-K | SF Vec Size |
| --- | --- | --- | --- | --- | --- |
| E2M1 | E2M1 | UE8M0, UE4M3 | F32 | 64 | 16 |

**Supported architectures:** sm\_100a, sm\_103a

**Constraints:**

- CtaGroup.ONE: Mma-M = 128; 8 <= Mma-N <= 256, step 8
- CtaGroup.TWO: Mma-M in {128, 256}; 16 <= Mma-N <= 256, step 16
- Transpose (MN-major) is not supported. Both A and B must be K-major.

**Execution Model:**

- `cute.gemm(...)` (PTX: `tcgen05.mma`) is asynchronous. Issue granularity is
  single-thread (for `.cta_group::1`) or single-thread in a CTA pair
  (for `.cta_group::2`), per PTX issue rules.
- In user code, issue `cute.gemm(...)` as warp-uniform and do not wrap it in
  `elect_one()`, as `elect_one()` insertion is handled by the compiler.
- For block-scaled MMA, pass A and B as paired operands in `cute.gemm(...)`:
  `[a, sfa]` and `[b, sfb]`.
- To observe/sequence MMA completion for dependent non-pipelined operations, call
  `cute.nvgpu.tcgen05.commit(...)` (PTX: `tcgen05.commit`) and follow the
  corresponding completion wait/synchronization path.

```python
# CORRECT: warp-uniform tcgen05 MMA
cute.gemm(mma_atom, d, [a, sfa], [b, sfb], c)

# Signal completion of prior tcgen05 MMA operations
with cute.arch.elect_one():
    cute.nvgpu.tcgen05.commit(mbar_ptr, None, cta_group)
```

<a id="cutlass.cute.nvgpu.tcgen05.MmaMXF4NVF4Op.descriptive_name"></a>
#### `cutlass.cute.nvgpu.tcgen05.MmaMXF4NVF4Op.descriptive_name`

```python
descriptive_name: ClassVar[str] = 'tcgen05 MXF4NVF4 BlockScaled MMA Operation'
```

<a id="cutlass.cute.nvgpu.tcgen05.MmaMXF4NVF4Op.__init__"></a>
#### `cutlass.cute.nvgpu.tcgen05.MmaMXF4NVF4Op.__init__`

```python
__init__( sf_dtype: Type[cutlass.cute.typing.Numeric], instruction_shape: cutlass.cute.typing.Shape, cta_group: CtaGroup, a_src: OperandSource, ) → None
```

<a id="cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind"></a>
### `cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind`

```python
class cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind(value)
```

Bases: `Enum`

Enum class for the kinds of SMEM layout atoms for SM100.

Given a swizzle kind, an SMEM layout atom is the compact layout of smallest size that can be
used to construct an SMEM layout using blocked product for operand A or B such that the
resulting layout is legal for both TMA and UMMA.

Note that there are other ways of creating legal layouts for operand A and B.

<a id="cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind.MN_INTER"></a>
#### `cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind.MN_INTER`

```python
MN_INTER = 1
```

<a id="cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind.MN_SW32"></a>
#### `cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind.MN_SW32`

```python
MN_SW32 = 2
```

<a id="cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind.MN_SW64"></a>
#### `cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind.MN_SW64`

```python
MN_SW64 = 3
```

<a id="cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind.MN_SW128"></a>
#### `cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind.MN_SW128`

```python
MN_SW128 = 4
```

<a id="cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind.MN_SW128_32B"></a>
#### `cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind.MN_SW128_32B`

```python
MN_SW128_32B = 5
```

<a id="cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind.K_INTER"></a>
#### `cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind.K_INTER`

```python
K_INTER = 6
```

<a id="cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind.K_SW32"></a>
#### `cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind.K_SW32`

```python
K_SW32 = 7
```

<a id="cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind.K_SW64"></a>
#### `cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind.K_SW64`

```python
K_SW64 = 8
```

<a id="cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind.K_SW128"></a>
#### `cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind.K_SW128`

```python
K_SW128 = 9
```

<a id="cutlass.cute.nvgpu.tcgen05.make_smem_layout_atom"></a>
### `cutlass.cute.nvgpu.tcgen05.make_smem_layout_atom`

```python
cutlass.cute.nvgpu.tcgen05.make_smem_layout_atom( kind: SmemLayoutAtomKind, element_type: Type[cutlass.cute.typing.Numeric], *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.ComposedLayout
```

Makes a SMEM layout Atom.

This function creates a composed layout in unit of elements consistent with the requested layout
Atom kind and element data type.

**Parameters:**

- **kind** ([*SmemLayoutAtomKind*](#cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind)) – The kind of layout Atom
- **element\_type** (*Type*[*Numeric*]) – The element data type to construct the layout for

**Returns:**

The SMEM layout atom

**Return type:**

ComposedLayout

<a id="cutlass.cute.nvgpu.tcgen05.tile_to_mma_shape"></a>
### `cutlass.cute.nvgpu.tcgen05.tile_to_mma_shape`

```python
cutlass.cute.nvgpu.tcgen05.tile_to_mma_shape( atom: cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout, mma_tile_shape: cutlass.cute.typing.Shape, order: cutlass.cute.typing.IntTuple | None = None, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout
```

Tiles a layout to an MMA shape.

<a id="cutlass.cute.nvgpu.tcgen05.commit"></a>
### `cutlass.cute.nvgpu.tcgen05.commit`

```python
cutlass.cute.nvgpu.tcgen05.commit( mbar_ptr: cutlass.cute.typing.Pointer, mask: ~typing.Any | None = None, cta_group: ~cutlass.cute.nvgpu.tcgen05.mma.CtaGroup = <CtaGroup.ONE>, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

Perform an arrive operation on a mbarrier upon completion of previous MMA operations.

**Single-Thread Execution Required - DSL Does NOT Handle Automatically**: This operation
**must** be wrapped in `cute.arch.elect_one()`. Without `elect_one()`, all 32
threads in the warp will execute the commit, causing 32x redundant `tcgen05.commit` PTX instructions.

```python
# CORRECT: Wrap tcgen05.commit in elect_one
with cute.arch.elect_one():
    tcgen05.commit(barrier_ptr, None, cta_group)

# WRONG: Without elect_one, all threads execute (32x redundant)
tcgen05.commit(barrier_ptr, None, cta_group)
```

**Parameters:**

- **mbar\_ptr** (*Pointer*) – A pointer to the mbarrier in SMEM
- **mask** (*Int*) – An optional multicast mask for the CTAs in the cluster to signal arrival to
- **cta\_group** ([*CtaGroup*](#cutlass.cute.nvgpu.tcgen05.CtaGroup)) – The CTA group size for the operation (ONE or TWO)

> **See also**
>
> - `cute.arch.elect_one()` - **REQUIRED** wrapper for single-thread execution
> - `cute.arch.mbarrier_arrive()` - General barrier arrive operation

<a id="cutlass.cute.nvgpu.tcgen05.is_tmem_load"></a>
### `cutlass.cute.nvgpu.tcgen05.is_tmem_load`

```python
cutlass.cute.nvgpu.tcgen05.is_tmem_load(atom: CopyAtom) → bool
```

Returns whether a CopyAtom instance is a TMEM load.

<a id="cutlass.cute.nvgpu.tcgen05.is_tmem_store"></a>
### `cutlass.cute.nvgpu.tcgen05.is_tmem_store`

```python
cutlass.cute.nvgpu.tcgen05.is_tmem_store(atom: CopyAtom) → bool
```

Returns whether a CopyAtom instance is a TMEM store.

<a id="cutlass.cute.nvgpu.tcgen05.get_tmem_copy_properties"></a>
### `cutlass.cute.nvgpu.tcgen05.get_tmem_copy_properties`

```python
cutlass.cute.nvgpu.tcgen05.get_tmem_copy_properties( atom: CopyAtom, ) → Tuple[int, int, int, Pack | Unpack]
```

Returns the properties of a TMEM copy atom (number of data paths, bits, repetitions,
and whether packing/unpacking is used).

<a id="cutlass.cute.nvgpu.tcgen05.find_tmem_tensor_col_offset"></a>
### `cutlass.cute.nvgpu.tcgen05.find_tmem_tensor_col_offset`

```python
cutlass.cute.nvgpu.tcgen05.find_tmem_tensor_col_offset( tmem_tensor: cutlass.cute.typing.Tensor, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Int
```

Computes the TMEM column offset given a TMEM tensor.

**Parameters:**

**tmem\_tensor** (*Tensor*) – The TMEM tensor to use to compute the columns offset

**Returns:**

The columns offset

**Return type:**

Int

<a id="cutlass.cute.nvgpu.tcgen05.make_tmem_copy"></a>
### `cutlass.cute.nvgpu.tcgen05.make_tmem_copy`

```python
cutlass.cute.nvgpu.tcgen05.make_tmem_copy( atom: CopyAtom, tmem_tensor: cutlass.cute.typing.Tensor, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → TiledCopy
```

Makes a Tiled Copy instance from a TMEM Copy Atom and a TMEM tensor.

<a id="cutlass.cute.nvgpu.tcgen05.make_s2t_copy"></a>
### `cutlass.cute.nvgpu.tcgen05.make_s2t_copy`

```python
cutlass.cute.nvgpu.tcgen05.make_s2t_copy( atom: CopyAtom, tmem_tensor: cutlass.cute.typing.Tensor, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → TiledCopy
```

Makes a Tiled Copy instance from a TMEM Copy Atom and a TMEM tensor.

<a id="cutlass.cute.nvgpu.tcgen05.get_s2t_smem_desc_tensor"></a>
### `cutlass.cute.nvgpu.tcgen05.get_s2t_smem_desc_tensor`

```python
cutlass.cute.nvgpu.tcgen05.get_s2t_smem_desc_tensor( atom: CopyAtom, smem_tensor: cutlass.cute.typing.Tensor, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Tensor
```

Returns the SMEM descriptor tensor from a S2T copy atom and a SMEM tensor.

<a id="cutlass.cute.nvgpu.tcgen05.make_umma_smem_desc"></a>
### `cutlass.cute.nvgpu.tcgen05.make_umma_smem_desc`

```python
cutlass.cute.nvgpu.tcgen05.make_umma_smem_desc( src: cutlass.cute.typing.Pointer, layout: cutlass.cute.typing.Layout, major: str, next_src: cutlass.cute.typing.Pointer | None = None, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → Any
```

Construct shared memory descriptor for UMMA.

The make\_umma\_smem\_desc operation accepts an input cute.ptr (optionally a nextSrc
pointer for the second buffer in a circular buffer scheme), alongside a cute.layout
and a major attr, then constructs the shared memory descriptor and returns it.
The layout must be describing the buffer pointed to by the input pointer and the
iterator must carry valid swizzle information.

There are 5 supported swizzle variants:
- S<0, 4, 3> | SWIZZLE\_NONE
- S<1, 4, 3> | SWIZZLE\_32B
- S<2, 4, 3> | SWIZZLE\_64B
- S<3, 4, 3> | SWIZZLE\_128B
- S<2, 5, 2> | SWIZZLE\_128B\_BASE32B

The cute.ptr must carry shared address space and must be aligned to 16B.

**Parameters:**

- **src** (*Pointer*) – The source pointer to shared memory
- **layout** (*Layout*) – The layout describing the buffer
- **major** (*str*) – The major mode attribute
- **next\_src** (*Optional*[*Pointer*]) – Optional next source pointer for circular buffer scheme

**Returns:**

The shared memory descriptor

**Return type:**

SmemDescType
