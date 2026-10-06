<!-- source: https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_common.html -->

<a id="module-cutlass.cute.nvgpu"></a>
# Common

<a id="cutlass.cute.nvgpu.OperandMajorMode"></a>
### `cutlass.cute.nvgpu.OperandMajorMode`

```python
class cutlass.cute.nvgpu.OperandMajorMode(value)
```

Bases: `Enum`

An enumeration for the majorness of the input operands of the MMA.

<a id="cutlass.cute.nvgpu.OutputMajorMode"></a>
### `cutlass.cute.nvgpu.OutputMajorMode`

```python
class cutlass.cute.nvgpu.OutputMajorMode(value)
```

Bases: `Enum`

Major mode for the output operand D(M, N).

M = M-major (column-major): stride=(1, M), contiguous along M.
N = N-major (row-major): stride=(N, 1), contiguous along N.

<a id="cutlass.cute.nvgpu.OutputMajorMode.M"></a>
#### `cutlass.cute.nvgpu.OutputMajorMode.M`

```python
M = 'm'
```

<a id="cutlass.cute.nvgpu.OutputMajorMode.N"></a>
#### `cutlass.cute.nvgpu.OutputMajorMode.N`

```python
N = 'n'
```

<a id="cutlass.cute.nvgpu.OpError"></a>
### `cutlass.cute.nvgpu.OpError`

```python
class cutlass.cute.nvgpu.OpError(*args: Any, **kwargs: Any)
```

Bases: `DSLBaseError`

An exception class for Op construction errors.

<a id="cutlass.cute.nvgpu.normalize_field_to_ir_name"></a>
### `cutlass.cute.nvgpu.normalize_field_to_ir_name`

```python
cutlass.cute.nvgpu.normalize_field_to_ir_name( field: Any, admissible_fields: Any, ) → str
```

Normalize a field specifier to its IR logical field name.

Accepted inputs:

- Enum value present in admissible\_fields (must expose \_to\_ir\_field\_name()).
- Exact string IR name (e.g., “accum\_c”, “neg\_a”, “sf\_a”).

Any other form is rejected.

<a id="cutlass.cute.nvgpu.MmaUniversalOp"></a>
### `cutlass.cute.nvgpu.MmaUniversalOp`

```python
class cutlass.cute.nvgpu.MmaUniversalOp(abacc_dtype: Type[cutlass.cute.typing.Numeric])
```

Bases: `MmaOp`

The universal MMA Operation.

This Operation currently expects the A/B operands as well as the accumulator to share the same
data types.

**Supported architectures:** all (universal FMA)

**Parameters:**

**abacc\_dtype** (*Type*[*Numeric*]) – The data type for the A/B operands and the accumulator

<a id="cutlass.cute.nvgpu.MmaUniversalOp.abacc_dtype"></a>
#### `cutlass.cute.nvgpu.MmaUniversalOp.abacc_dtype`

```python
abacc_dtype: Type[cutlass.cute.typing.Numeric]
```

<a id="cutlass.cute.nvgpu.MmaUniversalTrait"></a>
### `cutlass.cute.nvgpu.MmaUniversalTrait`

```python
class cutlass.cute.nvgpu.MmaUniversalTrait(value: cutlass._mlir.ir.Value)
```

Bases: `Trait`

<a id="cutlass.cute.nvgpu.CopyUniversalOp"></a>
### `cutlass.cute.nvgpu.CopyUniversalOp`

```python
class cutlass.cute.nvgpu.CopyUniversalOp
```

Bases: `CopyOp`

The universal Copy Operation.

This operation is equivalent to the `a = b` assignment without any extra
memory attributes. For advanced memory features (memory order, memory scope,
cache eviction priority, invariant loads, etc.) please use the specialized copy
operations instead:

- [`CopyG2ROp`](#cutlass.cute.nvgpu.CopyG2ROp) – global memory to register
- [`CopyR2GOp`](#cutlass.cute.nvgpu.CopyR2GOp) – register to global memory
- [`CopyS2ROp`](#cutlass.cute.nvgpu.CopyS2ROp) – shared memory to register
- [`CopyR2SOp`](#cutlass.cute.nvgpu.CopyR2SOp) – register to shared memory

When creating a Copy Atom out of this operation, the expected usage pattern is

```python
op = cute.nvgpu.CopyUniversalOp()
atom = cute.make_copy_atom(op, tensor_dtype, num_bits_per_copy=64)
```

- `tensor_dtype` is the data type used to build the reference TV Layout (either the source or the destination TV Layout) in unit of tensor elements and is used for partitioning by `TiledCopy` for example
- `num_bits_per_copy` is a kw argument specifying the number of bits to copy per Atom execution. This can be larger than the width of the above data type. When not provided, the compiler will do a best effort at auto-vectorizing.

<a id="cutlass.cute.nvgpu.CopyUniversalTrait"></a>
### `cutlass.cute.nvgpu.CopyUniversalTrait`

```python
class cutlass.cute.nvgpu.CopyUniversalTrait(value: cutlass._mlir.ir.Value)
```

Bases: `Trait`

<a id="cutlass.cute.nvgpu.CopyG2ROp"></a>
### `cutlass.cute.nvgpu.CopyG2ROp`

```python
class cutlass.cute.nvgpu.CopyG2ROp
```

Bases: `CopyOp`

The G2R copy operation.

When creating a Copy Atom out of this operation, the expected usage pattern is

```python
op = cute.nvgpu.CopyG2ROp()
atom = cute.make_copy_atom(
    op,
    tensor_dtype,
    num_bits_per_copy=64,
    memory_order=cute.nvgpu.MemoryOrder.VOLATILE,
    memory_scope=cute.nvgpu.MemoryScope.SYS,
    l2_prefetch_size=cute.nvgpu.L2PrefetchSize.NONE,
    l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.EVICT_NORMAL,
    load_cache_mode=cute.nvgpu.LoadCacheMode.ALWAYS,
    shared_space=cute.nvgpu.SharedSpace.CTA,
    invariant=False,
)
```

<a id="cutlass.cute.nvgpu.CopyG2RTrait"></a>
### `cutlass.cute.nvgpu.CopyG2RTrait`

```python
class cutlass.cute.nvgpu.CopyG2RTrait(value: cutlass._mlir.ir.Value)
```

Bases: `Trait`

<a id="cutlass.cute.nvgpu.CopyG2RTrait.unpack"></a>
#### `cutlass.cute.nvgpu.CopyG2RTrait.unpack`

```python
unpack( *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, cache_policy: cutlass.cute.typing.Int64 | None = None, **kwargs: Any, ) → cutlass._mlir.ir.Value
```

<a id="cutlass.cute.nvgpu.CopyR2GOp"></a>
### `cutlass.cute.nvgpu.CopyR2GOp`

```python
class cutlass.cute.nvgpu.CopyR2GOp
```

Bases: `CopyOp`

The R2G copy operation.

When creating a Copy Atom out of this operation, the expected usage pattern is

```python
op = cute.nvgpu.CopyR2GOp()
atom = cute.make_copy_atom(
    op,
    tensor_dtype,
    num_bits_per_copy=64,
    memory_order=cute.nvgpu.MemoryOrder.RELEASE,
    memory_scope=cute.nvgpu.MemoryScope.CLUSTER,
    l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.EVICT_NORMAL,
    shared_space=cute.nvgpu.SharedSpace.CTA,
)
```

<a id="cutlass.cute.nvgpu.CopyR2GTrait"></a>
### `cutlass.cute.nvgpu.CopyR2GTrait`

```python
class cutlass.cute.nvgpu.CopyR2GTrait(value: cutlass._mlir.ir.Value)
```

Bases: `Trait`

<a id="cutlass.cute.nvgpu.CopyR2GTrait.unpack"></a>
#### `cutlass.cute.nvgpu.CopyR2GTrait.unpack`

```python
unpack( *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, cache_policy: cutlass.cute.typing.Int64 | None = None, **kwargs: Any, ) → cutlass._mlir.ir.Value
```

<a id="cutlass.cute.nvgpu.CopyS2ROp"></a>
### `cutlass.cute.nvgpu.CopyS2ROp`

```python
class cutlass.cute.nvgpu.CopyS2ROp
```

Bases: `CopyOp`

The S2R copy operation.

When creating a Copy Atom out of this operation, the expected usage pattern is

```python
op = cute.nvgpu.CopyS2ROp()
atom = cute.make_copy_atom(
    op,
    tensor_dtype,
    num_bits_per_copy=64,
    memory_order=cute.nvgpu.MemoryOrder.WEAK,
    memory_scope=cute.nvgpu.MemoryScope.CTA,
    shared_space=cute.nvgpu.SharedSpace.CTA,
)
```

<a id="cutlass.cute.nvgpu.CopyS2RTrait"></a>
### `cutlass.cute.nvgpu.CopyS2RTrait`

```python
class cutlass.cute.nvgpu.CopyS2RTrait(value: cutlass._mlir.ir.Value)
```

Bases: `Trait`

<a id="cutlass.cute.nvgpu.CopyR2SOp"></a>
### `cutlass.cute.nvgpu.CopyR2SOp`

```python
class cutlass.cute.nvgpu.CopyR2SOp
```

Bases: `CopyOp`

The R2S copy operation.

When creating a Copy Atom out of this operation, the expected usage pattern is

```python
op = cute.nvgpu.CopyR2SOp()
atom = cute.make_copy_atom(
    op,
    tensor_dtype,
    num_bits_per_copy=64,
    memory_order=cute.nvgpu.MemoryOrder.WEAK,
    memory_scope=cute.nvgpu.MemoryScope.CTA,
    shared_space=cute.nvgpu.SharedSpace.CTA,
)
```

<a id="cutlass.cute.nvgpu.CopyR2STrait"></a>
### `cutlass.cute.nvgpu.CopyR2STrait`

```python
class cutlass.cute.nvgpu.CopyR2STrait(value: cutlass._mlir.ir.Value)
```

Bases: `Trait`

<a id="cutlass.cute.nvgpu.MemoryOrder"></a>
### `cutlass.cute.nvgpu.MemoryOrder`

```python
class cutlass.cute.nvgpu.MemoryOrder(value)
```

Bases: `Enum`

An enumeration.

<a id="cutlass.cute.nvgpu.MemoryScope"></a>
### `cutlass.cute.nvgpu.MemoryScope`

```python
class cutlass.cute.nvgpu.MemoryScope(value)
```

Bases: `Enum`

An enumeration.

<a id="cutlass.cute.nvgpu.L2PrefetchSize"></a>
### `cutlass.cute.nvgpu.L2PrefetchSize`

```python
class cutlass.cute.nvgpu.L2PrefetchSize(value)
```

Bases: `Enum`

An enumeration.

<a id="cutlass.cute.nvgpu.CacheEvictionPriority"></a>
### `cutlass.cute.nvgpu.CacheEvictionPriority`

```python
class cutlass.cute.nvgpu.CacheEvictionPriority(value)
```

Bases: `Enum`

An enumeration.

<a id="cutlass.cute.nvgpu.LoadCacheMode"></a>
### `cutlass.cute.nvgpu.LoadCacheMode`

```python
class cutlass.cute.nvgpu.LoadCacheMode(value)
```

Bases: `Enum`

An enumeration.

<a id="cutlass.cute.nvgpu.StoreCacheMode"></a>
### `cutlass.cute.nvgpu.StoreCacheMode`

```python
class cutlass.cute.nvgpu.StoreCacheMode(value)
```

Bases: `Enum`

An enumeration.

<a id="cutlass.cute.nvgpu.SharedSpace"></a>
### `cutlass.cute.nvgpu.SharedSpace`

```python
class cutlass.cute.nvgpu.SharedSpace(value)
```

Bases: `Enum`

An enumeration.

<a id="cutlass.cute.nvgpu.make_tiled_tma_atom_A"></a>
### `cutlass.cute.nvgpu.make_tiled_tma_atom_A`

```python
cutlass.cute.nvgpu.make_tiled_tma_atom_A( op: CopyBulkTensorTileG2SOp | CopyBulkTensorTileG2SMulticastOp, gmem_tensor: cutlass.cute.typing.Tensor, smem_layout: cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout, mma_tiler_mnk: cutlass.cute.typing.Shape, tiled_mma: TiledMma, cluster_shape_vmnk: cutlass.cute.typing.Shape | None = None, *, internal_type: Type[cutlass.cute.typing.Numeric] | None = None, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → TmaInfo
```

Makes a TMA Copy atom mapping to `.tile` mode for `cp.async.bulk.tensor` PTX operation
accounting for the MK projections of the TiledMMA for A tensor loads.

Given

- a GMEM tensor
- a SMEM layout
- a MMA Tiler
- a TiledMma
- a Cluster-level shape

this function figures out the bulk tensor asynchronous copy instruction to use with the maximum
“TMA vector length” to copy tiles of the GMEM tensor to an SMEM buffer with the provided
layout and consistent with the provided Tiler & tiled\_mma (considering the M-mode & K-mode).
The Cluster-level shape is used to determine the multicast factor across the N-mode for A tensor loads.

This function returns two results:

1. the Copy Atom
2. the so-called TMA tensor used to map logical coordinates of the GMEM tensor to coordinates
   that the TMA unit can consume. TMA tensors have so-called basis stride elements so that the
   associated layout can output coordinates. Otherwise, TMA tensors can be partitioned
   similarly to any other CuTe tensors using the algebra.

**Parameters:**

- **op** (*Union*[[*CopyBulkTensorTileG2SOp*](cutedsl_cute_dsl_api_cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp), [*CopyBulkTensorTileG2SMulticastOp*](cutedsl_cute_dsl_api_cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp)]) – The Copy Operation to construct an Atom for
- **gmem\_tensor** (*Tensor*) – The GMEM tensor to be loaded by this copy atom
- **smem\_layout** (*Union*[*Layout*, *ComposedLayout*]) – Shared memory layout to load the tensor into (PDSL)
- **mma\_tiler\_mnk** (*Shape*) – The MMA Tiler shape (TILE\_M, TILE\_N, TILE\_K) in MNK dimensions
- **tiled\_mma** ([*atom.TiledMma*](cutedsl_cute_dsl_api_cute.md#cutlass.cute.TiledMma)) – The TiledMMA that will consume the load as operands
- **cluster\_shape\_vmnk** (*Shape*) – The Cluster-level shape in VMNK dimensions
- **internal\_type** (*Type*[*Numeric*]) – Optional element-format override used when the
  tensor element type does not match the copy type

**Returns:**

A TmaInfo containing the Copy Atom, TMA tensor, and SMEM layout

**Return type:**

[TmaInfo](cutedsl_cute_dsl_api_cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.TmaInfo)

<a id="cutlass.cute.nvgpu.make_tiled_tma_atom_B"></a>
### `cutlass.cute.nvgpu.make_tiled_tma_atom_B`

```python
cutlass.cute.nvgpu.make_tiled_tma_atom_B( op: CopyBulkTensorTileG2SOp | CopyBulkTensorTileG2SMulticastOp, gmem_tensor: cutlass.cute.typing.Tensor, smem_layout: cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout, mma_tiler_mnk: cutlass.cute.typing.Shape, tiled_mma: TiledMma, cluster_shape_vmnk: cutlass.cute.typing.Shape | None = None, *, internal_type: Type[cutlass.cute.typing.Numeric] | None = None, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → TmaInfo
```

Makes a TMA Copy atom mapping to `.tile` mode for `cp.async.bulk.tensor` PTX operation
accounting for the NK projections of the TiledMMA for B tensor loads.

Given

- a GMEM tensor
- a SMEM layout
- a MMA Tiler
- a TiledMma
- a Cluster-level shape

this function figures out the bulk tensor asynchronous copy instruction to use with the maximum
“TMA vector length” to copy tiles of the GMEM tensor to an SMEM buffer with the provided
layout and consistent with the provided Tiler & tiled\_mma (considering the N-mode & K-mode).
The Cluster-level shape is used to determine the multicast factor across the M-mode for B tensor loads.

This function returns two results:

1. the Copy Atom
2. the so-called TMA tensor used to map logical coordinates of the GMEM tensor to coordinates
   that the TMA unit can consume. TMA tensors have so-called basis stride elements so that the
   associated layout can output coordinates. Otherwise, TMA tensors can be partitioned
   similarly to any other CuTe tensors using the algebra.

**Parameters:**

- **op** (*Union*[[*CopyBulkTensorTileG2SOp*](cutedsl_cute_dsl_api_cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp), [*CopyBulkTensorTileG2SMulticastOp*](cutedsl_cute_dsl_api_cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp)]) – The Copy Operation to construct an Atom for
- **gmem\_tensor** (*Tensor*) – The GMEM tensor to be loaded by this copy atom
- **smem\_layout** (*Union*[*Layout*, *ComposedLayout*]) – Shared memory layout to load the tensor into (PDSL)
- **mma\_tiler\_mnk** (*Shape*) – The MMA Tiler shape (TILE\_M, TILE\_N, TILE\_K) in MNK dimensions
- **tiled\_mma** (*core.TiledMma*) – The TiledMMA that will consume the load as operands
- **cluster\_shape\_vmnk** (*Shape*) – The Cluster-level shape in VMNK dimensions
- **internal\_type** (*Type*[*Numeric*]) – Optional element-format override used when the
  tensor element type does not match the copy type

**Returns:**

A TmaInfo containing the Copy Atom, TMA tensor, and SMEM layout

**Return type:**

[TmaInfo](cutedsl_cute_dsl_api_cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.TmaInfo)

<a id="cutlass.cute.nvgpu.make_im2col_tma_atom_A"></a>
### `cutlass.cute.nvgpu.make_im2col_tma_atom_A`

```python
cutlass.cute.nvgpu.make_im2col_tma_atom_A( op: CopyBulkTensorIm2ColG2SOp | CopyBulkTensorIm2ColG2SMulticastOp, gmem_tensor: cutlass.cute.typing.Tensor, smem_layout: cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout, mma_tiler_mnk: cutlass.cute.typing.Shape, tiled_mma: TiledMma, filter_trs: Tuple[int, int, int], upper_padding_dhw: Tuple[int, int, int], lower_padding_dhw: Tuple[int, int, int], stride_dhw: Tuple[int, int, int], dilation_dhw: Tuple[int, int, int], cluster_shape_vmnk: cutlass.cute.typing.Shape | None = None, *, internal_type: Type[cutlass.cute.typing.Numeric] | None = None, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → TmaInfo
```

Makes a TMA Copy atom mapping to `.im2col` mode for `cp.async.bulk.tensor` PTX operation accounting for the MK projections of the TiledMMA for A tensor loads.

Given

- a GMEM tensor
- a SMEM layout
- a MMA Tiler
- a TiledMma
- a filter shape
- a padding shape
- a stride shape
- a dilation shape
- a Cluster-level shape

this function figures out the bulk tensor asynchronous copy instruction to use with the maximum
“TMA vector length” to copy tiles of the GMEM tensor to/from an SMEM buffer with the provided
layout while maintaining consistency with the provided Tiler.

This function returns two results:

1. the Copy Atom
2. the TMA tensor used to map logical coordinates of the GMEM tensor to coordinates
   that the TMA unit can consume. TMA tensors have so-called basis stride elements so that the
   associated layout can output coordinates. Otherwise, TMA tensors can be partitioned
   similarly to any other CuTe tensors using the algebra.

**Parameters:**

- **op** (*Union*[*CopyBulkTensorIm2ColG2SOp*, *CopyBulkTensorIm2ColG2SMulticastOp*]) – The Copy Operation to construct an Atom for
- **gmem\_tensor** (*Tensor*) – The GMEM tensor to be loaded by this copy atom
- **smem\_layout** (*Union*[*Layout*, *ComposedLayout*]) – Shared memory layout to load the tensor into (PDSL)
- **mma\_tiler\_mnk** (*Shape*) – The MMA Tiler shape (TILE\_M, TILE\_N, TILE\_K) in MNK dimensions
- **tiled\_mma** ([*atom.TiledMma*](cutedsl_cute_dsl_api_cute.md#cutlass.cute.TiledMma)) – The TiledMMA that will consume the load as operands
- **filter\_trs** (*Tuple*[*int*, *int*, *int*]) – The filter shape (T, R, S) in TRS dimensions
- **upper\_padding\_dhw** (*Tuple*[*int*, *int*, *int*]) – The upper padding shape (D, H, W) in DHW dimensions
- **lower\_padding\_dhw** (*Tuple*[*int*, *int*, *int*]) – The lower padding shape (D, H, W) in DHW dimensions
- **stride\_dhw** (*Tuple*[*int*, *int*, *int*]) – The stride shape (D, H, W) in DHW dimensions
- **dilation\_dhw** (*Tuple*[*int*, *int*, *int*]) – The dilation shape (D, H, W) in DHW dimensions
- **cluster\_shape\_vmnk** (*Shape*) – The Cluster-level shape in VMNK dimensions
- **internal\_type** (*Type*[*Numeric*]) – Optional element-format override used when the
  tensor element type does not match the copy type

**Returns:**

A TmaInfo containing the Copy Atom, TMA tensor, and SMEM layout

**Return type:**

[TmaInfo](cutedsl_cute_dsl_api_cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.TmaInfo)
