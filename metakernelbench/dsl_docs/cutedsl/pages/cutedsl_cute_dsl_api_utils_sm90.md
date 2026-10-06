<!-- source: https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/utils_sm90.html -->

<a id="module-cutlass.utils.sm90"></a>
# Utilities for SM90

<a id="cutlass.utils.sm90.get_smem_store_op"></a>
### `cutlass.utils.sm90.get_smem_store_op`

```python
cutlass.utils.sm90.get_smem_store_op( layout_d: LayoutEnum, elem_ty_d: Type[cutlass.cutlass_dsl.Numeric], elem_ty_acc: Type[cutlass.cutlass_dsl.Numeric], *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → CopyAtom
```

Selects the largest vectorized smem store atom available subject to constraint of gmem layout.

<a id="parameters"></a>
## Parameters:

**layout\_dLayoutEnum**

The layout enum of the output tensor D.

**elem\_ty\_dType[Numeric]**

The element type for output tensor D.

**elem\_ty\_accType[Numeric]**

The element type for accumulator.

<a id="returns"></a>
## Returns:

Either SmemStoreMatrix or SimtSyncCopy, based on the input parameters.

<a id="cutlass.utils.sm90.make_smem_layout_a"></a>
### `cutlass.utils.sm90.make_smem_layout_a`

```python
cutlass.utils.sm90.make_smem_layout_a( a_layout: LayoutEnum, mma_tiler_mnk: cutlass.cute.typing.Tile, a_dtype: Type[cutlass.cutlass_dsl.Numeric], num_stages: int, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout
```

This function helps with:

1. Get the partitioned shape of the A tensor based on the MMA tiler.
2. Select the heuristic SMEM layout atom based on the A tensor’s majorness, the data type, and the major mode size.
3. cute.Tile the SMEM layout atom to the MMA tile shape.
4. Stage the SMEM layout based on the number of stages.

**Parameters:**

- **a\_layout** ([*LayoutEnum*](cutedsl_cute_dsl_api_utils.md#cutlass.utils.LayoutEnum)) – The layout enum for tensor A
- **mma\_tiler\_mnk** (*cute.cute.Tile*) – The MMA tile shape
- **a\_dtype** (*Type*[*Numeric*]) – The element type for tensor A
- **num\_stages** (*int*) – The number of pipeline stages for tensor A

**Returns:**

SMEM layout for tensor A

**Return type:**

Union[cute.Layout, cute.ComposedLayout]

<a id="cutlass.utils.sm90.make_smem_layout_b"></a>
### `cutlass.utils.sm90.make_smem_layout_b`

```python
cutlass.utils.sm90.make_smem_layout_b( b_layout: LayoutEnum, mma_tiler_mnk: cutlass.cute.typing.Tile, b_dtype: Type[cutlass.cutlass_dsl.Numeric], num_stages: int, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout
```

This function helps with:

1. Get the partitioned shape of the B tensor based on the MMA tiler.
2. Select the heuristic SMEM layout atom based on the B tensor’s majorness, the data type, and the major mode size.
3. cute.Tile the SMEM layout atom to the MMA tile shape.
4. Stage the SMEM layout based on the number of stages.

**Parameters:**

- **b\_layout** ([*LayoutEnum*](cutedsl_cute_dsl_api_utils.md#cutlass.utils.LayoutEnum)) – The layout enum for tensor B
- **mma\_tiler\_mnk** (*cute.cute.Tile*) – The MMA tile shape
- **b\_dtype** (*Type*[*Numeric*]) – The element type for tensor B
- **num\_stages** (*int*) – The number of pipeline stages for tensor B

**Returns:**

SMEM layout for tensor B

**Return type:**

Union[cute.Layout, cute.ComposedLayout]

<a id="cutlass.utils.sm90.make_smem_layout_epi"></a>
### `cutlass.utils.sm90.make_smem_layout_epi`

```python
cutlass.utils.sm90.make_smem_layout_epi( epi_dtype: Type[cutlass.cutlass_dsl.Numeric], epi_layout: LayoutEnum, epi_tile: cutlass.cute.typing.Tile, epi_stage: int, smem_trg_shape: cutlass.cute.typing.Layout | None = None, smem_order: tuple | None = None, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout
```

This function helps:

1. Select the heuristic SMEM layout atom based on the epilog tile shape,
   the epilog tensor’s majorness, and the element type.
2. cute.Tile the SMEM layout atom to the epilog tile shape.
3. Stage the SMEM layout based on the number of stages.

**Parameters:**

- **epi\_dtype** (*Type*[*Numeric*]) – The element type for the epilog tensor.
- **epi\_layout** ([*LayoutEnum*](cutedsl_cute_dsl_api_utils.md#cutlass.utils.LayoutEnum)) – The layout enum for the epilog tensor.
- **epi\_tile** (*cute.cute.Tile*) – The epilogue tile shape.
- **epi\_stage** (*int*) – The stage of the epilog tensor.
- **smem\_trg\_shape** (*cute.Layout* | *None*) – Target shape for SMEM layout (optional).
- **smem\_order** (*tuple* | *None*) – Order for SMEM layout (optional).

**Returns:**

SMEM layout for epilog tensors (usually C & D which are processed in the epilog)

**Return type:**

Union[cute.Layout, cute.ComposedLayout]

<a id="cutlass.utils.sm90.compute_tile_shape_or_override"></a>
### `cutlass.utils.sm90.compute_tile_shape_or_override`

```python
cutlass.utils.sm90.compute_tile_shape_or_override( tile_shape_mnk: tuple[int, int, int], element_type: type[cutlass.cutlass_dsl.Numeric], is_cooperative: bool = False, epi_tile_override: tuple[int, int] | None = None, ) → tuple[int, int]
```

Compute the epilogue tile shape or use override if provided.

**Parameters:**

- **tile\_shape\_mnk** (*Tuple*[*int*, *int*, *int*]) – CTA tile shape (M,N,K)
- **element\_type** (*type*[*Numeric*]) – Data type of elements
- **is\_cooperative** (*bool*) – Whether to use cooperative approach
- **epi\_tile\_override** (*Tuple*[*int*, *int**] or* *None*) – Optional override for epilogue tile shape

**Returns:**

Computed epilogue tile shape

**Return type:**

Tuple[int, int]
