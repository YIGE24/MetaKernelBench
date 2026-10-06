<!-- source: https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/utils.html -->

<a id="cutlass-utils"></a>
# cutlass.utils

The `cutlass.utils` module contains utilities for developing kernels with CuTe DSL.

<a id="cutlass.utils.get_smem_capacity_in_bytes"></a>
### `cutlass.utils.get_smem_capacity_in_bytes`

```python
cutlass.utils.get_smem_capacity_in_bytes( compute_capability: str | None = None, ) → int
```

Get the shared memory capacity in bytes for a given compute capability.

Returns the maximum shared memory capacity in bytes available for the specified
GPU compute capability.

**Parameters:**

**compute\_capability** (*Optional*[*str*]) – The compute capability string (e.g. “70”, “75”, “80”)

**Returns:**

The shared memory capacity in bytes

**Return type:**

int

**Raises:**

**ValueError** – If the compute capability is not supported

<a id="cutlass.utils.get_kernel_smem_size"></a>
### `cutlass.utils.get_kernel_smem_size`

```python
cutlass.utils.get_kernel_smem_size( kernel: Callable, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → int
```

Get the total static shared memory allocation in bytes for a kernel.

Uses `cute.kernel_smem_size` to query the total smem bytes that will be
allocated by a kernel. The result is lowered to a compile-time constant by
`InferKernelSmemUsagePass`.

Must be called from within a `@cute.jit` body after the kernel’s
`.launch()` has been called, which triggers tracing and registers the
kernel’s MLIR symbol.

**Parameters:**

**kernel** (*Callable*) – A `@cute.kernel`-decorated function. The MLIR symbol is
retrieved automatically from state stored by the DSL after `.launch()`.

**Returns:**

Total shared memory allocated by the kernel, in bytes.

**Return type:**

int (i64 MLIR value during tracing)

<a id="cutlass.utils.SmemAllocator"></a>
### `cutlass.utils.SmemAllocator`

```python
class cutlass.utils.SmemAllocator
```

Bases: `object`

A helper class for managing shared memory allocation on GPU.

This class manages shared memory and provides APIs for allocation of raw bytes,
numeric types, arrays, and tensors with specified layouts and alignments.

> **Note**
>
> - SmemAllocator will automatically calculate the usage upon kernel launch.
> - There is no need to explicitly specify shared memory size in kernel launch.
> - Currently only supports static layouts. Dynamic layouts are not supported.

**Examples**:

```python
smem = SmemAllocator()

# Allocate raw bytes
buf_ptr = smem.allocate(100)  # 100 bytes

# Allocate numeric type
int8_ptr = smem.allocate(Int8)  # 1 byte

# Define a struct
@cute.struct
class SharedStorage:
    alpha: cutlass.Float32
    x: cutlass.Int32

# Allocate struct
struct_ptr = smem.allocate(SharedStorage)  # 8 bytes

# use of struct members
struct_ptr.alpha = 1.0
struct_ptr.x = 2
x_ptr = struct_ptr.x.ptr

# Allocate array
int8_array = smem.allocate_array(Int8, 10)  # 10 bytes

# Allocate tensor
layout = cute.make_layout((16, 16))
tensor = smem.allocate_tensor(Int8, layout)  # 256 bytes
```

<a id="cutlass.utils.SmemAllocator.capacity_in_bytes"></a>
#### `cutlass.utils.SmemAllocator.capacity_in_bytes`

```python
static capacity_in_bytes( compute_capability: str | None = None, ) → int
```

Get the shared memory capacity in bytes for a given compute capability.

Returns the maximum shared memory capacity in bytes available for the specified
GPU compute capability.

**Parameters:**

**compute\_capability** (*Optional*[*str*]) – The compute capability string (e.g. “70”, “75”, “80”)

**Returns:**

The shared memory capacity in bytes

**Return type:**

int

**Raises:**

**ValueError** – If the compute capability is not supported

<a id="cutlass.utils.SmemAllocator.__init__"></a>
#### `cutlass.utils.SmemAllocator.__init__`

```python
__init__( *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, )
```

Initialize a new SmemAllocator instance.

**Parameters:**

- **loc** (*Optional*[*ir.Location*]) – Source location information for debugging, defaults to None
- **ip** (*Optional*[*ir.InsertionPoint*]) – Insertion point for MLIR operations, defaults to None

<a id="cutlass.utils.SmemAllocator.calculate_partition_size"></a>
#### `cutlass.utils.SmemAllocator.calculate_partition_size`

```python
calculate_partition_size( partition: SmemPartition, *, cumulative: bool = False, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cutlass_dsl.Int32
```

Get the size of shared memory allocation at given smem partition.

**Parameters:**

- **partition** (*SmemPartition*) – The smem partition to query
- **cumulative** (*bool*, *optional*) – Whether to return the cumulative size of all partitions up to and including the given partition
- **loc** (*Optional*[*ir.Location*]) – Source location information for debugging, defaults to None
- **ip** (*Optional*[*ir.InsertionPoint*]) – Insertion point for MLIR operations, defaults to None

<a id="cutlass.utils.SmemAllocator.calculate_total_usage"></a>
#### `cutlass.utils.SmemAllocator.calculate_total_usage`

```python
calculate_total_usage( *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cutlass_dsl.Int32
```

Get total kernel smem usage calculated by allocator.

**Parameters:**

- **loc** (*Optional*[*ir.Location*]) – Source location information for debugging, defaults to None
- **ip** (*Optional*[*ir.InsertionPoint*]) – Insertion point for MLIR operations, defaults to None

<a id="cutlass.utils.SmemAllocator._allocated_bytes"></a>
#### `cutlass.utils.SmemAllocator._allocated_bytes`

```python
property _allocated_bytes: cutlass.cutlass_dsl.Int32
```

<a id="cutlass.utils.SmemAllocator._smem_alloca"></a>
#### `cutlass.utils.SmemAllocator._smem_alloca`

```python
_smem_alloca( layout: cutlass.cute.typing.Layout, dtype: cutlass.cutlass_dsl.NumericMeta, byte_alignment: int, swizzle: cutlass._mlir.ir.register_value_caster | None = None, struct_fields: list[tuple[str, int, int]] | None = None, *, partition: SmemPartition = SmemPartition.USER, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Pointer
```

Allocate shared memory using cute.memref.alloca with given layout, data type, and alignment.

**Returns:**

An iterator (pointer) to the allocated shared memory.

**Return type:**

cute.Pointer

<a id="cutlass.utils.SmemAllocator.allocate"></a>
#### `cutlass.utils.SmemAllocator.allocate`

```python
allocate( size_or_type: int, byte_alignment: int = 1, *, partition: SmemPartition = SmemPartition.USER, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Pointer
```
#### `allocate`

```python
allocate( size_or_type: Type[cutlass.cutlass_dsl.Numeric], byte_alignment: int = 1, *, partition: SmemPartition = SmemPartition.USER, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Pointer
```
#### `allocate`

```python
allocate( size_or_type: struct, byte_alignment: int = 1, *, partition: SmemPartition = SmemPartition.USER, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Pointer
```

Allocate a block of memory with specified size and alignment.

This method allocates a block of shared memory with the specified size and alignment requirements.
It supports allocating raw bytes, numeric types(as scalar value), and struct types.

**Parameters:**

- **size\_or\_type** (*Union*[*int*, *Type*[*Numeric*], [*cute.struct*](cutedsl_cute_dsl_api_cute.md#cutlass.cute.struct)]) – The allocation specification, which can be:
  - An integer specifying the number of bytes to allocate
  - A Numeric type (e.g., Int8, Float32) to allocate space for one element
  - A struct type to allocate space for the entire struct
- **byte\_alignment** (*int*, *optional*) – The minimum byte alignment requirement for the allocation, defaults to 1
- **loc** (*Optional*[*ir.Location*]) – Source location information for debugging, defaults to None
- **ip** (*Optional*[*ir.InsertionPoint*]) – Insertion point for MLIR operations, defaults to None

**Returns:**

For raw bytes and numeric types, returns a pointer to the allocated memory.
For struct types, returns an initialized struct instance at the allocated location.

**Return type:**

cute.Pointer

**Raises:**

- **ValueError** – If size is negative or alignment is less than 1
- **TypeError** – If size\_or\_type is not an integer, Numeric type, or struct
- **RuntimeError** – If allocation would exceed available shared memory

<a id="cutlass.utils.SmemAllocator.allocate_array"></a>
#### `cutlass.utils.SmemAllocator.allocate_array`

```python
allocate_array( element_type: Type[cutlass.cutlass_dsl.Numeric], num_elems: int = 1, *, byte_alignment: int = 1, partition: SmemPartition = SmemPartition.USER, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Pointer
```

Allocate an array of elements in shared memory.

**Parameters:**

- **element\_type** (*Union*[*Type*[*Numeric*]]) – The type of elements to allocate
- **num\_elems** (*int*, *optional*) – Number of elements to allocate, defaults to 1

**Returns:**

Pointer to the start of the allocated array

**Return type:**

cute.Pointer

**Raises:**

- **ValueError** – If num\_elems is less than 1
- **TypeError** – If element\_type is not a Numeric type

<a id="cutlass.utils.SmemAllocator.allocate_tensor"></a>
#### `cutlass.utils.SmemAllocator.allocate_tensor`

```python
allocate_tensor( element_type: Type[cutlass.cutlass_dsl.Numeric], layout: int | cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout, byte_alignment: int = 1, swizzle: cutlass._mlir.ir.register_value_caster | None = None, *, partition: SmemPartition = SmemPartition.USER, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Tensor
```

Allocate a tensor in shared memory.

Note: Currently only supports static layouts. Dynamic layouts are not supported.

**Parameters:**

- **element\_type** (*Union*[*Type*[*Numeric*]]) – The type of elements in the tensor
- **layout** (*Union*[*int*, *cute.Layout*, *cute.ComposedLayout*]) – The layout specification for the tensor. Must be a static layout.
- **byte\_alignment** (*int*, *optional*) – The byte alignment requirement, defaults to 1
- **swizzle** ([*cute.Swizzle*](cutedsl_cute_dsl_api_cute.md#cutlass.cute.Swizzle), *optional*) – Swizzle for position-dependent swizzling, defaults to None

**Returns:**

The allocated tensor with specified properties

**Return type:**

cute.Tensor

**Raises:**

- **TypeError** – If element\_type is not a Numeric type, or if swizzle conflicts with layout
- **ValueError** – If allocation is not byte-aligned
- **NotImplementedError** – If dynamic layout is specified

<a id="cutlass.utils.TmemAllocator"></a>
### `cutlass.utils.TmemAllocator`

```python
class cutlass.utils.TmemAllocator( alloc_result_dst_smem_ptr: cutlass.cute.typing.Pointer, barrier_for_retrieve: NamedBarrier, allocator_warp_id: int = 0, is_two_cta: bool = False, num_allocated_columns: int = 0, two_cta_tmem_dealloc_mbar_ptr: cutlass.cute.typing.Pointer | None = None, )
```

Bases: `object`

A class for managing tensor memory allocation on GPUs.

This class manages allocation/deallocation of tensor memory, including the mbarrier
synchronization for two cta use case.

**Variables:**

- **\_alloc\_result\_dst\_smem\_ptr** – The smem pointer that holds the base address of allocated tensor memory.
- **\_barrier\_for\_retrieve** – The barrier for retrieving tensor memory ptr.
- **\_allocator\_warp\_id** – The warp id of the allocator warp.
- **\_is\_two\_cta** – Whether the allocator is for two cta.
- **\_num\_allocated\_columns** – The number of columns allocated in the tensor memory.
- **\_two\_cta\_tmem\_dealloc\_mbar\_ptr** – The mbarrier pointer required when deallocating tensor memory for two cta.
- **\_arch** – The architecture of the GPU.

<a id="cutlass.utils.TmemAllocator.__init__"></a>
#### `cutlass.utils.TmemAllocator.__init__`

```python
__init__( alloc_result_dst_smem_ptr: cutlass.cute.typing.Pointer | None = None, *, barrier_for_retrieve: NamedBarrier, allocator_warp_id: int = 0, is_two_cta: bool = False, num_allocated_columns: int = 0, two_cta_tmem_dealloc_mbar_ptr: cutlass.cute.typing.Pointer | None = None, arch: str = 'sm_100', initialize_mbarrier: bool = True, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, )
```

Initialize a TmemAllocator instance for managing tensor memory on Blackwell GPUs.

This initializer sets up the allocator’s state, including the shared memory (smem) pointer
holding the base address of the allocated tensor memory, barrier synchronization for
retrieving the tensor memory pointer, allocator warp ID, whether the allocator is being used
for a 2-SM configuration, number of allocated columns in tensor
memory, and the optional mbarrier pointer for deallocation in the 2-SM case.

If is\_two\_cta is set to True, this will initialize the mbarrier pointer required for tensor
memory deallocation across two CTAs.

Auto-allocated smem pointers: If alloc\_result\_dst and two\_cta\_tmem\_dealloc ptrs are omitted,
the allocator creates them to low address of the smem region,
so they can be treated as reserved partition allocations and survive kernel smem resize if needed.

**Parameters:**

- **alloc\_result\_dst\_smem\_ptr** (*Optional*[*cute.Pointer*]) – Shared memory pointer holding the base address of allocated tensor memory. If None, the allocator auto-allocates it in the reserved address.
- **barrier\_for\_retrieve** ([*pipeline.NamedBarrier*](cutedsl_cute_dsl_api_pipeline.md#cutlass.pipeline.NamedBarrier)) – The named barrier for retrieving the tensor memory pointer.
- **allocator\_warp\_id** (*int*, *optional*) – The warp ID of the allocator warp, defaults to 0.
- **is\_two\_cta** (*bool*, *optional*) – Whether the allocator should coordinate two CTAs, defaults to False.
- **num\_allocated\_columns** (*int*, *optional*) – The number of columns allocated in tensor memory, defaults to 0.
- **two\_cta\_tmem\_dealloc\_mbar\_ptr** (*Optional*[*cute.Pointer*]) – Mbarrier pointer for two-CTA tensor memory deallocation. If None and is\_two\_cta, the allocator auto-allocates it in the reserved address.
- **initialize\_mbarrier** (*bool*, *optional*) – Whether to initialize the mbarrier for two cta, defaults to True.
- **loc** (*Any*, *optional*) – Optional codegen location for debugging and error reporting.
- **ip** (*Any*, *optional*) – Optional insertion point for codegen.

**Raises:**

**ValueError** – If only provided one of required smem ptr in two cta.

<a id="cutlass.utils.TmemAllocator.check_valid_num_columns"></a>
#### `cutlass.utils.TmemAllocator.check_valid_num_columns`

```python
check_valid_num_columns(num_columns: int) → bool
```

Check if the number of columns is valid.

This method checks if the number of columns is valid.
It checks if the number of columns is larger than 0, smaller than max capacity, a multiple of 32, and a power of two.

<a id="cutlass.utils.TmemAllocator.wait_for_alloc"></a>
#### `cutlass.utils.TmemAllocator.wait_for_alloc`

```python
wait_for_alloc( *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

Wait for the allocator warp to finish allocation.

This method is used to synchronize the allocator warp with the other warps before retrieving tmem ptr.

<a id="cutlass.utils.TmemAllocator.retrieve_ptr"></a>
#### `cutlass.utils.TmemAllocator.retrieve_ptr`

```python
retrieve_ptr( dtype: Type[cutlass.cutlass_dsl.Numeric] = cutlass.cutlass_dsl.Float32, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Pointer
```

Retrieve the pointer to the allocated tensor memory.

This method can be called by all warps after allocation has been performed
by the allocator warp.

<a id="cutlass.utils.TmemAllocator.reserve"></a>
#### `cutlass.utils.TmemAllocator.reserve`

```python
reserve( num_columns: int, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → TmemBufferPool
```

Reserve a block of tensor memory and return a pool for sub-allocation.

This method allocates a block of tensor memory, waits for the allocation
to complete, and returns a TmemBufferPool that can be used to sub-allocate
regions within that block without manual offset calculations.

Example usage:

```console
tmem_pool = tmem_allocator.reserve(tmem_total_size)

# Allocate and create tensors in one call
tCtAcc = tmem_pool.allocate_tensor(tCtAcc_layout, cutlass.Float32)
tCtSFA = tmem_pool.allocate_tensor(tCtSFA_layout, sf_dtype)

# Or allocate pointer only, then create tensor manually
sfb_ptr = tmem_pool.allocate(tCtSFB_layout, sf_dtype)
tCtSFB = cute.make_tensor(sfb_ptr, tCtSFB_layout)
```

**Parameters:**

**num\_columns** (*int*) – The total number of columns to reserve.

**Returns:**

A TmemBufferPool for sub-allocating within the reserved region.

**Return type:**

[TmemBufferPool](#cutlass.utils.TmemBufferPool)

<a id="cutlass.utils.TmemBufferPool"></a>
### `cutlass.utils.TmemBufferPool`

```python
class cutlass.utils.TmemBufferPool(base_ptr: cutlass.cute.typing.Pointer, total_cols: int)
```

Bases: `object`

A pool for sub-allocating from a reserved chunk of tensor memory.

This class enables sub-allocation from a pre-reserved TMEM region,
eliminating the need for manual offset calculations when allocating
multiple tensors in TMEM.

Example usage:

```console
tmem_pool = tmem_allocator.reserve(tmem_total_size)

# Allocate and create tensors in one call
tCtAcc = tmem_pool.allocate_tensor(tCtAcc_layout, cutlass.Float32)
tCtSFA = tmem_pool.allocate_tensor(tCtSFA_layout, sf_dtype)

# Or allocate pointer only, then create tensor manually
sfb_ptr = tmem_pool.allocate(tCtSFB_layout, sf_dtype)
tCtSFB = cute.make_tensor(sfb_ptr, tCtSFB_layout)
```

**Variables:**

- **\_base\_ptr** – The base pointer to the reserved TMEM region.
- **\_total\_cols** – The total number of columns in the pool.
- **\_current\_offset** – The current offset within the pool (in columns).

<a id="cutlass.utils.TmemBufferPool.__init__"></a>
#### `cutlass.utils.TmemBufferPool.__init__`

```python
__init__( base_ptr: cutlass.cute.typing.Pointer, total_cols: int, )
```

Initialize a TmemBufferPool instance.

**Parameters:**

- **base\_ptr** (*cute.Pointer*) – The base pointer to the reserved TMEM region.
- **total\_cols** (*int*) – The total number of columns in the pool.

<a id="cutlass.utils.TmemBufferPool.base_ptr"></a>
#### `cutlass.utils.TmemBufferPool.base_ptr`

```python
property base_ptr: cutlass.cute.typing.Pointer
```

Return the base pointer of the pool.

<a id="cutlass.utils.TmemBufferPool.total_cols"></a>
#### `cutlass.utils.TmemBufferPool.total_cols`

```python
property total_cols: int
```

Return the total number of columns in the pool.

<a id="cutlass.utils.TmemBufferPool.current_offset"></a>
#### `cutlass.utils.TmemBufferPool.current_offset`

```python
property current_offset: int
```

Return the current offset within the pool.

<a id="cutlass.utils.TmemBufferPool.remaining_cols"></a>
#### `cutlass.utils.TmemBufferPool.remaining_cols`

```python
property remaining_cols: int
```

Return the number of remaining columns available for allocation.

<a id="cutlass.utils.TmemBufferPool.allocate"></a>
#### `cutlass.utils.TmemBufferPool.allocate`

```python
allocate( size: int | cutlass.cute.typing.Layout, dtype: Type[cutlass.cutlass_dsl.Numeric], *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Pointer
```

Allocate a sub-region from the pool and return a pointer.

This method allocates a contiguous region of TMEM columns from the pool
and returns a pointer to the start of that region.

**Parameters:**

- **size** (*Union*[*int*, *cute.Layout*]) – The allocation size, which can be:
  - int: explicit number of columns to allocate
  - cute.Layout: a TMEM layout that, combined with dtype, determines the size
- **dtype** (*Type*[*Numeric*]) – The data type for the returned pointer and for computing
  layout size (when size is a Layout).

**Returns:**

A pointer to the allocated region with the specified dtype.

**Return type:**

cute.Pointer

**Raises:**

**AssertionError** – If there are not enough columns remaining in the pool.

Example usage:

```console
# Allocate with explicit column count
acc_ptr = pool.allocate(64, cutlass.Float32)

# Allocate based on layout and dtype
sfa_ptr = pool.allocate(tCtSFA_layout, sf_dtype)
```

<a id="cutlass.utils.TmemBufferPool.allocate_tensor"></a>
#### `cutlass.utils.TmemBufferPool.allocate_tensor`

```python
allocate_tensor( layout: cutlass.cute.typing.Layout, dtype: Type[cutlass.cutlass_dsl.Numeric], *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Tensor
```

Allocate a sub-region from the pool and return a tensor.

This is a convenience method that combines allocate() and cute.make\_tensor()
into a single call.

**Parameters:**

- **layout** (*cute.Layout*) – The TMEM layout for the tensor.
- **dtype** (*Type*[*Numeric*]) – The data type for the tensor elements.

**Returns:**

A tensor backed by the allocated TMEM region.

**Return type:**

cute.Tensor

**Raises:**

**AssertionError** – If there are not enough columns remaining in the pool.

Example usage:

```console
tCtAcc = pool.allocate_tensor(tCtAcc_layout, cutlass.Float32)
tCtSFA = pool.allocate_tensor(tCtSFA_layout, sf_dtype)
```

<a id="cutlass.utils.get_num_tmem_alloc_cols"></a>
### `cutlass.utils.get_num_tmem_alloc_cols`

```python
cutlass.utils.get_num_tmem_alloc_cols( tmem_tensors: cutlass.cute.typing.Tensor | List[cutlass.cute.typing.Tensor], rounding: bool = True, *, arch: str = 'sm_100', loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → int
```

Get the total number of TMEM allocation columns for the given TMEM tensors.

**Parameters:**

- **tmem\_tensors** (*Union*[*cute.Tensor*, *List*[*cute.Tensor*]]) – The TMEM tensors to get the number of allocation columns for.
- **rounding** (*bool*) – Whether to round up the number of allocation columns to the nearest power of 2.
- **arch** (*str*) – The architecture of the GPU.

**Returns:**

The total number of TMEM allocation columns.

**Return type:**

int

**Raises:**

**ValueError** – If the number of TMEM allocation columns exceeds the maximum capacity or is less than 32.

<a id="cutlass.utils.compute_tmem_cols_from_layout"></a>
### `cutlass.utils.compute_tmem_cols_from_layout`

```python
cutlass.utils.compute_tmem_cols_from_layout( layout: cutlass.cute.typing.Layout, dtype: Type[cutlass.cutlass_dsl.Numeric], *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → int
```

Compute the number of TMEM columns required for a layout with a given dtype.

This function calculates the column offset by recasting the layout to Int32
and computing its cosize, similar to how find\_tmem\_tensor\_col\_offset works
but without requiring a tensor.

**Parameters:**

- **layout** (*cute.Layout*) – The TMEM layout to compute columns for.
- **dtype** (*Type*[*Numeric*]) – The data type of the elements in the layout.

**Returns:**

The number of TMEM columns (always a Python int).

**Return type:**

int

**Raises:**

**ValueError** – If the layout size cannot be determined at compile time.

<a id="cutlass.utils.LayoutEnum"></a>
### `cutlass.utils.LayoutEnum`

```python
class cutlass.utils.LayoutEnum(value)
```

Bases: `Enum`

An enumeration.

<a id="cutlass.utils.LayoutEnum.ROW_MAJOR"></a>
#### `cutlass.utils.LayoutEnum.ROW_MAJOR`

```python
ROW_MAJOR = 'row_major'
```

<a id="cutlass.utils.LayoutEnum.COL_MAJOR"></a>
#### `cutlass.utils.LayoutEnum.COL_MAJOR`

```python
COL_MAJOR = 'col_major'
```

<a id="cutlass.utils.LayoutEnum.mma_major_mode"></a>
#### `cutlass.utils.LayoutEnum.mma_major_mode`

```python
mma_major_mode() → OperandMajorMode
```

<a id="cutlass.utils.LayoutEnum.sm90_mma_major_mode"></a>
#### `cutlass.utils.LayoutEnum.sm90_mma_major_mode`

```python
sm90_mma_major_mode() → OperandMajorMode
```

<a id="cutlass.utils.LayoutEnum.is_k_major_a"></a>
#### `cutlass.utils.LayoutEnum.is_k_major_a`

```python
is_k_major_a() → bool
```

<a id="cutlass.utils.LayoutEnum.is_m_major_a"></a>
#### `cutlass.utils.LayoutEnum.is_m_major_a`

```python
is_m_major_a() → bool
```

<a id="cutlass.utils.LayoutEnum.is_n_major_b"></a>
#### `cutlass.utils.LayoutEnum.is_n_major_b`

```python
is_n_major_b() → bool
```

<a id="cutlass.utils.LayoutEnum.is_k_major_b"></a>
#### `cutlass.utils.LayoutEnum.is_k_major_b`

```python
is_k_major_b() → bool
```

<a id="cutlass.utils.LayoutEnum.is_n_major_c"></a>
#### `cutlass.utils.LayoutEnum.is_n_major_c`

```python
is_n_major_c() → bool
```

<a id="cutlass.utils.LayoutEnum.is_m_major_c"></a>
#### `cutlass.utils.LayoutEnum.is_m_major_c`

```python
is_m_major_c() → bool
```

<a id="cutlass.utils.LayoutEnum.from_tensor"></a>
#### `cutlass.utils.LayoutEnum.from_tensor`

```python
static from_tensor( tensor: cutlass.cute.typing.Tensor, ) → LayoutEnum
```

<a id="cutlass.utils.WorkTileInfo"></a>
### `cutlass.utils.WorkTileInfo`

```python
class cutlass.utils.WorkTileInfo( tile_idx: cutlass.cute.typing.Coord, is_valid_tile: cutlass.cutlass_dsl.Boolean, )
```

Bases: `object`

A class to represent information about a work tile.

**Variables:**

- **tile\_idx** – The index of the tile.
- **is\_valid\_tile** – Whether the tile is valid.

<a id="cutlass.utils.WorkTileInfo.__init__"></a>
#### `cutlass.utils.WorkTileInfo.__init__`

```python
__init__( tile_idx: cutlass.cute.typing.Coord, is_valid_tile: cutlass.cutlass_dsl.Boolean, )
```

<a id="cutlass.utils.WorkTileInfo.is_valid_tile"></a>
#### `cutlass.utils.WorkTileInfo.is_valid_tile`

```python
property is_valid_tile
```

<a id="cutlass.utils.WorkTileInfo.tile_idx"></a>
#### `cutlass.utils.WorkTileInfo.tile_idx`

```python
property tile_idx
```

<a id="cutlass.utils.PersistentTileSchedulerParams"></a>
### `cutlass.utils.PersistentTileSchedulerParams`

```python
class cutlass.utils.PersistentTileSchedulerParams(**kwargs)
```

Bases: `object`

A class to represent parameters for a persistent tile scheduler.

This class is designed to manage and compute the layout of clusters and tiles
in a batched gemm problem.

**Variables:**

- **cluster\_shape\_mn** – Shape of the cluster in (m, n) dimensions (K dimension cta count must be 1).
- **problem\_layout\_ncluster\_mnl** – Layout of the problem in terms of
  number of clusters in (m, n, l) dimensions.

<a id="cutlass.utils.PersistentTileSchedulerParams.__init__"></a>
#### `cutlass.utils.PersistentTileSchedulerParams.__init__`

```python
__init__( problem_shape_ntile_mnl: cutlass.cute.typing.Shape, cluster_shape_mnk: cutlass.cute.typing.Shape, swizzle_size: int = 1, raster_along_m: bool = True, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

Initializes the PersistentTileSchedulerParams with the given parameters.

**Parameters:**

- **problem\_shape\_ntile\_mnl** (*cute.Shape*) – The shape of the problem in terms of
  number of CTA (Cooperative Thread Array) in (m, n, l) dimensions.
- **cluster\_shape\_mnk** (*cute.Shape*) – The shape of the cluster in (m, n) dimensions.
- **swizzle\_size** (*int*) – Swizzling size in the unit of cluster. 1 means no swizzle
- **raster\_along\_m** (*bool*) – Rasterization order of clusters. Only used when swizzle\_size > 1.
  True means along M, false means along N.

**Raises:**

**ValueError** – If cluster\_shape\_k is not 1.

<a id="cutlass.utils.PersistentTileSchedulerParams.get_grid_shape"></a>
#### `cutlass.utils.PersistentTileSchedulerParams.get_grid_shape`

```python
get_grid_shape( max_active_clusters: cutlass.cutlass_dsl.Int32, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → Tuple[cutlass.cutlass_dsl.Integer, cutlass.cutlass_dsl.Integer, cutlass.cutlass_dsl.Integer]
```

Computes the grid shape based on the maximum active clusters allowed.

**Parameters:**

**max\_active\_clusters** (*Int32*) – The maximum number of active clusters that
can run in one wave.

**Returns:**

A tuple containing the grid shape in (m, n, persistent\_clusters).
- m: self.cluster\_shape\_m.
- n: self.cluster\_shape\_n.
- persistent\_clusters: Number of persistent clusters that can run.

<a id="cutlass.utils.StaticPersistentTileScheduler"></a>
### `cutlass.utils.StaticPersistentTileScheduler`

```python
class cutlass.utils.StaticPersistentTileScheduler( params: PersistentTileSchedulerParams, num_persistent_clusters: cutlass.cutlass_dsl.Int32, current_work_linear_idx: cutlass.cutlass_dsl.Int32, cta_id_in_cluster: cutlass.cute.typing.Coord, num_tiles_executed: cutlass.cutlass_dsl.Int32, )
```

Bases: `object`

A scheduler for static persistent tile execution in CUTLASS/CuTe kernels.

**Variables:**

- **params** – Tile schedule related params, including cluster shape and problem\_layout\_ncluster\_mnl
- **num\_persistent\_clusters** – Number of persistent clusters that can be launched
- **cta\_id\_in\_cluster** – ID of the CTA within its cluster
- **\_num\_tiles\_executed** – Counter for executed tiles
- **\_current\_work\_linear\_idx** – Current cluster index

<a id="cutlass.utils.StaticPersistentTileScheduler.__init__"></a>
#### `cutlass.utils.StaticPersistentTileScheduler.__init__`

```python
__init__( params: PersistentTileSchedulerParams, num_persistent_clusters: cutlass.cutlass_dsl.Int32, current_work_linear_idx: cutlass.cutlass_dsl.Int32, cta_id_in_cluster: cutlass.cute.typing.Coord, num_tiles_executed: cutlass.cutlass_dsl.Int32, )
```

Initializes the StaticPersistentTileScheduler with the given parameters.

**Parameters:**

- **params** ([*PersistentTileSchedulerParams*](#cutlass.utils.PersistentTileSchedulerParams)) – Tile schedule related params, including cluster shape and problem\_layout\_ncluster\_mnl.
- **num\_persistent\_clusters** (*Int32*) – Number of persistent clusters that can be launched.
- **current\_work\_linear\_idx** (*Int32*) – Current cluster index.
- **cta\_id\_in\_cluster** (*cute.Coord*) – ID of the CTA within its cluster.
- **num\_tiles\_executed** (*Int32*) – Counter for executed tiles.

<a id="cutlass.utils.StaticPersistentTileScheduler.create"></a>
#### `cutlass.utils.StaticPersistentTileScheduler.create`

```python
static create( params: PersistentTileSchedulerParams, block_idx: Tuple[cutlass.cutlass_dsl.Integer, cutlass.cutlass_dsl.Integer, cutlass.cutlass_dsl.Integer], grid_dim: Tuple[cutlass.cutlass_dsl.Integer, cutlass.cutlass_dsl.Integer, cutlass.cutlass_dsl.Integer], *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → StaticPersistentTileScheduler
```

Initialize the static persistent tile scheduler.

**Parameters:**

- **params** ([*PersistentTileSchedulerParams*](#cutlass.utils.PersistentTileSchedulerParams)) – Parameters for the persistent
  tile scheduler.
- **block\_idx** (*Tuple*[*Integer*, *Integer*, *Integer*]) – The 3d block index in the format (bidx, bidy, bidz).
- **grid\_dim** (*Tuple*[*Integer*, *Integer*, *Integer*]) – The 3d grid dimensions for kernel launch.

**Returns:**

A StaticPersistentTileScheduler object.

**Return type:**

[StaticPersistentTileScheduler](#cutlass.utils.StaticPersistentTileScheduler)

<a id="cutlass.utils.StaticPersistentTileScheduler.get_grid_shape"></a>
#### `cutlass.utils.StaticPersistentTileScheduler.get_grid_shape`

```python
static get_grid_shape( params: PersistentTileSchedulerParams, max_active_clusters: cutlass.cutlass_dsl.Int32, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → Tuple[cutlass.cutlass_dsl.Integer, cutlass.cutlass_dsl.Integer, cutlass.cutlass_dsl.Integer]
```

Calculates the grid shape to be launched on GPU using problem shape,
threadblock shape, and active cluster size.

**Parameters:**

- **params** ([*PersistentTileSchedulerParams*](#cutlass.utils.PersistentTileSchedulerParams)) – Parameters for grid shape calculation.
- **max\_active\_clusters** (*Int32*) – Maximum active clusters allowed.

**Returns:**

The calculated 3d grid shape.

**Return type:**

Tuple[Integer, Integer, Integer]

<a id="cutlass.utils.StaticPersistentTileScheduler._get_current_work_for_linear_idx"></a>
#### `cutlass.utils.StaticPersistentTileScheduler._get_current_work_for_linear_idx`

```python
_get_current_work_for_linear_idx( current_work_linear_idx: cutlass.cutlass_dsl.Int32, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → WorkTileInfo
```

Compute current tile coord given current\_work\_linear\_idx and cta\_id\_in\_cluster.

**Parameters:**

**current\_work\_linear\_idx** (*Int32*) – The linear index of the current work.

**Returns:**

An object containing information about the current tile coordinates
and validity status.

**Return type:**

[WorkTileInfo](#cutlass.utils.WorkTileInfo)

<a id="cutlass.utils.StaticPersistentTileScheduler._get_cluster_work_idx_with_fastdivmod"></a>
#### `cutlass.utils.StaticPersistentTileScheduler._get_cluster_work_idx_with_fastdivmod`

```python
_get_cluster_work_idx_with_fastdivmod( current_work_linear_idx: cutlass.cutlass_dsl.Int32, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → Tuple[cutlass.cutlass_dsl.Int32, cutlass.cutlass_dsl.Int32, cutlass.cutlass_dsl.Int32]
```

FastDivmod optimized CLUSTER coordinate calculation.

CRITICAL: This should mimic problem\_layout\_ncluster\_mnl.get\_hier\_coord()
which returns CLUSTER coordinates, not tile coordinates!

**Parameters:**

**current\_work\_linear\_idx** (*Int32*) – Linear index in the work space

**Returns:**

Cluster coordinates (m, n, l) or None if FastDivmod not available

**Return type:**

Tuple[Int32, Int32, Int32] or None

<a id="cutlass.utils.StaticPersistentTileScheduler.num_tiles_executed"></a>
#### `cutlass.utils.StaticPersistentTileScheduler.num_tiles_executed`

```python
property num_tiles_executed
```

<a id="cutlass.utils.StaticPersistentRuntimeTileScheduler"></a>
### `cutlass.utils.StaticPersistentRuntimeTileScheduler`

```python
class cutlass.utils.StaticPersistentRuntimeTileScheduler(**kwargs)
```

Bases: [`StaticPersistentTileScheduler`](#cutlass.utils.StaticPersistentTileScheduler)

A scheduler for static persistent runtime tile execution in CUTLASS/CuTe kernels.
This scheduler will always launch all the SMs and the scheduler will generate the real tile info for each SM.

**Variables:**

- **params** – Tile schedule related params, including cluster shape and problem\_layout\_ncluster\_mnl
- **num\_persistent\_clusters** – Number of persistent clusters that can be launched
- **cta\_id\_in\_cluster** – ID of the CTA within its cluster
- **\_num\_tiles\_executed** – Counter for executed tiles
- **\_current\_work\_linear\_idx** – Current cluster index

<a id="cutlass.utils.StaticPersistentRuntimeTileScheduler.__init__"></a>
#### `cutlass.utils.StaticPersistentRuntimeTileScheduler.__init__`

```python
__init__( params: PersistentTileSchedulerParams, num_persistent_clusters: cutlass.cutlass_dsl.Int32, current_work_linear_idx: cutlass.cutlass_dsl.Int32, cta_id_in_cluster: cutlass.cute.typing.Coord, num_tiles_executed: cutlass.cutlass_dsl.Int32, inner_mode: int = 1, )
```

Initializes the StaticPersistentTileScheduler with the given parameters.

**Parameters:**

- **params** ([*PersistentTileSchedulerParams*](#cutlass.utils.PersistentTileSchedulerParams)) – Tile schedule related params, including cluster shape and problem\_layout\_ncluster\_mnl.
- **num\_persistent\_clusters** (*Int32*) – Number of persistent clusters that can be launched.
- **current\_work\_linear\_idx** (*Int32*) – Current cluster index.
- **cta\_id\_in\_cluster** (*cute.Coord*) – ID of the CTA within its cluster.
- **num\_tiles\_executed** (*Int32*) – Counter for executed tiles.
- **inner\_mode** (*int*) – The inner mode along which the linear index will be decomposed first.

<a id="cutlass.utils.StaticPersistentRuntimeTileScheduler.create"></a>
#### `cutlass.utils.StaticPersistentRuntimeTileScheduler.create`

```python
static create( params: PersistentTileSchedulerParams, block_idx: Tuple[cutlass.cutlass_dsl.Integer, cutlass.cutlass_dsl.Integer, cutlass.cutlass_dsl.Integer], grid_dim: Tuple[cutlass.cutlass_dsl.Integer, cutlass.cutlass_dsl.Integer, cutlass.cutlass_dsl.Integer], inner_mode: int = 1, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → StaticPersistentRuntimeTileScheduler
```

Initialize the static persistent tile scheduler.

**Parameters:**

- **params** ([*PersistentTileSchedulerParams*](#cutlass.utils.PersistentTileSchedulerParams)) – Parameters for the persistent
  tile scheduler.
- **block\_idx** (*Tuple*[*Integer*, *Integer*, *Integer*]) – The 3d block index in the format (bidx, bidy, bidz).
- **grid\_dim** (*Tuple*[*Integer*, *Integer*, *Integer*]) – The 3d grid dimensions for kernel launch.
- **inner\_mode** (*int*) – The inner mode along which the linear index will be decomposed first.

**Returns:**

A StaticPersistentRuntimeTileScheduler object.

**Return type:**

[StaticPersistentRuntimeTileScheduler](#cutlass.utils.StaticPersistentRuntimeTileScheduler)

<a id="cutlass.utils.StaticPersistentRuntimeTileScheduler._get_current_work_for_linear_idx"></a>
#### `cutlass.utils.StaticPersistentRuntimeTileScheduler._get_current_work_for_linear_idx`

```python
_get_current_work_for_linear_idx( current_work_linear_idx: cutlass.cutlass_dsl.Int32, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → WorkTileInfo
```

Compute current tile coord given current\_work\_linear\_idx and cta\_id\_in\_cluster.

**Parameters:**

**current\_work\_linear\_idx** (*Int32*) – The linear index of the current work.

**Returns:**

An object containing information about the current tile coordinates
and validity status.

**Return type:**

[WorkTileInfo](#cutlass.utils.WorkTileInfo)

<a id="cutlass.utils.TensorMapUpdateMode"></a>
### `cutlass.utils.TensorMapUpdateMode`

```python
class cutlass.utils.TensorMapUpdateMode(value)
```

Bases: `Enum`

Enum class defining tensor map update modes.

Modes:
GMEM: Update tensormap in global memory
SMEM: Load tensormap from global memory to shared memory,
update it in shared memory, then store back to global memory

<a id="cutlass.utils.TensorMapUpdateMode.GMEM"></a>
#### `cutlass.utils.TensorMapUpdateMode.GMEM`

```python
GMEM = 1
```

<a id="cutlass.utils.TensorMapUpdateMode.SMEM"></a>
#### `cutlass.utils.TensorMapUpdateMode.SMEM`

```python
SMEM = 2
```

<a id="cutlass.utils.TensorMapManager"></a>
### `cutlass.utils.TensorMapManager`

```python
class cutlass.utils.TensorMapManager( tensormap_update_mode: TensorMapUpdateMode, bytes_per_tensormap: int, )
```

Bases: `object`

Manages TensorMap operations including initialization and updates.
Provides utilities to convert tensormap pointer to across different memory spaces.

<a id="cutlass.utils.TensorMapManager.tensormap_update_mode"></a>
#### `cutlass.utils.TensorMapManager.tensormap_update_mode`

```python
tensormap_update_mode: TensorMapUpdateMode
```

<a id="cutlass.utils.TensorMapManager.bytes_per_tensormap"></a>
#### `cutlass.utils.TensorMapManager.bytes_per_tensormap`

```python
bytes_per_tensormap: int
```

<a id="cutlass.utils.TensorMapManager.get_tensormap_ptr"></a>
#### `cutlass.utils.TensorMapManager.get_tensormap_ptr`

```python
get_tensormap_ptr( ptr: cutlass.cute.typing.Pointer, address_space: cutlass.base_dsl.address_space.AddressSpace = cutlass.base_dsl.address_space.AddressSpace.gmem, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Pointer
```

<a id="cutlass.utils.TensorMapManager.fence_tensormap_initialization"></a>
#### `cutlass.utils.TensorMapManager.fence_tensormap_initialization`

```python
fence_tensormap_initialization( *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

<a id="cutlass.utils.TensorMapManager.fence_tensormap_update"></a>
#### `cutlass.utils.TensorMapManager.fence_tensormap_update`

```python
fence_tensormap_update( tensormap_ptr: cutlass.cute.typing.Pointer, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

<a id="cutlass.utils.TensorMapManager.__init__"></a>
#### `cutlass.utils.TensorMapManager.__init__`

```python
__init__( tensormap_update_mode: TensorMapUpdateMode, bytes_per_tensormap: int, ) → None
```

<a id="cutlass.utils.GroupSearchResult"></a>
### `cutlass.utils.GroupSearchResult`

```python
class cutlass.utils.GroupSearchResult(**kwargs)
```

Bases: `object`

The result of the group search for grouped gemm.

**Parameters:**

- **group\_idx** (*Int32*) – The result group index
- **cta\_tile\_idx\_m** (*Int32*) – CTA tile index along M dimension after rasterization
- **cta\_tile\_idx\_n** (*Int32*) – CTA tile index along N dimension after rasterization
- **problem\_shape\_m** (*Int32*) – The M dimension of the gemm problem
- **problem\_shape\_n** (*Int32*) – The N dimension of the gemm problem
- **problem\_shape\_k** (*Int32*) – The K dimension of the gemm problem
- **cta\_tile\_count\_k** (*Int32*) – Number of tiles along K dimension

<a id="cutlass.utils.GroupSearchResult.__init__"></a>
#### `cutlass.utils.GroupSearchResult.__init__`

```python
__init__( group_idx: cutlass.cutlass_dsl.Int32, cta_tile_idx_m: cutlass.cutlass_dsl.Int32, cta_tile_idx_n: cutlass.cutlass_dsl.Int32, problem_shape_m: cutlass.cutlass_dsl.Int32, problem_shape_n: cutlass.cutlass_dsl.Int32, problem_shape_k: cutlass.cutlass_dsl.Int32, cta_tile_count_k: cutlass.cutlass_dsl.Int32, ) → None
```

<a id="cutlass.utils.GroupedGemmGroupSearchState"></a>
### `cutlass.utils.GroupedGemmGroupSearchState`

```python
class cutlass.utils.GroupedGemmGroupSearchState(**kwargs)
```

Bases: `object`

The state of group index search for grouped gemm.

The state will be initialized once and updated in every round of group index search.

**Parameters:**

- **start\_group\_idx** (*Int32*) – The group idx to start the search with
- **tile\_count\_prev\_group** (*Int32*) – Number of tiles before the matched group
- **tile\_count\_searched** (*Int32*) – Number of tiles we have searched. When the matched group
  is found, it records the number of tiles including the
  matched group

<a id="cutlass.utils.GroupedGemmGroupSearchState.__init__"></a>
#### `cutlass.utils.GroupedGemmGroupSearchState.__init__`

```python
__init__( start_group_idx: cutlass.cutlass_dsl.Int32, tile_count_prev_group: cutlass.cutlass_dsl.Int32, tile_count_searched: cutlass.cutlass_dsl.Int32, found: cutlass.cutlass_dsl.Boolean, ) → None
```

<a id="cutlass.utils.create_initial_search_state"></a>
### `cutlass.utils.create_initial_search_state`

```python
cutlass.utils.create_initial_search_state() → GroupedGemmGroupSearchState
```

Create an initial search state for grouped gemm.

**Returns:**

A new search state with initial values

**Return type:**

[GroupedGemmGroupSearchState](#cutlass.utils.GroupedGemmGroupSearchState)

<a id="cutlass.utils.GroupedGemmTileSchedulerHelper"></a>
### `cutlass.utils.GroupedGemmTileSchedulerHelper`

```python
class cutlass.utils.GroupedGemmTileSchedulerHelper(**kwargs)
```

Bases: `object`

A helper to translate the raw block index (x, y, z) from tile scheduler to real CTA
tile index for grouped gemm.

**Parameters:**

- **group\_count** (*int*) – Number of groups in current grouped gemm problem
- **tile\_sched\_params** ([*PersistentTileSchedulerParams*](#cutlass.utils.PersistentTileSchedulerParams)) – Parameter used to create the tile scheduler this helper
  works with
- **cluster\_tile\_shape\_mnk** (*tuple*[*int*, *int*, *int*]) – The shape of cluster tile as (m, n, k)
- **search\_state** ([*GroupedGemmGroupSearchState*](#cutlass.utils.GroupedGemmGroupSearchState)) – The initial search state

<a id="cutlass.utils.GroupedGemmTileSchedulerHelper.__init__"></a>
#### `cutlass.utils.GroupedGemmTileSchedulerHelper.__init__`

```python
__init__( group_count: int, tile_sched_params: PersistentTileSchedulerParams, cluster_tile_shape_mnk: tuple[int, int, int], search_state: GroupedGemmGroupSearchState, ) → None
```

<a id="cutlass.utils.GroupedGemmTileSchedulerHelper.delinearize_z"></a>
#### `cutlass.utils.GroupedGemmTileSchedulerHelper.delinearize_z`

```python
delinearize_z( cta_tile_coord: tuple, problem_shape_mnkl: cutlass.cute.typing.Tensor, ) → GroupSearchResult
```

Delinearize the linear z index and return GroupSearchResult.

This function should be used by warps that need to know the CTA tile index on M
and N dimensions.

**Parameters:**

- **cta\_tile\_coord** (*tuple* *of* *Int32*) – The raw CTA coordinate from tile scheduler
- **problem\_shape\_mnkl** (*cute.Tensor*) – Tensor containing gemm problem size (M, N, K, L) for
  each group

**Returns:**

The search result containing group index and tile coordinates

**Return type:**

[GroupSearchResult](#cutlass.utils.GroupSearchResult)

<a id="cutlass.utils.GroupedGemmTileSchedulerHelper.search_cluster_tile_count_k"></a>
#### `cutlass.utils.GroupedGemmTileSchedulerHelper.search_cluster_tile_count_k`

```python
search_cluster_tile_count_k( cta_tile_coord: tuple, problem_shape_mnkl: cutlass.cute.typing.Tensor, ) → Tuple[cutlass.cutlass_dsl.Int32, cutlass.cutlass_dsl.Int32]
```

Search the matched group for given linear index and compute the number of tiles
along K dimension for the matched group.

This function should be used by warps that are only interested in the number of
tiles along K dimension.

**Parameters:**

- **cta\_tile\_coord** (*tuple* *of* *Int32*) – The raw CTA coordinate from tile scheduler
- **problem\_shape\_mnkl** (*cute.Tensor*) – Tensor containing gemm problem size (M, N, K, L) for
  all groups

**Returns:**

A tuple containing cluster count along K dimension and the group index

**Return type:**

Tuple[Int32, Int32]

<a id="cutlass.utils.GroupedGemmTileSchedulerHelper._prefix_sum"></a>
#### `cutlass.utils.GroupedGemmTileSchedulerHelper._prefix_sum`

```python
_prefix_sum( value_per_thread: cutlass.cutlass_dsl.Int32, ) → cutlass.cutlass_dsl.Int32
```

Perform prefix sum within a full warp.

**Parameters:**

**value\_per\_thread** (*Int32*) – The value for this thread to contribute to the prefix
sum

**Returns:**

The prefix sum result for this thread

**Return type:**

Int32

<a id="cutlass.utils.GroupedGemmTileSchedulerHelper._get_problem_for_group"></a>
#### `cutlass.utils.GroupedGemmTileSchedulerHelper._get_problem_for_group`

```python
_get_problem_for_group( problem_shape_mnkl: cutlass.cute.typing.Tensor, group_idx: cutlass.cutlass_dsl.Int32, ) → cutlass.cute.typing.Tensor
```

Load gemm problem (m,n,k,l) for the specified group from global memory to
register.

**Parameters:**

- **problem\_shape\_mnkl** (*cute.Tensor*) – Tensor in global memory with layout
  (group\_count, 4):(4, 1)
- **group\_idx** (*Int32*) – The index of the group to load

**Returns:**

The problem shape tensor for the specified group

**Return type:**

cute.Tensor

<a id="cutlass.utils.GroupedGemmTileSchedulerHelper._get_cluster_tile_count_mn"></a>
#### `cutlass.utils.GroupedGemmTileSchedulerHelper._get_cluster_tile_count_mn`

```python
_get_cluster_tile_count_mn( problem_shape: cutlass.cute.typing.Tensor, ) → cutlass.cutlass_dsl.Int32
```

Compute total cluster count.

**Parameters:**

**problem\_shape** (*cute.Tensor*) – Tensor containing problem shape (m, n, k, l)

**Returns:**

The total cluster tile count for M and N dimensions

**Return type:**

Int32

<a id="cutlass.utils.GroupedGemmTileSchedulerHelper._compute_cta_tile_coord"></a>
#### `cutlass.utils.GroupedGemmTileSchedulerHelper._compute_cta_tile_coord`

```python
_compute_cta_tile_coord( cluster_tile_idx: cutlass.cutlass_dsl.Int32, cta_tile_coord_in_cluster: tuple, cluster_tile_count_m: cutlass.cutlass_dsl.Int32, cluster_tile_count_n: cutlass.cutlass_dsl.Int32, ) → tuple
```

Compute CTA tile indices along M and N dimensions based on the linear index
within a group.

It uses the AlongM mode to decompose the linear index onto M and N dimensions.

**Parameters:**

- **cluster\_tile\_idx** (*Int32*) – The linear index within a group
- **cta\_tile\_coord\_in\_cluster** (*tuple* *of* *Int32*) – CTA indices along M and N dimensions within a
  cluster
- **cluster\_tile\_count\_m** (*Int32*) – The number of clusters along M dimension of the
  matched group
- **cluster\_tile\_count\_n** (*Int32*) – The number of clusters along N dimension of the
  matched group

**Returns:**

A tuple containing CTA tile indices along M and N dimensions

**Return type:**

tuple of (Int32, Int32)

<a id="cutlass.utils.GroupedGemmTileSchedulerHelper._group_search"></a>
#### `cutlass.utils.GroupedGemmTileSchedulerHelper._group_search`

```python
_group_search( linear_idx: cutlass.cutlass_dsl.Int32, problem_shape_mnkl: cutlass.cute.typing.Tensor, init_group_idx: cutlass.cutlass_dsl.Int32, init_tile_count_searched: cutlass.cutlass_dsl.Int32, ) → GroupedGemmGroupSearchState
```

Search which group the linear index belongs to.

**Parameters:**

- **linear\_idx** (*Int32*) – The linear index to be decomposed
- **problem\_shape\_mnkl** (*cute.Tensor*) – Tensor containing gemm problem size (M, N, K, L) for
  all groups
- **init\_group\_idx** (*Int32*) – The group idx to start the search with
- **init\_tile\_count\_searched** (*Int32*) – The number of tiles we have searched

**Returns:**

The updated search state

**Return type:**

[GroupedGemmGroupSearchState](#cutlass.utils.GroupedGemmGroupSearchState)

<a id="cutlass.utils.GroupedGemmTileSchedulerHelper._group_search_and_load_problem_shape"></a>
#### `cutlass.utils.GroupedGemmTileSchedulerHelper._group_search_and_load_problem_shape`

```python
_group_search_and_load_problem_shape( linear_idx: cutlass.cutlass_dsl.Int32, problem_shape_mnkl: cutlass.cute.typing.Tensor, start_group_idx: cutlass.cutlass_dsl.Int32, tile_count_searched: cutlass.cutlass_dsl.Int32, ) → Tuple[cutlass.cutlass_dsl.Int32, cutlass.cute.typing.Tensor]
```

Perform group search and load problem shape for the matched group.

**Parameters:**

- **linear\_idx** (*Int32*) – The linear index to be decomposed
- **problem\_shape\_mnkl** (*cute.Tensor*) – Tensor containing gemm problem size (M, N, K, L) for
  all groups
- **start\_group\_idx** (*Int32*) – The group idx to start the search with
- **tile\_count\_searched** (*Int32*) – The number of tiles we have searched

**Returns:**

A tuple containing the final group index and the problem shape tensor

**Return type:**

Tuple[Int32, cute.Tensor]

<a id="cutlass.utils.HardwareInfo"></a>
### `cutlass.utils.HardwareInfo`

```python
class cutlass.utils.HardwareInfo(device_id: int = 0)
```

Bases: `object`

device\_id: CUDA device ID to get the hardware info.

<a id="cutlass.utils.HardwareInfo.__init__"></a>
#### `cutlass.utils.HardwareInfo.__init__`

```python
__init__(device_id: int = 0)
```

<a id="cutlass.utils.HardwareInfo.get_max_active_clusters"></a>
#### `cutlass.utils.HardwareInfo.get_max_active_clusters`

```python
get_max_active_clusters( cluster_size: int, stream: cuda.bindings.driver.CUstream | None = None, ) → int
```

Get the maximum number of active clusters for a given cluster size.

When a stream from a green context is provided, the occupancy calculation
will reflect the reduced SM partition of the green context.

**Parameters:**

- **cluster\_size** (*int*) – Number of blocks per cluster (must be between 1 and 32)
- **stream** (*driver.CUstream*, *optional*) – Optional CUDA stream handle. If provided (especially from a green context),
  the occupancy calculation reflects the stream’s SM partition.

**Returns:**

Maximum number of active clusters

**Return type:**

int

<a id="cutlass.utils.HardwareInfo.get_l2_cache_size_in_bytes"></a>
#### `cutlass.utils.HardwareInfo.get_l2_cache_size_in_bytes`

```python
get_l2_cache_size_in_bytes() → int
```

<a id="cutlass.utils.HardwareInfo.get_device_multiprocessor_count"></a>
#### `cutlass.utils.HardwareInfo.get_device_multiprocessor_count`

```python
get_device_multiprocessor_count() → int
```

<a id="cutlass.utils.HardwareInfo._checkCudaErrors"></a>
#### `cutlass.utils.HardwareInfo._checkCudaErrors`

```python
_checkCudaErrors(result: Any) → Any
```

<a id="cutlass.utils.HardwareInfo._cudaGetErrorEnum"></a>
#### `cutlass.utils.HardwareInfo._cudaGetErrorEnum`

```python
_cudaGetErrorEnum(error: Any) → str
```

<a id="cutlass.utils.HardwareInfo._cuda_driver_version_ge"></a>
#### `cutlass.utils.HardwareInfo._cuda_driver_version_ge`

```python
_cuda_driver_version_ge(major: int, minor: int) → bool
```

<a id="cutlass.utils.HardwareInfo._cuda_driver_version_lt"></a>
#### `cutlass.utils.HardwareInfo._cuda_driver_version_lt`

```python
_cuda_driver_version_lt(major: int, minor: int) → bool
```

<a id="cutlass.utils.HardwareInfo._empty_kernel"></a>
#### `cutlass.utils.HardwareInfo._empty_kernel`

```python
_empty_kernel() → None
```

<a id="cutlass.utils.HardwareInfo._host_function"></a>
#### `cutlass.utils.HardwareInfo._host_function`

```python
_host_function() → None
```

<a id="cutlass.utils.HardwareInfo._get_device_function"></a>
#### `cutlass.utils.HardwareInfo._get_device_function`

```python
_get_device_function() → cuda.bindings.driver.CUfunction
```

Get a device function by compiling a dummy kernel using cuteDSL pipeline.

<a id="cutlass.utils.TransformMode"></a>
### `cutlass.utils.TransformMode`

```python
class cutlass.utils.TransformMode(value)
```

Bases: `Enum`

An enumeration for the possible transform modes of a mixed-input GEMM.

<a id="cutlass.utils.TransformMode.ConvertOnly"></a>
#### `cutlass.utils.TransformMode.ConvertOnly`

```python
ConvertOnly = 1
```

<a id="cutlass.utils.TransformMode.ConvertScale"></a>
#### `cutlass.utils.TransformMode.ConvertScale`

```python
ConvertScale = 2
```

<a id="cutlass.utils.scale_tma_partition"></a>
### `cutlass.utils.scale_tma_partition`

```python
cutlass.utils.scale_tma_partition( tCsS: cutlass.cute.typing.Tensor, tCgS: cutlass.cute.typing.Tensor, tma_atom_s: CopyAtom, block_in_cluster_coord_vmnk: cutlass.cute.typing.Coord, scale_cta_layout: cutlass.cute.typing.Layout, ) → tuple[cutlass.cute.typing.Tensor, cutlass.cute.typing.Tensor]
```

Perform TMA partition for scale tensor.
This method partitions the global memory and shared memory buffer for the scale tensor for TMA load.
:param tCsS: Input scale shared memory tensor
:type tCsS: cute.Tensor
:param tCgS: Input scale global memory tensor
:type tCgS: cute.Tensor
:param tma\_atom\_s: TMA copy atom for scale tensor
:type tma\_atom\_s: cute.CopyAtom
:param block\_in\_cluster\_coord\_vmnk: CTA coord in the cluster
:type block\_in\_cluster\_coord\_vmnk: cute.Coord
:param scale\_cta\_layout: Layout of CTA from the view of the scale tensor
:type scale\_cta\_layout: cute.Layout
:return: A tuple containing (tSsS, tSgS) where:

> - tSsS: Partitioned scale tensor in shared memory
> - tSgS: Partitioned scale tensor in global memory

**Return type:**

tuple[cute.Tensor, cute.Tensor]

<a id="cutlass.utils.transform_partition"></a>
### `cutlass.utils.transform_partition`

```python
cutlass.utils.transform_partition( transform_a_source: tcgen05.OperandSource, scale_mode: TransformMode, copy_atom_a_input: cute.CopyAtom, copy_atom_a_transform: cute.CopyAtom, sA_input: cute.Tensor, A_transform: cute.Tensor, transform_local_tidx: cutlass.Int32, ) → tuple[cute.TiledCopy | None, cute.TiledCopy | None, cute.Tensor, cute.Tensor]
```

Partition tensors for transform input and output.
This method sets up the copy atoms and partitions the shared/tensor memory
for the transformation of tensor A.
:param transform\_a\_source: Where the transformed tensor A is stored (TMEM or SMEM)
:type transform\_a\_source: tcgen05.OperandSource
:param scale\_mode: The transform mode (ConvertOnly or ConvertScale)
:type scale\_mode: TransformMode
:param copy\_atom\_a\_input: Copy atom for loading A from shared memory
:type copy\_atom\_a\_input: cute.CopyAtom
:param copy\_atom\_a\_transform: Copy atom for storing transformed A
:type copy\_atom\_a\_transform: cute.CopyAtom
:param sA\_input: Input tensor A in shared memory
:type sA\_input: cute.Tensor
:param A\_transform: Transformed tensor A in tensor or shared memory
:type A\_transform: cute.Tensor
:param transform\_local\_tidx: Local thread index for transformation warps
:type transform\_local\_tidx: cutlass.Int32
:return: A tuple containing (src\_copy\_a, dst\_copy\_a, tAsA\_input, tA\_transform) where:

> - src\_copy\_a: Tiled copy for source tensor
> - dst\_copy\_a: Tiled copy for destination tensor
> - tAsA\_input: Partitioned input tensor A
> - tA\_transform: Partitioned transformed tensor A

**Return type:**

tuple[Optional[[cute.TiledCopy](cutedsl_cute_dsl_api_cute.md#cutlass.cute.TiledCopy)], Optional[[cute.TiledCopy](cutedsl_cute_dsl_api_cute.md#cutlass.cute.TiledCopy)], cute.Tensor, cute.Tensor]

<a id="cutlass.utils.scale_partition"></a>
### `cutlass.utils.scale_partition`

```python
cutlass.utils.scale_partition( src_copy_a: cute.TiledCopy, tCsS: cute.Tensor, transform_local_tidx: cutlass.Int32, mma_dtype: type[cutlass.Numeric], ) → tuple[cute.TiledCopy, cute.Tensor, cute.Tensor, cute.Tensor]
```

Partition the scale tensor for transformation.
This method prepares the copy atom and partitions the shared memory for the scale tensor.
:param src\_copy\_a: Tiled copy for the source tensor
:type src\_copy\_a: cute.TiledCopy
:param tCsS: Scale tensor in shared memory
:type tCsS: cute.Tensor
:param transform\_local\_tidx: Local thread index for transformation warps
:type transform\_local\_tidx: cutlass.Int32
:param mma\_dtype: Data type for the MMA operation
:type mma\_dtype: type[cutlass.Numeric]
:return: A tuple containing (smem\_thr\_copy\_S, tSsS\_trans, tSrS\_copy, tSrS) where:

> - smem\_thr\_copy\_S: Tiled copy for the scale tensor
> - tSsS\_trans: Partitioned scale tensor for transformation
> - tSrS\_copy: Register fragment for the scale tensor
> - tSrS: View of scale tensor used for transformation computation

**Return type:**

tuple[[cute.TiledCopy](cutedsl_cute_dsl_api_cute.md#cutlass.cute.TiledCopy), cute.Tensor, cute.Tensor, cute.Tensor]

<a id="cutlass.utils.get_gmem_layout_scale"></a>
### `cutlass.utils.get_gmem_layout_scale`

```python
cutlass.utils.get_gmem_layout_scale( scale_shape_mkl: tuple[int, int, int], scale_granularity_m: int, scale_granularity_k: int, scale_major_mode: OperandMajorMode, ) → cutlass.cute.typing.Layout
```

Get the layout of the scale tensor in global memory.
:param scale\_shape\_mkl: The shape of the scale tensor (M, K, L).
:type scale\_shape\_mkl: tuple[int, int, int]
:return: The layout of the scale tensor in global memory.
:rtype: cute.Layout

<a id="cutlass.utils.get_smem_layout_scale"></a>
### `cutlass.utils.get_smem_layout_scale`

```python
cutlass.utils.get_smem_layout_scale( mma_tiler: tuple[int, int, int], use_2cta_instrs: bool, scale_granularity_m: int, scale_granularity_k: int, scale_major_mode: cutlass.cute.nvgpu.OperandMajorMode, a_scale_dtype: type[cutlass.Numeric], num_scale_load2trans_stage: int, ) → tuple[tuple[int, int], cute.ComposedLayout, cute.ComposedLayout]
```

Get the layout of the scale tensor in shared memory.
:return: A tuple containing (scale\_tile\_shape, smem\_layout\_scale\_per\_stage, smem\_layout\_scale) where:

> - scale\_tile\_shape: The tile shape
> - smem\_layout\_scale\_per\_stage: Shared memory layout for scale tensor per stage
> - smem\_layout\_scale: Shared memory layout for scale tensor

**Return type:**

tuple[tuple[int, int], cute.ComposedLayout, cute.ComposedLayout]

<a id="cutlass.utils.compute_smem_layout"></a>
### `cutlass.utils.compute_smem_layout`

```python
cutlass.utils.compute_smem_layout( tiled_mma: cute.TiledMma, mma_tiler_mnk: tuple[int, int, int], a_dtype: type[cutlass.Numeric], b_dtype: type[cutlass.Numeric], load2trans_stage_count: int, trans2mma_stage_count: int, ) → tuple[cute.ComposedLayout, cute.ComposedLayout, cute.ComposedLayout]
```

Compute shared memory layouts for tensor A, transformed A and tensor B.
:param tiled\_mma: The tiled MMA object defining the core computation.
:type tiled\_mma: cute.TiledMma
:param mma\_tiler\_mnk: The shape (M, N, K) of the MMA tiler.
:type mma\_tiler\_mnk: tuple[int, int, int]
:param a\_dtype: Data type of operand A.
:type a\_dtype: type[cutlass.Numeric]
:param b\_dtype: Data type of operand B.
:type b\_dtype: type[cutlass.Numeric]
:param load2trans\_stage\_count: Number of stages for load-to-transform pipeline.
:type load2trans\_stage\_count: int
:param trans2mma\_stage\_count: Number of stages for transform-to-MMA pipeline.
:type trans2mma\_stage\_count: int
:return: A tuple containing (smem\_layout\_a, smem\_layout\_a\_transform, smem\_layout\_b) where:

> - smem\_layout\_a: Shared memory layout for tensor A
> - smem\_layout\_a\_transform: Shared memory layout for transformed tensor A
> - smem\_layout\_b: Shared memory layout for tensor B

**Return type:**

tuple[cute.ComposedLayout, cute.ComposedLayout, cute.ComposedLayout]

<a id="cutlass.utils.get_transform_a_source"></a>
### `cutlass.utils.get_transform_a_source`

```python
cutlass.utils.get_transform_a_source( a_major_mode: OperandMajorMode, ) → OperandSource
```

Determine the operand source for transformed A tensor based on the operand major mode.

<a id="cutlass.utils.get_tma_atom_kind"></a>
### `cutlass.utils.get_tma_atom_kind`

```python
cutlass.utils.get_tma_atom_kind( mcast: cutlass.Boolean, use_2cta_instrs: bool, is_b: bool, ) → cpasync.CopyBulkTensorTileG2SMulticastOp | cpasync.CopyBulkTensorTileG2SOp
```

Get the TMA atom kind based on 1) whether it’s a multicast operation,
2) whether 2CTA tcgen05.mma instruction is enabled, and
3) whether it’s a B tensor

<a id="cutlass.utils.get_copy_atom_a_transform"></a>
### `cutlass.utils.get_copy_atom_a_transform`

```python
cutlass.utils.get_copy_atom_a_transform( mma_dtype: type[cutlass.Numeric], use_2cta_instrs: bool, transform_a_source: tcgen05.OperandSource, a_smem_shape: cute.Shape, a_dtype: type[cutlass.Numeric], ) → cute.CopyAtom
```

Determine the copy atom for transformed A tensor based on the operand source and tile size.

<a id="cutlass.utils.is_valid_scale_granularity"></a>
### `cutlass.utils.is_valid_scale_granularity`

```python
cutlass.utils.is_valid_scale_granularity( scale_granularity_m: int, scale_granularity_k: int, a_dtype: type[cutlass.Numeric], k: int, mma_tiler_k: int, ) → bool
```

Check if the scale granularity settings are valid for the given data type and problem size.

<a id="cutlass.utils.get_divisibility"></a>
### `cutlass.utils.get_divisibility`

```python
cutlass.utils.get_divisibility( contiguous_dim_size: int, upper_bound: int = 128, ) → int
```

Calculate the largest power of 2 divisibility factor for memory alignment.

<a id="cutlass.utils.cluster_shape_to_tma_atom_A"></a>
### `cutlass.utils.cluster_shape_to_tma_atom_A`

```python
cutlass.utils.cluster_shape_to_tma_atom_A( cluster_shape_mnk: cutlass.cute.typing.Shape, atom_thr_id: cutlass.cute.typing.Layout, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → CopyBulkTensorTileG2SMulticastOp | CopyBulkTensorTileG2SOp
```

Select the appropriate TMA copy atom for A based on the number of SMs and the multicast flag.

**Parameters:**

- **cluster\_shape\_mnk** (*cute.Shape*) – The shape of the cluster
- **atom\_thr\_id** (*cute.Layout*) – The thread ID of the atom

**Returns:**

The appropriate TMA copy atom kind

**Return type:**

[cpasync.CopyBulkTensorTileG2SMulticastOp](cutedsl_cute_dsl_api_cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp) or [cpasync.CopyBulkTensorTileG2SOp](cutedsl_cute_dsl_api_cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp)

**Raises:**

- **ValueError** – If the atom\_sm\_cnt is invalid
- **ValueError** – If the cluster shape is not divisible by the atom SM count

<a id="cutlass.utils.cluster_shape_to_tma_atom_B"></a>
### `cutlass.utils.cluster_shape_to_tma_atom_B`

```python
cutlass.utils.cluster_shape_to_tma_atom_B( cluster_shape_mnk: cutlass.cute.typing.Shape, atom_thr_id: cutlass.cute.typing.Layout, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → CopyBulkTensorTileG2SMulticastOp | CopyBulkTensorTileG2SOp
```

Select the appropriate TMA copy atom for Bbased on the number of SMs and the multicast flag.

**Parameters:**

- **cluster\_shape\_mnk** (*cute.Shape*) – The shape of the cluster
- **atom\_thr\_id** (*cute.Layout*) – The thread ID of the atom

**Returns:**

The appropriate TMA copy atom kind

**Return type:**

[cpasync.CopyBulkTensorTileG2SMulticastOp](cutedsl_cute_dsl_api_cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp) or [cpasync.CopyBulkTensorTileG2SOp](cutedsl_cute_dsl_api_cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp)

**Raises:**

- **ValueError** – If the atom\_sm\_cnt is invalid
- **ValueError** – If the cluster shape is not divisible by the atom SM count

<a id="cutlass.utils.cluster_shape_to_tma_atom_SFB"></a>
### `cutlass.utils.cluster_shape_to_tma_atom_SFB`

```python
cutlass.utils.cluster_shape_to_tma_atom_SFB( cluster_shape_mnk: cutlass.cute.typing.Shape, atom_thr_id: cutlass.cute.typing.Layout, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → CopyBulkTensorTileG2SMulticastOp | CopyBulkTensorTileG2SOp
```

Select the appropriate TMA copy atom for SFB based on the number of SMs and the multicast flag.

**Parameters:**

- **cluster\_shape\_mnk** (*cute.Shape*) – The shape of the cluster
- **atom\_thr\_id** (*cute.Layout*) – The thread ID of the atom

**Returns:**

The appropriate TMA copy atom kind

**Return type:**

[cpasync.CopyBulkTensorTileG2SMulticastOp](cutedsl_cute_dsl_api_cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp) or [cpasync.CopyBulkTensorTileG2SOp](cutedsl_cute_dsl_api_cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp)

**Raises:**

- **ValueError** – If the atom\_sm\_cnt is invalid
- **ValueError** – If the cluster shape is not divisible by the atom SM count

<a id="cutlass.utils.compute_epilogue_tile_shape"></a>
### `cutlass.utils.compute_epilogue_tile_shape`

```python
cutlass.utils.compute_epilogue_tile_shape( cta_tile_shape: cutlass.cute.typing.Shape, use_2cta_instrs: bool, layout_d: LayoutEnum, elem_ty_d: Type[cutlass.cutlass_dsl.Numeric], *, layout_c: LayoutEnum | None = None, elem_ty_c: Type[cutlass.cutlass_dsl.Numeric] | None = None, tmem_warp_shape_mn: Tuple[int, int] | None = None, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Tile
```

Attempts to compute a reasonable epilogue tile based on block tile shape or allows the user to provide one.

**Parameters:**

- **cta\_tile\_shape** (*cute.Shape*) – A tuple or list representing the dimensions of the CTA tile, where
  cta\_tile\_shape[0] corresponds to the height (M) and cta\_tile\_shape[1]
  corresponds to the width (N) of the tile.
- **use\_2cta\_instrs** (*bool*) – A flag indicating whether the configuration is for a 2SM setup.
- **layout\_d** ([*LayoutEnum*](#cutlass.utils.LayoutEnum)) – The layout enum of the output tensor D.
- **elem\_ty\_d** (*Type*[*Numeric*]) – The element type of output tensor D.
- **layout\_c** ([*LayoutEnum*](#cutlass.utils.LayoutEnum), *optional*) – The layout enum of the input tensor C. Defaults to None.
- **elem\_ty\_c** (*Union*[*Type*[*Numeric*], *None*], *optional*) – The element type for input tensor C. Defaults to None.
- **tmem\_warp\_shape\_mn** (*Tuple*[*int*, *int*], *optional*) – Optional (warp\_m, warp\_n) override for the tmem
  subpartition layout. When omitted, the layout is derived from
  `cta_tile_shape` and `use_2cta_instrs`.

**Returns:**

Returns epilog tiler, which is used in subsequent epilog partitions.

**Return type:**

cute.Tile

**Raises:**

**ValueError** – If the computed tile cute.size does not meet minimum requirements based on CTA dimensions.

<a id="cutlass.utils.get_permutation_mnk"></a>
### `cutlass.utils.get_permutation_mnk`

```python
cutlass.utils.get_permutation_mnk( tile_shape_mnk: cutlass.cute.typing.Shape, sf_vec_size: int, use_mxf8f6f4: bool, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → Tuple[int, int, int]
```

Get the permutation of M, N, K for the tiled MMA.

**Parameters:**

- **tile\_shape\_mnk** (*cute.Shape*) – The shape of the tile
- **sf\_vec\_size** (*int*) – The vector size of the Scale Factor.
- **use\_mxf8f6f4** (*bool*) – Whether to use MXF8F6F4 or MXF4NVF4.

**Returns:**

The permutation of M, N, K

**Return type:**

Tuple[int, int, int]

**Raises:**

**ValueError** – If the tile shape is not divisible by the sf\_vec\_size

<a id="cutlass.utils.get_smem_layout_atom_ab"></a>
### `cutlass.utils.get_smem_layout_atom_ab`

```python
cutlass.utils.get_smem_layout_atom_ab( major_mode: OperandMajorMode, element_type: Type[cutlass.cutlass_dsl.Numeric], smem_shape_mn_k: cutlass.cute.typing.Tile, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → SmemLayoutAtomKind
```

Simple heuristics to select the optimal SMEM layout atom based on the
majorness, the data type, and the major mode size.

**Parameters:**

- **major\_mode** ([*cutlass.cute.nvgpu.OperandMajorMode*](cutedsl_cute_dsl_api_cute_nvgpu_common.md#cutlass.cute.nvgpu.OperandMajorMode)) – The major mode for the SMEM tensor is K major.
- **element\_type** (*Type*[*Numeric*]) – The element type for the SMEM tensor.
- **smem\_shape\_mn\_k** (*cute.Tile*) – The shape of the SMEM tensor.

**Returns:**

The SMEM layout atom kind

**Return type:**

[cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind](cutedsl_cute_dsl_api_cute_nvgpu_tcgen05.md#cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind)

<a id="cutlass.utils.get_smem_layout_atom_epi"></a>
### `cutlass.utils.get_smem_layout_atom_epi`

```python
cutlass.utils.get_smem_layout_atom_epi( layout: LayoutEnum, element_type: Type[cutlass.cutlass_dsl.Numeric], epi_tile: cutlass.cute.typing.Tile, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → SmemLayoutAtomKind
```

Simple heuristics to select the optimal SMEM layout atom for epilog tensors.

**Parameters:**

- **layout** ([*LayoutEnum*](#cutlass.utils.LayoutEnum)) – The layout enum for the SMEM tensor.
- **element\_type** (*Type*[*Numeric*]) – The element type for the SMEM tensor.
- **epi\_tile** (*cute.Tile*) – The epilogue tile shape.

**Returns:**

The SMEM layout atom kind

**Return type:**

[cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind](cutedsl_cute_dsl_api_cute_nvgpu_tcgen05.md#cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind)

<a id="cutlass.utils.get_smem_store_op"></a>
### `cutlass.utils.get_smem_store_op`

```python
cutlass.utils.get_smem_store_op( layout_d: LayoutEnum, elem_ty_d: Type[cutlass.cutlass_dsl.Numeric], elem_ty_acc: Type[cutlass.cutlass_dsl.Numeric], tiled_tmem_load: TiledCopy, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → CopyAtom
```

Selects the largest vectorized smem store atom available subject to
constraint of gmem layout and chosen TMEM\_LOAD’s thread-value ownership.

**Parameters:**

- **layout\_d** ([*LayoutEnum*](#cutlass.utils.LayoutEnum)) – The layout enum of the output tensor D.
- **elem\_ty\_d** (*Type*[*Numeric*]) – The element type for output tensor D.
- **elem\_ty\_acc** (*Type*[*Numeric*]) – The element type for accumulator.
- **tiled\_tmem\_load** ([*cute.TiledCopy*](cutedsl_cute_dsl_api_cute.md#cutlass.cute.TiledCopy)) – An instance of TiledCopy that represents the tmem load operation.

**Returns:**

Either SmemStoreMatrix or SimtSyncCopy, based on the input parameters.

**Return type:**

[cute.CopyAtom](cutedsl_cute_dsl_api_cute.md#cutlass.cute.CopyAtom)

<a id="cutlass.utils.get_tmem_load_op"></a>
### `cutlass.utils.get_tmem_load_op`

```python
cutlass.utils.get_tmem_load_op( cta_tile_shape: cutlass.cute.typing.Shape, layout_d: LayoutEnum, elem_ty_d: Type[cutlass.cutlass_dsl.Numeric], elem_ty_acc: Type[cutlass.cutlass_dsl.Numeric], epi_tile: cutlass.cute.typing.Tile, use_2cta_instrs: bool, *, tmem_warp_shape_mn: Tuple[int, int] | None = None, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → CopyAtom
```

Finds a performant TMEM\_LOAD copy op for the selected epilogue
tile (epi\_tile), element types, and tcgen05.mma instruction used.

**Parameters:**

- **cta\_tile\_shape** (*cute.Shape*) – A tuple or list representing the dimensions of the CTA tile.
- **layout\_d** ([*LayoutEnum*](#cutlass.utils.LayoutEnum)) – The layout enum of the output tensor D.
- **elem\_ty\_d** (*Type*[*Numeric*]) – The element type for output tensor D.
- **elem\_ty\_acc** (*Type*[*Numeric*]) – The element type for accumulation.
- **epi\_tile** (*cute.Tile*) – The epilogue tile configuration.
- **use\_2cta\_instrs** (*bool*) – A flag indicating whether the configuration is for 2 SMs.

**Returns:**

An instance of Sm100TmemLoad with the computed configuration.

**Return type:**

[cute.CopyAtom](cutedsl_cute_dsl_api_cute.md#cutlass.cute.CopyAtom)

**Raises:**

**ValueError** – If the function cannot handle the given combination of accumulation
and dimension types, or if it cannot determine the appropriate configuration based on
the input parameters.

<a id="cutlass.utils.make_smem_layout"></a>
### `cutlass.utils.make_smem_layout`

```python
cutlass.utils.make_smem_layout( leading_mode: OperandMajorMode, smem_tile_shape: cutlass.cute.typing.Tile, a_dtype: Type[cutlass.cutlass_dsl.Numeric], num_stages: int, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout
```

Construct a staged SMEM layout for an operand given its major mode and tile shape.

This helper:

1. Selects a SMEM layout atom using simple heuristics based on the operand’s major mode,
   element type, and the size of the major dimension in `smem_tile_shape`.
2. Tiles the atom to `smem_tile_shape` and appends a staging dimension of length `num_stages`.
3. Orders the `(M, N, stage)` axes so the major dimension is contiguous, then coalesces.

**Parameters:**

- **leading\_mode** ([*cutlass.cute.nvgpu.OperandMajorMode*](cutedsl_cute_dsl_api_cute_nvgpu_common.md#cutlass.cute.nvgpu.OperandMajorMode)) – Operand major mode (`MN` or `K`) of the staged operand.
- **smem\_tile\_shape** (*cute.Tile*) – 2D SMEM tile shape to stage (before the staging dimension is appended).
- **a\_dtype** (*Type*[*Numeric*]) – Element type of the staged operand.
- **num\_stages** (*int*) – Number of pipeline stages (depth of the staging dimension).

**Returns:**

Staged SMEM layout for the operand.

**Return type:**

Union[cute.Layout, cute.ComposedLayout]

<a id="cutlass.utils.make_smem_layout_a"></a>
### `cutlass.utils.make_smem_layout_a`

```python
cutlass.utils.make_smem_layout_a( tiled_mma: TiledMma, mma_tiler_mnk: cutlass.cute.typing.Tile, a_dtype: Type[cutlass.cutlass_dsl.Numeric], num_stages: int, *, is_k_major: bool | None = None, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout
```

This function helps with:

1. Get the partitioned shape of the A tensor based on the tiled\_mma & MMA tiler.
2. Select the heuristic SMEM layout atom based on the A tensor’s majorness, the data type, and the major mode size.
3. cute.Tile the SMEM layout atom to the MMA tile shape.
4. Stage the SMEM layout based on the number of stages.

**Parameters:**

- **tiled\_mma** ([*cute.TiledMma*](cutedsl_cute_dsl_api_cute.md#cutlass.cute.TiledMma)) – The tiled MMA used to partition tensor A
- **mma\_tiler\_mnk** (*cute.cute.Tile*) – The MMA tile shape
- **a\_dtype** (*Type*[*Numeric*]) – The element type for tensor A
- **num\_stages** (*int*) – The number of pipeline stages for tensor A

**Returns:**

SMEM layout for tensor A

**Return type:**

Union[cute.Layout, cute.ComposedLayout]

<a id="cutlass.utils.make_smem_layout_b"></a>
### `cutlass.utils.make_smem_layout_b`

```python
cutlass.utils.make_smem_layout_b( tiled_mma: TiledMma, mma_tiler_mnk: cutlass.cute.typing.Tile, b_dtype: Type[cutlass.cutlass_dsl.Numeric], num_stages: int, *, is_k_major: bool | None = None, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout
```

This function helps:

1. Get the partitioned shape of the B tensor based on the tiled\_mma & MMA tiler.
2. Select the heuristic SMEM layout atom based on the B tensor’s majorness, the data type, and the major mode size.
3. cute.Tile the SMEM layout atom to the MMA tile shape.
4. Stage the SMEM layout based on the number of stages.

**Parameters:**

- **tiled\_mma** ([*cute.TiledMma*](cutedsl_cute_dsl_api_cute.md#cutlass.cute.TiledMma)) – The tiled MMA which is used to partition the B tensor.
- **mma\_tiler\_mnk** (*cute.cute.Tile*) – The MMA tile shape.
- **b\_dtype** (*Type*[*Numeric*]) – The element type for the B tensor.
- **num\_stages** (*int*) – The stage of the B tensor.

**Returns:**

SMEM layout for the B tensor.

**Return type:**

Union[cute.Layout, cute.ComposedLayout]

<a id="cutlass.utils.make_smem_layout_epi"></a>
### `cutlass.utils.make_smem_layout_epi`

```python
cutlass.utils.make_smem_layout_epi( epi_dtype: Type[cutlass.cutlass_dsl.Numeric], epi_layout: LayoutEnum, epi_tile: cutlass.cute.typing.Tile, epi_stage: int, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout
```

This function helps:

1. Select the heuristic SMEM layout atom based on the epilog tile shape,
   the epilog tensor’s majorness, and the element type.
2. cute.Tile the SMEM layout atom to the epilog tile shape.
3. Stage the SMEM layout based on the number of stages.

**Parameters:**

- **epi\_dtype** (*Type*[*Numeric*]) – The element type for the epilog tensor.
- **epi\_layout** ([*LayoutEnum*](#cutlass.utils.LayoutEnum)) – The layout enum for the epilog tensor.
- **epi\_tile** (*cute.cute.Tile*) – The epilogue tile shape.
- **epi\_stage** (*int*) – The stage of the epilog tensor.

**Returns:**

SMEM layout for epilog tensors (usually C & D which are processed in the epilog)

**Return type:**

Union[cute.Layout, cute.ComposedLayout]

<a id="cutlass.utils.make_trivial_tiled_mma"></a>
### `cutlass.utils.make_trivial_tiled_mma`

```python
cutlass.utils.make_trivial_tiled_mma( *args: Any, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, **kwargs: Any, ) → TiledMma
```

Make a tiled MMA atom with given data type, leading dimension, cta group and mma tile shape.
By default, the MMA atom is created with SMEM operand source for A.

Supports two calling conventions:

**New (recommended):** separate `a_dtype` and `b_dtype`:

```console
make_trivial_tiled_mma(
    a_dtype, b_dtype, a_leading_mode, b_leading_mode,
    acc_dtype, cta_group, mma_tiler_mn, [a_source])
```

**Legacy (deprecated):** single `ab_dtype`:

```console
make_trivial_tiled_mma(
    ab_dtype, a_leading_mode, b_leading_mode,
    acc_dtype, cta_group, mma_tiler_mn, [a_source])
```

<a id="cutlass.utils.make_blockscaled_trivial_tiled_mma"></a>
### `cutlass.utils.make_blockscaled_trivial_tiled_mma`

```python
cutlass.utils.make_blockscaled_trivial_tiled_mma( *args: Any, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, **kwargs: Any, ) → TiledMma
```

Make a BlockScaled tiled MMA atom with given data type, leading dimension, cta group and mma tile shape.
By default, the MMA atom is created with SMEM operand source for A.

Supports two calling conventions:

**New (recommended):** separate `a_dtype` and `b_dtype`:

```console
make_blockscaled_trivial_tiled_mma(
    a_dtype, b_dtype, a_leading_mode, b_leading_mode,
    sf_dtype, sf_vec_size, cta_group, mma_tiler_mn, [a_source])
```

**Legacy (deprecated):** single `ab_dtype`:

```console
make_blockscaled_trivial_tiled_mma(
    ab_dtype, a_leading_mode, b_leading_mode,
    sf_dtype, sf_vec_size, cta_group, mma_tiler_mn, [a_source])
```

<a id="cutlass.utils.sm90_get_smem_layout_atom"></a>
### `cutlass.utils.sm90_get_smem_layout_atom`

```python
cutlass.utils.sm90_get_smem_layout_atom( layout: LayoutEnum, element_type: Type[cutlass.cutlass_dsl.Numeric], major_mode_size: int, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → Any
```

Select the optimal shared memory layout atom based on parameters.

**Parameters:**

- **layout** ([*LayoutEnum*](#cutlass.utils.LayoutEnum)) – Layout enum of the tensor
- **element\_type** (*type*[*Numeric*]) – Data type of the elements
- **major\_mode\_size** (*int*) – Size of the major mode dimension

**Returns:**

Selected shared memory layout atom kind

**Return type:**

[cutlass.cute.nvgpu.warpgroup.SmemLayoutAtomKind](cutedsl_cute_dsl_api_cute_nvgpu_warpgroup.md#cutlass.cute.nvgpu.warpgroup.SmemLayoutAtomKind)

<a id="cutlass.utils.sm90_make_trivial_tiled_mma"></a>
### `cutlass.utils.sm90_make_trivial_tiled_mma`

```python
cutlass.utils.sm90_make_trivial_tiled_mma( a_dtype: Type[cutlass.cutlass_dsl.Numeric], b_dtype: Type[cutlass.cutlass_dsl.Numeric], a_leading_mode: OperandMajorMode, b_leading_mode: OperandMajorMode, acc_dtype: Type[cutlass.cutlass_dsl.Numeric], atom_layout_mnk: Tuple[int, int, int], tiler_mn: Tuple[int, int], a_source: OperandSource = cutlass._mlir.dialects.cute.MmaFragKind.smem_desc, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → TiledMma
```

Make a tiled MMA atom with given data type, leading dimension, cta group and mma tile shape.
By default, the MMA atom is created with SMEM operand source for A.

**Parameters:**

- **a\_dtype** (*type*[*Numeric*]) – Data type of operand A.
- **b\_dtype** (*type*[*Numeric*]) – Data type of operand B.
- **a\_leading\_mode** ([*cutlass.cute.nvgpu.OperandMajorMode*](cutedsl_cute_dsl_api_cute_nvgpu_common.md#cutlass.cute.nvgpu.OperandMajorMode)) – Leading dimension of operand A (1 for K, 0 for M/N).
- **b\_leading\_mode** ([*cutlass.cute.nvgpu.OperandMajorMode*](cutedsl_cute_dsl_api_cute_nvgpu_common.md#cutlass.cute.nvgpu.OperandMajorMode)) – Leading dimension of operand B (1 for K, 0 for M/N).
- **acc\_dtype** (*type*[*Numeric*]) – Data type of the accumulator.
- **atom\_layout\_mnk** (*Tuple*[*int*, *int*, *int*]) – A integer tuple describing the tiling of Atom across threads.
- **tiler\_mn** (*Tuple*[*int*, *int*]) – The shape (M, N) of the cta tiler.

**Returns:**

A tiled MMA atom.

**Return type:**

[cute.TiledMma](cutedsl_cute_dsl_api_cute.md#cutlass.cute.TiledMma)

**Raises:**

**TypeError** – If the data type is not supported.

<a id="cutlass.utils.ClcDynamicPersistentTileSchedulerParams"></a>
### `cutlass.utils.ClcDynamicPersistentTileSchedulerParams`

```python
class cutlass.utils.ClcDynamicPersistentTileSchedulerParams(**kwargs)
```

Bases: `object`

A class to represent parameters for a dynamic persistent tile scheduler.

This class is designed to manage and compute the layout of clusters and tiles
in a batched gemm problem.

**Variables:**

**cluster\_shape\_mn** – Shape of the cluster in (m, n) dimensions (K dimension cta count must be 1).

<a id="cutlass.utils.ClcDynamicPersistentTileSchedulerParams.__init__"></a>
#### `cutlass.utils.ClcDynamicPersistentTileSchedulerParams.__init__`

```python
__init__( problem_shape_ntile_mnl: cutlass.cute.typing.Shape, cluster_shape_mnk: cutlass.cute.typing.Shape, swizzle_size: int = 1, raster_along_m: bool = True, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

Initializes the ClcDynamicPersistentTileSchedulerParams with the given parameters.

**Parameters:**

- **problem\_shape\_ntile\_mnl** (*cute.Shape*) – The shape of the problem in terms of
  number of CTA (Cooperative Thread Array) in (m, n, l) dimensions.
- **cluster\_shape\_mnk** (*cute.Shape*) – The shape of the cluster in (m, n) dimensions.
- **swizzle\_size** (*int*) – Swizzling size in the unit of cluster. 1 means no swizzle
- **raster\_along\_m** (*bool*) – Rasterization order of clusters. Only used when swizzle\_size > 1.
  True means along M, false means along N.

**Raises:**

**ValueError** – If cluster\_shape\_k is not 1.

<a id="cutlass.utils.ClcDynamicPersistentTileSchedulerParams.get_grid_shape"></a>
#### `cutlass.utils.ClcDynamicPersistentTileSchedulerParams.get_grid_shape`

```python
get_grid_shape( *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → Tuple[cutlass.cutlass_dsl.Integer, cutlass.cutlass_dsl.Integer, cutlass.cutlass_dsl.Integer]
```

Computes the grid shape based on the problem shape and cluster shape.

**Returns:**

the grid is the CTA numbers that has aligned with cluster shape.

<a id="cutlass.utils.ClcDynamicPersistentTileScheduler"></a>
### `cutlass.utils.ClcDynamicPersistentTileScheduler`

```python
class cutlass.utils.ClcDynamicPersistentTileScheduler(**kwargs)
```

Bases: `object`

A scheduler for dynamic persistent tile execution in CUTLASS/CuTe kernels.

**Variables:**

- **params** – Tile schedule related params, including cluster shape.
- **cta\_id\_in\_cluster** – ID of the CTA within its cluster
- **\_num\_tiles\_executed** – Counter for executed tiles

<a id="cutlass.utils.ClcDynamicPersistentTileScheduler.__init__"></a>
#### `cutlass.utils.ClcDynamicPersistentTileScheduler.__init__`

```python
__init__( params: ClcDynamicPersistentTileSchedulerParams, cta_id_in_cluster: cutlass.cute.typing.Coord, num_tiles_executed: cutlass.cutlass_dsl.Int32, clc_response_ptr: cutlass.cute.typing.Pointer, block_idx: Tuple[cutlass.cutlass_dsl.Integer, cutlass.cutlass_dsl.Integer, cutlass.cutlass_dsl.Integer], insert_fence: bool = True, )
```

Initializes the ClcDynamicPersistentTileScheduler with the given parameters.

**Parameters:**

- **params** ([*ClcDynamicPersistentTileSchedulerParams*](#cutlass.utils.ClcDynamicPersistentTileSchedulerParams)) – Tile schedule related params, including cluster shape.
- **cta\_id\_in\_cluster** (*cute.Coord*) – ID of the CTA within its cluster.
- **num\_tiles\_executed** (*Int32*) – Counter for executed tiles.
- **clc\_response\_ptr** (*cute.Pointer*) – Pointer of the clc rsponse.
- **block\_idx** (*Tuple*[*Integer*, *Integer*, *Integer*]) – The block index.
- **insert\_fence** (*bool*) – Whether to insert a fence to ensure generic-async proxy order.
  CLC issue is in async proxy while loading the response from shared memory is in
  generic proxy. A cross-proxy fence is needed to ensure producer’s next issue
  won’t race with consumer’s current loading. Therefore the scheduler inserts a
  fence by default after loading the response.
  Developers may insert the fence in pipeline acquire/release functions. In that case,
  the fence here can be omitted.

<a id="cutlass.utils.ClcDynamicPersistentTileScheduler.create"></a>
#### `cutlass.utils.ClcDynamicPersistentTileScheduler.create`

```python
create( params: ClcDynamicPersistentTileSchedulerParams, block_idx: Tuple[cutlass.cutlass_dsl.Integer, cutlass.cutlass_dsl.Integer, cutlass.cutlass_dsl.Integer], grid_dim: Tuple[cutlass.cutlass_dsl.Integer, cutlass.cutlass_dsl.Integer, cutlass.cutlass_dsl.Integer], clc_response_ptr: cutlass.cute.typing.Pointer, insert_fence: bool = True, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → ClcDynamicPersistentTileScheduler
```

Initialize the dynamic persistent tile scheduler.

**Parameters:**

- **params** ([*ClcDynamicPersistentTileSchedulerParams*](#cutlass.utils.ClcDynamicPersistentTileSchedulerParams)) – Parameters for the persistent
  tile scheduler.
- **block\_idx** (*Tuple*[*Integer*, *Integer*, *Integer*]) – The 3d block index in the format (bidx, bidy, bidz).
- **grid\_dim** (*Tuple*[*Integer*, *Integer*, *Integer*]) – The 3d grid dimensions for kernel launch.

**Returns:**

A ClcDynamicPersistentTileScheduler object.

**Return type:**

[ClcDynamicPersistentTileScheduler](#cutlass.utils.ClcDynamicPersistentTileScheduler)

<a id="cutlass.utils.ClcDynamicPersistentTileScheduler.get_grid_shape"></a>
#### `cutlass.utils.ClcDynamicPersistentTileScheduler.get_grid_shape`

```python
get_grid_shape( *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → Tuple[cutlass.cutlass_dsl.Integer, cutlass.cutlass_dsl.Integer, cutlass.cutlass_dsl.Integer]
```

Calculates the grid shape to be launched on GPU using problem shape,
threadblock shape, and active cluster size.

**Parameters:**

**params** ([*ClcDynamicPersistentTileSchedulerParams*](#cutlass.utils.ClcDynamicPersistentTileSchedulerParams)) – Parameters for grid shape calculation.

**Returns:**

The calculated 3d grid shape.

**Return type:**

Tuple[Integer, Integer, Integer]

<a id="cutlass.utils.ClcDynamicPersistentTileScheduler._swizzle_and_rasterize"></a>
#### `cutlass.utils.ClcDynamicPersistentTileScheduler._swizzle_and_rasterize`

```python
_swizzle_and_rasterize( x_idx: cutlass.cutlass_dsl.Int32, y_idx: cutlass.cutlass_dsl.Int32, z_idx: cutlass.cutlass_dsl.Int32, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → Tuple[cutlass.cutlass_dsl.Int32, cutlass.cutlass_dsl.Int32, cutlass.cutlass_dsl.Int32]
```

Swizzle and rasterize the given coordinates for leader CTA of the cluster.
x\_idx, y\_idx, and z\_idx must be divisible by cluster shape x, y, and z respectively. They should not be offset
by the ID of the CTA in the cluster.

<a id="cutlass.utils.ClcDynamicPersistentTileScheduler.work_tile_info_from_clc_response"></a>
#### `cutlass.utils.ClcDynamicPersistentTileScheduler.work_tile_info_from_clc_response`

```python
work_tile_info_from_clc_response( result_addr: cutlass.cute.typing.Pointer, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → WorkTileInfo
```

Simulates parsing CLC response data in Python.
result\_addr: 16-byte response data (simulating shared memory access)

<a id="cutlass.utils.ClcDynamicPersistentTileScheduler.num_tiles_executed"></a>
#### `cutlass.utils.ClcDynamicPersistentTileScheduler.num_tiles_executed`

```python
property num_tiles_executed
```

<a id="cutlass.utils.print_latex"></a>
### `cutlass.utils.print_latex`

```python
cutlass.utils.print_latex(x: cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout, *, color: ~typing.Callable = <function tikz_color_bwx8>, render_func: ~typing.Callable[[str], None] | None = None) → None
```

Prints a layout.

**Parameters:**

- **x** (*Union*[*Layout*, *ComposedLayout*]) – A layout
- **color** (*Callable*) – A function that returns TiKZ colors
- **render\_func** (*Callable*[[*str*], *None*] | *None*) – Optional callback fed the `{tikzpicture}` body
  (without the standalone-document wrapper) in a single call,
  instead of printing a full LaTeX document to stdout.

<a id="cutlass.utils.print_latex_tv"></a>
### `cutlass.utils.print_latex_tv`

```python
cutlass.utils.print_latex_tv(layout_tv: cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout, tile_mn: cutlass.cute.typing.IntTuple | cutlass.cute.typing.Layout, *, color: ~typing.Callable = <function tikz_color_tv>, palette: str | ~typing.Callable | None = None, title: str | None = None, axis_labels: bool = False, render_func: ~typing.Callable[[str], None] | None = None) → None
```

Prints a tv layout for a tile M N. Everything must be static.

**Parameters:**

- **layout\_tv** (*Union*[*Layout*, *ComposedLayout*]) – A static thread value layout
- **tile\_mn** (*Union*[*IntTuple*, *Layout*]) – A static M N tile
- **color** (*Callable*) – A function `color(tid, vid) -> str` returning a TikZ
  fill color for the cell owned by thread `tid` value `vid`.
  Used when `palette` is `None`; ignored otherwise.
- **palette** (*Optional*[*Union*[*str*, *Callable*]]) – Optional richer cell coloring that supersedes
  `color`. Either the name of a built-in palette (a key of
  `PALETTES`: the color `"pastel"`, `"rainbow"`,
  `"rainbow_dual"` or the monochrome `"white"`, `"bw"`,
  `"bw_dual"`) or a factory `palette(num_tid, num_vid) -> cell`
  where
  `cell(tid, vid)` returns either a TikZ fill string (one fill
  spanning the cell) or a list of ``` Band``s (stacked horizontal
  fills).  The factory form lets a palette capture the thread /
  value counts -- needed to spread hues evenly over the wheel --
  without widening the per-cell ``(tid, vid) ``` contract.
- **title** (*Optional*[*str*]) – Optional title drawn above the figure, one line per
  `\n`-separated segment (e.g. the operator / function /
  tensor that produced this layout). LaTeX specials are escaped.
- **axis\_labels** (*bool*) – When `True`, annotate the M (row, downward)
  and N (column, rightward) axis directions. The picture’s
  coordinate basis runs M down and N right; these labels make
  that orientation explicit so a reader can tell whether a
  thread’s values are contiguous along the memory-major axis.
- **render\_func** (*Callable*[[*str*], *None*] | *None*) – Optional callback fed the `{tikzpicture}` body
  (without the standalone-document wrapper) in a single call,
  instead of printing a full LaTeX document to stdout.

<a id="cutlass.utils.Band"></a>
### `cutlass.utils.Band`

```python
class cutlass.utils.Band(lo: float, hi: float, color: str)
```

Bases: `NamedTuple`

One horizontal fill band of a TV cell. `lo` and `hi` are
fractions in `[0, 1]` along the screen-vertical M axis (0 = the
cell’s top edge, 1 = its bottom edge); `color` is a TikZ fill spec.
A single `Band(0.0, 1.0, ...)` is an ordinary one-fill cell; the
`"rainbow_dual"` palette stacks two half-height bands.

<a id="cutlass.utils.Band.lo"></a>
#### `cutlass.utils.Band.lo`

```python
lo: float
```

Alias for field number 0

<a id="cutlass.utils.Band.hi"></a>
#### `cutlass.utils.Band.hi`

```python
hi: float
```

Alias for field number 1

<a id="cutlass.utils.Band.color"></a>
#### `cutlass.utils.Band.color`

```python
color: str
```

Alias for field number 2

<a id="cutlass.utils.Band._asdict"></a>
#### `cutlass.utils.Band._asdict`

```python
_asdict()
```

Return a new dict which maps field names to their values.

<a id="cutlass.utils.Band._field_defaults"></a>
#### `cutlass.utils.Band._field_defaults`

```python
_field_defaults = {}
```

<a id="cutlass.utils.Band._fields"></a>
#### `cutlass.utils.Band._fields`

```python
_fields = ('lo', 'hi', 'color')
```

<a id="cutlass.utils.Band._make"></a>
#### `cutlass.utils.Band._make`

```python
classmethod _make(iterable)
```

Make a new Band object from a sequence or iterable

<a id="cutlass.utils.Band._replace"></a>
#### `cutlass.utils.Band._replace`

```python
_replace(**kwds)
```

Return a new Band object replacing specified fields with new values

<a id="cutlass.utils.is_fp8_dtype"></a>
### `cutlass.utils.is_fp8_dtype`

```python
cutlass.utils.is_fp8_dtype(dtype: Type[cutlass.cute.typing.Numeric]) → bool
```

Check if dtype is a float8 type that doesn’t support dlpack.
params dtype: The cutlass numeric type to check
type dtype: Type[cutlass.Numeric]
return: True if the dtype is Float8E5M2 or Float8E4M3FN, False otherwise

<a id="cutlass.utils.create_cute_tensor_for_fp8"></a>
### `cutlass.utils.create_cute_tensor_for_fp8`

```python
cutlass.utils.create_cute_tensor_for_fp8( storage_tensor: Any, dtype: Type[cutlass.cute.typing.Numeric], leading_dim: int, source_f32_tensor: Any | None = None, assumed_align: int = 16, mark_dynamic_layout: bool = True, ) → cutlass.cute.typing.Tensor
```

Create cute tensor, handling float8 types that don’t support dlpack.

For float8 types, the storage\_tensor should use byte storage (for DLPack compatibility).
The source\_f32\_tensor provides the actual float32 values to convert to fp8.

params storage\_tensor: Tensor for DLPack (byte storage for fp8, otherwise the actual dtype)
params dtype: Target cutlass dtype
params leading\_dim: Leading dimension for dynamic layout
paramas source\_f32\_tensor: Float32 source data for fp8 conversion (required for fp8)
params assumed\_align: Assumed alignment for the DLPack tensor
params mark\_dynamic\_layout: Whether to mark the resulting tensor layout dynamic
return: A cute tensor with the appropriate dtype and layout
