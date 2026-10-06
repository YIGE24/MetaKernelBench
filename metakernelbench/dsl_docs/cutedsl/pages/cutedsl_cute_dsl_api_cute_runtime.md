<!-- source: https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_runtime.html -->

<a id="runtime"></a>
# Runtime

Description

<a id="module-cutlass.cute.runtime"></a>
## API documentation

<a id="cutlass.cute.runtime._Pointer"></a>
### `cutlass.cute.runtime._Pointer`

```python
class cutlass.cute.runtime._Pointer(*args: Any, **kwargs: Any)
```

Bases: `Pointer`

Runtime representation of a pointer that can inter-operate with various data structures,
including numpy arrays and device memory.

**Parameters:**

- **pointer** (*int* *or* *pointer-like object*) – The pointer to the data
- **dtype** (*Type*) – Data type of the elements pointed to
- **mem\_space** (*AddressSpace*, *optional*) – Memory space where the pointer resides, defaults to generic
- **assumed\_align** (*int*, *optional*) – Assumed alignment of input pointer in bytes, defaults to None

**Variables:**

- **\_pointer** – The underlying pointer
- **\_dtype** – Data type of the elements
- **\_addr\_space** – Memory space of the pointer
- **\_assumed\_align** – Alignment of the pointer in bytes
- **\_desc** – C-type descriptor for the pointer
- **\_c\_pointer** – C-compatible pointer representation

<a id="cutlass.cute.runtime._Pointer.__init__"></a>
#### `cutlass.cute.runtime._Pointer.__init__`

```python
__init__( pointer: int, dtype: Type[cutlass.cute.typing.Numeric], mem_space: cutlass.cute.typing.AddressSpace = cutlass.cute.typing.AddressSpace.generic, assumed_align: int | None = None, ) → None
```

<a id="cutlass.cute.runtime._Pointer.size_in_bytes"></a>
#### `cutlass.cute.runtime._Pointer.size_in_bytes`

```python
size_in_bytes() → int
```

<a id="cutlass.cute.runtime._Pointer.mlir_type"></a>
#### `cutlass.cute.runtime._Pointer.mlir_type`

```python
property mlir_type: cutlass._mlir.ir.Type
```

<a id="cutlass.cute.runtime._Pointer.dtype"></a>
#### `cutlass.cute.runtime._Pointer.dtype`

```python
property dtype: Type[cutlass.cute.typing.Numeric]
```

<a id="cutlass.cute.runtime._Pointer.memspace"></a>
#### `cutlass.cute.runtime._Pointer.memspace`

```python
property memspace: cutlass.cute.typing.AddressSpace
```

<a id="cutlass.cute.runtime._Pointer.align"></a>
#### `cutlass.cute.runtime._Pointer.align`

```python
align( min_align: int, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Pointer
```

<a id="cutlass.cute.runtime._Tensor"></a>
### `cutlass.cute.runtime._Tensor`

```python
class cutlass.cute.runtime._Tensor(*args: Any, **kwargs: Any)
```

Bases: `Tensor`

<a id="cutlass.cute.runtime._Tensor.__init__"></a>
#### `cutlass.cute.runtime._Tensor.__init__`

```python
__init__( tensor: object, assumed_align: int | None = None, use_32bit_stride: bool = False, *, enable_tvm_ffi: bool = False, ) → None
```

<a id="cutlass.cute.runtime._Tensor.load_dltensor"></a>
#### `cutlass.cute.runtime._Tensor.load_dltensor`

```python
load_dltensor() → None
```

Lazily load the DLTensorWrapper.

This function loads the DLTensorWrapper when needed,
avoiding overhead in the critical path of calling JIT functions.

<a id="cutlass.cute.runtime._Tensor.mark_layout_dynamic"></a>
#### `cutlass.cute.runtime._Tensor.mark_layout_dynamic`

```python
mark_layout_dynamic( leading_dim: int | None = None, ) → _Tensor
```

Marks the tensor layout as dynamic based on the leading dimension.

**Parameters:**

**leading\_dim** (*int*, *optional*) – The leading dimension of the layout, defaults to None

When `leading_dim` is None, the leading dimension is deduced as follows:

- If exactly one dimension has stride 1, that dimension is used.
- If multiple dimensions have stride 1 but exactly one of them has size > 1,
  that dimension is used.
- If multiple dimensions have stride 1 but none or more than one has size > 1,
  an error is raised.
- If no dimension has stride 1, all strides remain dynamic.

When `leading_dim` is explicitly specified, marks the layout as dynamic while setting the
stride at `leading_dim` to 1. Also validates that the specified `leading_dim` is consistent
with the existing layout by checking that the corresponding stride of that dimension is 1.

Limitation: only support flat layout for now. Will work on supporting nested layout in the future.

**Returns:**

The tensor with dynamic layout

**Return type:**

[\_Tensor](#cutlass.cute.runtime._Tensor)

<a id="cutlass.cute.runtime._Tensor.mark_compact_shape_dynamic"></a>
#### `cutlass.cute.runtime._Tensor.mark_compact_shape_dynamic`

```python
mark_compact_shape_dynamic( mode: int, stride_order: tuple[int, ...] | None = None, divisibility: int = 1, ) → _Tensor
```

Marks the tensor shape as dynamic and propagates dynamic and divisibility information to the corresponding strides.

**Parameters:**

- **mode** (*int*) – The mode of the compact shape, defaults to 0
- **stride\_order** – Consistent with torch.Tensor.dim\_order. Defaults to None.

Indicates the order of the modes (dimensions) if the current layout were converted to row-major order.
It starts from the outermost to the innermost dimension.
:type stride\_order: tuple[int, …], optional
:param divisibility: The divisibility constraint for the compact shape, defaults to 1
:type divisibility: int, optional
:return: The tensor with dynamic compact shape
:rtype: \_Tensor

If `stride_order` is not provided, the stride ordering will be automatically deduced from the layout.
Automatic deduction is only possible when exactly one dimension has a stride of 1 (compact layout).
An error is raised if automatic deduction fails.

If `stride_order` is explicitly specified, it does the consistency check with the layout.

For example:
- Layout: (4,2):(1,4) has stride\_order: (1,0) indicates the innermost dimension is 0(4:1), the outermost dimension is 1(2:4)
- Layout: (5,3,2,4):(3,1,15,30) has stride\_order: (3,2,0,1) indicates the innermost dimension is 1(3:1), the outermost dimension is 3(4:30).

Using torch.Tensor.dim\_order() to get the stride order of the torch tensor.
.. code-block:: python
a = torch.empty(3, 4)
t = cute.runtime.from\_dlpack(a)
t = t.mark\_compact\_shape\_dynamic(mode=0, stride\_order=a.dim\_order())

<a id="cutlass.cute.runtime._Tensor.element_type"></a>
#### `cutlass.cute.runtime._Tensor.element_type`

```python
property element_type: Type[cutlass.cute.typing.Numeric]
```

<a id="cutlass.cute.runtime._Tensor.dtype"></a>
#### `cutlass.cute.runtime._Tensor.dtype`

```python
property dtype: Type[cutlass.cute.typing.Numeric]
```

<a id="cutlass.cute.runtime._Tensor.memspace"></a>
#### `cutlass.cute.runtime._Tensor.memspace`

```python
property memspace: cutlass.cute.typing.AddressSpace
```

<a id="cutlass.cute.runtime._Tensor.size_in_bytes"></a>
#### `cutlass.cute.runtime._Tensor.size_in_bytes`

```python
property size_in_bytes: int
```

<a id="cutlass.cute.runtime._Tensor.mlir_type"></a>
#### `cutlass.cute.runtime._Tensor.mlir_type`

```python
property mlir_type: cutlass._mlir.ir.Type
```

<a id="cutlass.cute.runtime._Tensor.iterator"></a>
#### `cutlass.cute.runtime._Tensor.iterator`

```python
property iterator: _Pointer
```

<a id="cutlass.cute.runtime._Tensor.layout"></a>
#### `cutlass.cute.runtime._Tensor.layout`

```python
property layout: NoReturn
```

<a id="cutlass.cute.runtime._Tensor.shape"></a>
#### `cutlass.cute.runtime._Tensor.shape`

```python
property shape: cutlass.cute.typing.Shape
```

<a id="cutlass.cute.runtime._Tensor.stride"></a>
#### `cutlass.cute.runtime._Tensor.stride`

```python
property stride: cutlass.cute.typing.Stride
```

<a id="cutlass.cute.runtime._Tensor.leading_dim"></a>
#### `cutlass.cute.runtime._Tensor.leading_dim`

```python
property leading_dim: int | tuple[int, ...] | None
```

Get the leading dimension of this Tensor.

**Returns:**

The leading dimension index or indices

**Return type:**

int or tuple or None

The return value depends on the tensor’s stride pattern:

- If a single leading dimension is found, returns an integer index
- If nested leading dimensions are found, returns a tuple of indices
- If no leading dimension is found, returns None

<a id="cutlass.cute.runtime._Tensor.fill"></a>
#### `cutlass.cute.runtime._Tensor.fill`

```python
fill(value: cutlass.cute.typing.Numeric) → None
```

<a id="cutlass.cute.runtime._Tensor.data_ptr"></a>
#### `cutlass.cute.runtime._Tensor.data_ptr`

```python
property data_ptr: int
```

<a id="cutlass.cute.runtime._Tensor.dynamic_shapes_mask"></a>
#### `cutlass.cute.runtime._Tensor.dynamic_shapes_mask`

```python
property dynamic_shapes_mask: tuple[int, ...]
```

Get the mask of dynamic shapes in the tensor.

<a id="cutlass.cute.runtime._Tensor.dynamic_strides_mask"></a>
#### `cutlass.cute.runtime._Tensor.dynamic_strides_mask`

```python
property dynamic_strides_mask: tuple[int, ...]
```

Get the mask of dynamic strides in the tensor.

<a id="cutlass.cute.runtime._FakeTensor"></a>
### `cutlass.cute.runtime._FakeTensor`

```python
class cutlass.cute.runtime._FakeTensor(*args: Any, **kwargs: Any)
```

Bases: `Tensor`

Fake Tensor implementation as a placeholder.
It mimics the interface of Tensor, but does not hold real data or allow indexing.
Used for compilation or testing situations where only shape/type/layout information is needed.
All attempts to access or mutate data will raise errors.

<a id="cutlass.cute.runtime._FakeTensor.__init__"></a>
#### `cutlass.cute.runtime._FakeTensor.__init__`

```python
__init__( dtype: Type[cutlass.cute.typing.Numeric], shape: tuple[int | cutlass.cute.typing.SymInt, ...], *, stride: tuple[int | cutlass.cute.typing.SymInt, ...], memspace: cutlass.cute.typing.AddressSpace = cutlass.cute.typing.AddressSpace.gmem, assumed_align: int | None = None, use_32bit_stride: bool = False, ) → None
```

<a id="cutlass.cute.runtime._FakeTensor.mlir_type"></a>
#### `cutlass.cute.runtime._FakeTensor.mlir_type`

```python
property mlir_type: cutlass._mlir.ir.Type
```

<a id="cutlass.cute.runtime._FakeTensor.element_type"></a>
#### `cutlass.cute.runtime._FakeTensor.element_type`

```python
property element_type: Type[cutlass.cute.typing.Numeric]
```

<a id="cutlass.cute.runtime._FakeTensor.dtype"></a>
#### `cutlass.cute.runtime._FakeTensor.dtype`

```python
property dtype: Type[cutlass.cute.typing.Numeric]
```

<a id="cutlass.cute.runtime._FakeTensor.memspace"></a>
#### `cutlass.cute.runtime._FakeTensor.memspace`

```python
property memspace: cutlass.cute.typing.AddressSpace
```

<a id="cutlass.cute.runtime._FakeTensor.iterator"></a>
#### `cutlass.cute.runtime._FakeTensor.iterator`

```python
property iterator: NoReturn
```

<a id="cutlass.cute.runtime._FakeTensor.shape"></a>
#### `cutlass.cute.runtime._FakeTensor.shape`

```python
property shape: cutlass.cute.typing.Shape
```

<a id="cutlass.cute.runtime._FakeTensor.stride"></a>
#### `cutlass.cute.runtime._FakeTensor.stride`

```python
property stride: cutlass.cute.typing.Stride
```

<a id="cutlass.cute.runtime._FakeTensor.leading_dim"></a>
#### `cutlass.cute.runtime._FakeTensor.leading_dim`

```python
property leading_dim: int | tuple[int, ...] | None
```

<a id="cutlass.cute.runtime._FakeTensor.dynamic_shapes_mask"></a>
#### `cutlass.cute.runtime._FakeTensor.dynamic_shapes_mask`

```python
property dynamic_shapes_mask: tuple[int, ...]
```

<a id="cutlass.cute.runtime._FakeTensor.dynamic_strides_mask"></a>
#### `cutlass.cute.runtime._FakeTensor.dynamic_strides_mask`

```python
property dynamic_strides_mask: tuple[int, ...]
```

<a id="cutlass.cute.runtime._FakeTensor.fill"></a>
#### `cutlass.cute.runtime._FakeTensor.fill`

```python
fill(value: cutlass.cute.typing.Numeric) → None
```

<a id="cutlass.cute.runtime.make_fake_compact_tensor"></a>
### `cutlass.cute.runtime.make_fake_compact_tensor`

```python
cutlass.cute.runtime.make_fake_compact_tensor( dtype: Type[cutlass.cute.typing.Numeric], shape: tuple[int | cutlass.cute.typing.SymInt, ...], *, stride_order: tuple[int, ...] | None = None, memspace: cutlass.cute.typing.AddressSpace = cutlass.cute.typing.AddressSpace.gmem, assumed_align: int | None = None, use_32bit_stride: bool = False, ) → _FakeTensor
```

Create a fake tensor descriptor with a compact layout derived from shape.

This is the usual builder for `cute.compile(...)` when the logical
tensor is compact and you want the runtime stride tuple to be derived
automatically from `shape` and `stride_order`. Each entry in
`shape` may be a static Python `int` or a dynamic
`SymInt`. Dynamic dimensions become
runtime-bound scalar parameters on the compiled callable.

**Parameters:**

- **dtype** (*Type*[*Numeric*]) – Data type of the tensor elements.
- **shape** (*tuple*[*int* | *SymInt*, *...*]) – Tensor extents in elements, one per mode. Each entry may be
  static (`int`) or dynamic (`SymInt`).
- **stride\_order** (*tuple*[*int*, *...*], *optional*) – Permutation describing which mode is fastest-changing.
  `0` means the innermost / stride-1 mode, `len(shape)-1` the
  slowest-changing mode. If omitted, the default is left-to-right
  order `(0, 1, ..., n-1)`.
- **memspace** (*AddressSpace*, *optional*) – Memory space where the fake tensor resides. Defaults to AddressSpace.gmem.
- **assumed\_align** (*int*, *optional*) – Assumed byte alignment of the base pointer. If
  `None`, defaults to one element width in bytes (and at least 1).
- **use\_32bit\_stride** (*bool*, *optional*) – Use 32-bit symbolic strides instead of 64-bit
  ones for dynamic layouts. This only affects dynamically-derived
  stride entries and is useful when the compact layout provably fits
  in int32.

**Returns:**

An instance of a fake tensor with the given properties and compact layout.

**Return type:**

[\_FakeTensor](#cutlass.cute.runtime._FakeTensor)

Use [`make_fake_tensor()`](#cutlass.cute.runtime.make_fake_tensor) instead when the logical layout is
non-compact or when you need to spell the stride tuple explicitly.

**Examples:**

```python
@cute.jit
def foo(x: cute.Tensor):
    ...

x = make_fake_compact_tensor(
    cutlass.Float32, (100, cute.sym_int32(divisibility=8)), stride_order=(1, 0)
)

# Compiled function will take a tensor with the type:
#   tensor<ptr<f32, generic> o (100,?{div=8}):(?{i32 div=8},1)>
compiled_foo = cute.compile(foo, x)

# Default stride order is left-to-right order (0, 1, ..., n-1)
y = make_fake_compact_tensor(cutlass.Float32, (8, 3, 2))
assert y.stride == (1, 8, 24)
```

<a id="cutlass.cute.runtime.make_fake_tensor"></a>
### `cutlass.cute.runtime.make_fake_tensor`

```python
cutlass.cute.runtime.make_fake_tensor( dtype: Type[cutlass.cute.typing.Numeric], shape: tuple[int | cutlass.cute.typing.SymInt, ...], stride: tuple[int | cutlass.cute.typing.SymInt, ...], *, memspace: cutlass.cute.typing.AddressSpace = cutlass.cute.typing.AddressSpace.gmem, assumed_align: int | None = None, ) → _FakeTensor
```

Create a fake tensor descriptor with an explicit layout.

Use this builder for `cute.compile(...)` when the logical tensor
layout is not compact, when you already know the exact stride tuple,
or when you want fake-tensor layout to match an external contract
exactly. `shape` and `stride` are both expressed in elements, not
bytes.

**Parameters:**

- **dtype** (*Type*[*Numeric*]) – Data type of the tensor elements.
- **shape** (*tuple*[*int* | *SymInt*, *...*]) – Tensor extents in elements, one per mode. Each entry may be
  static (`int`) or dynamic (`SymInt`).
  Dynamic dimensions become runtime-bound scalar parameters on the
  compiled callable.
- **stride** (*tuple*[*int* | *SymInt*, *...*]) – Explicit stride tuple in elements. Must have the same rank
  as `shape`. Each entry may be static (`int`) or dynamic
  (`SymInt`).
- **memspace** (*AddressSpace*, *optional*) – Memory space where the fake tensor resides. Defaults to AddressSpace.gmem.
- **assumed\_align** (*int*, *optional*) – Assumed byte alignment of the base pointer. If
  `None`, defaults to one element width in bytes (and at least 1).

**Returns:**

An instance of a fake tensor with the given properties.

**Return type:**

[\_FakeTensor](#cutlass.cute.runtime._FakeTensor)

If the same runtime symbolic quantity appears in multiple positions,
reuse the same `SymInt` object at every
occurrence. Different `SymInt` objects are treated as distinct
runtime parameters even if they share the same `symbol` string.

Use [`make_fake_compact_tensor()`](#cutlass.cute.runtime.make_fake_compact_tensor) instead when the layout is
compact and you want the stride tuple inferred from `shape` and a
mode order.

**Examples:**

```python
@cute.jit
def foo(x: cute.Tensor):
    ...

sym_m = cute.sym_int64(symbol="M")
sym_ld = cute.sym_int64(divisibility=16, symbol="LD")

# Row-major logical layout: contiguous K dimension, explicit leading dim.
x = make_fake_tensor(
    cutlass.Float16,
    shape=(sym_m, 128),
    stride=(sym_ld, 1),
)

compiled_foo = cute.compile(foo, x)
```

<a id="cutlass.cute.runtime.from_dlpack"></a>
### `cutlass.cute.runtime.from_dlpack`

```python
cutlass.cute.runtime.from_dlpack( tensor_dlpack: object, assumed_align: int | None = None, use_32bit_stride: bool = False, *, enable_tvm_ffi: bool = False, force_tf32: bool = False, ) → cutlass.cute.typing.Tensor
```

Convert from tensor object supporting \_\_dlpack\_\_() to a CuTe Tensor.

**Parameters:**

- **tensor\_dlpack** (*object*) – Tensor object that supports the DLPack protocol
- **assumed\_align** (*int*, *optional*) – Assumed alignment of the tensor (bytes), defaults to None,
  if None, will use the element size bytes as the assumed alignment.
- **use\_32bit\_stride** (*bool*, *optional*) – Whether to use 32-bit stride, defaults to False. When True, the dynamic
  stride bitwidth will be set to 32 for small problem size (cosize(layout) <= Int32\_max) for better performance.
  This is only applied when the dimension is dynamic.
- **enable\_tvm\_ffi** (*bool*, *optional*) – Whether to enable TVM-FFI, defaults to False. When True, the tensor will be converted to
  a TVM-FFI function compatible tensor.
- **force\_tf32** (*bool*, *optional*) – Whether to force the element type to TFloat32 if the element type is Float32.

**Returns:**

A CuTe Tensor object

**Return type:**

Tensor

For packed subbyte torch dtypes such as `torch.float4_e2m1fn_x2`,
`from_dlpack` returns the logical element layout expected by CuTe instead
of the packed storage layout. For example, a torch tensor with shape
`(128, 128)` and dtype `torch.float4_e2m1fn_x2` is exposed as a logical
FP4 tensor with shape `(128, 256)`.

**Examples:**

```python
import torch
from cutlass.cute.runtime import from_dlpack
x = torch.randn(100, 100)
y = from_dlpack(x)
y.shape
# (100, 100)
type(y)
# <class 'cutlass.cute.Tensor'>
```

<a id="cutlass.cute.runtime.make_ptr"></a>
### `cutlass.cute.runtime.make_ptr`

```python
cutlass.cute.runtime.make_ptr( dtype: Type[cutlass.cute.typing.Numeric], value: int | _Pointer, mem_space: cutlass.cute.typing.AddressSpace = cutlass.cute.typing.AddressSpace.generic, assumed_align: int | None = None, ) → cutlass.cute.typing.Pointer
```

Create a pointer from a memory address

**Parameters:**

- **dtype** (*Type*[*Numeric*]) – Data type of the pointer elements
- **value** (*Union*[*int*, *ctypes.\_Pointer*]) – Memory address as integer or ctypes pointer
- **mem\_space** (*AddressSpace*, *optional*) – Memory address space, defaults to AddressSpace.generic
- **align\_bytes** (*int*, *optional*) – Alignment in bytes, defaults to None

**Returns:**

A pointer object

**Return type:**

Pointer

```python
import numpy as np
import ctypes

from cutlass import Float32
from cutlass.cute.runtime import make_ptr

# Create a numpy array
a = np.random.randn(16, 32).astype(np.float32)

# Get pointer address as integer
ptr_address = a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

# Create pointer from address
y = make_ptr(cutlass.Float32, ptr_address)

# Check properties
print(y.element_type)
print(type(y))  # <class 'cutlass.cute.Pointer'>
```

<a id="cutlass.cute.runtime.nullptr"></a>
### `cutlass.cute.runtime.nullptr`

```python
cutlass.cute.runtime.nullptr( dtype: Type[cutlass.cute.typing.Numeric], mem_space: cutlass.cute.typing.AddressSpace = cutlass.cute.typing.AddressSpace.generic, assumed_align: int | None = None, ) → cutlass.cute.typing.Pointer
```

Create a null pointer which is useful for compilation

**Parameters:**

- **dtype** (*Type*[*Numeric*]) – Data type of the pointer elements
- **mem\_space** (*AddressSpace*, *optional*) – Memory address space, defaults to AddressSpace.generic

**Returns:**

A null pointer object

**Return type:**

Pointer

<a id="cutlass.cute.runtime.TensorAdapter"></a>
### `cutlass.cute.runtime.TensorAdapter`

```python
class cutlass.cute.runtime.TensorAdapter(arg: object)
```

Bases: `object`

Convert a DLPack protocol supported tensor/array to a cute tensor.

<a id="cutlass.cute.runtime.TensorAdapter.__init__"></a>
#### `cutlass.cute.runtime.TensorAdapter.__init__`

```python
__init__(arg: object) → None
```
