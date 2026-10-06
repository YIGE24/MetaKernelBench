<!-- source: https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute.html -->

<a id="module-cutlass.cute"></a>
# cutlass.cute

<a id="cutlass.cute.Swizzle"></a>
### `cutlass.cute.Swizzle`

```python
class cutlass.cute.Swizzle(*args: Any, **kwargs: Any)
```

Bases: `Value`

Swizzle is a transformation that permutes the elements of a layout.

Swizzles are used to rearrange data elements to improve memory access patterns
and computational efficiency.

Swizzle is defined by three parameters:
- MBase: The number of least-significant bits to keep constant
- BBits: The number of bits in the mask
- SShift: The distance to shift the mask

The mask is applied to the least-significant bits of the layout.

```console
0bxxxxxxxxxxxxxxxYYYxxxxxxxZZZxxxx
                              ^--^ MBase is the number of least-sig bits to keep constant
                 ^-^       ^-^     BBits is the number of bits in the mask
                   ^---------^     SShift is the distance to shift the YYY mask
                                      (pos shifts YYY to the right, neg shifts YYY to the left)

e.g. Given
0bxxxxxxxxxxxxxxxxYYxxxxxxxxxZZxxx

the result is
0bxxxxxxxxxxxxxxxxYYxxxxxxxxxAAxxx where AA = ZZ `xor` YY
```

<a id="cutlass.cute.struct"></a>
### `cutlass.cute.struct`

```python
class cutlass.cute.struct(cls: type)
```

Bases: `object`

Decorator to abstract C structure in Python DSL.

**Usage:**

```python
# Supports base_dsl scalar int/float elements, array and nested struct:
@cute.struct
class complex:
    real : cutlass.Float32
    imag : cutlass.Float32

@cute.struct
class StorageA:
    mbarA : cute.struct.MemRange[cutlass.Int64, stage]
    compA : complex
    intA : cutlass.Int16

# Supports alignment for its elements:
@cute.struct
class StorageB:
    a: cute.struct.Align[
        cute.struct.MemRange[cutlass.Float32, size_a], 1024
    ]
    b: cute.struct.Align[
        cute.struct.MemRange[cutlass.Float32, size_b], 1024
    ]
    x: cute.struct.Align[cutlass.Int32, 16]
    compA: cute.struct.Align[complex, 16]

# Statically get size and alignment:
size = StorageB.__sizeof__()
align = StorageB.__alignof__()

# Allocate and referencing elements:
storage = allocator.allocate(StorageB)

storage.a[0] ...
storage.x.ptr ...
storage.compA.real.ptr ...
```

**Parameters:**

**cls** – The struct class with annotations.

**Returns:**

The decorated struct class.

<a id="cutlass.cute.struct._MemRangeMeta"></a>
#### `cutlass.cute.struct._MemRangeMeta`

```python
class _MemRangeMeta( name: str, bases: tuple[type, ...], dct: Dict[str, Any], )
```

Bases: `type`

A metaclass for creating MemRange classes.

This metaclass is used to dynamically create MemRange classes with specific
data types and sizes.

**Variables:**

- **\_dtype** – The data type of the MemRange.
- **\_size** – The size of the MemRange.

<a id="cutlass.cute.struct._MemRangeMeta._dtype"></a>
##### `cutlass.cute.struct._MemRangeMeta._dtype`

```python
_dtype: Type[cutlass.cute.typing.Numeric] | None = None
```

<a id="cutlass.cute.struct._MemRangeMeta._size"></a>
##### `cutlass.cute.struct._MemRangeMeta._size`

```python
_size: int | None = None
```

<a id="cutlass.cute.struct._MemRangeMeta.size"></a>
##### `cutlass.cute.struct._MemRangeMeta.size`

```python
property size: int | None
```

<a id="cutlass.cute.struct._MemRangeMeta.elem_width"></a>
##### `cutlass.cute.struct._MemRangeMeta.elem_width`

```python
property elem_width: int
```

<a id="cutlass.cute.struct._MemRangeMeta.size_in_bytes"></a>
##### `cutlass.cute.struct._MemRangeMeta.size_in_bytes`

```python
property size_in_bytes: int
```

<a id="cutlass.cute.struct.MemRange"></a>
#### `cutlass.cute.struct.MemRange`

```python
class MemRange
```

Bases: `object`

Defines a range of memory by MemRange[T, size].

<a id="cutlass.cute.struct.MemRange._dtype"></a>
##### `cutlass.cute.struct.MemRange._dtype`

```python
_dtype: Type[cutlass.cute.typing.Numeric] | None = None
```

<a id="cutlass.cute.struct.MemRange._size"></a>
##### `cutlass.cute.struct.MemRange._size`

```python
_size: int | None = None
```

<a id="cutlass.cute.struct._MemRangeData"></a>
#### `cutlass.cute.struct._MemRangeData`

```python
class _MemRangeData( dtype: Type[cutlass.cute.typing.Numeric] | None, size: int | None, base: cutlass.cute.typing.Pointer | None, )
```

Bases: `object`

Represents a range of memory.

**Parameters:**

- **dtype** – The data type.
- **size** – The size of the memory range in bytes.
- **base** – The base address of the memory range.

<a id="cutlass.cute.struct._MemRangeData.__init__"></a>
##### `cutlass.cute.struct._MemRangeData.__init__`

```python
__init__( dtype: Type[cutlass.cute.typing.Numeric] | None, size: int | None, base: cutlass.cute.typing.Pointer | None, ) → None
```

Initializes a new memory range.

**Parameters:**

- **dtype** – The data type.
- **size** – Size of the memory range in bytes. A size of **0** is accepted, but in that
  case the range can only be used for its address (e.g. as a partition marker).
- **base** – The base address of the memory range.

<a id="cutlass.cute.struct._MemRangeData.data_ptr"></a>
##### `cutlass.cute.struct._MemRangeData.data_ptr`

```python
data_ptr( *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Pointer
```

Returns start pointer to the data in this memory range.

**Returns:**

A pointer to the start of the memory range.

**Raises:**

**AssertionError** – If the size of the memory range is negative.

<a id="cutlass.cute.struct._MemRangeData.get_tensor"></a>
##### `cutlass.cute.struct._MemRangeData.get_tensor`

```python
get_tensor( layout: cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout, swizzle: cutlass._mlir.ir.register_value_caster | None = None, dtype: Type[cutlass.cute.typing.Numeric] | None = None, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Tensor
```

Creates a tensor from the memory range.

**Parameters:**

- **layout** – The layout of the tensor.
- **swizzle** – Optional swizzle pattern.
- **dtype** – Optional data type; defaults to the memory range’s data type if not specified.

**Returns:**

A tensor representing the memory range.

**Raises:**

- **TypeError** – If the layout is incompatible with the swizzle.
- **AssertionError** – If the size of the memory range is not greater than zero.

<a id="cutlass.cute.struct._AlignMeta"></a>
#### `cutlass.cute.struct._AlignMeta`

```python
class _AlignMeta( name: str, bases: tuple[type, ...], dct: Dict[str, Any], )
```

Bases: `type`

Aligns the given object by setting its alignment attribute.

**Parameters:**

- **v** – The object to align. Must be a struct, MemRange, or a scalar type.
- **align** – The alignment value to set.

**Raises:**

**TypeError** – If the object is not a struct, MemRange, or a scalar type.

**Variables:**

- **\_dtype** – The data type to be aligned.
- **\_align** – The alignment of the data type.

<a id="cutlass.cute.struct._AlignMeta._dtype"></a>
##### `cutlass.cute.struct._AlignMeta._dtype`

```python
_dtype: Any | None = None
```

<a id="cutlass.cute.struct._AlignMeta._align"></a>
##### `cutlass.cute.struct._AlignMeta._align`

```python
_align: int | None = None
```

<a id="cutlass.cute.struct._AlignMeta.dtype"></a>
##### `cutlass.cute.struct._AlignMeta.dtype`

```python
property dtype: Any | None
```

<a id="cutlass.cute.struct._AlignMeta.align"></a>
##### `cutlass.cute.struct._AlignMeta.align`

```python
property align: int | None
```

<a id="cutlass.cute.struct.Align"></a>
#### `cutlass.cute.struct.Align`

```python
class Align
```

Bases: `object`

Aligns the given type by Align[T, alignment].

<a id="cutlass.cute.struct.Align._dtype"></a>
##### `cutlass.cute.struct.Align._dtype`

```python
_dtype: Any | None = None
```

<a id="cutlass.cute.struct.Align._align"></a>
##### `cutlass.cute.struct.Align._align`

```python
_align: int | None = None
```

<a id="cutlass.cute.struct._ScalarData"></a>
#### `cutlass.cute.struct._ScalarData`

```python
class _ScalarData(*args: Any, **kwargs: Any)
```

Bases: `register_value_caster`

Represents a scalar value at a given pointer location in memory.

This class provides utility methods to get a scalar pointer.
It wraps a pointer to a scalar element and enables element-wise memory operations.

**Variables:**

**\_ptr** – The underlying pointer to the scalar value.

<a id="cutlass.cute.struct._ScalarData.__init__"></a>
##### `cutlass.cute.struct._ScalarData.__init__`

```python
__init__( ptr: cutlass._mlir.ir.register_value_caster, ) → None
```

<a id="cutlass.cute.struct._ScalarData.to_llvm_ptr"></a>
##### `cutlass.cute.struct._ScalarData.to_llvm_ptr`

```python
to_llvm_ptr( *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass._mlir.ir.Value
```

<a id="cutlass.cute.struct._ScalarData.ptr"></a>
##### `cutlass.cute.struct._ScalarData.ptr`

```python
property ptr: cutlass.cute.typing.Pointer
```

Get the underlying pointer.

**Returns:**

The pointer to the scalar value.

**Return type:**

Pointer

<a id="cutlass.cute.struct._ScalarData.dtype"></a>
##### `cutlass.cute.struct._ScalarData.dtype`

```python
property dtype: Type[cutlass.cute.typing.Numeric]
```

Get the data type of the scalar value.

**Returns:**

The numeric data type of the underlying pointer.

**Return type:**

Type[Numeric]

<a id="cutlass.cute.struct._ScalarData.value"></a>
##### `cutlass.cute.struct._ScalarData.value`

```python
property value: cutlass._mlir.ir.Value
```

Get the raw MLIR value of the underlying pointer.

Deprecated since version Using: `struct.scalar` as pointer is deprecated.
Use explicit `struct.scalar.ptr` for pointer instead.

**Returns:**

The MLIR value of the underlying pointer.

**Return type:**

ir.Value

<a id="cutlass.cute.struct._is_scalar_type"></a>
#### `cutlass.cute.struct._is_scalar_type`

```python
static _is_scalar_type(dtype: Any) → bool
```

Checks if the given type is a scalar numeric type.

**Parameters:**

**dtype** – The type to check.

**Returns:**

True if the type is a subclass of Numeric, False otherwise.

<a id="cutlass.cute.struct._install_dynamic_expression_protocol"></a>
#### `cutlass.cute.struct._install_dynamic_expression_protocol`

```python
static _install_dynamic_expression_protocol( cls: type, decorator: Any, ) → None
```

<a id="cutlass.cute.struct.__init__"></a>
#### `cutlass.cute.struct.__init__`

```python
__init__(cls: type) → None
```

Initializes a new struct decorator instance.

**Parameters:**

**cls** – The class representing the structured data type.

**Raises:**

**TypeError** – If the struct is empty.

<a id="cutlass.cute.struct.size_in_bytes"></a>
#### `cutlass.cute.struct.size_in_bytes`

```python
size_in_bytes() → int
```

Returns the size of the struct in bytes.

**Returns:**

The size of the struct.

<a id="cutlass.cute.struct.align_offset"></a>
#### `cutlass.cute.struct.align_offset`

```python
static align_offset(offset: Any, align: int) → Any
```

Return the round-up offset up to the next multiple of align.

<a id="cutlass.cute.E"></a>
### `cutlass.cute.E`

```python
cutlass.cute.E( mode: int | List[int], ) → ScaledBasis | int
```

Create a unit ScaledBasis element with the specified mode.

This function creates a ScaledBasis with value 1 and the given mode.
The mode represents the coordinate axis or dimension in the layout.

**Parameters:**

**mode** (*Union*[*int*, *List*[*int*]]) – The mode (dimension) for the basis element, either a single integer or a list of integers

**Returns:**

A ScaledBasis with value 1 and the specified mode

**Return type:**

[ScaledBasis](#cutlass.cute.ScaledBasis)

**Raises:**

**TypeError** – If mode is not an integer or a list

**Examples:**

```python
# Create a basis element for the first dimension (mode 0)
e0 = E(0)

# Create a basis element for the second dimension (mode 1)
e1 = E(1)

# Create a basis element for a hierarchical dimension
e_hier = E([0, 1])
```

<a id="cutlass.cute.get_divisibility"></a>
### `cutlass.cute.get_divisibility`

```python
cutlass.cute.get_divisibility(x: cutlass.cute.typing.Int) → int
```

<a id="cutlass.cute.is_static"></a>
### `cutlass.cute.is_static`

```python
cutlass.cute.is_static(x: object) → bool
```

Check if a value is statically known at compile time.

In CuTe, static values are those whose values are known at compile time,
as opposed to dynamic values which are only known at runtime.

This function checks if a value is static by recursively traversing its type hierarchy
and checking if all components are static.

Static values include:
- Python literals (bool, int, float, None)
- Static ScaledBasis objects
- Static ComposedLayout objects
- Static IR types
- Tuples containing only static values

Dynamic values include:
- Numeric objects (representing runtime values)
- Dynamic expressions
- Any tuple containing dynamic values

**Parameters:**

**x** (*Any*) – The value to check

**Returns:**

True if the value is static, False otherwise

**Return type:**

bool

**Raises:**

**TypeError** – If an unsupported type is provided

<a id="cutlass.cute.has_underscore"></a>
### `cutlass.cute.has_underscore`

```python
cutlass.cute.has_underscore(a: cutlass.cute.typing.XTuple) → bool
```

<a id="cutlass.cute.pretty_str"></a>
### `cutlass.cute.pretty_str`

```python
cutlass.cute.pretty_str(arg: object) → str
```

Constructs a concise readable pretty string.

<a id="cutlass.cute.printf"></a>
### `cutlass.cute.printf`

```python
cutlass.cute.printf( *args: Any, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, end: str = '\n', ) → None
```

Print one or more values with optional formatting.

This function provides printf-style formatted printing capabilities. It can print values directly
or format them using C-style format strings. The function supports printing various types including
layouts, numeric values, tensors, pointers, and other CuTe objects.

The function accepts either:
1. A list of values to print directly
2. A format string followed by values to format

**Parameters:**

- **args** (*Any*) – Variable length argument list containing either:
  - One or more values to print directly
  - A format string followed by values to format
- **loc** (*Optional*[*Location*]) – Source location information for debugging, defaults to None
- **ip** (*Optional*[*InsertionPoint*]) – Insertion point for code generation, defaults to None
- **end** (*Optional*[*str*]) – Suffix for the printed value, defaults to newline

**Raises:**

- **ValueError** – If no arguments are provided
- **TypeError** – If an unsupported argument type is passed

**Examples:**

Direct printing of values:

```python
a = cute.make_layout(shape=(10, 10), stride=(10, 1))
b = cutlass.Float32(1.234)
cute.printf(a, b)  # Prints values directly
```

Formatted printing:

```python
# Using format string with generic format specifiers
cute.printf("a={}, b={}", a, b)

# Using format string with C-style format specifiers
cute.printf("a={}, b=%.2f", a, b)
```

<a id="cutlass.cute.front"></a>
### `cutlass.cute.front`

```python
cutlass.cute.front( input: Any, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → Any
```

Recursively get the first element of input.

This function traverses a hierarchical structure (like a layout or tensor)
and returns the first element at the deepest level. It’s particularly useful
for accessing the first stride value in a layout to determine properties like
majorness.

**Parameters:**

- **input** (*Union*[*Tensor*, *Layout*, *Stride*]) – The hierarchical structure to traverse
- **loc** (*source location*, *optional*) – Source location where it’s called, defaults to None
- **ip** (*insertion pointer*, *optional*) – Insertion pointer for IR generation, defaults to None

**Returns:**

The first element at the deepest level of the input structure

**Return type:**

Union[int, float, bool, ir.Value]

<a id="cutlass.cute.is_major"></a>
### `cutlass.cute.is_major`

```python
cutlass.cute.is_major( mode: int | List[int], stride: cutlass.cute.typing.Stride, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → bool
```

Check whether a mode in stride is the major mode.

<a id="cutlass.cute.assume"></a>
### `cutlass.cute.assume`

```python
cutlass.cute.assume( src: Any, divby: int | None = None, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → Any
```

<a id="cutlass.cute.make_swizzle"></a>
### `cutlass.cute.make_swizzle`

```python
cutlass.cute.make_swizzle( b: int, m: int, s: int, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass._mlir.ir.register_value_caster
```

<a id="cutlass.cute.static"></a>
### `cutlass.cute.static`

```python
cutlass.cute.static( value: Any, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → Any
```

<a id="cutlass.cute.get_leaves"></a>
### `cutlass.cute.get_leaves`

```python
cutlass.cute.get_leaves( value: Any, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → Any
```

<a id="cutlass.cute.depth"></a>
### `cutlass.cute.depth`

```python
cutlass.cute.depth( a: cutlass.cute.typing.XTuple | cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout, ) → int
```

Returns the depth (nesting level) of a tuple, layout, or tensor.

The depth of a tuple is the maximum depth of its elements plus 1.
For an empty tuple, the depth is 1. For layouts and tensors, the depth
is determined by the depth of their shape. For non-tuple values (e.g., integers),
the depth is considered 0.

**Parameters:**

**a** (*Union*[*XTuple*, *Layout*, *ComposedLayout*, *Tensor*, *Any*]) – The object whose depth is to be determined

**Returns:**

The depth of the input object

**Return type:**

int

**Example:**

```python
depth(1)                # 0
depth((1, 2))           # 1
depth(((1, 2), (3, 4))) # 2
```

<a id="cutlass.cute.is_congruent"></a>
### `cutlass.cute.is_congruent`

```python
cutlass.cute.is_congruent( a: cutlass.cute.typing.XTuple | cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout | cutlass.cute.typing.Tensor, b: cutlass.cute.typing.XTuple | cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout | cutlass.cute.typing.Tensor, ) → bool
```

Returns whether a is congruent to b.

Congruence is an equivalence relation between hierarchical structures.

Two objects are congruent if:
\* They have the same rank, AND
\* They are both non-tuple values, OR
\* They are both tuples AND all corresponding elements are congruent.

Congruence requires type matching at each level – scalar values match with
scalar values, and tuples match with tuples of the same rank.

**Parameters:**

- **a** (*Union*[*XTuple*, *Layout*, *ComposedLayout*, *Tensor*]) – First object to compare
- **b** (*Union*[*XTuple*, *Layout*, *ComposedLayout*, *Tensor*]) – Second object to compare

**Returns:**

True if a and b are congruent, False otherwise

**Return type:**

bool

<a id="cutlass.cute.is_weakly_congruent"></a>
### `cutlass.cute.is_weakly_congruent`

```python
cutlass.cute.is_weakly_congruent( a: cutlass.cute.typing.XTuple | cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout | cutlass.cute.typing.Tensor, b: cutlass.cute.typing.XTuple | cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout | cutlass.cute.typing.Tensor, ) → bool
```

Returns whether a is weakly congruent to b.

Weak congruence is a partial order on hierarchical structures.

Object X is weakly congruent to object Y if:
\* X is a non-tuple value, OR
\* X and Y are both tuples of the same rank AND all corresponding elements are weakly congruent.

Weak congruence allows scalar values to match with tuples, making it useful
for determining whether an object has a hierarchical structure “up to” another.

**Parameters:**

- **a** (*Union*[*XTuple*, *Layout*, *ComposedLayout*, *Tensor*]) – First object to compare
- **b** (*Union*[*XTuple*, *Layout*, *ComposedLayout*, *Tensor*]) – Second object to compare

**Returns:**

True if a and b are weakly congruent, False otherwise

**Return type:**

bool

<a id="cutlass.cute.group_modes"></a>
### `cutlass.cute.group_modes`

```python
cutlass.cute.group_modes( input: cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout | cutlass.cute.typing.Tensor | cutlass.cute.typing.XTuple, begin: int, end: int | None = None, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout | cutlass.cute.typing.Tensor | cutlass.cute.typing.XTuple
```

Group modes of a hierarchical tuple or layout into a single mode.

This function groups a range of modes from the input object into a single mode,
creating a hierarchical structure. For tuples, it creates a nested tuple containing
the specified range of elements. For layouts and other CuTe objects, it creates
a hierarchical representation where the specified modes are grouped together.

**Parameters:**

- **input** (*Layout*, *ComposedLayout*, *tuple*, *Shape*, *Stride*, *etc.*) – Input object to group modes from (layout, tuple, etc.)
- **beg** (*int*) – Beginning index of the range to group (inclusive)
- **end** (*int*) – Ending index of the range to group (exclusive)
- **loc** (*optional*) – Source location for MLIR, defaults to None
- **ip** (*optional*) – Insertion point, defaults to None

**Returns:**

A new object with the specified modes grouped

**Return type:**

Same type as input with modified structure

**Examples:**

```python
# Group modes in a tuple
t = (2, 3, 4, 5)
grouped = group_modes(t, 1, 3)  # (2, (3, 4), 5)

# Group modes in a layout
layout = make_layout((2, 3, 4, 5))
grouped_layout = group_modes(layout, 1, 3)  # Layout with shape (2, (3, 4), 5)

# Group modes in a shape
shape = make_shape(2, 3, 4, 5)
grouped_shape = group_modes(shape, 0, 2)  # Shape ((2, 3), 4, 5)
```

<a id="cutlass.cute.slice_"></a>
### `cutlass.cute.slice_`

```python
cutlass.cute.slice_( src: cutlass.cute.typing.Layout | cutlass._mlir.ir.register_value_caster | cutlass.cute.typing.Tensor | cutlass.cute.typing.XTuple, coord: cutlass.cute.typing.Coord, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Layout | cutlass._mlir.ir.register_value_caster | cutlass.cute.typing.Tensor | cutlass.cute.typing.XTuple
```

Perform a slice operation on a source object using the given coordinate.

This function implements CuTe’s slicing operation which extracts a subset of elements
from a source object (tensor, layout, etc.) based on a coordinate pattern. The slice
operation preserves the structure of the source while selecting specific elements.

**Parameters:**

- **src** (*Union*[*Tensor*, *Layout*, *IntTuple*, *Value*]) – Source object to be sliced (tensor, layout, tuple, etc.)
- **coord** (*Coord*) – Coordinate pattern specifying which elements to select
- **loc** (*Optional*[*Location*]) – Source location information, defaults to None
- **ip** (*Optional*[*InsertionPoint*]) – Insertion point for IR generation, defaults to None

**Returns:**

A new object containing the sliced elements

**Return type:**

Union[Tensor, Layout, IntTuple, tuple]

**Raises:**

**ValueError** – If the coordinate pattern is incompatible with source

**Examples:**

```python
# Layout slicing
layout = make_layout((4,4))

# Select 1st index of first mode and keep all elements in second mode
sub_layout = slice_(layout, (1, None))
```
```python
# Basic tensor slicing
tensor = make_tensor(...)           # Create a 2D tensor

# Select 1st index of first mode and keep all elements in second mode
sliced = slice_(tensor, (1, None))
```
```python
# Select 2nd index of second mode and keep all elements in first mode
sliced = slice_(tensor, (None, 2))
```
> **Note**
>
> - None represents keeping all elements in that mode
> - Slicing preserves the layout/structure of the original object
> - Can be used for:
>   \* Extracting sub-tensors/sub-layouts
>   \* Creating views into data
>   \* Selecting specific patterns of elements

<a id="cutlass.cute.prepend"></a>
### `cutlass.cute.prepend`

```python
cutlass.cute.prepend( input: cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout | cutlass.cute.typing.XTuple, elem: Any, up_to_rank: int | None = None, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout | cutlass.cute.typing.XTuple
```

Extend input to rank up\_to\_rank by prepending elem in front of input.

This function extends the input object by prepending elements to reach a desired rank.
It supports various CuTe types including shapes, layouts, tensors etc.

**Parameters:**

- **input** (*Union*[*Shape*, *Stride*, *Coord*, *IntTuple*, *Tile*, *Layout*, *ComposedLayout*, *Tensor*]) – Source to be prepended to
- **elem** (*Union*[*Shape*, *Stride*, *Coord*, *IntTuple*, *Tile*, *Layout*]) – Element to prepend to input
- **up\_to\_rank** (*Union*[*None*, *int*], *optional*) – The target rank after extension, defaults to None
- **loc** (*Optional*[*Location*]) – Source location for MLIR, defaults to None
- **ip** (*Optional*[*InsertionPoint*]) – Insertion point, defaults to None

**Returns:**

The extended result with prepended elements

**Return type:**

Union[Shape, Stride, Coord, IntTuple, Tile, Layout, ComposedLayout, Tensor]

**Raises:**

- **ValueError** – If up\_to\_rank is less than input’s current rank
- **TypeError** – If input or elem has unsupported type

**Examples:**

```python
# Prepend to a Shape
shape = (4,4)
prepend(shape, 2)                   # Returns (2,4,4)

# Prepend to a Layout
layout = make_layout((8,8))
prepend(layout, make_layout((2,)))  # Returns (2,8,8):(1,1,8)

# Prepend with target rank
coord = (1,1)
prepend(coord, 0, up_to_rank=4)     # Returns (0,0,1,1)
```

<a id="cutlass.cute.append"></a>
### `cutlass.cute.append`

```python
cutlass.cute.append( input: cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout | cutlass.cute.typing.XTuple, elem: Any, up_to_rank: int | None = None, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout | cutlass.cute.typing.XTuple
```

Extend input to rank up\_to\_rank by appending elem to the end of input.

This function extends the input object by appending elements to reach a desired rank.
It supports various CuTe types including shapes, layouts, tensors etc.

**Parameters:**

- **input** (*Union*[*Shape*, *Stride*, *Coord*, *IntTuple*, *Tile*, *Layout*, *ComposedLayout*, *Tensor*]) – Source to be appended to
- **elem** (*Union*[*Shape*, *Stride*, *Coord*, *IntTuple*, *Tile*, *Layout*]) – Element to append to input
- **up\_to\_rank** (*Union*[*None*, *int*], *optional*) – The target rank after extension, defaults to None
- **loc** (*Optional*[*Location*]) – Source location for MLIR, defaults to None
- **ip** (*Optional*[*InsertionPoint*]) – Insertion point, defaults to None

**Returns:**

The extended result with appended elements

**Return type:**

Union[Shape, Stride, Coord, IntTuple, Tile, Layout, ComposedLayout, Tensor]

**Raises:**

- **ValueError** – If up\_to\_rank is less than input’s current rank
- **TypeError** – If input or elem has unsupported type

**Examples:**

```python
# Append to a Shape
shape = (4,4)
append(shape, 2)                   # Returns (4,4,2)

# Append to a Layout
layout = make_layout((8,8))
append(layout, make_layout((2,)))  # Returns (8,8,2):(1,8,1)

# Append with target rank
coord = (1,1)
append(coord, 0, up_to_rank=4)     # Returns (1,1,0,0)
```
> **Note**
>
> - The function preserves the structure of the input while extending it
> - Can be used to extend tensors, layouts, shapes and other CuTe types
> - When up\_to\_rank is specified, fills remaining positions with elem
> - Useful for tensor reshaping and layout transformations

<a id="cutlass.cute.prepend_ones"></a>
### `cutlass.cute.prepend_ones`

```python
cutlass.cute.prepend_ones( t: cutlass.cute.typing.Tensor, up_to_rank: int | None = None, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Tensor
```

<a id="cutlass.cute.append_ones"></a>
### `cutlass.cute.append_ones`

```python
cutlass.cute.append_ones( t: cutlass.cute.typing.Layout | cutlass.cute.typing.Tensor, up_to_rank: int | None = None, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Layout | cutlass.cute.typing.Tensor
```

<a id="cutlass.cute.repeat_as_tuple"></a>
### `cutlass.cute.repeat_as_tuple`

```python
cutlass.cute.repeat_as_tuple(x: Any, n: int) → tuple
```

Creates a tuple with x repeated n times.

This function creates a tuple by repeating the input value x n times.

**Parameters:**

- **x** (*Any*) – The value to repeat
- **n** (*int*) – Number of times to repeat x

**Returns:**

A tuple containing x repeated n times

**Return type:**

tuple

**Examples:**

```python
repeat_as_tuple(1, 1)     # Returns (1,)
repeat_as_tuple(1, 3)     # Returns (1, 1, 1)
repeat_as_tuple(None, 4)  # Returns (None, None, None, None)
```

<a id="cutlass.cute.repeat"></a>
### `cutlass.cute.repeat`

```python
cutlass.cute.repeat(x: Any, n: int) → Any
```

Creates an object by repeating x n times.

This function creates an object by repeating the input value x n times.
If n=1, returns x directly, otherwise returns a tuple of x repeated n times.

**Parameters:**

- **x** (*Any*) – The value to repeat
- **n** (*int*) – Number of times to repeat x

**Returns:**

x if n=1, otherwise a tuple containing x repeated n times

**Return type:**

Union[Any, tuple]

**Raises:**

**ValueError** – If n is less than 1

**Examples:**

```python
repeat(1, 1)     # Returns 1
repeat(1, 3)     # Returns (1, 1, 1)
repeat(None, 4)  # Returns (None, None, None, None)
```

<a id="cutlass.cute.repeat_like"></a>
### `cutlass.cute.repeat_like`

```python
cutlass.cute.repeat_like(x: Any, target: Any) → Any
```

Creates an object congruent to target and filled with x.

This function recursively creates a nested tuple structure that matches the structure
of the target, with each leaf node filled with the value x.

**Parameters:**

- **x** (*Any*) – The value to fill the resulting structure with
- **target** (*Union*[*tuple*, *Any*]) – The structure to mimic

**Returns:**

A structure matching target but filled with x

**Return type:**

Union[tuple, Any]

**Examples:**

```python
repeat_like(0, (1, 2, 3))      # Returns (0, 0, 0)
repeat_like(1, ((1, 2), 3))    # Returns ((1, 1), 1)
repeat_like(2, 5)              # Returns 2
```

<a id="cutlass.cute.flatten"></a>
### `cutlass.cute.flatten`

```python
cutlass.cute.flatten( a: cutlass.cute.typing.Layout | cutlass.cute.typing.Tensor | cutlass.cute.typing.XTuple, ) → cutlass.cute.typing.Layout | cutlass.cute.typing.Tensor | cutlass.cute.typing.XTuple
```

Flattens a CuTe data structure into a simpler form.

For tuples, this function flattens the structure into a single-level tuple.
For layouts, it returns a new layout with flattened shape and stride.
For tensors, it returns a new tensor with flattened layout.
For other types, it returns the input unchanged.

**Parameters:**

**a** (*Union*[*IntTuple*, *Coord*, *Shape*, *Stride*, *Layout*, *Tensor*]) – The structure to flatten

**Returns:**

The flattened structure

**Return type:**

Union[tuple, Any]

**Examples:**

```python
flatten((1, 2, 3))                      # Returns (1, 2, 3)
flatten(((1, 2), (3, 4)))               # Returns (1, 2, 3, 4)
flatten(5)                              # Returns 5
flatten(Layout(shape, stride))          # Returns Layout(flatten(shape), flatten(stride))
flatten(Tensor(layout))                 # Returns Tensor(flatten(layout))
```

<a id="cutlass.cute.filter_zeros"></a>
### `cutlass.cute.filter_zeros`

```python
cutlass.cute.filter_zeros( input: cutlass.cute.typing.Layout | cutlass.cute.typing.Tensor, *, target_profile: cutlass.cute.typing.Stride | None = None, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Layout | cutlass.cute.typing.Tensor
```

Filter out zeros from a layout or tensor.

This function removes zero-stride dimensions from a layout or tensor.
Refer to NVIDIA/cutlass
for more layout algebra operations.

**Parameters:**

- **input** (*Layout* *or* *Tensor*) – The input layout or tensor to filter
- **target\_profile** (*Stride*, *optional*) – Target stride profile for the filtered result, defaults to None
- **loc** (*optional*) – Source location for MLIR, defaults to None
- **ip** (*optional*) – Insertion point, defaults to None

**Returns:**

The filtered layout or tensor with zeros removed

**Return type:**

Layout or Tensor

**Raises:**

**TypeError** – If input is not a Layout or Tensor

<a id="cutlass.cute.filter"></a>
### `cutlass.cute.filter`

```python
cutlass.cute.filter( input: cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout | cutlass.cute.typing.Tensor, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout | cutlass.cute.typing.Tensor
```

Filter a layout or tensor.

This function filters a layout or tensor according to CuTe’s filtering rules.

**Parameters:**

- **input** (*Layout* *or* *Tensor*) – The input layout or tensor to filter
- **loc** (*optional*) – Source location for MLIR, defaults to None
- **ip** (*optional*) – Insertion point, defaults to None

**Returns:**

The filtered layout or tensor

**Return type:**

Layout or Tensor

**Raises:**

**TypeError** – If input is not a Layout or Tensor

<a id="cutlass.cute.shape_div"></a>
### `cutlass.cute.shape_div`

```python
cutlass.cute.shape_div( lhs: cutlass.cute.typing.Shape, rhs: cutlass.cute.typing.Shape, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Shape
```

Perform element-wise division of shapes.

This function performs element-wise division between two shapes.

**Parameters:**

- **lhs** (*Shape*) – Left-hand side shape
- **rhs** (*Shape*) – Right-hand side shape
- **loc** (*optional*) – Source location for MLIR, defaults to None
- **ip** (*optional*) – Insertion point, defaults to None

**Returns:**

The result of element-wise division

**Return type:**

Shape

<a id="cutlass.cute.ceil_div"></a>
### `cutlass.cute.ceil_div`

```python
cutlass.cute.ceil_div( input: cutlass.cute.typing.Shape, tiler: cutlass.cute.typing.Tiler, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Shape
```

Compute the ceiling division of a target shape by a tiling specification.

This function computes the number of tiles required to cover the target domain.
It is equivalent to the second mode of zipped\_divide(input, tiler).

**Parameters:**

- **input** (*Shape*) – A tuple of integers representing the dimensions of the target domain.
- **tiler** (*Union*[*Layout*, *Shape*, *Tile*]) – The tiling specification.
- **loc** (*optional*) – Optional location information for IR diagnostics.
- **ip** (*optional*) – Optional instruction pointer or context for underlying IR functions.

**Returns:**

A tuple of integers representing the number of tiles required along each dimension,
i.e. the result of the ceiling division of the input dimensions by the tiler dimensions.

**Return type:**

Shape

Example:

```python
import cutlass.cute as cute
@cute.jit
def foo():
    input = (10, 6)
    tiler = (3, 4)
    result = cute.ceil_div(input, tiler)
    print(result)  # Outputs: (4, 2)
```

<a id="cutlass.cute.round_up"></a>
### `cutlass.cute.round_up`

```python
cutlass.cute.round_up( a: cutlass.cute.typing.IntTuple, b: cutlass.cute.typing.IntTuple, ) → cutlass.cute.typing.IntTuple
```

Rounds up elements of a using elements of b.

<a id="cutlass.cute.make_layout"></a>
### `cutlass.cute.make_layout`

```python
cutlass.cute.make_layout( shape: cutlass.cute.typing.Shape | Iterable[cutlass.cute.typing.Layout], *, stride: cutlass.cute.typing.Stride | None = None, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Layout
```

Create a CuTe Layout object from shape and optional stride information.

A Layout in CuTe represents the mapping between logical and physical coordinates of a tensor.
This function creates a Layout object that defines how tensor elements are arranged in memory.

As an alternative to a shape, an iterable of `Layout` objects may be
passed, in which case each layout becomes a separate mode of the result (the
`stride` argument is ignored). This mirrors CuTe’s variadic
`make_layout(layoutA, layoutB, ...)`.

**Parameters:**

- **shape** (*Union*[*Shape*, *Iterable*[*Layout*]]) – Shape of the layout defining the size of each mode, or an iterable of Layout objects to concatenate (each becomes a mode)
- **stride** (*Union*[*Stride*, *None*]) – Optional stride values for each mode, defaults to None (ignored when shape is an iterable of layouts)
- **loc** (*Optional*[*Location*]) – Source location information, defaults to None
- **ip** (*Optional*[*InsertionPoint*]) – Insertion point for IR generation, defaults to None

**Returns:**

A new Layout object with the specified shape and stride

**Return type:**

Layout

**Examples:**

```python
# Create a 2D compact left-most layout with shape (4,4)
layout = make_layout((4,4))                     # compact left-most layout

# Create a left-most layout with custom strides
layout = make_layout((4,4), stride=(1,4))       # left-most layout with strides (1,4)

# Create a layout for a 3D tensor
layout = make_layout((32,16,8))                 # left-most layout

# Create a layout with custom strides
layout = make_layout((2,2,2), stride=(4,1,2))   # layout with strides (4,1,2)

# Concatenate layouts: each becomes a mode of the result
mode0 = make_layout(64, stride=1)
mode1 = make_layout(128, stride=64)
combined = make_layout([mode0, mode1])          # (64,128):(1,64)
```
> **Note**
>
> - If stride is not provided, a default compact left-most stride is computed based on the shape
> - The resulting layout maps logical coordinates to physical memory locations
> - The layout object can be used for tensor creation and memory access patterns
> - Strides can be used to implement:
>   \* Row-major vs column-major layouts
>   \* Padding and alignment
>   \* Blocked/tiled memory arrangements
>   \* Interleaved data formats
> - Stride is keyword only argument to improve readability, e.g.
>   \* make\_layout((3,4), (1,4)) can be confusing with make\_layout(((3,4), (1,4)))
>   \* make\_layout((3,4), stride=(1,4)) is more readable
> - When passing an iterable of layouts, each layout becomes a separate mode

<a id="cutlass.cute.make_identity_layout"></a>
### `cutlass.cute.make_identity_layout`

```python
cutlass.cute.make_identity_layout( shape: cutlass.cute.typing.Shape, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Layout
```

Create an identity layout with the given shape.

An identity layout maps logical coordinates directly to themselves without any transformation.
This is equivalent to a layout with stride (1@0,1@1,…,1@(N-1)).

**Parameters:**

- **shape** (*Shape*) – The shape of the layout
- **loc** (*Optional*[*Location*]) – Source location information, defaults to None
- **ip** (*Optional*[*InsertionPoint*]) – Insertion point for IR generation, defaults to None

**Returns:**

A new identity Layout object with the specified shape

**Return type:**

Layout

**Examples:**

```python
# Create a 2D identity layout with shape (4,4)
layout = make_identity_layout((4,4))     # stride=(1@0,1@1)

# Create a 3D identity layout
layout = make_identity_layout((32,16,8)) # stride=(1@0,1@1,1@2)
```
> **Note**
>
> - An identity layout is a special case where each coordinate maps to itself
> - Useful for direct coordinate mapping without any transformation

<a id="cutlass.cute.make_ordered_layout"></a>
### `cutlass.cute.make_ordered_layout`

```python
cutlass.cute.make_ordered_layout( shape: cutlass.cute.typing.Shape, order: cutlass.cute.typing.Shape, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Layout
```

Create a layout with a specific ordering of dimensions.

This function creates a layout where the dimensions are ordered according to the
specified order parameter, allowing for custom dimension ordering in the layout.

**Parameters:**

- **shape** (*Shape*) – The shape of the layout
- **order** (*Shape*) – The ordering of dimensions
- **loc** (*Optional*[*Location*]) – Source location information, defaults to None
- **ip** (*Optional*[*InsertionPoint*]) – Insertion point for IR generation, defaults to None

**Returns:**

A new Layout object with the specified shape and dimension ordering

**Return type:**

Layout

**Examples:**

```python
# Create a row-major layout
layout = make_ordered_layout((4,4), order=(1,0))

# Create a column-major layout
layout = make_ordered_layout((4,4), order=(0,1))         # stride=(1,4)

# Create a layout with custom dimension ordering for a 3D tensor
layout = make_ordered_layout((32,16,8), order=(2,0,1))   # stride=(128,1,16)
```
> **Note**
>
> - The order parameter specifies the ordering of dimensions from fastest-varying to slowest-varying
> - For a 2D tensor, (0,1) creates a column-major layout, while (1,0) creates a row-major layout
> - The length of order must match the rank of the shape

<a id="cutlass.cute.make_layout_like"></a>
### `cutlass.cute.make_layout_like`

```python
cutlass.cute.make_layout_like( input: cutlass.cute.typing.Layout | cutlass.cute.typing.Tensor, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Layout
```

<a id="cutlass.cute.make_composed_layout"></a>
### `cutlass.cute.make_composed_layout`

```python
cutlass.cute.make_composed_layout( inner: Any, offset: cutlass.cute.typing.IntTuple, outer: cutlass.cute.typing.Layout, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.ComposedLayout
```

Create a composed layout by composing an inner transformation with an outer layout.

A composed layout applies a sequence of transformations
to coordinates. The composition is defined as (inner ∘ offset ∘ outer), where the operations
are applied from right to left.

**Parameters:**

- **inner** (*Union*[*Layout*, [*Swizzle*](#cutlass.cute.Swizzle)]) – The inner transformation (can be a Layout or Swizzle)
- **offset** (*IntTuple*) – An integral offset applied between transformations
- **outer** (*Layout*) – The outer (right-most) layout that is applied first
- **loc** (*Optional*[*Location*]) – Source location information, defaults to None
- **ip** (*Optional*[*InsertionPoint*]) – Insertion point for IR generation, defaults to None

**Returns:**

A new ComposedLayout representing the composition

**Return type:**

ComposedLayout

**Examples:**

```python
# Create a basic layout
inner = make_layout(...)
outer = make_layout((4,4), stride=(E(0), E(1)))

# Create a composed layout with an offset
composed = make_composed_layout(inner, (2,0), outer)
```
> **Note**
>
> - The composition applies transformations in the order: outer → offset → inner
> - The stride divisibility condition must be satisfied for valid composition
> - Certain compositions (like Swizzle with scaled basis) are invalid and will raise errors
> - Composed layouts inherit many properties from the outer layout

<a id="cutlass.cute.size_in_bytes"></a>
### `cutlass.cute.size_in_bytes`

```python
cutlass.cute.size_in_bytes( dtype: Type[cutlass.cute.typing.Numeric], layout: cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout | None, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Int
```

Calculate the size in bytes based on its data type and layout. The result is rounded up to the nearest byte.

Supports both regular Numeric types.
:param dtype: The DSL numeric data type
:type dtype: Union[Type[Numeric]]
:param layout: The layout of the elements. If None, the function returns 0
:type layout: Layout, optional
:param loc: Location information for diagnostics, defaults to None
:type loc: optional
:param ip: Instruction pointer for diagnostics, defaults to None
:type ip: optional
:return: The total size in bytes. Returns 0 if the layout is None
:rtype: int

<a id="cutlass.cute.coalesce"></a>
### `cutlass.cute.coalesce`

```python
cutlass.cute.coalesce( input: cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout | cutlass.cute.typing.Tensor, *, target_profile: cutlass.cute.typing.Coord | None = None, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout | cutlass.cute.typing.Tensor
```

<a id="cutlass.cute.crd2idx"></a>
### `cutlass.cute.crd2idx`

```python
cutlass.cute.crd2idx( coord: cutlass.cute.typing.Coord, layout: cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout | tuple | int, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Int
```

Convert a multi-dimensional coordinate into a value using the specified layout.

This function computes the inner product of the flattened coordinate and stride:

> index = sum(flatten(coord)[i] \* flatten(stride)[i] for i in range(len(coord)))

**Parameters:**

- **coord** (*Coord*) – A tuple or list representing the multi-dimensional coordinate
  (e.g., (i, j) for a 2D layout).
- **layout** (*Layout* *or* *ComposedLayout*) – A layout object that defines the memory storage layout, including shape and stride,
  used to compute the inner product.
- **loc** (*optional*) – Optional location information for IR diagnostics.
- **ip** (*optional*) – Optional instruction pointer or context for underlying IR functions.

**Returns:**

The result of applying the layout transformation to the provided coordinate.

**Return type:**

Any type that the layout maps to

**Example:**

```python
import cutlass.cute as cute
@cute.jit
def foo():
    L = cute.make_layout((5, 4), stride=(4, 1))
    idx = cute.crd2idx((2, 3), L)
    # Computed as: 2 * 4 + 3 = 11
    print(idx)
foo()  # Expected output: 11
```

<a id="cutlass.cute.idx2crd"></a>
### `cutlass.cute.idx2crd`

```python
cutlass.cute.idx2crd( idx: cutlass.cute.typing.IntTuple, shape: cutlass.cute.typing.Shape, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.IntTuple
```

Convert a linear index back into a nested coordinate using the specified layout.

Mapping from a linear index to the corresponding nested coordinate in the layout’s coordinate space.
It essentially “unfolds” a linear index into its constituent coordinate components.

**Parameters:**

- **idx** (*: int/Integer/Tuple*) – The linear index to convert back to coordinates.
- **shape** (*Shape*) – Shape of the layout defining the size of each mode
- **loc** (*optional*) – Optional location information for IR diagnostics.
- **ip** (*optional*) – Optional instruction pointer or context for underlying IR functions.

**Returns:**

The result of applying the layout transformation to the provided coordinate.

**Return type:**

Coord

**Examples:**

```python
import cutlass.cute as cute
@cute.jit
def foo():
    coord = cute.idx2crd(11, (5, 4))
    # idx2crd is always lexicographical ordering (left-to-right)
    # For shape (m, n, l, ...), coord = (idx % m, idx // m % n, idx // m // n % l, ...
    # Computed as: (11 % 5, 11 // 5 % 4) = (1, 2)
    cute.printf("coord: {}", coord)

foo()  # Expected output: (1, 2)
```

<a id="cutlass.cute.increment_coord"></a>
### `cutlass.cute.increment_coord`

```python
cutlass.cute.increment_coord( coord: cutlass.cute.typing.Coord, shape: cutlass.cute.typing.Shape, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Coord
```

Colexicographically increment a coordinate within a coordinate space defined by a shape.

Increments the leftmost mode first. When a mode reaches its
shape limit, it wraps to 0 and carries to the next mode.

**Parameters:**

- **coord** (*Coord*) – The coordinate to increment.
- **shape** (*Shape*) – The shape defining the coordinate space bounds.
- **loc** (*optional*) – Optional location information for IR diagnostics.
- **ip** (*optional*) – Optional instruction pointer or context for underlying IR functions.

**Returns:**

The incremented coordinate.

**Return type:**

Coord

**Raises:**

**ValueError** – If the coordinate and shape are not congruent or if the coordinate contains an underscore.

**Example:**

```python
import cutlass.cute as cute
@cute.jit
def foo():
    coord = cute.increment_coord((2, 0, 0), (3, 3, 3))
    # Increments colexicographically: (2,0,0) -> (0,1,0)
    cute.printf("coord: {}", coord)
foo()  # Expected output: coord: (0, 1, 0)
```

<a id="cutlass.cute.recast_layout"></a>
### `cutlass.cute.recast_layout`

```python
cutlass.cute.recast_layout( new_type_bits: int, old_type_bits: int, src_layout: cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout
```

Recast a layout from one data type to another.

**Parameters:**

- **new\_type\_bits** (*int*) – The new data type bits
- **old\_type\_bits** (*int*) – The old data type bits
- **src\_layout** (*Union*[*Layout*, *ComposedLayout*]) – The layout to recast
- **loc** (*optional*) – Optional location information for IR diagnostics.
- **ip** (*optional*) – Optional instruction pointer or context for underlying IR functions.

**Returns:**

The recast layout

**Return type:**

Layout or ComposedLayout

**Example:**

```python
import cutlass.cute as cute
@cute.jit
def foo():
    # Create a layout
    L = cute.make_layout((2, 3, 4))
    # Recast the layout to a different data type
    L_recast = cute.recast_layout(16, 8, L)
    print(L_recast)
foo()  # Expected output: (2, 3, 4)
```

<a id="cutlass.cute.slice_and_offset"></a>
### `cutlass.cute.slice_and_offset`

```python
cutlass.cute.slice_and_offset( coord: cutlass.cute.typing.Coord, src: cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → tuple
```

<a id="cutlass.cute.recast_ptr"></a>
### `cutlass.cute.recast_ptr`

```python
cutlass.cute.recast_ptr( ptr: cutlass.cute.typing.Pointer, swizzle_: cutlass._mlir.ir.register_value_caster | None = None, dtype: Type[cutlass.cute.typing.Numeric] | None = None, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Pointer
```

<a id="cutlass.cute.make_ptr"></a>
### `cutlass.cute.make_ptr`

```python
cutlass.cute.make_ptr( dtype: Type[cutlass.cute.typing.Numeric] | cutlass._mlir.dialects.cute.SparseElemType, value: int | cutlass.cute.typing.Integer | cutlass._mlir.ir.Value, mem_space: cutlass.cute.typing.AddressSpace | None = None, *, assumed_align: int | None = None, swizzle_: cutlass._mlir.ir.register_value_caster | None = None, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Pointer
```

<a id="cutlass.cute.composition"></a>
### `cutlass.cute.composition`

```python
cutlass.cute.composition( lhs: cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout | cutlass.cute.typing.Tensor, rhs: cutlass.cute.typing.Layout | cutlass.cute.typing.Shape | cutlass.cute.typing.Tile, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout | cutlass.cute.typing.Tensor
```

Compose two layout representations using the CuTe layout algebra.

Compose a left-hand layout (or tensor) with a right-hand operand into a new layout R, such that
for every coordinate c in the domain of the right-hand operand, the composed layout satisfies:

> R(c) = A(B(c))

where A is the left-hand operand provided as `lhs` and B is the right-hand operand provided as
`rhs`. In this formulation, B defines the coordinate domain while A applies its transformation to
B’s output, and the resulting layout R inherits the stride and shape adjustments from A.

**Satisfies:**

cute.shape(cute.composition(lhs, rhs)) is compatible with cute.shape(rhs)

**Parameters:**

- **lhs** (*Layout* *or* *Tensor*) – The left-hand operand representing the transformation to be applied.
- **rhs** (*Layout*, *Shape**, or* *Tile**, or* *int* *or* *tuple*) – The right-hand operand defining the coordinate domain. If provided as an int or tuple,
  it will be converted to a tile layout.
- **loc** (*optional*) – Optional location information for IR diagnostics.
- **ip** (*optional*) – Optional instruction pointer or context for underlying IR functions.

**Returns:**

A new composed layout R, such that for all coordinates c in the domain of `rhs`,
R(c) = lhs(rhs(c)).

**Return type:**

Layout or Tensor

**Example:**

```python
import cutlass.cute as cute
@cute.jit
def foo():
    # Create a layout that maps (i,j) to i*4 + j
    L1 = cute.make_layout((2, 3), stride=(4, 1))
    # Create a layout that maps (i,j) to i*3 + j
    L2 = cute.make_layout((3, 4), stride=(3, 1))
    # Compose L1 and L2
    L3 = cute.composition(L1, L2)
    # L3 now maps coordinates through L2 then L1
```

<a id="cutlass.cute.complement"></a>
### `cutlass.cute.complement`

```python
cutlass.cute.complement( input: cutlass.cute.typing.Layout, cotarget: cutlass.cute.typing.Layout | cutlass.cute.typing.Shape, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Layout
```

Compute the complement layout of the input layout with respect to the cotarget.

The complement of a layout A with respect to cotarget n is a layout A\* such that
for every k in Z\_n and c in the domain of A, there exists a unique c\* in the domain
of A\* where k = A(c) + A\*(c\*).

This operation is useful for creating layouts that partition a space in complementary ways,
such as row and column layouts that together cover a matrix.

**Parameters:**

- **input** (*Layout*) – The layout to compute the complement of
- **cotarget** (*Union*[*Layout*, *Shape*]) – The target layout or shape that defines the codomain
- **loc** (*optional*) – Optional location information for IR diagnostics
- **ip** (*optional*) – Optional instruction pointer or context for underlying IR functions

**Returns:**

The complement layout

**Return type:**

Layout

**Example:**

```python
import cutlass.cute as cute
@cute.jit
def foo():
    # Create a right-major layout for a 4x4 matrix
    row_layout = cute.make_layout((4, 4), stride=(4, 1))
    # Create a left-major layout that complements the row layout
    col_layout = cute.complement(row_layout, 16)
    # The two layouts are complementary under 16
```

<a id="cutlass.cute.right_inverse"></a>
### `cutlass.cute.right_inverse`

```python
cutlass.cute.right_inverse( input: cutlass.cute.typing.Layout, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Layout
```

<a id="cutlass.cute.left_inverse"></a>
### `cutlass.cute.left_inverse`

```python
cutlass.cute.left_inverse( input: cutlass.cute.typing.Layout, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Layout
```

<a id="cutlass.cute.logical_product"></a>
### `cutlass.cute.logical_product`

```python
cutlass.cute.logical_product( block: cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout, tiler: cutlass.cute.typing.Tile, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout
```

<a id="cutlass.cute.zipped_product"></a>
### `cutlass.cute.zipped_product`

```python
cutlass.cute.zipped_product( block: cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout, tiler: cutlass.cute.typing.Layout, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout
```

<a id="cutlass.cute.tiled_product"></a>
### `cutlass.cute.tiled_product`

```python
cutlass.cute.tiled_product( block: cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout, tiler: cutlass.cute.typing.Layout, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout
```

<a id="cutlass.cute.flat_product"></a>
### `cutlass.cute.flat_product`

```python
cutlass.cute.flat_product( block: cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout, tiler: cutlass.cute.typing.Layout, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout
```

<a id="cutlass.cute.raked_product"></a>
### `cutlass.cute.raked_product`

```python
cutlass.cute.raked_product( block: cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout, tiler: cutlass.cute.typing.Layout, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout
```

<a id="cutlass.cute.blocked_product"></a>
### `cutlass.cute.blocked_product`

```python
cutlass.cute.blocked_product( block: cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout, tiler: cutlass.cute.typing.Layout, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout
```

<a id="cutlass.cute.logical_divide"></a>
### `cutlass.cute.logical_divide`

```python
cutlass.cute.logical_divide( target: cutlass.cute.typing.Layout | cutlass.cute.typing.Tensor, tiler: cutlass.cute.typing.Tiler, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Layout | cutlass.cute.typing.Tensor
```

<a id="cutlass.cute.zipped_divide"></a>
### `cutlass.cute.zipped_divide`

```python
cutlass.cute.zipped_divide( target: cutlass.cute.typing.Layout | cutlass.cute.typing.Tensor, tiler: cutlass.cute.typing.Tiler, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Layout | cutlass.cute.typing.Tensor
```

`zipped_divide` is `logical_divide` with Tiler modes and Rest modes gathered together: `(Tiler,Rest)`

- When Tiler is Layout, this has no effect as `logical_divide` results in the same.
- When Tiler is `Tile` (nested tuple of `Layout`) or `Shape`, this zips modes into standard form
  `((BLK_A,BLK_B),(a,b,x,y))`

For example, if `target` has shape `(s, t, r)` and `tiler` has shape `(BLK_A, BLK_B)`,
then the result will have shape `((BLK_A, BLK_B), (ceil_div(s, BLK_A), ceil_div(t, BLK_B), r))`.

**Parameters:**

- **target** (*Layout* *or* *Tensor*) – The layout or tensor to partition.
- **tiler** (*Tiler*) – The tiling specification (can be a Layout, Shape, Tile).
- **loc** (*optional*) – Optional MLIR IR location information.
- **ip** (*optional*) – Optional MLIR IR insertion point.

**Returns:**

A zipped (partitioned) version of the target.

**Return type:**

Layout or Tensor

**Example:**

```python
layout = cute.make_layout((128, 64), stride=(64, 1))
tiler = (8, 8)
result = cute.zipped_divide(layout, tiler)  # result shape: ((8, 8), (16, 8))
```

<a id="cutlass.cute.tiled_divide"></a>
### `cutlass.cute.tiled_divide`

```python
cutlass.cute.tiled_divide( target: cutlass.cute.typing.Layout | cutlass.cute.typing.Tensor, tiler: cutlass.cute.typing.Tiler, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Layout | cutlass.cute.typing.Tensor
```

<a id="cutlass.cute.flat_divide"></a>
### `cutlass.cute.flat_divide`

```python
cutlass.cute.flat_divide( target: cutlass.cute.typing.Layout | cutlass.cute.typing.Tensor, tiler: cutlass.cute.typing.Tile, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Layout | cutlass.cute.typing.Tensor
```

<a id="cutlass.cute.max_common_layout"></a>
### `cutlass.cute.max_common_layout`

```python
cutlass.cute.max_common_layout( a: cutlass.cute.typing.Layout | cutlass.cute.typing.Tensor, b: cutlass.cute.typing.Layout | cutlass.cute.typing.Tensor, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Layout
```

<a id="cutlass.cute.max_common_vector"></a>
### `cutlass.cute.max_common_vector`

```python
cutlass.cute.max_common_vector( a: cutlass.cute.typing.Layout | cutlass.cute.typing.Tensor, b: cutlass.cute.typing.Layout | cutlass.cute.typing.Tensor, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → int
```

<a id="cutlass.cute.tile_to_shape"></a>
### `cutlass.cute.tile_to_shape`

```python
cutlass.cute.tile_to_shape( atom: cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout, trg_shape: cutlass.cute.typing.Shape, order: cutlass.cute.typing.Shape, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout
```

<a id="cutlass.cute.local_partition"></a>
### `cutlass.cute.local_partition`

```python
cutlass.cute.local_partition( target: cutlass.cute.typing.Tensor, tiler: cutlass.cute.typing.Layout | cutlass.cute.typing.Shape, index: int | cutlass.cute.typing.Numeric, proj: cutlass.cute.typing.XTuple = 1, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Tensor
```

<a id="cutlass.cute.local_tile"></a>
### `cutlass.cute.local_tile`

```python
cutlass.cute.local_tile( input: cutlass.cute.typing.Tensor, tiler: cutlass.cute.typing.Tiler, coord: cutlass.cute.typing.Coord, proj: cutlass.cute.typing.XTuple | None = None, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Tensor
```

Partition a tensor into tiles using a tiler and extract a single tile at the provided coordinate.

The `local_tile` operation applies a `zipped_divide` to split the `input` tensor by the `tiler`
and then slices out a single tile using the provided coord. This is commonly used for extracting block-,
thread-, or CTA-level tiles for parallel operations.

\[\text{local\_tile}(input, tiler, coord) = \text{zipped\_divide}(input, tiler)[coord]\]

This function corresponds to the CUTE/C++ local\_tile utility:
<https://docs.nvidia.com/cutlass/media/docs/cpp/cute/03_tensor.html#local-tile>

**Parameters:**

- **input** (*Tensor*) – The input tensor to partition into tiles.
- **tiler** (*Tiler*) – The tiling specification (can be a Layout, Shape, Tile).
- **coord** (*Coord*) – The coordinate to select within the remainder (“rest”) modes after tiling.
  This selects which tile to extract.
- **proj** (*XTuple*, *optional*) – (Optional) Projection onto tiling modes; specify to project out unused tiler modes,
  e.g., when working with projections of tilers in multi-mode partitioning.
  Default is None for no projection.
- **loc** (*Any*, *optional*) – (Optional) MLIR location, for diagnostic/debugging.
- **ip** (*Any*, *optional*) – (Optional) MLIR insertion point, used in IR building context.

**Returns:**

A new tensor representing the local tile selected at the given coordinate.

**Return type:**

Tensor

**Examples**

1. Tiling a 2D tensor and extracting a tile:

   > ```python
# input: (16, 24)
tensor : cute.Tensor
tiler = (2, 4)
coord = (1, 1)

# output: (8, 6)
# - zipped_divide(tensor, tiler)     -> ((2, 4), (8, 6))
# - local_tile(tensor, tiler, coord) -> (8, 6)
result = cute.local_tile(tensor, tiler=tiler, coord=coord)
```
2. Using a stride projection for specialized tiling:

   > ```python
# input: (16, 24)
tensor : cute.Tensor
tiler = (2, 2, 4)
coord = (0, 1, 1)
proj = (1, None, 1)

# output: (8, 6)
# projected_tiler: (2, 4)
# projected_coord: (0, 1)
# - zipped_divide(tensor, projected_tiler)               -> ((2, 4), (8, 6))
# - local_tile(tensor, projected_tiler, projected_coord) -> (8, 6)
result = cute.local_tile(tensor, tiler=tiler, coord=coord, proj=proj)
```

<a id="cutlass.cute.make_layout_image_mask"></a>
### `cutlass.cute.make_layout_image_mask`

```python
cutlass.cute.make_layout_image_mask( lay: cutlass.cute.typing.Layout, coord: cutlass.cute.typing.Coord, mode: int, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Int16
```

Makes a 16-bit integer mask of the image of a layout sliced at a given mode
and accounting for the offset given by the input coordinate for the other modes.

<a id="cutlass.cute.leading_dim"></a>
### `cutlass.cute.leading_dim`

```python
cutlass.cute.leading_dim( shape: cutlass.cute.typing.Shape, stride: cutlass.cute.typing.Stride, ) → int | Tuple[int, ...] | None
```

Find the leading dimension of a shape and stride.

**Parameters:**

- **shape** (*Shape*) – The shape of the tensor or layout
- **stride** (*Stride*) – The stride of the tensor or layout

**Returns:**

The leading dimension index or indices

**Return type:**

Union[int, Tuple[int, …], None]

The return value depends on the stride pattern:

> - If a single leading dimension is found, returns an integer index
> - If nested leading dimensions are found, returns a tuple of indices
> - If no leading dimension is found, returns None

<a id="cutlass.cute.make_layout_tv"></a>
### `cutlass.cute.make_layout_tv`

```python
cutlass.cute.make_layout_tv( thr_layout: cutlass.cute.typing.Layout, val_layout: cutlass.cute.typing.Layout, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → Tuple[cutlass.cute.typing.Shape, cutlass.cute.typing.Layout]
```

Create a thread-value layout by repeating the val\_layout over the thr\_layout.

This function creates a thread-value layout that maps between `(thread_idx, value_idx)`
coordinates and logical `(M,N)` coordinates. The thread and value layouts must be compact to ensure
proper partitioning.

This implements the thread-value partitioning pattern where data is partitioned
across threads and values within each thread.

**Parameters:**

- **thr\_layout** (*Layout*) – Layout mapping from `(TileM,TileN)` coordinates to thread IDs (must be compact)
- **val\_layout** (*Layout*) – Layout mapping from `(ValueM,ValueN)` coordinates to value IDs within each thread
- **loc** (*Optional*[*Location*], *optional*) – Source location for MLIR, defaults to None
- **ip** (*Optional*[*InsertionPoint*], *optional*) – Insertion point, defaults to None

**Returns:**

A tuple containing `tiler_mn` and `layout_tv`

**Return type:**

Tuple[Shape, Layout]

**where:**

- `tiler_mn` is tiler and `shape(tiler_mn)` is compatible with `shape(zipped_divide(x, tiler_mn))[0]`
- `layout_tv`: Thread-value layout mapping (thread\_idx, value\_idx) -> (M,N)

**Example:**

**The below code creates a TV Layout that maps thread/value coordinates to the logical coordinates in a `(4,6)` tensor:**

- *Tiler MN*: `(4,6)`
- *TV Layout*: `((3,2),(2,2)):((8,2),(4,1))`

```python
thr_layout = cute.make_layout((2, 3), stride=(3, 1))
val_layout = cute.make_layout((2, 2), stride=(2, 1))
tiler_mn, layout_tv = cute.make_layout_tv(thr_layout, val_layout)
```

Table 4 TV Layout

|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | 0 | 1 | 2 | 3 | 4 | 5 |
| 0 | T0, V0 | T0, V1 | T1, V0 | T1, V1 | T2, V0 | T2, V1 |
| 1 | T0, V2 | T0, V3 | T1, V2 | T1, V3 | T2, V2 | T2, V3 |
| 2 | T3, V0 | T3, V1 | T4, V0 | T4, V1 | T5, V0 | T5, V1 |
| 3 | T3, V2 | T3, V3 | T4, V2 | T4, V3 | T5, V2 | T5, V3 |

<a id="cutlass.cute.get_nonswizzle_portion"></a>
### `cutlass.cute.get_nonswizzle_portion`

```python
cutlass.cute.get_nonswizzle_portion( layout: cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout
```

Extract the non-swizzle portion from a layout.

For a simple Layout, the entire layout is considered non-swizzled and is returned as-is.
For a ComposedLayout, the inner layout (non-swizzled portion) is extracted and returned,
effectively separating the base layout from any swizzle transformation that may be applied.

**Parameters:**

- **layout** (*Union*[*Layout*, *ComposedLayout*]) – A Layout or ComposedLayout from which to extract the non-swizzle portion.
- **loc** (*optional*) – Optional location information for IR diagnostics.
- **ip** (*optional*) – Optional

**Returns:**

The non-swizzle portion of the input layout. For Layout objects, returns the layout itself.
For ComposedLayout objects, returns the outer layout component.

**Return type:**

Layout

**Raises:**

**TypeError** – If the layout is neither a Layout nor a ComposedLayout.

<a id="cutlass.cute.get_swizzle_portion"></a>
### `cutlass.cute.get_swizzle_portion`

```python
cutlass.cute.get_swizzle_portion( layout: cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass._mlir.ir.register_value_caster
```

Extract or create the swizzle portion from a layout.

For a simple Layout (which has no explicit swizzle), a default identity swizzle is created.
For a ComposedLayout, the outer layout is checked and returned if it is a Swizzle object.
Otherwise, a default identity swizzle is created. The default identity swizzle has parameters
(0, 4, 3), which represents a no-op swizzle transformation.

**Parameters:**

- **layout** (*Union*[*Layout*, *ComposedLayout*]) – A Layout or ComposedLayout from which to extract the swizzle portion.
- **loc** (*optional*) – Optional location information for IR diagnostics.
- **ip** (*optional*) – Optional

**Returns:**

The swizzle portion of the layout. For Layout objects or ComposedLayout objects without
a Swizzle outer component, returns a default identity swizzle (0, 4, 3). For ComposedLayout
objects with a Swizzle outer component, returns that swizzle.

**Return type:**

[Swizzle](#cutlass.cute.Swizzle)

**Raises:**

**TypeError** – If the layout is neither a Layout nor a ComposedLayout.

<a id="cutlass.cute.nullspace"></a>
### `cutlass.cute.nullspace`

```python
cutlass.cute.nullspace( layout: cutlass.cute.typing.Layout, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Layout
```

Computes the nullspace (kernel) of a layout.

Returns a layout l such that layout(l(i)) == 0 for all i < size(l),
nullspace(l) == make\_layout(1, stride=0),
and size(l) == size(layout) / size(filter\_zeros(layout))

**Parameters:**

- **layout** (*Layout*) – The layout to compute the nullspace of.
- **loc** (*optional*) – Optional location information for IR diagnostics.
- **ip** (*optional*) – Optional

**Returns:**

The nullspace of the layout

**Return type:**

Layout

**Raises:**

**TypeError** – If the layout is not a Layout.

<a id="cutlass.cute.ScaledBasis"></a>
### `cutlass.cute.ScaledBasis`

```python
class cutlass.cute.ScaledBasis(value: Any, mode: int | List[int])
```

Bases: `object`

A class representing a scaled basis element in CuTe’s layout algebra.

ScaledBasis is used to represent elements in the layout algebra, particularly
in the context of composition operations. It consists of a value (scale) and
a mode that identifies mode of the basis element.

**Parameters:**

- **value** (*Union*[*int*, *Integer*, *Ratio*, *ir.Value*]) – The scale value
- **mode** (*Union*[*int*, *List*[*int*]]) – The mode identifying the basis element

**Raises:**

**TypeError** – If mode is not an integer or list of integers

**Examples:**

```python
# Create a scaled basis with integer scale and mode
sb1 = ScaledBasis(2, 0)  # 2 * E(0)

# Create a scaled basis with a Ratio scale
sb2 = ScaledBasis(Ratio(1, 2), 1)  # (1/2) * E(1)

# Create a scaled basis with a list of modes
sb3 = ScaledBasis(4, [0, 1])  # 4 * E([0, 1])

# Scaled basis elements are commonly used in layout strides
layout = make_layout((4, 8), stride=(ScaledBasis(2, 0), ScaledBasis(1, 1)))

# This creates a layout with strides (2@0, 1@1) representing
# a coordinate system where each dimension has its own basis

# Example: Mapping coordinates to indices using the layout
coord = (2, 3)
idx = crd2idx(coord, layout)  # Maps (2, 3) to (4, 3)
```

<a id="cutlass.cute.ScaledBasis.__init__"></a>
#### `cutlass.cute.ScaledBasis.__init__`

```python
__init__( value: Any, mode: int | List[int], ) → None
```

<a id="cutlass.cute.ScaledBasis.is_static"></a>
#### `cutlass.cute.ScaledBasis.is_static`

```python
is_static() → bool
```

Check if the value is statically known.

**Returns:**

True if the value is not a dynamic expression

**Return type:**

bool

<a id="cutlass.cute.ScaledBasis.to"></a>
#### `cutlass.cute.ScaledBasis.to`

```python
to( dtype: type, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → Any
```

Convert to another type.

**Parameters:**

- **dtype** (*type*) – The target type for conversion
- **loc** (*Location*, *optional*) – The source location for the operation, defaults to None
- **ip** (*InsertionPoint*, *optional*) – The insertion point for the operation, defaults to None

**Returns:**

The ScaledBasis converted to the specified type

**Raises:**

**TypeError** – If conversion to the specified type is not supported

<a id="cutlass.cute.ScaledBasis.value"></a>
#### `cutlass.cute.ScaledBasis.value`

```python
property value: Any
```

Get the scale value.

**Returns:**

The scale value

<a id="cutlass.cute.ScaledBasis.mode"></a>
#### `cutlass.cute.ScaledBasis.mode`

```python
property mode: List[int]
```

Get the mode identifying the basis element.

**Returns:**

The mode as a list of integers

**Return type:**

List[int]

<a id="cutlass.cute.Atom"></a>
### `cutlass.cute.Atom`

```python
class cutlass.cute.Atom(op: Op, trait: Trait)
```

Bases: `ABC`

Atom base class.

An Atom is the composition of

- a MMA or Copy Operation;
- an internal MMA or Copy Trait.

An Operation is a pure Python class that is used to model a specific MMA or Copy instruction.
The Trait wraps the underlying IR Value and provides access to the metadata of the instruction
encoded using CuTe Layouts. When the Trait can be constructed straighforwardly from an
Operation, the `make_mma_atom` or `make_copy_atom` API should be used. There are cases where
constructing the metadata is not trivial and requires more information, for example to determine
the number of bytes copied per TMA instruction (“the TMA vector length”). In such cases,
dedicated helper functions are provided with an appropriate API such that the Atom is
constructed internally in an optimal fashion for the user.

<a id="cutlass.cute.Atom.__init__"></a>
#### `cutlass.cute.Atom.__init__`

```python
__init__( op: Op, trait: Trait, ) → None
```

<a id="cutlass.cute.Atom.op"></a>
#### `cutlass.cute.Atom.op`

```python
property op: Op
```

<a id="cutlass.cute.Atom.type"></a>
#### `cutlass.cute.Atom.type`

```python
property type: cutlass._mlir.ir.Type
```

<a id="cutlass.cute.Atom.set"></a>
#### `cutlass.cute.Atom.set`

```python
set( modifier: Any, value: Any, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

Sets runtime fields of the Atom.

Some Atoms have runtime state, for example a tcgen05 MMA Atom

```python
tiled_mma = cute.make_tiled_mma(some_tcgen05_mma_op)
tiled_mma.set(cute.nvgpu.tcgen05.Field.ACCUMULATE, True)
```

The `set` method provides a way to the user to modify such runtime state. Modifiable
fields are provided by arch-specific enumerations, for example `tcgen05.Field`. The Atom
instance internally validates the field as well as the value provided by the user to set
the field to.

<a id="cutlass.cute.Atom.get"></a>
#### `cutlass.cute.Atom.get`

```python
get( field: Any, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → Any
```

Gets runtime fields of the Atom.

Some Atoms have runtime state, for example a tcgen05 MMA Atom

```python
tiled_mma = cute.make_tiled_mma(some_tcgen05_mma_op)
accum = tiled_mma.get(cute.nvgpu.tcgen05.Field.ACCUMULATE)
```

The `get` method provides a way to the user to access such runtime state. Modifiable
fields are provided by arch-specific enumerations, for example `tcgen05.Field`. The Atom
instance internally validates the field as well as the value provided by the user to set
the field to.

<a id="cutlass.cute.Atom.with_"></a>
#### `cutlass.cute.Atom.with_`

```python
with_( *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, **kwargs: Any, ) → Atom
```

Returns a new Atom with the new Operation and Trait with the given runtime state. The runtime state
is provided as keyword arguments and it is Atom-specific.

```python
tiled_copy = cute.make_tiled_copy(tma_copy_op)
new_tiled_copy = tiled_copy.with_(tma_bar_ptr=tma_bar_ptr, cache_policy=cute.CacheEvictionPriority.EVICT_LAST)
```

The `with_` method provides a way to the user to modify such runtime state or create an executable Atom
(e.g. an Executable TMA Load Atom).

<a id="cutlass.cute.Atom._unpack"></a>
#### `cutlass.cute.Atom._unpack`

```python
_unpack( *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, **kwargs: Any, ) → cutlass._mlir.ir.Value
```

<a id="cutlass.cute.Atom._abc_impl"></a>
#### `cutlass.cute.Atom._abc_impl`

```python
_abc_impl = <_abc._abc_data object>
```

<a id="cutlass.cute.MmaAtom"></a>
### `cutlass.cute.MmaAtom`

```python
class cutlass.cute.MmaAtom(op: Op, trait: Trait)
```

Bases: [`Atom`](#cutlass.cute.Atom)

The MMA Atom class.

<a id="cutlass.cute.MmaAtom.thr_id"></a>
#### `cutlass.cute.MmaAtom.thr_id`

```python
property thr_id
```

<a id="cutlass.cute.MmaAtom.shape_mnk"></a>
#### `cutlass.cute.MmaAtom.shape_mnk`

```python
property shape_mnk
```

<a id="cutlass.cute.MmaAtom.tv_layout_A"></a>
#### `cutlass.cute.MmaAtom.tv_layout_A`

```python
property tv_layout_A
```

<a id="cutlass.cute.MmaAtom.tv_layout_B"></a>
#### `cutlass.cute.MmaAtom.tv_layout_B`

```python
property tv_layout_B
```

<a id="cutlass.cute.MmaAtom.tv_layout_C"></a>
#### `cutlass.cute.MmaAtom.tv_layout_C`

```python
property tv_layout_C
```

<a id="cutlass.cute.MmaAtom.make_fragment_A"></a>
#### `cutlass.cute.MmaAtom.make_fragment_A`

```python
make_fragment_A( input: Any, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass._mlir.ir.OpResult
```

<a id="cutlass.cute.MmaAtom.make_fragment_B"></a>
#### `cutlass.cute.MmaAtom.make_fragment_B`

```python
make_fragment_B( input: Any, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass._mlir.ir.OpResult
```

<a id="cutlass.cute.MmaAtom.make_fragment_C"></a>
#### `cutlass.cute.MmaAtom.make_fragment_C`

```python
make_fragment_C( input: Any, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass._mlir.ir.OpResult
```

<a id="cutlass.cute.MmaAtom._abc_impl"></a>
#### `cutlass.cute.MmaAtom._abc_impl`

```python
_abc_impl = <_abc._abc_data object>
```

<a id="cutlass.cute.CopyAtom"></a>
### `cutlass.cute.CopyAtom`

```python
class cutlass.cute.CopyAtom(op: Op, trait: Trait)
```

Bases: [`Atom`](#cutlass.cute.Atom)

The Copy Atom class.

<a id="cutlass.cute.CopyAtom.value_type"></a>
#### `cutlass.cute.CopyAtom.value_type`

```python
property value_type: Type[cutlass.cute.typing.Numeric]
```

<a id="cutlass.cute.CopyAtom.thr_id"></a>
#### `cutlass.cute.CopyAtom.thr_id`

```python
property thr_id: cutlass.cute.typing.Layout
```

<a id="cutlass.cute.CopyAtom.layout_src_tv"></a>
#### `cutlass.cute.CopyAtom.layout_src_tv`

```python
property layout_src_tv: cutlass.cute.typing.Layout
```

<a id="cutlass.cute.CopyAtom.layout_dst_tv"></a>
#### `cutlass.cute.CopyAtom.layout_dst_tv`

```python
property layout_dst_tv: cutlass.cute.typing.Layout
```

<a id="cutlass.cute.CopyAtom._abc_impl"></a>
#### `cutlass.cute.CopyAtom._abc_impl`

```python
_abc_impl = <_abc._abc_data object>
```

<a id="cutlass.cute.TiledCopy"></a>
### `cutlass.cute.TiledCopy`

```python
class cutlass.cute.TiledCopy(op: Op, trait: Trait)
```

Bases: [`CopyAtom`](#cutlass.cute.CopyAtom)

The tiled Copy class.

<a id="cutlass.cute.TiledCopy.layout_tv_tiled"></a>
#### `cutlass.cute.TiledCopy.layout_tv_tiled`

```python
property layout_tv_tiled: cutlass.cute.typing.Layout
```

<a id="cutlass.cute.TiledCopy.tiler_mn"></a>
#### `cutlass.cute.TiledCopy.tiler_mn`

```python
property tiler_mn: cutlass.cute.typing.Tile
```

<a id="cutlass.cute.TiledCopy.layout_src_tv_tiled"></a>
#### `cutlass.cute.TiledCopy.layout_src_tv_tiled`

```python
property layout_src_tv_tiled: cutlass.cute.typing.Layout
```

<a id="cutlass.cute.TiledCopy.layout_dst_tv_tiled"></a>
#### `cutlass.cute.TiledCopy.layout_dst_tv_tiled`

```python
property layout_dst_tv_tiled: cutlass.cute.typing.Layout
```

<a id="cutlass.cute.TiledCopy.size"></a>
#### `cutlass.cute.TiledCopy.size`

```python
property size: int
```

<a id="cutlass.cute.TiledCopy.get_slice"></a>
#### `cutlass.cute.TiledCopy.get_slice`

```python
get_slice( thr_idx: int | cutlass.cute.typing.Int32, ) → ThrCopy
```

<a id="cutlass.cute.TiledCopy.retile"></a>
#### `cutlass.cute.TiledCopy.retile`

```python
retile( src: cutlass.cute.typing.Tensor, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Tensor
```

<a id="cutlass.cute.TiledCopy._abc_impl"></a>
#### `cutlass.cute.TiledCopy._abc_impl`

```python
_abc_impl = <_abc._abc_data object>
```

<a id="cutlass.cute.TiledMma"></a>
### `cutlass.cute.TiledMma`

```python
class cutlass.cute.TiledMma(op: Op, trait: Trait)
```

Bases: [`MmaAtom`](#cutlass.cute.MmaAtom)

The tiled MMA class.

<a id="cutlass.cute.TiledMma.tv_layout_A_tiled"></a>
#### `cutlass.cute.TiledMma.tv_layout_A_tiled`

```python
property tv_layout_A_tiled
```

<a id="cutlass.cute.TiledMma.tv_layout_B_tiled"></a>
#### `cutlass.cute.TiledMma.tv_layout_B_tiled`

```python
property tv_layout_B_tiled
```

<a id="cutlass.cute.TiledMma.tv_layout_C_tiled"></a>
#### `cutlass.cute.TiledMma.tv_layout_C_tiled`

```python
property tv_layout_C_tiled
```

<a id="cutlass.cute.TiledMma.permutation_mnk"></a>
#### `cutlass.cute.TiledMma.permutation_mnk`

```python
property permutation_mnk
```

<a id="cutlass.cute.TiledMma.thr_layout_vmnk"></a>
#### `cutlass.cute.TiledMma.thr_layout_vmnk`

```python
property thr_layout_vmnk
```

<a id="cutlass.cute.TiledMma.size"></a>
#### `cutlass.cute.TiledMma.size`

```python
property size: int
```

<a id="cutlass.cute.TiledMma.get_tile_size"></a>
#### `cutlass.cute.TiledMma.get_tile_size`

```python
get_tile_size(mode_idx: int) → cutlass.cute.typing.Shape
```

<a id="cutlass.cute.TiledMma.get_slice"></a>
#### `cutlass.cute.TiledMma.get_slice`

```python
get_slice( thr_idx: int | cutlass.cute.typing.Int32, ) → ThrMma
```

<a id="cutlass.cute.TiledMma._partition_shape"></a>
#### `cutlass.cute.TiledMma._partition_shape`

```python
_partition_shape( operand_id: Any, shape: cutlass.cute.typing.Shape, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.XTuple
```

<a id="cutlass.cute.TiledMma.partition_shape_A"></a>
#### `cutlass.cute.TiledMma.partition_shape_A`

```python
partition_shape_A( shape_mk: cutlass.cute.typing.Shape, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.XTuple
```

<a id="cutlass.cute.TiledMma.partition_shape_B"></a>
#### `cutlass.cute.TiledMma.partition_shape_B`

```python
partition_shape_B( shape_nk: cutlass.cute.typing.Shape, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.XTuple
```

<a id="cutlass.cute.TiledMma.partition_shape_C"></a>
#### `cutlass.cute.TiledMma.partition_shape_C`

```python
partition_shape_C( shape_mn: cutlass.cute.typing.Shape, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.XTuple
```

<a id="cutlass.cute.TiledMma._thrfrg"></a>
#### `cutlass.cute.TiledMma._thrfrg`

```python
_thrfrg( operand_id: Any, input: cutlass.cute.typing.Layout, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Layout
```
#### `_thrfrg`

```python
_thrfrg( operand_id: Any, input: cutlass.cute.typing.Tensor, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Tensor
```

<a id="cutlass.cute.TiledMma._thrfrg_A"></a>
#### `cutlass.cute.TiledMma._thrfrg_A`

```python
_thrfrg_A( input: cutlass.cute.typing.Layout | cutlass.cute.typing.Tensor, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Layout | cutlass.cute.typing.Tensor
```

<a id="cutlass.cute.TiledMma._thrfrg_B"></a>
#### `cutlass.cute.TiledMma._thrfrg_B`

```python
_thrfrg_B( input: cutlass.cute.typing.Layout | cutlass.cute.typing.Tensor, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Layout | cutlass.cute.typing.Tensor
```

<a id="cutlass.cute.TiledMma._thrfrg_C"></a>
#### `cutlass.cute.TiledMma._thrfrg_C`

```python
_thrfrg_C( input: cutlass.cute.typing.Layout | cutlass.cute.typing.Tensor, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Layout | cutlass.cute.typing.Tensor
```

<a id="cutlass.cute.TiledMma._abc_impl"></a>
#### `cutlass.cute.TiledMma._abc_impl`

```python
_abc_impl = <_abc._abc_data object>
```

<a id="cutlass.cute.ThrMma"></a>
### `cutlass.cute.ThrMma`

```python
class cutlass.cute.ThrMma( op: Op, trait: Trait, thr_idx: int | cutlass.cute.typing.Int32, )
```

Bases: [`TiledMma`](#cutlass.cute.TiledMma)

The thread MMA class for modeling a thread-slice of a tiled MMA.

<a id="cutlass.cute.ThrMma.__init__"></a>
#### `cutlass.cute.ThrMma.__init__`

```python
__init__( op: Op, trait: Trait, thr_idx: int | cutlass.cute.typing.Int32, ) → None
```

<a id="cutlass.cute.ThrMma.thr_idx"></a>
#### `cutlass.cute.ThrMma.thr_idx`

```python
property thr_idx: int | cutlass.cute.typing.Int32
```

<a id="cutlass.cute.ThrMma.partition_A"></a>
#### `cutlass.cute.ThrMma.partition_A`

```python
partition_A( input_mk: cutlass.cute.typing.Tensor, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Tensor
```

<a id="cutlass.cute.ThrMma.partition_B"></a>
#### `cutlass.cute.ThrMma.partition_B`

```python
partition_B( input_nk: cutlass.cute.typing.Tensor, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Tensor
```

<a id="cutlass.cute.ThrMma.partition_C"></a>
#### `cutlass.cute.ThrMma.partition_C`

```python
partition_C( input_mn: cutlass.cute.typing.Tensor, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Tensor
```

<a id="cutlass.cute.ThrMma._abc_impl"></a>
#### `cutlass.cute.ThrMma._abc_impl`

```python
_abc_impl = <_abc._abc_data object>
```

<a id="cutlass.cute.ThrCopy"></a>
### `cutlass.cute.ThrCopy`

```python
class cutlass.cute.ThrCopy( op: Op, trait: Trait, thr_idx: int | cutlass.cute.typing.Int32, )
```

Bases: [`TiledCopy`](#cutlass.cute.TiledCopy)

The thread Copy class for modeling a thread-slice of a tiled Copy.

<a id="cutlass.cute.ThrCopy.__init__"></a>
#### `cutlass.cute.ThrCopy.__init__`

```python
__init__( op: Op, trait: Trait, thr_idx: int | cutlass.cute.typing.Int32, ) → None
```

<a id="cutlass.cute.ThrCopy.thr_idx"></a>
#### `cutlass.cute.ThrCopy.thr_idx`

```python
property thr_idx: int | cutlass.cute.typing.Int32
```

<a id="cutlass.cute.ThrCopy.partition_S"></a>
#### `cutlass.cute.ThrCopy.partition_S`

```python
partition_S( src: cutlass.cute.typing.Tensor, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Tensor
```

<a id="cutlass.cute.ThrCopy.partition_D"></a>
#### `cutlass.cute.ThrCopy.partition_D`

```python
partition_D( dst: cutlass.cute.typing.Tensor, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Tensor
```

<a id="cutlass.cute.ThrCopy._abc_impl"></a>
#### `cutlass.cute.ThrCopy._abc_impl`

```python
_abc_impl = <_abc._abc_data object>
```

<a id="cutlass.cute.TensorSSA"></a>
### `cutlass.cute.TensorSSA`

```python
class cutlass.cute.TensorSSA(*args: Any, **kwargs: Any)
```

Bases: `Vector`

A class representing thread local data from CuTe Tensor in value semantic and immutable.

**Parameters:**

- **value** (*ir.Value*) – Flatten vector as ir.Value holding logic data of SSA Tensor
- **shape** (*Shape*) – The nested shape in CuTe of the vector
- **dtype** (*Type*[*Numeric*]) – Data type of the tensor elements

**Variables:**

- **\_shape** – The nested shape in CuTe of the vector
- **\_dtype** – Data type of the tensor elements

**Raises:**

**ValueError** – If shape is not static

<a id="cutlass.cute.TensorSSA.__init__"></a>
#### `cutlass.cute.TensorSSA.__init__`

```python
__init__( value: cutlass._mlir.ir.Value, shape: cutlass.cute.typing.Shape, dtype: Type[cutlass.cute.typing.Numeric] | None = None, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

Create a [`TensorSSA`](#cutlass.cute.TensorSSA) object: an immutable, thread-local tensor backed by a flattened MLIR vector.

**Parameters:**

- **value** (`ir.Value`) – A `ir.Value` holding the flattened MLIR vector value of the tensor.
- **shape** (*Shape*) – The logical (possibly nested) shape of the tensor.
- **dtype** (*Type*[*Numeric*], *optional*) – The data type of the tensor elements. If None,
  this is inferred from the MLIR element type.

**Keyword Arguments:**

- **loc** – Optional location for op construction.
- **ip** – Optional insertion point for op construction.

**Raises:**

**ValueError** – If `value` is not an `ir.Value`, is not of vector type,
or if `shape` is not statically known.

> **Note**
>
> - Instances are immutable and represent per-thread local SSA values using value semantics.
> - The tensor’s broadcast shape and static element type are registered; dynamic shapes are not supported.

<a id="cutlass.cute.TensorSSA.from_vector"></a>
#### `cutlass.cute.TensorSSA.from_vector`

```python
static from_vector( value: cutlass._mlir.ir.Value, *, dtype: Type[cutlass.cute.typing.Numeric] | None = None, shape: cutlass.cute.typing.Shape | None = None, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → TensorSSA
```

Construct a [`TensorSSA`](#cutlass.cute.TensorSSA) from a given MLIR vector value.

This helper interprets the given 1D or n-D MLIR vector value and returns a TensorSSA view.
If the input is an n-D vector, it shape-casts it into a 1D vector holding the same number of elements.

**Parameters:**

- **value** – The ir.Value representing an MLIR vector value (1D or n-D).
- **dtype** – Optional explicit type of the elements. Deduced from MLIR type if not provided.
- **loc** – Optional MLIR location.
- **ip** – Optional MLIR insertion point.

**Returns:**

A TensorSSA view over the vector value.

<a id="cutlass.cute.TensorSSA.to_vector"></a>
#### `cutlass.cute.TensorSSA.to_vector`

```python
to_vector( *, force_flatten: bool = False, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass._mlir_helpers.vector.Vector
```

Convert the tensor to `Vector` carrying the tensor’s dtype.

Returns a `Vector` wrapping the underlying MLIR vector value;
the DSL dtype is propagated so callers can use `Vector.reduce()`,
`Vector.to()`, and element-wise arithmetic.

<a id="cutlass.cute.TensorSSA.dtype"></a>
#### `cutlass.cute.TensorSSA.dtype`

```python
property dtype: Type[cutlass.cute.typing.Numeric]
```

<a id="cutlass.cute.TensorSSA.element_type"></a>
#### `cutlass.cute.TensorSSA.element_type`

```python
property element_type: Type[cutlass.cute.typing.Numeric]
```

<a id="cutlass.cute.TensorSSA._wrap_like"></a>
#### `cutlass.cute.TensorSSA._wrap_like`

```python
_wrap_like( result_ir: cutlass._mlir.ir.Value, ) → TensorSSA
```

Preserve CuTe nested shape when the math foundation wraps a
per-element op’s result back into a TensorSSA.

<a id="cutlass.cute.TensorSSA._count"></a>
#### `cutlass.cute.TensorSSA._count`

```python
property _count: int
```

Total element count — flatten CuTe nested shape before multiplying.

Overrides `Vector._count`, which assumes a flat MLIR shape tuple.
TensorSSA carries a possibly-nested CuTe shape (e.g. `((4, 2), 8)`),
so the base implementation’s `result *= dim` produces garbage for
nested shapes (tuple-repetition instead of arithmetic). `numel`
picks up this override automatically.

<a id="cutlass.cute.TensorSSA.shape"></a>
#### `cutlass.cute.TensorSSA.shape`

```python
property shape: cutlass.cute.typing.Shape
```

<a id="cutlass.cute.TensorSSA._apply_op"></a>
#### `cutlass.cute.TensorSSA._apply_op`

```python
_apply_op( op: Callable, other: object, flip: bool = False, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → TensorSSA
```

<a id="cutlass.cute.TensorSSA.apply_op"></a>
#### `cutlass.cute.TensorSSA.apply_op`

```python
apply_op( op: Callable, other: object, flip: bool = False, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → TensorSSA
```

Apply a binary operation to this tensor and another operand.

This public API method wraps the internal `_apply_op` for external usage, allowing custom operations to be performed on tensors.

**Parameters:**

- **op** (*Callable*) – The operation function (e.g., `operator.add`, `operator.mul`, etc.).
- **other** ([*TensorSSA*](#cutlass.cute.TensorSSA) *or* *ArithValue* *or* *scalar*) – The other operand. Can be a [`TensorSSA`](#cutlass.cute.TensorSSA), ArithValue, or scalar.
- **flip** (*bool*, *optional*) – If `True`, flips the operands (applies operation as `op(other, self)`).
- **loc** (*object*, *optional*) – MLIR location, optional.
- **ip** (*object*, *optional*) – MLIR insertion point, optional.

**Returns:**

The result of applying the binary operation.

**Return type:**

[TensorSSA](#cutlass.cute.TensorSSA)

**Example**

```python
import operator

tensor1 = cute.Tensor(...)
tensor2 = cute.Tensor(...)
result = tensor1.apply_op(operator.add, tensor2)
# Equivalent to: tensor1 + tensor2
```

<a id="cutlass.cute.TensorSSA.broadcast_to"></a>
#### `cutlass.cute.TensorSSA.broadcast_to`

```python
broadcast_to( target_shape: cutlass.cute.typing.Shape, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → TensorSSA
```

Broadcast the tensor to the target shape.

This method broadcasts the tensor to match a target shape following NumPy-style
broadcasting rules. Dimensions of size 1 can be broadcast to any size, and
missing dimensions are added with size 1.

**Parameters:**

- **target\_shape** (*Shape*) – The desired output shape
- **loc** (*Optional*[*Location*]) – Source location for MLIR operations, defaults to None
- **ip** (*Optional*[*InsertionPoint*]) – Insertion point for MLIR operations, defaults to None

**Returns:**

A new tensor broadcast to the target shape

**Return type:**

[TensorSSA](#cutlass.cute.TensorSSA)

**Raises:**

**ValueError** – If shapes are incompatible for broadcasting

**Examples:**

```python
# Broadcast a (1, 4) tensor to (3, 4)
src = cute.full((1, 4), 1.0, Float32)
dst = src.broadcast_to((3, 4))
# dst now has shape (3, 4) with the first row replicated
```

<a id="cutlass.cute.TensorSSA._flatten_shape_and_coord"></a>
#### `cutlass.cute.TensorSSA._flatten_shape_and_coord`

```python
_flatten_shape_and_coord( crd: cutlass.cute.typing.Coord, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → Tuple[cutlass.cute.typing.Shape, cutlass.cute.typing.Coord]
```

<a id="cutlass.cute.TensorSSA._build_result"></a>
#### `cutlass.cute.TensorSSA._build_result`

```python
_build_result( res_vect: cutlass._mlir.ir.Value, res_shp: cutlass.cute.typing.Shape, *, row_major: bool = False, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → TensorSSA
```

<a id="cutlass.cute.TensorSSA.reshape"></a>
#### `cutlass.cute.TensorSSA.reshape`

```python
reshape( shape: cutlass.cute.typing.Shape, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → TensorSSA
```

Reshape the tensor to a new shape.

**Parameters:**

**shape** (*Shape*) – The new shape to reshape to.

**Returns:**

A new tensor with the same elements but with the new shape.

**Return type:**

[TensorSSA](#cutlass.cute.TensorSSA)

**Raises:**

- **NotImplementedError** – If dynamic size is not supported
- **ValueError** – If the new shape is not compatible with the current shape

<a id="cutlass.cute.TensorSSA.to"></a>
#### `cutlass.cute.TensorSSA.to`

```python
to( dtype: Type[cutlass.cute.typing.Numeric], *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → TensorSSA
```

Convert the tensor to a different numeric type.

**Parameters:**

**dtype** (*Type*[*Numeric*]) – The target numeric type to cast to.

**Returns:**

A new tensor with the same shape but with elements cast to the target type.

**Return type:**

[TensorSSA](#cutlass.cute.TensorSSA)

**Raises:**

- **TypeError** – If dtype is not a subclass of Numeric.
- **NotImplementedError** – If dtype is an unsigned integer type.

<a id="cutlass.cute.TensorSSA.bitcast"></a>
#### `cutlass.cute.TensorSSA.bitcast`

```python
bitcast( dtype: Type[cutlass.cute.typing.Numeric], *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → TensorSSA
```

Reinterpret the bits of this tensor as a different element type.

Total bit width is preserved; the element count adjusts proportionally.
For example, a `TensorSSA` of shape `(4,)` with `Float32` bitcast
to `Float16` yields a `TensorSSA` of shape `(8,)` with `Float16`
(4 × 32 = 8 × 16 bits). Multi-dimensional shapes are flattened.

**Parameters:**

**dtype** (*Type*[*Numeric*]) – Target DSL element type (e.g. `Int32`, `Float16`).

**Returns:**

A new [`TensorSSA`](#cutlass.cute.TensorSSA) with bits reinterpreted as `dtype`.

**Return type:**

[TensorSSA](#cutlass.cute.TensorSSA)

**Raises:**

**TypeError** – If `dtype` is not a subclass of `Numeric`.

<a id="cutlass.cute.TensorSSA.ir_value"></a>
#### `cutlass.cute.TensorSSA.ir_value`

```python
ir_value( *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → TensorSSA
```

<a id="cutlass.cute.TensorSSA.ir_value_int8"></a>
#### `cutlass.cute.TensorSSA.ir_value_int8`

```python
ir_value_int8( *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass._mlir.ir.Value
```

Returns int8 ir value of Boolean tensor.
When we need to store Boolean tensor ssa, use ir\_value\_int8().

**Parameters:**

- **loc** (*Optional*[*Location*], *optional*) – Source location information, defaults to None
- **ip** (*Optional*[*InsertionPoint*], *optional*) – Insertion point for MLIR operations, defaults to None

**Returns:**

The int8 value of this Boolean

**Return type:**

ir.Value

<a id="cutlass.cute.TensorSSA.reduce"></a>
#### `cutlass.cute.TensorSSA.reduce`

```python
reduce( op: cutlass._mlir.dialects.cute.ReductionOp, init_val: object, reduction_profile: cutlass.cute.typing.Coord, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → TensorSSA | cutlass._mlir.ir.Value
```

Perform reduce on selected modes with given predefined reduction op.

**Parameters:**

- **op** (*operator*) – The reduction operator to use (operator.add or operator.mul)
- **init\_val** (*numeric*) – The initial value for the reduction
- **reduction\_profile** (*Coord*) – Specifies which dimensions to reduce. Dimensions marked with None are kept.

**Returns:**

The reduced tensor

**Return type:**

[TensorSSA](#cutlass.cute.TensorSSA)

**Examples:**

```python
reduce(f32 o (4,))
  => f32

reduce(f32 o (4, 5))
  => f32
reduce(f32 o (4, (5, 4)), reduction_profile=(None, 1))
  => f32 o (4,)
reduce(f32 o (4, (5, 4)), reduction_profile=(None, (None, 1)))
  => f32 o (4, (5,))
```

<a id="cutlass.cute.print_tensor"></a>
### `cutlass.cute.print_tensor`

```python
cutlass.cute.print_tensor( tensor: cutlass.cute.typing.Tensor | TensorSSA, *, verbose: bool = False, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

Print content of the tensor in human readable format.

Outputs the tensor data in a structured format showing both metadata
and the actual data values. The output includes tensor type information,
layout details, and a formatted array representation of the values.

**Parameters:**

- **tensor** (*Tensor*) – The tensor to print
- **verbose** (*bool*) – If True, includes additional debug information in the output
- **loc** (*source location*, *optional*) – Source location where it’s called, defaults to None
- **ip** (*insertion pointer*, *optional*) – Insertion pointer for IR generation, defaults to None

**Raises:**

**NotImplementedError** – If the tensor type doesn’t support trivial dereferencing

**Example output:**

```text
tensor(raw_ptr<@..., Float32, generic, align(4)> o (8,5):(5,1), data=
       [[-0.4326, -0.5434,  0.1238,  0.7132,  0.8042],
        [-0.8462,  0.9871,  0.4389,  0.7298,  0.6948],
        [ 0.3426,  0.5856,  0.1541,  0.2923,  0.6976],
        [-0.1649,  0.8811,  0.1788,  0.1404,  0.2568],
        [-0.2944,  0.8593,  0.4171,  0.8998,  0.1766],
        [ 0.8814,  0.7919,  0.7390,  0.4566,  0.1576],
        [ 0.9159,  0.7577,  0.6918,  0.0754,  0.0591],
        [ 0.6551,  0.1626,  0.1189,  0.0292,  0.8655]])
```

<a id="cutlass.cute.make_tensor"></a>
### `cutlass.cute.make_tensor`

```python
cutlass.cute.make_tensor( iterator: cutlass.cute.typing.Pointer | cutlass.cute.typing.IntTuple | cutlass._mlir.ir.Value, layout: cutlass.cute.typing.Shape | cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Tensor
```

Creates a tensor by composing an engine (iterator/pointer) with a layout.

A tensor is defined as T = E ∘ L, where E is an engine (array, pointer, or counting iterator)
and L is a layout that maps logical coordinates to physical offsets. The tensor
evaluates coordinates by applying the layout mapping and dereferencing the engine
at the resulting offset.

**Parameters:**

- **iterator** (*Union*[*Pointer*, *IntTuple*, *ir.Value*]) – Engine component that provides data access capabilities. Can be:
  - A pointer (Pointer type)
  - An integer or integer tuple for coordinate tensors
  - A shared memory descriptor (SmemDescType)
- **layout** (*Union*[*Shape*, *Layout*, *ComposedLayout*]) – Layout component that defines the mapping from logical coordinates to
  physical offsets. Can be:
  - A shape tuple that will be converted to a layout
  - A Layout object
  - A ComposedLayout object (must be a normal layout)
- **loc** (*Optional*[*Location*]) – Source location for MLIR operation tracking, defaults to None
- **ip** (*Optional*[*InsertionPoint*]) – Insertion point for MLIR operation, defaults to None

**Returns:**

A tensor object representing the composition E ∘ L

**Return type:**

Tensor

**Raises:**

- **TypeError** – If iterator type is not a supported type
- **ValueError** – If layout is a composed layout with customized inner functions

**Examples:**

```python
# Create a tensor with row-major layout from a pointer
ptr = make_ptr(Float32, base_ptr, AddressSpace.gmem)
layout = make_layout((64, 128), stride=(128, 1))
tensor = make_tensor(ptr, layout)

# Create a tensor with hierarchical layout in shared memory
smem_ptr = make_ptr(Float16, base_ptr, AddressSpace.smem)
layout = make_layout(((128, 8), (1, 4, 1)), stride=((32, 1), (0, 8, 4096)))
tensor = make_tensor(smem_ptr, layout)

# Create a coordinate tensor
layout = make_layout(2, stride=16 * E(0))
tensor = make_tensor(5, layout)  # coordinate tensor with iterator starting at 5
```

Notes

- The engine (iterator) must support random access operations
- Common engine types include raw pointers, arrays, and random-access iterators
- The layout defines both the shape (logical dimensions) and stride (physical mapping)
- Supports both direct coordinate evaluation T(c) and partial evaluation (slicing)
- ComposedLayouts must be “normal” layouts (no inner functions)
- For coordinate tensors, the iterator is converted to a counting sequence

<a id="cutlass.cute.make_identity_tensor"></a>
### `cutlass.cute.make_identity_tensor`

```python
cutlass.cute.make_identity_tensor( shape: cutlass.cute.typing.Shape, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Tensor
```

Creates an identity tensor with the given shape.

An identity tensor maps each coordinate to itself, effectively creating a counting
sequence within the shape’s bounds. This is useful for generating coordinate indices
or creating reference tensors for layout transformations.

**Parameters:**

- **shape** (*Shape*) – The shape defining the tensor’s dimensions. Can be a simple integer
  sequence or a hierarchical structure ((m,n),(p,q))
- **loc** (*Optional*[*Location*]) – Source location for MLIR operation tracking, defaults to None
- **ip** (*Optional*[*InsertionPoint*]) – Insertion point for MLIR operation, defaults to None

**Returns:**

A tensor that maps each coordinate to itself

**Return type:**

Tensor

**Examples:**

```python
# Create a simple 1D coord tensor
tensor = make_identity_tensor(6)  # [0,1,2,3,4,5]

# Create a 2D coord tensor
tensor = make_identity_tensor((3,2))  # [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1)]

# Create hierarchical coord tensor
tensor = make_identity_tensor(((2,1),3))
# [((0,0),0),((1,0),0),((0,0),1),((1,0),1),((0,0),2),((1,0),2)]
```

Notes

- The shape parameter follows CuTe’s IntTuple concept
- Coordinates are ordered colexicographically
- Useful for generating reference coordinates in layout transformations

<a id="cutlass.cute.make_fragment_like"></a>
### `cutlass.cute.make_fragment_like`

```python
cutlass.cute.make_fragment_like( src: cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout | cutlass.cute.typing.Tensor, dtype: Type[cutlass.cute.typing.Numeric] | None = None, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Layout | cutlass.cute.typing.Tensor
```

<a id="cutlass.cute.make_rmem_tensor"></a>
### `cutlass.cute.make_rmem_tensor`

```python
cutlass.cute.make_rmem_tensor( layout_or_shape: cutlass.cute.typing.Layout | cutlass.cute.typing.Shape, dtype: Type[cutlass.cute.typing.Numeric], *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Tensor
```

Creates a tensor in register memory with the specified layout/shape and data type.

This function allocates a tensor in register memory (rmem) usually on stack with
either a provided layout or creates a new layout from the given shape. The tensor
will have elements of the specified numeric data type.

**Parameters:**

- **layout\_or\_shape** (*Union*[*Layout*, *Shape*]) – Either a Layout object defining the tensor’s memory organization,
  or a Shape defining its dimensions
- **dtype** (*Type*[*Numeric*]) – The data type for tensor elements (must be a Numeric type)
- **loc** (*Optional*[*Location*]) – Source location for MLIR operation tracking, defaults to None
- **ip** (*Optional*[*InsertionPoint*]) – Insertion point for MLIR operation, defaults to None

**Returns:**

A tensor allocated in register memory

**Return type:**

Tensor

**Examples:**

```python
# Create rmem tensor with explicit layout
layout = make_layout((128, 32))
tensor = make_rmem_tensor(layout, cutlass.Float16)

# Create rmem tensor directly from shape
tensor = make_rmem_tensor((64, 64), cutlass.Float32)
```

Notes

- Uses 32-byte alignment to support .128 load/store operations
- Boolean types are stored as 8-bit integers
- Handles both direct shapes and Layout objects

<a id="cutlass.cute.make_rmem_tensor_like"></a>
### `cutlass.cute.make_rmem_tensor_like`

```python
cutlass.cute.make_rmem_tensor_like( src: cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout | cutlass.cute.typing.Tensor | TensorSSA, dtype: Type[cutlass.cute.typing.Numeric] | None = None, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Tensor
```

**Creates a tensor in register memory with the same shape as the input layout but**

compact col-major strides. This is equivalent to calling make\_rmem\_tensor(make\_layout\_like(tensor)).

This function allocates a tensor in register memory (rmem) usually on stack with
with the compact layout like the source. The tensor will have elements of the
specified numeric data type or the same as the source.

**Parameters:**

- **src** (*Union*[*Layout*, *ComposedLayout*, *Tensor*]) – The source layout or tensor whose shape will be matched
- **dtype** (*Type*[*Numeric*], *optional*) – The element type for the fragment tensor, defaults to None
- **loc** (*Location*, *optional*) – Source location for MLIR operations, defaults to None
- **ip** (*InsertionPoint*, *optional*) – Insertion point for MLIR operations, defaults to None

**Returns:**

A new layout or fragment tensor with matching shape

**Return type:**

Union[Layout, Tensor]

**Examples:**

Creating a rmem tensor from a tensor:

```python
smem_tensor = cute.make_tensor(smem_ptr, layout)
rmem_tensor = cute.make_rmem_tensor_like(smem_tensor, cutlass.Float32)
# frag_tensor will be a register-backed tensor with the same shape
```

Creating a fragment with a different element type:

```python
tensor = cute.make_tensor(gmem_ptr, layout)
rmem_bool_tensor = cute.make_rmem_tensor_like(tensor, cutlass.Boolean)
# bool_frag will be a register-backed tensor with Boolean elements
```

**Notes**

- When used with a Tensor, if a type is provided, it will create a new
  fragment tensor with that element type.
- For layouts with ScaledBasis strides, the function creates a fragment
  from the shape only.
- This function is commonly used in GEMM and other tensor operations to
  create register storage for intermediate results.

<a id="cutlass.cute.recast_tensor"></a>
### `cutlass.cute.recast_tensor`

```python
cutlass.cute.recast_tensor( src: cutlass.cute.typing.Tensor, dtype: Type[cutlass.cute.typing.Numeric], swizzle_: object | None = None, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Tensor
```

Recast a tensor to a different data type by changing the element interpretation.

This function reinterprets the memory of a tensor with a different element type,
adjusting both the iterator pointer type and the layout to maintain consistency.

**Parameters:**

- **src** (*Tensor*) – The source tensor to recast
- **dtype** (*Type*[*Numeric*]) – The target data type for tensor elements
- **swizzle** (*Optional*, *unused*) – Optional swizzle parameter (reserved for future use), defaults to None
- **loc** (*Optional*[*Location*]) – Source location for MLIR operation tracking, defaults to None
- **ip** (*Optional*[*InsertionPoint*]) – Insertion point for MLIR operation, defaults to None

**Returns:**

A new tensor with the same memory but reinterpreted as dtype

**Return type:**

Tensor

**Raises:**

**TypeError** – If dtype is not a subclass of Numeric

**Examples:**

```python
# Create a Float32 tensor
tensor_f32 = make_rmem_tensor((4, 8), Float32)

# Recast to Int32 to manipulate bits
tensor_i32 = recast_tensor(tensor_f32, Int32)

# Both tensors share the same memory, but interpret it differently
```

<a id="cutlass.cute.find"></a>
### `cutlass.cute.find`

```python
cutlass.cute.find( t: cutlass.cute.typing.XTuple, x: int, hierarchical: bool = True, ) → int | Tuple[int, ...] | None
```

Find the first position of a value `x` in a hierarchical structure `t`.

Searches for the first occurrence of x in t, optionally excluding positions
where a comparison value matches. The search can traverse nested structures
and returns either a single index or a tuple of indices for nested positions.

**Parameters:**

- **t** (*XTuple*) – The search space
- **x** (*int*) – The static integer x to search for

**Returns:**

Index if found at top level, tuple of indices showing nested position, or None if not found

**Return type:**

Union[int, Tuple[int, …], None]

<a id="cutlass.cute.find_if"></a>
### `cutlass.cute.find_if`

```python
cutlass.cute.find_if( t: cutlass.cute.typing.XTuple, pred_fn: Callable[[cutlass.cute.typing.XTuple, int | Tuple[int, ...]], bool], hierarchical: bool = True, ) → int | Tuple[int, ...] | None
```

<a id="cutlass.cute.transform_leaf"></a>
### `cutlass.cute.transform_leaf`

```python
cutlass.cute.transform_leaf( f: Callable[[...], cutlass.cute.typing.XTuple], *args: cutlass.cute.typing.XTuple, ) → cutlass.cute.typing.XTuple
```

Apply a function to the leaf nodes of nested tuple structures.

This function traverses nested tuple structures in parallel and applies the function f
to corresponding leaf nodes. All input tuples must have the same nested structure.

**Parameters:**

- **f** (*Callable*) – Function to apply to leaf nodes
- **args** – One or more nested tuple structures with matching profiles

**Returns:**

A new nested tuple with the same structure as the inputs, but with leaf values transformed by f

**Raises:**

**TypeError** – If the input tuples have different nested structures

**Example:**

```python
>>> transform_leaf(lambda x: x + 1, (1, 2))
(2, 3)
>>> transform_leaf(lambda x, y: x + y, (1, 2), (3, 4))
(4, 6)
>>> transform_leaf(lambda x: x * 2, ((1, 2), (3, 4)))
((2, 4), (6, 8))
```

<a id="cutlass.cute.basis_value"></a>
### `cutlass.cute.basis_value`

```python
cutlass.cute.basis_value( e: ScaledBasis | Any, ) → cutlass.cute.typing.Int | cutlass._mlir.ir.Value | Ratio
```

Extract the value from a ScaledBasis or return the input as-is.

If the input is a ScaledBasis, returns its value component.
Otherwise, returns the input unchanged.

**Parameters:**

**e** (*Any*) – The input element (ScaledBasis or any other type)

**Returns:**

The value of the ScaledBasis or the input itself

**Return type:**

Any

**Examples:**

```python
>>> basis_value(ScaledBasis(5, 0))
5
>>> basis_value(42)
42
```

<a id="cutlass.cute.basis_get"></a>
### `cutlass.cute.basis_get`

```python
cutlass.cute.basis_get( basis: ScaledBasis | cutlass.cute.typing.Numeric | int, t: cutlass.cute.typing.XTuple | cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.XTuple | cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout
```

Apply the mode indices from a ScaledBasis to get an element from a tuple, layout, or composed layout.

If the basis is a ScaledBasis or Numeric with mode indices, this function uses those
indices to extract the corresponding element from the tuple using hierarchical
indexing. If the basis is not a ScaledBasis or has no modes, returns the tuple, layout, or composed layout as-is.

**Parameters:**

- **basis** ([*ScaledBasis*](#cutlass.cute.ScaledBasis)) – The basis element (ScaledBasis)
- **t** (*Union*[*XTuple*, *Layout*, *ComposedLayout*]) – The tuple, layout, or composed layout to index into

**Returns:**

The element at the position specified by the basis modes, or t itself

**Return type:**

Union[XTuple, Layout, ComposedLayout]

**Examples:**

```python
>>> basis_get(ScaledBasis(2, 1), (10, 20, 30))
20
>>> basis_get(ScaledBasis(2, [0, 1]), ((10, 20), (30, 40)))
20
>>> basis_get(5, (10, 20, 30))  # Non-basis returns tuple as-is
(10, 20, 30)
```

<a id="cutlass.cute.flatten_to_tuple"></a>
### `cutlass.cute.flatten_to_tuple`

```python
cutlass.cute.flatten_to_tuple( a: cutlass.cute.typing.XTuple, ) → Tuple[Any, ...]
```

Flattens a potentially nested tuple structure into a flat tuple.

This function recursively traverses the input structure and flattens it into
a single-level tuple, preserving the order of elements.

**Parameters:**

**a** (*Union*[*IntTuple*, *Coord*, *Shape*, *Stride*]) – The structure to flatten

**Returns:**

A flattened tuple containing all elements from the input

**Return type:**

tuple

**Examples:**

```python
flatten_to_tuple((1, 2, 3))       # Returns (1, 2, 3)
flatten_to_tuple(((1, 2), 3))     # Returns (1, 2, 3)
flatten_to_tuple((1, (2, (3,))))  # Returns (1, 2, 3)
```

<a id="cutlass.cute.unflatten"></a>
### `cutlass.cute.unflatten`

```python
cutlass.cute.unflatten( sequence: Tuple[Any, ...] | List[Any] | Iterable[Any], profile: cutlass.cute.typing.XTuple, ) → cutlass.cute.typing.XTuple
```

Unflatten a flat tuple into a nested tuple structure according to a profile.

This function transforms a flat sequence of elements into a nested tuple structure
that matches the structure defined by the profile parameter. It traverses the profile
structure and populates it with elements from the sequence.

sequence must be long enough to fill the profile. Raises RuntimeError if it is not.

**Parameters:**

- **sequence** (*Union*[*Tuple*[*Any*, *...*], *List*[*Any*], *Iterable*[*Any*]]) – A flat sequence of elements to be restructured
- **profile** (*XTuple*) – A nested tuple structure that defines the shape of the output

**Returns:**

A nested tuple with the same structure as profile but containing elements from sequence

**Return type:**

XTuple

**Examples:**

```python
unflatten([1, 2, 3, 4], ((0, 0), (0, 0)))  # Returns ((1, 2), (3, 4))
```

<a id="cutlass.cute.product"></a>
### `cutlass.cute.product`

```python
cutlass.cute.product( a: cutlass.cute.typing.IntTuple | cutlass.cute.typing.Shape, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.IntTuple
```

<a id="cutlass.cute.product_like"></a>
### `cutlass.cute.product_like`

```python
cutlass.cute.product_like( a: cutlass.cute.typing.IntTuple, target_profile: cutlass.cute.typing.XTuple, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.IntTuple
```

Return product of the given IntTuple or Shape at leaves of target\_profile.

This function computes products according to the structure defined by target\_profile.

**Parameters:**

- **a** (*IntTuple* *or* *Shape*) – The input tuple or shape
- **target\_profile** (*XTuple*) – The profile that guides how products are computed
- **loc** (*optional*) – Source location for MLIR, defaults to None
- **ip** (*optional*) – Insertion point, defaults to None

**Returns:**

The resulting tuple with products computed according to target\_profile

**Return type:**

IntTuple or Shape

**Raises:**

- **TypeError** – If inputs have incompatible types
- **ValueError** – If inputs have incompatible shapes

<a id="cutlass.cute.product_each"></a>
### `cutlass.cute.product_each`

```python
cutlass.cute.product_each( a: cutlass.cute.typing.IntTuple, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.IntTuple
```

<a id="cutlass.cute.elem_less"></a>
### `cutlass.cute.elem_less`

```python
cutlass.cute.elem_less( lhs: cutlass.cute.typing.Shape | cutlass.cute.typing.IntTuple | cutlass.cute.typing.Coord, rhs: cutlass.cute.typing.Shape | cutlass.cute.typing.IntTuple | cutlass.cute.typing.Coord, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Boolean
```

<a id="cutlass.cute.tuple_cat"></a>
### `cutlass.cute.tuple_cat`

```python
cutlass.cute.tuple_cat( *tuples: cutlass.cute.typing.XTuple, ) → Tuple[Any, ...]
```

Concatenate multiple tuples into a single tuple.

This function takes any number of tuples and concatenates them into a single tuple.
Non-tuple arguments are treated as single-element tuples.

**Parameters:**

**tuples** (*tuple* *or* *any*) – Variable number of tuples to concatenate

**Returns:**

A single concatenated tuple

**Return type:**

tuple

**Examples:**

```python
>>> tuple_cat((1, 2), (3, 4))
(1, 2, 3, 4)
>>> tuple_cat((1,), (2, 3), (4,))
(1, 2, 3, 4)
>>> tuple_cat(1, (2, 3))
(1, 2, 3)
```

<a id="cutlass.cute.transform_apply"></a>
### `cutlass.cute.transform_apply`

```python
cutlass.cute.transform_apply( *args: cutlass.cute.typing.XTuple, f: Callable[[...], cutlass.cute.typing.XTuple], g: Callable[[...], cutlass.cute.typing.XTuple], ) → cutlass.cute.typing.XTuple
```

Transform elements of tuple(s) with f, then apply g to all results.

This function applies f to corresponding elements across input tuple(s),
then applies g to all transformed results. It mimics the C++ CuTe implementation.

Supports multiple signatures:
- transform\_apply(t, f, g): For single tuple, computes g(f(t[0]), f(t[1]), …)
- transform\_apply(t0, t1, f, g): For two tuples, computes g(f(t0[0], t1[0]), f(t0[1], t1[1]), …)
- transform\_apply(t0, t1, t2, …, f, g): For multiple tuples of same length

For non-tuple inputs, f is applied to the input(s) and g is applied to that single result.

**Parameters:**

- **args** – One or more tuples (or non-tuples) to transform
- **f** (*Callable*) – The function to apply to each element (or corresponding elements across tuples)
- **g** (*Callable*) – The function to apply to all transformed elements
- **loc** (*optional*) – Source location for MLIR, defaults to None
- **ip** (*optional*) – Insertion point, defaults to None

**Returns:**

The result of applying g to all transformed elements

**Return type:**

any

**Examples:**

```python
>>> transform_apply((1, 2, 3), f=lambda x: x * 2, g=lambda *args: sum(args))
12  # (1*2 + 2*2 + 3*2) = 12
>>> transform_apply((1, 2), f=lambda x: (x, x+1), g=tuple_cat)
(1, 2, 2, 3)
>>> transform_apply((1, 2), (3, 4), f=lambda x, y: x + y, g=lambda *args: args)
(4, 6)
```

<a id="cutlass.cute.filter_tuple"></a>
### `cutlass.cute.filter_tuple`

```python
cutlass.cute.filter_tuple( *args: cutlass.cute.typing.XTuple, f: Callable[[...], Tuple[Any, ...]], ) → Tuple[Any, ...]
```

Filter and flatten tuple elements by applying a function.

The function f should return tuples, which are then concatenated together
to produce the final result. This is useful for filtering and transforming
tuple structures in a single pass.

**Parameters:**

- **t** (*Union*[*tuple*, *ir.Value*, *int*]) – The tuple to filter
- **f** (*Callable*) – The function to apply to each element of t
- **loc** (*optional*) – Source location for MLIR, defaults to None
- **ip** (*optional*) – Insertion point, defaults to None

**Returns:**

A concatenated tuple of all results

**Return type:**

tuple

**Examples:**

```python
>>> # Keep only even numbers, wrapped in tuples
>>> filter_tuple((1, 2, 3, 4), lambda x: (x,) if x % 2 == 0 else ())
(2, 4)
>>> # Duplicate each element
>>> filter_tuple((1, 2, 3), lambda x: (x, x))
(1, 1, 2, 2, 3, 3)
```

<a id="cutlass.cute.unwrap"></a>
### `cutlass.cute.unwrap`

```python
cutlass.cute.unwrap(x: cutlass.cute.typing.XTuple) → cutlass.cute.typing.XTuple
```

Unwraps the input tuple if it is a single-element tuple, otherwise returns the input.

Example:
>>> unwrap((1,))
1
>>> unwrap(((1, 2, 3),))
(1, 2, 3)
>>> unwrap((1, 2, 3))
(1, 2, 3)

<a id="cutlass.cute.wrap"></a>
### `cutlass.cute.wrap`

```python
cutlass.cute.wrap(x: cutlass.cute.typing.XTuple) → Tuple[Any, ...]
```

Wraps the input into a tuple if not a tuple.

<a id="cutlass.cute.domain_offset"></a>
### `cutlass.cute.domain_offset`

```python
cutlass.cute.domain_offset( coord: cutlass.cute.typing.Coord, tensor: cutlass.cute.typing.Tensor, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Tensor
```

Offset the tensor domain by the given coordinate.

This function creates a new tensor by offsetting the iterator/pointer of the input tensor
by the amount corresponding to the given coordinate in its layout.

**Parameters:**

- **coord** (*Coord*) – The coordinate offset to apply
- **tensor** (*Tensor*) – The source tensor to offset
- **loc** (*Optional*[*Location*]) – Source location for MLIR operation tracking, defaults to None
- **ip** (*Optional*[*InsertionPoint*]) – Insertion point for MLIR operation, defaults to None

**Returns:**

A new tensor with the offset iterator

**Return type:**

Tensor

**Raises:**

**ValueError** – If the tensor type doesn’t support domain offsetting

**Examples:**

```python
# Create a tensor with a row-major layout
ptr = make_ptr(Float32, base_ptr, AddressSpace.gmem)
layout = make_layout((64, 128), stride=(128, 1))
tensor = make_tensor(ptr, layout)

# Offset by coordinate (3, 5)
offset_tensor = domain_offset((3, 5), tensor)
# offset_tensor now points to element at (3, 5)
```

<a id="cutlass.cute.make_atom"></a>
### `cutlass.cute.make_atom`

```python
cutlass.cute.make_atom( ty: cutlass._mlir.ir.Type, values: List[cutlass._mlir.ir.Value] | None = None, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass._mlir.ir.OpResult
```

This is a wrapper around the \_cute\_ir.make\_atom operation, providing default value for the values argument.

<a id="cutlass.cute.make_mma_atom"></a>
### `cutlass.cute.make_mma_atom`

```python
cutlass.cute.make_mma_atom( op: MmaOp, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, **kwargs: Any, ) → MmaAtom
```

Makes an MMA Atom from an MMA Operation.

This function creates an MMA Atom from a given MMA Operation. Arbitrary kw arguments can be
provided for Op-specific additional parameters. They are not used as of today.

**Parameters:**

**op** (*MmaOp*) – The MMA Operation to construct an Atom for

**Returns:**

The MMA Atom

**Return type:**

[MmaAtom](#cutlass.cute.MmaAtom)

<a id="cutlass.cute.make_tiled_mma"></a>
### `cutlass.cute.make_tiled_mma`

```python
cutlass.cute.make_tiled_mma( op_or_atom: Op | MmaAtom, atom_layout_mnk: cutlass.cute.typing.Layout | Tuple[Any, ...] = (1, 1, 1), permutation_mnk: cutlass.cute.typing.Tiler | None = None, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, **kwargs: Any, ) → TiledMma
```

Makes a tiled MMA from an MMA Operation or an MMA Atom.

**Parameters:**

- **op\_or\_atom** (*Union*[*Op*, [*MmaAtom*](#cutlass.cute.MmaAtom)]) – The MMA Operation or Atom
- **atom\_layout\_mnk** (*Layout*) – A Layout describing the tiling of Atom across threads
- **permutation\_mnk** (*Tiler*) – A permutation Tiler describing the tiling of Atom across values including any permutation of such tiling

**Returns:**

The resulting tiled MMA

**Return type:**

[TiledMma](#cutlass.cute.TiledMma)

<a id="cutlass.cute.make_copy_atom"></a>
### `cutlass.cute.make_copy_atom`

```python
cutlass.cute.make_copy_atom( op: CopyOp, copy_internal_type: Type[cutlass.cute.typing.Numeric], *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, **kwargs: Any, ) → CopyAtom
```

Makes a Copy Atom from a Copy Operation.

This function creates a Copy Atom from a given Copy Operation. Arbitrary kw arguments can be
provided for Op-specific additional parameters.

Example:

```python
op = cute.nvgpu.CopyUniversalOp()
atom = cute.make_copy_atom(op, tensor_dtype, num_bits_per_copy=64)
```

**Parameters:**

- **op** (*CopyOp*) – The Copy Operation to construct an Atom for
- **copy\_internal\_type** (*Type*[*Numeric*]) – Element type used to construct the source/destination layouts in units of tensor elements

**Returns:**

The Copy Atom

**Return type:**

[CopyAtom](#cutlass.cute.CopyAtom)

<a id="cutlass.cute.make_tiled_copy_tv"></a>
### `cutlass.cute.make_tiled_copy_tv`

```python
cutlass.cute.make_tiled_copy_tv( atom: CopyAtom, thr_layout: cutlass.cute.typing.Layout, val_layout: cutlass.cute.typing.Layout, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → TiledCopy
```

Create a tiled copy given separate thread and value layouts.

A TV partitioner is inferred based on the input layouts. The input thread layout
must be compact.

**Parameters:**

- **atom** ([*CopyAtom*](#cutlass.cute.CopyAtom)) – Copy atom
- **thr\_layout** (*Layout*) – Layout mapping from `(TileM,TileN)` coordinates to thread IDs (must be compact)
- **val\_layout** (*Layout*) – Layout mapping from `(ValueM,ValueN)` coordinates to value IDs
- **loc** (*Optional*[*Location*], *optional*) – Source location for MLIR, defaults to None
- **ip** (*Optional*[*InsertionPoint*], *optional*) – Insertion point, defaults to None

**Returns:**

A tiled copy for the partitioner

**Return type:**

[TiledCopy](#cutlass.cute.TiledCopy)

<a id="cutlass.cute.make_tiled_copy"></a>
### `cutlass.cute.make_tiled_copy`

```python
cutlass.cute.make_tiled_copy( atom: CopyAtom, layout_tv: cutlass.cute.typing.Layout, tiler_mn: Any, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → TiledCopy
```

Create a tiled type given a TV partitioner and tiler.

**Parameters:**

- **atom** ([*CopyAtom*](#cutlass.cute.CopyAtom)) – Copy atom, e.g. smit\_copy and simt\_async\_copy, tma\_load, etc.
- **layout\_tv** (*Layout*) – Thread-value layout
- **tiler\_mn** (*Tiler*) – Tile size
- **loc** (*Optional*[*Location*], *optional*) – Source location for MLIR, defaults to None
- **ip** (*Optional*[*InsertionPoint*], *optional*) – Insertion point, defaults to None

**Returns:**

A tiled copy for the partitioner

**Return type:**

[TiledCopy](#cutlass.cute.TiledCopy)

<a id="cutlass.cute.make_tiled_copy_S"></a>
### `cutlass.cute.make_tiled_copy_S`

```python
cutlass.cute.make_tiled_copy_S( atom: CopyAtom, tiled_copy: TiledCopy, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → TiledCopy
```

Create a tiled copy out of the copy\_atom that matches the Src-Layout of tiled\_copy.

**Parameters:**

- **atom** ([*CopyAtom*](#cutlass.cute.CopyAtom)) – Copy atom
- **tiled\_copy** ([*TiledCopy*](#cutlass.cute.TiledCopy)) – Tiled copy
- **loc** (*Optional*[*Location*], *optional*) – Source location for MLIR, defaults to None
- **ip** (*Optional*[*InsertionPoint*], *optional*) – Insertion point, defaults to None

**Returns:**

A tiled copy for the partitioner

**Return type:**

[TiledCopy](#cutlass.cute.TiledCopy)

<a id="cutlass.cute.make_tiled_copy_D"></a>
### `cutlass.cute.make_tiled_copy_D`

```python
cutlass.cute.make_tiled_copy_D( atom: CopyAtom, tiled_copy: TiledCopy, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → TiledCopy
```

Create a tiled copy out of the copy\_atom that matches the Dst-Layout of tiled\_copy.

**Parameters:**

- **atom** ([*CopyAtom*](#cutlass.cute.CopyAtom)) – Copy atom
- **tiled\_copy** ([*TiledCopy*](#cutlass.cute.TiledCopy)) – Tiled copy
- **loc** (*Optional*[*Location*], *optional*) – Source location for MLIR, defaults to None
- **ip** (*Optional*[*InsertionPoint*], *optional*) – Insertion point, defaults to None

**Returns:**

A tiled copy for the partitioner

**Return type:**

[TiledCopy](#cutlass.cute.TiledCopy)

<a id="cutlass.cute.make_tiled_copy_A"></a>
### `cutlass.cute.make_tiled_copy_A`

```python
cutlass.cute.make_tiled_copy_A( atom: CopyAtom, tiled_mma: TiledMma, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → TiledCopy
```

Create a tiled copy out of the copy\_atom that matches the A-Layout of tiled\_mma.

**Parameters:**

- **atom** ([*CopyAtom*](#cutlass.cute.CopyAtom)) – Copy atom
- **tiled\_mma** ([*TiledMma*](#cutlass.cute.TiledMma)) – Tiled MMA
- **loc** (*Optional*[*Location*], *optional*) – Source location for MLIR, defaults to None
- **ip** (*Optional*[*InsertionPoint*], *optional*) – Insertion point, defaults to None

**Returns:**

A tiled copy for the partitioner

**Return type:**

[TiledCopy](#cutlass.cute.TiledCopy)

<a id="cutlass.cute.make_tiled_copy_B"></a>
### `cutlass.cute.make_tiled_copy_B`

```python
cutlass.cute.make_tiled_copy_B( atom: CopyAtom, tiled_mma: TiledMma, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → TiledCopy
```

Create a tiled copy out of the copy\_atom that matches the B-Layout of tiled\_mma.

**Parameters:**

- **atom** ([*CopyAtom*](#cutlass.cute.CopyAtom)) – Copy atom
- **tiled\_mma** ([*TiledMma*](#cutlass.cute.TiledMma)) – Tiled MMA
- **loc** (*Optional*[*Location*], *optional*) – Source location for MLIR, defaults to None
- **ip** (*Optional*[*InsertionPoint*], *optional*) – Insertion point, defaults to None

**Returns:**

A tiled copy for the partitioner

**Return type:**

[TiledCopy](#cutlass.cute.TiledCopy)

<a id="cutlass.cute.make_tiled_copy_C"></a>
### `cutlass.cute.make_tiled_copy_C`

```python
cutlass.cute.make_tiled_copy_C( atom: CopyAtom, tiled_mma: TiledMma, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → TiledCopy
```

Create a tiled copy out of the copy\_atom that matches the C-Layout of tiled\_mma.

**Parameters:**

- **atom** ([*CopyAtom*](#cutlass.cute.CopyAtom)) – Copy atom
- **tiled\_mma** ([*TiledMma*](#cutlass.cute.TiledMma)) – Tiled MMA
- **loc** (*Optional*[*Location*], *optional*) – Source location for MLIR, defaults to None
- **ip** (*Optional*[*InsertionPoint*], *optional*) – Insertion point, defaults to None

**Returns:**

A tiled copy for the partitioner

**Return type:**

[TiledCopy](#cutlass.cute.TiledCopy)

<a id="cutlass.cute.make_tiled_copy_C_atom"></a>
### `cutlass.cute.make_tiled_copy_C_atom`

```python
cutlass.cute.make_tiled_copy_C_atom( atom: CopyAtom, mma: TiledMma, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → TiledCopy
```

Create the smallest tiled copy that can retile LayoutC\_TV for use with pipelined epilogues with subtiled stores.

**Parameters:**

- **atom** ([*CopyAtom*](#cutlass.cute.CopyAtom)) – Copy atom
- **mma** ([*TiledMma*](#cutlass.cute.TiledMma)) – Tiled MMA
- **loc** (*Optional*[*Location*], *optional*) – Source location for MLIR, defaults to None
- **ip** (*Optional*[*InsertionPoint*], *optional*) – Insertion point, defaults to None

**Returns:**

A tiled copy for partitioner

**Return type:**

[TiledCopy](#cutlass.cute.TiledCopy)

**Raises:**

**ValueError** – If the number value of CopyAtom’s source layout is greater than the size of TiledMma’s LayoutC\_TV

<a id="cutlass.cute.make_cotiled_copy"></a>
### `cutlass.cute.make_cotiled_copy`

```python
cutlass.cute.make_cotiled_copy( atom: CopyAtom, atom_layout_tv: cutlass.cute.typing.Layout, data_layout: cutlass.cute.typing.Layout, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → TiledCopy
```

Produce a TiledCopy from thread and value offset maps.
The TV Layout maps threads and values to the codomain of the data\_layout.
It is verified that the intended codomain is valid within data\_layout.
Useful when threads and values don’t care about owning specific coordinates, but
care more about the vector-width and offsets between them.

**Parameters:**

- **atom** (*copy atom*, *e.g. simt\_copy and simt\_async\_copy*, *tgen05.st*, *etc.*)
- **atom\_layout\_tv** (*(**tid*, *vid**)* *-> data addr*)
- **data\_layout** (*data coord -> data addr*)
- **loc** (*source location for mlir* *(**optional**)*)
- **ip** (*insertion point* *(**optional**)*)

**Returns:**

A tuple of A tiled copy and atom

**Return type:**

tiled\_copy

<a id="cutlass.cute.copy_atom_call"></a>
### `cutlass.cute.copy_atom_call`

```python
cutlass.cute.copy_atom_call( atom: CopyAtom, src: cutlass.cute.typing.Tensor | List[cutlass.cute.typing.Tensor] | Tuple[cutlass.cute.typing.Tensor, ...], dst: cutlass.cute.typing.Tensor | List[cutlass.cute.typing.Tensor] | Tuple[cutlass.cute.typing.Tensor, ...], *, pred: cutlass.cute.typing.Tensor | None = None, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, **kwargs: Any, ) → None
```

Execute a single copy atom operation.

The copy\_atom\_call operation executes a copy atom with the given operands.
Source and destination tensors have layout profile `(V)`.

The `V-mode` represents either:

- A singular mode directly consumable by the provided Copy Atom
- A composite mode requiring recursive decomposition, structured as `(V, Rest...)`,

For src/dst layout like `(V, Rest...)`, the layout profile of `pred` must match `(Rest...)`.

> - Certain Atoms may require additional operation-specific keyword arguments.
> - Current implementation limits `V-mode` rank to 2 or less. Support for higher ranks is planned
>   for future releases.

Both `src` and `dst` operands are variadic, containing a variable number of tensors:

- For regular copy, `src` and `dst` each contain a single tensor.
- For copy with auxiliary operands, they contain the main tensor followed by
  auxiliary tensors. For example:

  - For static load to tensor memory, `dst` = [data, stat].
  - For SPARSIFY, `dst` = [data, metadata].
  - For TMA gather4, `src` = [coord0, coord1, coord2, coord3] (four 2D coordinate tensors).
  - For TMA scatter4, `dst` = [coord0, coord1, coord2, coord3] (four 2D coordinate tensors).

**Parameters:**

- **atom** ([*CopyAtom*](#cutlass.cute.CopyAtom)) – Copy atom specifying the transfer operation
- **src** (*Union*[*Tensor*, *List*[*Tensor*], *Tuple*[*Tensor*, *...*]]) – Source tensor(s) with layout profile `(V)`. Can be a single Tensor
  or a list/tuple of Tensors for operations with auxiliary source operands.
- **dst** (*Union*[*Tensor*, *List*[*Tensor*], *Tuple*[*Tensor*, *...*]]) – Destination tensor(s) with layout profile `(V)`. Can be a single Tensor
  or a list/tuple of Tensors for operations with auxiliary destination operands.
- **pred** (*Optional*[*Tensor*], *optional*) – Optional predication tensor for conditional transfers, defaults to None
- **loc** (*Any*, *optional*) – Source location information, defaults to None
- **ip** (*Any*, *optional*) – Insertion point, defaults to None
- **kwargs** (*Dict*[*str*, *Any*]) – Additional copy atom specific arguments

**Raises:**

**TypeError** – If source and destination element type bit widths differ

**Returns:**

None

**Return type:**

None

**Examples**:

```python
# Regular copy atom operation
cute.copy_atom_call(copy_atom, src, dst)

# Predicated copy atom operation
cute.copy_atom_call(copy_atom, src, dst, pred=pred)

# Static load to tensor memory: load with row-wise reduction (MAX, MIN, MAXABS, MINABS)
cute.copy_atom_call(loadtm_stat_atom, src, [data, stat])

# TMA gather4: combine four 2D coordinate tensors into single destination
cute.copy_atom_call(tma_gather4_atom, [coord0, coord1, coord2, coord3], dst)
```

<a id="cutlass.cute.mma_atom_call"></a>
### `cutlass.cute.mma_atom_call`

```python
cutlass.cute.mma_atom_call( atom: MmaAtom, d: cutlass.cute.typing.Tensor, a: cutlass.cute.typing.Tensor | List[cutlass.cute.typing.Tensor] | Tuple[cutlass.cute.typing.Tensor, ...], b: cutlass.cute.typing.Tensor | List[cutlass.cute.typing.Tensor] | Tuple[cutlass.cute.typing.Tensor, ...], c: cutlass.cute.typing.Tensor, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, **kwargs: Any, ) → None
```

Execute a single MMA atom operation.

The mma\_atom\_call operation executes an MMA atom with the given operands.
This performs a matrix multiplication and accumulation operation:
D = A \* B + C

Note: The tensors ‘d’, ‘a’, ‘b’, and ‘c’ must only have a single fragment.

The operands a and b are variadic, each containing a variable number of tensors:

- For regular MMA, a and b contain the MMA A and B tensors respectively.
- For MMA with auxiliary operands, a and b contain the MMA A and B tensors followed by
  their respective auxiliary tensors.

Auxiliary operands examples:

- For BlockScaledMMA, a = [A, SFA] and b = [B, SFB].
- For SparseMMA, a = [A, E] and b = [B].
- For BlockScaledSparseMMA, a = [A, SFA, E] and b = [B, SFB].

Runtime keyword arguments in `kwargs` are forwarded to the atom trait’s `unpack` logic.
For SM100 tcgen05 MMA atoms, you can pass `disable_output_lane` to control
per-lane output writes through `tcgen05.mma.disable_output_lane` lowering.
The expected mask length is 4 lanes for `cta_group::1` and 8 lanes for
`cta_group::2`.

**Parameters:**

- **atom** ([*MmaAtom*](#cutlass.cute.MmaAtom)) – The MMA atom to execute
- **d** (*Tensor*) – Destination tensor (output accumulator)
- **a** (*Union*[*Tensor*, *List*[*Tensor*], *Tuple*[*Tensor*, *...*]]) – A tensor or list of tensors containing the MMA A tensor and optional auxiliary tensors
- **b** (*Union*[*Tensor*, *List*[*Tensor*], *Tuple*[*Tensor*, *...*]]) – B tensor or list of tensors containing the MMA B tensor and optional auxiliary tensors
- **c** (*Tensor*) – Input accumulator tensor
- **loc** (*Optional*[*Location*], *optional*) – Source location for MLIR, defaults to None
- **ip** (*Optional*[*InsertionPoint*], *optional*) – Insertion point, defaults to None

Examples:

```python
# Regular MMA atom call
cute.mma_atom_call(mma_atom, d_tensor, a_tensor, b_tensor, c_tensor)

# Block-scaled MMA atom call
cute.mma_atom_call(mma_atom, d_tensor, [a_tensor, sfa_tensor],
                  [b_tensor, sfb_tensor], c_tensor)
```

<a id="cutlass.cute.basic_copy"></a>
### `cutlass.cute.basic_copy`

```python
cutlass.cute.basic_copy( src: cutlass.cute.typing.Tensor, dst: cutlass.cute.typing.Tensor, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

Performs a basic element-wise copy.

This functions **assumes** the following pre-conditions:
1. size(src) == size(dst)

When the src and dst shapes are static, the pre-conditions are actually verified and the
element-wise loop is fully unrolled.

**Parameters:**

- **src** (*Tensor*) – Source tensor
- **dst** (*Tensor*) – Destination tensor
- **loc** (*Optional*[*Location*], *optional*) – Source location for MLIR, defaults to None
- **ip** (*Optional*[*InsertionPoint*], *optional*) – Insertion point, defaults to None

<a id="cutlass.cute.basic_copy_if"></a>
### `cutlass.cute.basic_copy_if`

```python
cutlass.cute.basic_copy_if( pred: cutlass.cute.typing.Tensor, src: cutlass.cute.typing.Tensor, dst: cutlass.cute.typing.Tensor, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

Performs a basic predicated element-wise copy.

This functions **assumes** the following pre-conditions:
1. size(src) == size(dst)
2. size(src) == size(pred)

When all shapes are static, the pre-conditions are actually verified and the element-wise loop
is fully unrolled.

<a id="cutlass.cute.autovec_copy"></a>
### `cutlass.cute.autovec_copy`

```python
cutlass.cute.autovec_copy( src: cutlass.cute.typing.Tensor, dst: cutlass.cute.typing.Tensor, *, l1c_evict_priority: CacheEvictionPriority = cutlass._mlir.dialects.cute.CacheEvictionPriority.EVICT_NORMAL, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

Auto-vectorization SIMT copy policy.

Given a source and destination tensors that are statically shaped, this policy
figures out the largest safe vector width that the copy instruction can take
and performs the copy. Any extra memory attributes are forwarded to the specialized
copy op.

<a id="cutlass.cute.copy"></a>
### `cutlass.cute.copy`

```python
cutlass.cute.copy( atom: CopyAtom, src: cutlass.cute.typing.Tensor | List[cutlass.cute.typing.Tensor] | Tuple[cutlass.cute.typing.Tensor, ...], dst: cutlass.cute.typing.Tensor | List[cutlass.cute.typing.Tensor] | Tuple[cutlass.cute.typing.Tensor, ...], *, pred: cutlass.cute.typing.Tensor | None = None, unroll_factor: int | None = None, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, **kwargs: Any, ) → None
```

Facilitates data transfer between two tensors conforming to layout profile `(V, Rest...)`.

**Parameters:**

- **atom** ([*CopyAtom*](#cutlass.cute.CopyAtom)) – Copy atom specifying the transfer operation
- **src** (*Union*[*Tensor*, *List*[*Tensor*], *Tuple*[*Tensor*, *...*]]) – Source tensor or list of tensors with layout profile `(V, Rest...)`
- **dst** (*Union*[*Tensor*, *List*[*Tensor*], *Tuple*[*Tensor*, *...*]]) – Destination tensor or list of tensors with layout profile `(V, Rest...)`
- **pred** (*Optional*[*Tensor*], *optional*) – Optional predication tensor for conditional transfers, defaults to None
- **unroll\_factor** (*Optional*[*int*], *optional*) – Optional unroll count for loop over Rest… modes, defaults to None for fully unroll when Rest… modes are static
- **loc** (*Any*, *optional*) – Source location information, defaults to None
- **ip** (*Any*, *optional*) – Insertion point, defaults to None
- **kwargs** (*Dict*[*str*, *Any*]) – Additional copy atom specific arguments

**Raises:**

- **TypeError** – If source and destination element type bit widths differ
- **ValueError** – If source and destination ranks differ
- **ValueError** – If source and destination mode-1 sizes differ
- **NotImplementedError** – If `V-mode` rank exceeds 2

**Returns:**

None

**Return type:**

None

The `V-mode` represents either:

- A singular mode directly consumable by the provided Copy Atom
- A composite mode requiring recursive decomposition, structured as `(V, Rest...)`,
  and src/dst layout like `((V, Rest...), Rest...)`

The algorithm recursively processes the `V-mode`, decomposing it until reaching the minimum granularity
compatible with the provided Copy Atom’s requirements.

Source and destination tensors must be partitioned in accordance with the Copy Atom specifications.
Post-partitioning, both tensors will exhibit a `(V, Rest...)` layout profile.

The operands src and dst are variadic, each containing a variable number of tensors:

- For regular copy, src and dst contain single source and destination tensors respectively.
- For copy with auxiliary operands, src and dst contain the primary tensors followed by
  their respective auxiliary tensors.

**Precondition:** The size of mode 1 must be equal for both source and destination tensors:
`size(src, mode=[1]) == size(dst, mode=[1])`

**Examples**:

TMA copy operation with multicast functionality:

```python
cute.copy(tma_atom, src, dst, tma_bar_ptr=mbar_ptr, mcast_mask=mask, cache_policy=policy)
```

Optional predication is supported through an additional tensor parameter. For partitioned tensors with
logical profile `((ATOM_V,ATOM_REST),REST,...)`, the predication tensor must maintain profile
compatibility with `(ATOM_REST,REST,...)`.

For Copy Atoms requiring single-threaded execution, thread election is managed automatically by the
copy operation. External thread selection mechanisms are not necessary.

> **Note**
>
> - Certain Atoms may require additional operation-specific keyword arguments.
> - Current implementation limits `V-mode` rank to 2 or less. Support for higher ranks is planned
>   for future releases.

<a id="cutlass.cute.prefetch"></a>
### `cutlass.cute.prefetch`

```python
cutlass.cute.prefetch( atom: CopyAtom, src: cutlass.cute.typing.Tensor | List[cutlass.cute.typing.Tensor] | Tuple[cutlass.cute.typing.Tensor, ...], *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

The Prefetch algorithm.

The “prefetch” expects source tensors to be partitioned according to the provided Copy Atom.
Prefetch is used for loading tensors from global memory to L2.

Prefetch accepts Copy Atom but not all are allowed. Currently, only supports TMA prefetch.

For standard TMA modes (tiled, im2col), pass a single GMEM tensor:

```python
cute.prefetch(tma_prefetch, tAgA)
```

For 2D `tile::gather4` mode, pass a list `[data_tensor, gmem_index_tensor]`
(mirrors `cute.copy()` for the same mode); see `cute.nvgpu.cpasync.tma_partition()`
for the gather4 layout conventions:

```python
cute.prefetch(tma_atom, [tAgA, tAgI])
```

Pass the whole multi-stage partitioned tensors — the lowering’s internal loop walks every
rest-dimension entry, so a single `cute.prefetch` call covers every stage and lives outside
any per-stage loop.

For Copy Atoms that require single-threaded execution, the copy op automatically handles thread
election internally. Manual thread selection is not required in such cases.

<a id="cutlass.cute.gemm"></a>
### `cutlass.cute.gemm`

```python
cutlass.cute.gemm( atom: MmaAtom, d: cutlass.cute.typing.Tensor, a: cutlass.cute.typing.Tensor | List[cutlass.cute.typing.Tensor] | Tuple[cutlass.cute.typing.Tensor, ...], b: cutlass.cute.typing.Tensor | List[cutlass.cute.typing.Tensor] | Tuple[cutlass.cute.typing.Tensor, ...], c: cutlass.cute.typing.Tensor, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, **kwargs: Any, ) → None
```

The GEMM algorithm.

Computes `D <- A * B + C` where `C` and `D` can alias. Note that some MMA Atoms (e.g.
warpgroup-wide or tcgen05 MMAs) require manually setting an “accumulate” boolean field.

All tensors must be partitioned according to the provided MMA Atom.

For MMA Atoms that require single-threaded execution, the gemm op automatically handles thread
election internally. Manual thread selection is not required in such cases.

Following dispatch rules are supported:

- Dispatch [1]: (V) x (V) => (V) => (V,1,1) x (V,1,1) => (V,1,1)
- Dispatch [2]: (M) x (N) => (M,N) => (1,M,1) x (1,N,1) => (1,M,N)
- Dispatch [3]: (M,K) x (N,K) => (M,N) => (1,M,K) x (1,N,K) => (1,M,N)
- Dispatch [4]: (V,M) x (V,N) => (V,M,N) => (V,M,1) x (V,N,1) => (V,M,N)
- Dispatch [5]: (V,M,K) x (V,N,K) => (V,M,N)

The operands a and b are variadic, each containing a variable number of tensors:

- For regular GEMM, a and b contain the GEMM A and B tensors respectively.
- For GEMM with auxiliary operands, a and b contain the GEMM A and B tensors followed by
  their respective auxiliary tensors.

Auxiliary operands examples:

- For BlockScaledGemm, a = [A, SFA] and b = [B, SFB].
- For SparseGemm, a = [A, E] and b = [B].
- For BlockScaledSparseGemm, a = [A, SFA, E] and b = [B, SFB].

Runtime keyword arguments in `kwargs` are forwarded to the underlying MMA atom trait.
For SM100 tcgen05 MMA atoms, `disable_output_lane` provides a per-lane
write-disable mask for `tcgen05.mma.disable_output_lane` lowering.
The expected lane count is 4 for `cta_group::1` and 8 for `cta_group::2`.

**Parameters:**

- **atom** ([*MmaAtom*](#cutlass.cute.MmaAtom)) – MMA atom
- **d** (*Tensor*) – Destination tensor (output accumulator)
- **a** (*Union*[*Tensor*, *List*[*Tensor*], *Tuple*[*Tensor*, *...*]]) – A tensor or list of tensors containing the GEMM A tensor and optional auxiliary tensors
- **b** (*Union*[*Tensor*, *List*[*Tensor*], *Tuple*[*Tensor*, *...*]]) – B tensor or list of tensors containing the GEMM B tensor and optional auxiliary tensors
- **c** (*Tensor*) – Input accumulator tensor
- **loc** (*Optional*[*Location*], *optional*) – Source location for MLIR, defaults to None
- **ip** (*Optional*[*InsertionPoint*], *optional*) – Insertion point for MLIR, defaults to None
- **kwargs** (*dict*) – Additional keyword arguments

**Returns:**

None

**Return type:**

None

<a id="cutlass.cute.full"></a>
### `cutlass.cute.full`

```python
cutlass.cute.full( shape: cutlass.cute.typing.Shape, fill_value: cutlass._mlir.ir.Value | int | float | bool | cutlass.cute.typing.Numeric, dtype: Type[cutlass.cute.typing.Numeric], *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → TensorSSA
```

Return a new TensorSSA of given shape and type, filled with fill\_value.

**Parameters:**

- **shape** (*tuple*) – Shape of the new tensor.
- **fill\_value** (*scalar*) – Value to fill the tensor with.
- **dtype** (*Type*[*Numeric*]) – Data type of the tensor.

**Returns:**

Tensor of fill\_value with the specified shape and dtype.

**Return type:**

[TensorSSA](#cutlass.cute.TensorSSA)

<a id="cutlass.cute.full_like"></a>
### `cutlass.cute.full_like`

```python
cutlass.cute.full_like( a: TensorSSA | cutlass.cute.typing.Tensor, fill_value: object, dtype: None | Type[cutlass.cute.typing.Numeric] = None, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → TensorSSA
```

Return a full TensorSSA with the same shape and type as a given array.

**Parameters:**

- **a** (*array\_like*) – The shape and data-type of a define these same attributes of the returned array.
- **fill\_value** (*array\_like*) – Fill value.
- **dtype** (*Union*[*None*, *Type*[*Numeric*]], *optional*) – Overrides the data type of the result, defaults to None

**Returns:**

Tensor of fill\_value with the same shape and type as a.

**Return type:**

[TensorSSA](#cutlass.cute.TensorSSA)

> **See also**
>
> [`empty_like()`](#cutlass.cute.empty_like): Return an empty array with shape and type of input.
> [`ones_like()`](#cutlass.cute.ones_like): Return an array of ones with shape and type of input.
> [`zeros_like()`](#cutlass.cute.zeros_like): Return an array of zeros with shape and type of input.
> [`full()`](#cutlass.cute.full): Return a new array of given shape filled with value.

**Examples:**

```python
frg = cute.make_rmem_tensor((2, 3), Float32)
a = frg.load()
b = cute.full_like(a, 1.0)
```

<a id="cutlass.cute.empty_like"></a>
### `cutlass.cute.empty_like`

```python
cutlass.cute.empty_like( a: TensorSSA | cutlass.cute.typing.Tensor, dtype: None | Type[cutlass.cute.typing.Numeric] = None, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → TensorSSA
```

Return a new TensorSSA with the same shape and type as a given array, without initializing entries.

**Parameters:**

- **a** ([*TensorSSA*](#cutlass.cute.TensorSSA)) – The shape and data-type of a define these same attributes of the returned array.
- **dtype** (*Type*[*Numeric*], *optional*) – Overrides the data type of the result, defaults to None

**Returns:**

Uninitialized tensor with the same shape and type (unless overridden) as a.

**Return type:**

[TensorSSA](#cutlass.cute.TensorSSA)

<a id="cutlass.cute.ones_like"></a>
### `cutlass.cute.ones_like`

```python
cutlass.cute.ones_like( a: TensorSSA | cutlass.cute.typing.Tensor, dtype: None | Type[cutlass.cute.typing.Numeric] = None, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → TensorSSA
```

Return a TensorSSA of ones with the same shape and type as a given array.

**Parameters:**

- **a** ([*TensorSSA*](#cutlass.cute.TensorSSA)) – The shape and data-type of a define these same attributes of the returned array.
- **dtype** (*Type*[*Numeric*], *optional*) – Overrides the data type of the result, defaults to None

**Returns:**

Tensor of ones with the same shape and type (unless overridden) as a.

**Return type:**

[TensorSSA](#cutlass.cute.TensorSSA)

<a id="cutlass.cute.zeros_like"></a>
### `cutlass.cute.zeros_like`

```python
cutlass.cute.zeros_like( a: TensorSSA | cutlass.cute.typing.Tensor, dtype: None | Type[cutlass.cute.typing.Numeric] = None, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → TensorSSA
```

Return a TensorSSA of zeros with the same shape and type as a given array.

**Parameters:**

- **a** ([*TensorSSA*](#cutlass.cute.TensorSSA)) – The shape and data-type of a define these same attributes of the returned array.
- **dtype** (*Type*[*Numeric*], *optional*) – Overrides the data type of the result, defaults to None

**Returns:**

Tensor of zeros with the same shape and type (unless overridden) as a.

**Return type:**

[TensorSSA](#cutlass.cute.TensorSSA)

<a id="cutlass.cute.where"></a>
### `cutlass.cute.where`

```python
cutlass.cute.where( cond: TensorSSA, x: TensorSSA | cutlass.cute.typing.Numeric, y: TensorSSA | cutlass.cute.typing.Numeric, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → TensorSSA
```

Return elements chosen from x or y depending on condition; will auto broadcast x or y if needed.

**Parameters:**

- **cond** ([*TensorSSA*](#cutlass.cute.TensorSSA)) – Where True, yield x, where False, yield y.
- **x** (*Union*[[*TensorSSA*](#cutlass.cute.TensorSSA), *Numeric*]) – Values from which to choose when condition is True.
- **y** (*Union*[[*TensorSSA*](#cutlass.cute.TensorSSA), *Numeric*]) – Values from which to choose when condition is False.

**Returns:**

A tensor with elements from x where condition is True, and elements from y where condition is False.

**Return type:**

[TensorSSA](#cutlass.cute.TensorSSA)

<a id="cutlass.cute.any_"></a>
### `cutlass.cute.any_`

```python
cutlass.cute.any_( x: TensorSSA, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Boolean
```

Test whether any tensor element evaluates to True.

**Parameters:**

**x** ([*TensorSSA*](#cutlass.cute.TensorSSA)) – Input tensor.

**Returns:**

Returns a TensorSSA scalar containing True if any element of x is True, False otherwise.

**Return type:**

[TensorSSA](#cutlass.cute.TensorSSA)

<a id="cutlass.cute.all_"></a>
### `cutlass.cute.all_`

```python
cutlass.cute.all_( x: TensorSSA, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Boolean
```

Test whether all tensor elements evaluate to True.

**Parameters:**

**x** ([*TensorSSA*](#cutlass.cute.TensorSSA)) – Input tensor.

**Returns:**

Returns a TensorSSA scalar containing True if all elements of x are True, False otherwise.

**Return type:**

[TensorSSA](#cutlass.cute.TensorSSA)

<a id="cutlass.cute.union"></a>
### `cutlass.cute.union`

```python
class cutlass.cute.union(cls: type)
```

Bases: [`struct`](#cutlass.cute.struct)

Decorator to abstract C union in Python DSL.

Similar to cute.struct, but lays out objects as a union:
- All objects start at offset 0
- The alignment is the maximum alignment of all objects
- The size is the maximum size of all objects

**Usage:**

```python
# Define a union with scalar int/float elements:
@cute.union
class value_union:
    as_int : cutlass.Int32
    as_float : cutlass.Float32

@cute.union
class data_union:
    small : cutlass.Int16
    medium : cutlass.Int32
    large : cutlass.Int64

# Supports alignment for its elements:
@cute.union
class aligned_union:
    a: cute.struct.Align[cutlass.Float32, 16]
    b: cute.struct.Align[cutlass.Int32, 8]

# Statically get size and alignment:
size = data_union.__sizeof__()
align = data_union.__alignof__()

# Allocate and reference elements:
allocator = cutlass.utils.SmemAllocator()
value = allocator.allocate(data_union)

# Access union members (all at the same offset):
value.small.ptr ...
value.medium.ptr ...
value.large.ptr ...
```

**Parameters:**

**cls** – The union class with annotations.

**Returns:**

The decorated union class.

<a id="cutlass.cute.union.__init__"></a>
#### `cutlass.cute.union.__init__`

```python
__init__(cls: type) → None
```

Initializes a new cute.union decorator instance.

**Parameters:**

**cls** – The class representing the union data type.

**Raises:**

**TypeError** – If the union is empty.

<a id="cutlass.cute.union.size_in_bytes"></a>
#### `cutlass.cute.union.size_in_bytes`

```python
size_in_bytes() → int
```

Returns the size of the union in bytes.

**Returns:**

The size of the union.

<a id="cutlass.cute.FastDivmodDivisor"></a>
### `cutlass.cute.FastDivmodDivisor`

```python
class cutlass.cute.FastDivmodDivisor( divisor: cutlass.cute.typing.Integer, is_power_of_2: bool = None, )
```

Bases: `object`

First-class FastDivmod divisor with operator overloading support.

This class wraps a FastDivmod divisor and enables natural Python operator syntax.

Deprecated since version Use: [`FastDivmodDivisorV2`](#cutlass.cute.FastDivmodDivisorV2) instead. V2 additionally carries the
scalar divisor across kernel boundaries (2 MLIR values per object
instead of 1), so `.divisor` is readable inside kernels;
arithmetic is unchanged. This class keeps the legacy 1-value
serialization contract for existing integrations.

**Variables:**

- **divisor** – The original divisor value (publicly accessible)
- **\_divisor\_mlir** – The FastDivmod divisor MLIR value (internal)

**Example:**

```python
quotient, remainder = divmod(dividend, divisor)
quotient = dividend // divisor
remainder = dividend % divisor
```

<a id="cutlass.cute.FastDivmodDivisor.__init__"></a>
#### `cutlass.cute.FastDivmodDivisor.__init__`

```python
__init__( divisor: cutlass.cute.typing.Integer, is_power_of_2: bool | None = None, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

Create a FastDivmod divisor for optimized division operations.

**Parameters:**

- **divisor** – The divisor value (should be runtime-dynamic value)
- **is\_power\_of\_2** – Whether divisor is known to be a power of 2.
  Defaults to False.

<a id="cutlass.cute.FastDivmodDivisor.divisor"></a>
#### `cutlass.cute.FastDivmodDivisor.divisor`

```python
property divisor: cutlass.cute.typing.Integer
```

Get the original divisor value.

This allows users to access the divisor value that was used to create
this FastDivmodDivisor object. This is useful for passing the divisor
value to other functions or for storing it in data structures without
needing to manually track the divisor value separately.

**Returns:**

The original divisor value

**Return type:**

Integer

**Example:**

```python
batch_size = 32
batch_fdd = cute.fast_divmod_create_divisor(batch_size)
print(f"Divisor: {batch_fdd.divisor}")  # Access the divisor value
some_function(divisor=batch_fdd.divisor)  # Pass to other functions
```
> **Note**
>
> After this object crosses a kernel boundary (e.g. stored in a
> params structure passed to a `@cute.kernel`), the returned value
> still references host-side SSA and fails MLIR region isolation if
> used inside the kernel (OSS issue #3243). Use
> [`FastDivmodDivisorV2`](#cutlass.cute.FastDivmodDivisorV2) to read the divisor inside a kernel.

<a id="cutlass.cute.FastDivmodDivisor._divisor"></a>
#### `cutlass.cute.FastDivmodDivisor._divisor`

```python
property _divisor: cutlass._mlir.ir.Value
```

<a id="cutlass.cute.FastDivmodDivisorV2"></a>
### `cutlass.cute.FastDivmodDivisorV2`

```python
class cutlass.cute.FastDivmodDivisorV2( divisor: cutlass.cute.typing.Integer, is_power_of_2: bool = None, )
```

Bases: [`FastDivmodDivisor`](#cutlass.cute.FastDivmodDivisor)

FastDivmod divisor whose `.divisor` property is readable inside kernels.

Same arithmetic behavior as [`FastDivmodDivisor`](#cutlass.cute.FastDivmodDivisor) (`divmod`, `//`,
`%`), but serializes **two** MLIR values across region boundaries — the
encoded FastDivmod plus the scalar divisor — so `.divisor` resolves to
in-region SSA after the object crosses a kernel boundary (OSS issue #3243):

```python
@dataclass
class Params:
    fdd: cute.FastDivmodDivisorV2

@cute.kernel
def kernel(out: cute.Tensor, params: Params):
    out[0] = params.fdd.divisor  # OK: region-local SSA
```

[`FastDivmodDivisor`](#cutlass.cute.FastDivmodDivisor) keeps the legacy 1-value serialization contract
for backward compatibility; its `.divisor` is not readable inside a
kernel.

<a id="cutlass.cute.fast_divmod_create_divisor"></a>
### `cutlass.cute.fast_divmod_create_divisor`

```python
cutlass.cute.fast_divmod_create_divisor( divisor: cutlass.cute.typing.Integer, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → FastDivmodDivisor
```

Create a FastDivmod divisor for optimized division operations.

This function creates a FastDivmod divisor that precomputes auxiliary values
to enable fast division and modulus operations without using division instructions.

The returned FastDivmodDivisor object supports natural Python operator syntax.

**Parameters:**

**divisor** (*Integer*) – The divisor value (should be runtime-dynamic value)

**Returns:**

FastDivmodDivisor object with operator overloading support

**Return type:**

[FastDivmodDivisor](#cutlass.cute.FastDivmodDivisor)

**Example:**

```python
divisor = fast_divmod_create_divisor(batch_size)
quotient, remainder = divmod(linear_idx, divisor)
quotient = linear_idx // divisor
remainder = linear_idx % divisor
```

<a id="cutlass.cute.fast_divmod_create_divisor_v2"></a>
### `cutlass.cute.fast_divmod_create_divisor_v2`

```python
cutlass.cute.fast_divmod_create_divisor_v2( divisor: cutlass.cute.typing.Integer, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → FastDivmodDivisorV2
```

Create a FastDivmod divisor whose `.divisor` is readable inside kernels.

Behaves like [`fast_divmod_create_divisor()`](#cutlass.cute.fast_divmod_create_divisor), but the returned
[`FastDivmodDivisorV2`](#cutlass.cute.FastDivmodDivisorV2) serializes both the encoded FastDivmod and the
scalar divisor across kernel boundaries, so `.divisor` resolves to
region-local SSA inside a kernel (OSS issue #3243).

**Parameters:**

**divisor** (*Integer*) – The divisor value (should be runtime-dynamic value)

**Returns:**

FastDivmodDivisorV2 object with operator overloading support

**Return type:**

[FastDivmodDivisorV2](#cutlass.cute.FastDivmodDivisorV2)

**Example:**

```python
divisor = fast_divmod_create_divisor_v2(batch_size)
quotient, remainder = divmod(linear_idx, divisor)
d = divisor.divisor  # readable on host AND inside kernels
```

<a id="cutlass.cute.ffi"></a>
### `cutlass.cute.ffi`

```python
cutlass.cute.ffi( *, name: str | None = None, params_types: list | None = None, return_type: cutlass._mlir.ir.Type | None = None, inline: bool = True, source: str | BitCode | None = None, ) → cutlass.base_dsl.ffi.FFI
```
