<!-- source: https://tvm.apache.org/docs/tirx/api/tirx.html -->

<a id="module-tvm.tirx"></a>
# tvm.tirx

Namespace for Tensor-level IR

<a id="tvm.tirx.Buffer"></a>
### `tvm.tirx.Buffer`

```python
class tvm.tirx.Buffer(data, dtype, shape, strides, elem_offset, name, data_alignment, offset_factor, span, layout, allocated_addr)
```

Symbolic data buffer in TVM.

Buffer provide a way to represent data layout
specialization of data structure in TVM.

Do not construct directly, use `decl_buffer()` instead.
See the documentation of `decl_buffer()` for more details.

> **See also**
>
> `decl_buffer`
> :   Declare a buffer

<a id="tvm.tirx.Buffer.access_ptr"></a>
#### `tvm.tirx.Buffer.access_ptr`

```python
access_ptr(access_mask, ptr_type='handle', content_lanes=1, offset=0, extent=None)
```

Get an access pointer to the head of buffer.

This is the recommended method to get buffer data
ptress when interacting with external functions.

**Parameters:**

- **access\_mask** (int) – The access pattern MASK. Indicate whether the
  access will read or write to the data content.
- **ptr\_type** (str *or* tvm.ir.Type, *optional*) – The data type of the result pointer. Do not specify
  unless we want to cast pointer to specific type.
- **content\_lanes** (int, *optional*) – The number of lanes for the data type. This value
  is greater than one for vector types.
- **offset** (tvm.relax.Expr, *optional*) – The offset of pointer. We can use it to offset by
  the number of elements from the address of ptr.
- **extent** (tvm.relax.Expr, *optional*) – The extent of pointer.

Examples

```python
# Get access ptr for read
buffer.access_ptr("r")
# Get access ptr for read/write with bitmask
buffer.access_ptr(Buffer.READ | Buffer.WRITE)
# Get access ptr for read/write with str flag
buffer.access_ptr("rw")
# Get access ptr for read with offset
buffer.access_ptr("r", offset = 100)
# Get access ptr for read with extent
buffer.access_ptr("r", extent = 100)
```

<a id="tvm.tirx.Buffer.vload"></a>
#### `tvm.tirx.Buffer.vload`

```python
vload(begin, dtype=None, predicate=None)
```

Generate an Expr that loads dtype from begin index.

**Parameters:**

- **begin** (tvm.ir.Array *of* tvm.relax.Expr) – The beginning index in unit of Buffer.dtype
- **dtype** (str) – The data type to be loaded,
  can be vector type which have lanes that is multiple of Buffer.dtype
- **predicate** (*Optional*[tvm.relax.Expr]) – A vector mask of boolean values indicating which lanes of a vector are to be
  loaded. The number lanes of the mask must be equal to the number of lanes being loaded.

**Returns:**

**load** – The corresponding load expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.Buffer.vstore"></a>
#### `tvm.tirx.Buffer.vstore`

```python
vstore(begin, value, predicate=None)
```

Generate a Stmt that store value into begin index.

**Parameters:**

- **begin** (tvm.ir.Array *of* tvm.relax.Expr) – The beginning index in unit of Buffer.dtype
- **value** (tvm.relax.Expr) – The value to be stored.
- **predicate** (*Optional*[tvm.relax.Expr]) – A vector mask of boolean values indicating which lanes of a vector are to be
  stored. The number lanes of the mask must be equal to the number of lanes in
  value.

**Returns:**

**store** – The corresponding store stmt.

**Return type:**

[Stmt](#tvm.tirx.Stmt)

<a id="tvm.tirx.Buffer.scope"></a>
#### `tvm.tirx.Buffer.scope`

```python
scope()
```

Return the storage scope associated with this buffer.
:returns: **scope** – The storage scope associated with this buffer.
:rtype: str

<a id="tvm.tirx.Buffer.get_flattened_buffer"></a>
#### `tvm.tirx.Buffer.get_flattened_buffer`

```python
get_flattened_buffer()
```

Generate a Buffer that is a flattened version of this buffer.

**Returns:**

**flattened** – The corresponding flat buffer.

**Return type:**

[Buffer](#tvm.tirx.Buffer)

<a id="tvm.tirx.Buffer.with_allocated_addr"></a>
#### `tvm.tirx.Buffer.with_allocated_addr`

```python
with_allocated_addr(allocated_addr)
```

Return a new buffer with the allocated address.

<a id="tvm.tirx.Buffer.with_dtype"></a>
#### `tvm.tirx.Buffer.with_dtype`

```python
with_dtype(dtype)
```

Return a new buffer with the dtype.

<a id="tvm.tirx.Buffer.with_data"></a>
#### `tvm.tirx.Buffer.with_data`

```python
with_data(data)
```

Return a new buffer with the data.

<a id="tvm.tirx.Buffer.offset_of"></a>
#### `tvm.tirx.Buffer.offset_of`

```python
offset_of(indices)
```

Determine the offset of the provided indices in the flattened buffer.

**Parameters:**

**indices** (*Union*[tvm.relax.Expr, *List*[tvm.relax.Expr]]) – The indices of the element in the original buffer.

**Returns:**

**flattened\_indices** – The offset indices of the element in the flattened buffer.

**Return type:**

List[tvm.relax.Expr]

<a id="tvm.tirx.Buffer.byte_offset"></a>
#### `tvm.tirx.Buffer.byte_offset`

```python
property byte_offset
```

Get the byte offset of the buffer.

<a id="tvm.tirx.Buffer.elem_offset_of"></a>
#### `tvm.tirx.Buffer.elem_offset_of`

```python
elem_offset_of(indices, inner=True)
```

Get the element offset of the buffer at the given indices.
Note that indices subject to buffer’s layout mapping.

**Parameters:**

- **indices** (*Union*[tvm.relax.Expr, *List*[tvm.relax.Expr]]) – The indices of the element in the original buffer.
- **inner** (bool, *optional*) – If False, the offset is relative to the original buffer.
  Default is True.

**Returns:**

**offset** – The element offset of the buffer at the given indices.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.Buffer.byte_offset_of"></a>
#### `tvm.tirx.Buffer.byte_offset_of`

```python
byte_offset_of(indices, inner=True)
```

Get the byte offset of the buffer at the given indices.
Note that indices subject to buffer’s layout mapping.

**Parameters:**

- **indices** (*Union*[tvm.relax.Expr, *List*[tvm.relax.Expr]]) – The indices of the element in the original buffer.
- **inner** (bool, *optional*) – If False, the offset is relative to the original buffer.
  Default is True.

**Returns:**

**offset** – The byte offset of the buffer at the given indices.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.Buffer.is_scalar"></a>
#### `tvm.tirx.Buffer.is_scalar`

```python
is_scalar(alloc_or_decl=True)
```

Check if the buffer is a scalar.

**Parameters:**

**alloc\_or\_decl** (bool, *optional*) – Whether to consider alloc\_scalar and decl\_scalar as scalar. True for alloc\_scalar,
False for decl\_scalar.

**Returns:**

**bool**

**Return type:**

True if the buffer is a scalar, False otherwise.

<a id="tvm.tirx.Buffer.ptr_to"></a>
#### `tvm.tirx.Buffer.ptr_to`

```python
ptr_to(indices)
```

Get the pointer to the buffer at the given indices (logical indices).

Note that the bufferload inside requires LowerTIPp pass to apply the layout to get the physical indices.

<a id="tvm.tirx.Buffer.view"></a>
#### `tvm.tirx.Buffer.view`

```python
view(*args, **kwargs) → Buffer
```

Creates a new view of the buffer. (used by parser)

Supported signatures are `view(*shape, layout=None)`, where shape can contain
`-1` to indicate that the dimension size is auto-inferred, and
`view(dtype: Union[str, tvm.DataType])`.

**Returns:**

**view** – The corresponding view buffer.

**Return type:**

DeclBufferFrame

<a id="tvm.tirx.Buffer.local"></a>
#### `tvm.tirx.Buffer.local`

```python
local(*shape, layout=None) → Buffer
```

Create a thread-local view of this buffer.

When called with no shape arguments, auto-infers a 1D shape from
the layout’s non-thread component (i.e. `layout.storage().shard`).

**Parameters:**

- **shape** (tuple *of* tvm.relax.Expr) – The shape of the local view for indexing. If omitted, a 1D
  shape is computed automatically.
- **layout** (*optional*) – Override layout. If None, uses the storage layout
  (parent layout with thread axes removed).

**Returns:**

**local** – The corresponding local buffer.

**Return type:**

DeclBufferFrame

<a id="tvm.tirx.Buffer.permute"></a>
#### `tvm.tirx.Buffer.permute`

```python
permute(*dims) → Buffer
```

Permute the dimensions of the buffer.

**Parameters:**

**dims** (tuple *of* int) – The permutation of dimensions.

**Returns:**

**permuted** – The buffer with permuted dimensions.

**Return type:**

DeclBufferFrame

<a id="tvm.tirx.DataProducer"></a>
### `tvm.tirx.DataProducer`

```python
class tvm.tirx.DataProducer(*args: Any, **kwargs: Any)
```

<a id="tvm.tirx.Reduce"></a>
### `tvm.tirx.Reduce`

```python
class tvm.tirx.Reduce(combiner: CommReducer, src: list[Expr], rdom: list[IterVar], condition: Expr, value_index: int, init: list[Expr] | None = None, span: Span | None = None)
```

Reduce node.

**Parameters:**

- **combiner** ([*CommReducer*](#tvm.tirx.CommReducer)) – The combiner.
- **src** (list *of* tvm.relax.Expr) – The source expression.
- **rdom** (list *of* [*IterVar*](#tvm.tirx.IterVar)) – The iteration domain
- **condition** (tvm.relax.Expr) – The reduce condition.
- **value\_index** (int) – The value index.
- **init** (list *of* tvm.relax.Expr) – The initial value for output. This can be an int, float or ProducerLoad
- **span** (*Optional*[tvm.ir.Span]) – The location of this expression in the source code.

<a id="tvm.tirx.FloatImm"></a>
### `tvm.tirx.FloatImm`

```python
class tvm.tirx.FloatImm(dtype: str | PrimType, value: float, span: Span | None = None)
```

Float constant.

**Parameters:**

- **dtype** (str) – The data type
- **value** (float) – The constant value.
- **span** (*Optional*[tvm.ir.Span]) – The location of this expression in the source code.

<a id="tvm.tirx.IntImm"></a>
### `tvm.tirx.IntImm`

```python
class tvm.tirx.IntImm(dtype: str | PrimType, value: int, span: Span | None = None)
```

Int constant.

**Parameters:**

- **dtype** (str) – The data type
- **value** (int) – The constant value.
- **span** (*Optional*[tvm.ir.Span]) – The location of this expression in the source code.

<a id="tvm.tirx.StringImm"></a>
### `tvm.tirx.StringImm`

```python
class tvm.tirx.StringImm(value: str, span: Span | None = None)
```

String constant.

**Parameters:**

- **value** (str) – The value of the function.
- **span** (*Optional*[tvm.ir.Span]) – The location of this expression in the source code.

<a id="tvm.tirx.Cast"></a>
### `tvm.tirx.Cast`

```python
class tvm.tirx.Cast(dtype: str | PrimType, value, span: Span | None = None)
```

Cast expression.

**Parameters:**

- **dtype** (str) – The data type
- **value** (tvm.relax.Expr) – The value of the function.
- **span** (*Optional*[tvm.ir.Span]) – The location of this expression in the source code.

<a id="tvm.tirx.Add"></a>
### `tvm.tirx.Add`

```python
class tvm.tirx.Add(a: Expr, b: Expr, span: Span | None = None)
```

Add node.

**Parameters:**

- **a** (tvm.relax.Expr) – The left hand operand.
- **b** (tvm.relax.Expr) – The right hand operand.
- **span** (*Optional*[tvm.ir.Span]) – The location of this expression in the source code.

<a id="tvm.tirx.Sub"></a>
### `tvm.tirx.Sub`

```python
class tvm.tirx.Sub(a: Expr, b: Expr, span: Span | None = None)
```

Sub node.

**Parameters:**

- **a** (tvm.relax.Expr) – The left hand operand.
- **b** (tvm.relax.Expr) – The right hand operand.
- **span** (*Optional*[tvm.ir.Span]) – The location of this expression in the source code.

<a id="tvm.tirx.Mul"></a>
### `tvm.tirx.Mul`

```python
class tvm.tirx.Mul(a: Expr, b: Expr, span: Span | None = None)
```

Mul node.

**Parameters:**

- **a** (tvm.relax.Expr) – The left hand operand.
- **b** (tvm.relax.Expr) – The right hand operand.
- **span** (*Optional*[tvm.ir.Span]) – The location of this expression in the source code.

<a id="tvm.tirx.Div"></a>
### `tvm.tirx.Div`

```python
class tvm.tirx.Div(a: Expr, b: Expr, span: Span | None = None)
```

Div node.

**Parameters:**

- **a** (tvm.relax.Expr) – The left hand operand.
- **b** (tvm.relax.Expr) – The right hand operand.
- **span** (*Optional*[tvm.ir.Span]) – The location of this expression in the source code.

<a id="tvm.tirx.Mod"></a>
### `tvm.tirx.Mod`

```python
class tvm.tirx.Mod(a: Expr, b: Expr, span: Span | None = None)
```

Mod node.

**Parameters:**

- **a** (tvm.relax.Expr) – The left hand operand.
- **b** (tvm.relax.Expr) – The right hand operand.
- **span** (*Optional*[tvm.ir.Span]) – The location of this expression in the source code.

<a id="tvm.tirx.FloorDiv"></a>
### `tvm.tirx.FloorDiv`

```python
class tvm.tirx.FloorDiv(a: Expr, b: Expr, span: Span | None = None)
```

FloorDiv node.

**Parameters:**

- **a** (tvm.relax.Expr) – The left hand operand.
- **b** (tvm.relax.Expr) – The right hand operand.
- **span** (*Optional*[tvm.ir.Span]) – The location of this expression in the source code.

<a id="tvm.tirx.FloorMod"></a>
### `tvm.tirx.FloorMod`

```python
class tvm.tirx.FloorMod(a: Expr, b: Expr, span: Span | None = None)
```

FloorMod node.

**Parameters:**

- **a** (tvm.relax.Expr) – The left hand operand.
- **b** (tvm.relax.Expr) – The right hand operand.
- **span** (*Optional*[tvm.ir.Span]) – The location of this expression in the source code.

<a id="tvm.tirx.Min"></a>
### `tvm.tirx.Min`

```python
class tvm.tirx.Min(a: Expr, b: Expr, span: Span | None = None)
```

Min node.

**Parameters:**

- **a** (tvm.relax.Expr) – The left hand operand.
- **b** (tvm.relax.Expr) – The right hand operand.
- **span** (*Optional*[tvm.ir.Span]) – The location of this expression in the source code.

<a id="tvm.tirx.Max"></a>
### `tvm.tirx.Max`

```python
class tvm.tirx.Max(a: Expr, b: Expr, span: Span | None = None)
```

Max node.

**Parameters:**

- **a** (tvm.relax.Expr) – The left hand operand.
- **b** (tvm.relax.Expr) – The right hand operand.
- **span** (*Optional*[tvm.ir.Span]) – The location of this expression in the source code.

<a id="tvm.tirx.EQ"></a>
### `tvm.tirx.EQ`

```python
class tvm.tirx.EQ(a: Expr, b: Expr, span: Span | None = None)
```

EQ node.

**Parameters:**

- **a** (tvm.relax.Expr) – The left hand operand.
- **b** (tvm.relax.Expr) – The right hand operand.
- **span** (*Optional*[tvm.ir.Span]) – The location of this expression in the source code.

<a id="tvm.tirx.NE"></a>
### `tvm.tirx.NE`

```python
class tvm.tirx.NE(a: Expr, b: Expr, span: Span | None = None)
```

NE node.

**Parameters:**

- **a** (tvm.relax.Expr) – The left hand operand.
- **b** (tvm.relax.Expr) – The right hand operand.
- **span** (*Optional*[tvm.ir.Span]) – The location of this expression in the source code.

<a id="tvm.tirx.LT"></a>
### `tvm.tirx.LT`

```python
class tvm.tirx.LT(a: Expr, b: Expr, span: Span | None = None)
```

LT node.

**Parameters:**

- **a** (tvm.relax.Expr) – The left hand operand.
- **b** (tvm.relax.Expr) – The right hand operand.
- **span** (*Optional*[tvm.ir.Span]) – The location of this expression in the source code.

<a id="tvm.tirx.LE"></a>
### `tvm.tirx.LE`

```python
class tvm.tirx.LE(a: Expr, b: Expr, span: Span | None = None)
```

LE node.

**Parameters:**

- **a** (tvm.relax.Expr) – The left hand operand.
- **b** (tvm.relax.Expr) – The right hand operand.
- **span** (*Optional*[tvm.ir.Span]) – The location of this expression in the source code.

<a id="tvm.tirx.GT"></a>
### `tvm.tirx.GT`

```python
class tvm.tirx.GT(a: Expr, b: Expr, span: Span | None = None)
```

GT node.

**Parameters:**

- **a** (tvm.relax.Expr) – The left hand operand.
- **b** (tvm.relax.Expr) – The right hand operand.
- **span** (*Optional*[tvm.ir.Span]) – The location of this expression in the source code.

<a id="tvm.tirx.GE"></a>
### `tvm.tirx.GE`

```python
class tvm.tirx.GE(a: Expr, b: Expr, span: Span | None = None)
```

GE node.

**Parameters:**

- **a** (tvm.relax.Expr) – The left hand operand.
- **b** (tvm.relax.Expr) – The right hand operand.
- **span** (*Optional*[tvm.ir.Span]) – The location of this expression in the source code.

<a id="tvm.tirx.And"></a>
### `tvm.tirx.And`

```python
class tvm.tirx.And(a: Expr, b: Expr, span: Span | None = None)
```

And node.

**Parameters:**

- **a** (tvm.relax.Expr) – The left hand operand.
- **b** (tvm.relax.Expr) – The right hand operand.
- **span** (*Optional*[tvm.ir.Span]) – The location of this expression in the source code.

<a id="tvm.tirx.Or"></a>
### `tvm.tirx.Or`

```python
class tvm.tirx.Or(a: Expr, b: Expr, span: Span | None = None)
```

Or node.

**Parameters:**

- **a** (tvm.relax.Expr) – The left hand operand.
- **b** (tvm.relax.Expr) – The right hand operand.
- **span** (*Optional*[tvm.ir.Span]) – The location of this expression in the source code.

<a id="tvm.tirx.Not"></a>
### `tvm.tirx.Not`

```python
class tvm.tirx.Not(a: Expr, span: Span | None = None)
```

Not node.

**Parameters:**

- **a** (tvm.relax.Expr) – The input value
- **span** (*Optional*[tvm.ir.Span]) – The location of this expression in the source code.

<a id="tvm.tirx.Select"></a>
### `tvm.tirx.Select`

```python
class tvm.tirx.Select(condition: Expr, true_value: Expr, false_value: Expr, span: Span | None = None)
```

Select node.

> **Note**
>
> Select may compute both true\_value and false\_value.
> Use [`tvm.tirx.if_then_else`](#tvm.tirx.if_then_else) instead if you want to
> get a conditional expression that only evaluates
> the correct branch.

**Parameters:**

- **condition** (tvm.relax.Expr) – The condition expression.
- **true\_value** (tvm.relax.Expr) – The value to take when condition is true.
- **false\_value** (tvm.relax.Expr) – The value to take when condition is false.
- **span** (*Optional*[tvm.ir.Span]) – The location of this expression in the source code.

<a id="tvm.tirx.BufferLoad"></a>
### `tvm.tirx.BufferLoad`

```python
class tvm.tirx.BufferLoad(buffer: Buffer, indices: list[Expr], predicate: Expr | None = None, span: Span | None = None)
```

Buffer load node.

**Parameters:**

- **buffer** ([*Buffer*](#tvm.tirx.Buffer)) – The buffer to be loaded.
- **indices** (*List*[tvm.relax.Expr]) – The buffer indices to load values from.
- **span** (*Optional*[tvm.ir.Span]) – The location of this expression in the source code.
- **predicate** (*Optional*[tvm.relax.Expr]) – A vector mask of boolean values indicating which lanes of a vector are to be
  loaded. The number lanes of the mask must be equal to the number of lanes being loaded.

<a id="tvm.tirx.ProducerLoad"></a>
### `tvm.tirx.ProducerLoad`

```python
class tvm.tirx.ProducerLoad(producer: DataProducer, indices: list[Expr], span: Span | None = None)
```

Producer load node.

**Parameters:**

- **producer** ([*DataProducer*](#tvm.tirx.DataProducer)) – The buffer to be loaded.
- **indices** (*List*[tvm.relax.Expr]) – The buffer indices.
- **span** (*Optional*[tvm.ir.Span]) – The location of this expression in the source code.

<a id="tvm.tirx.Ramp"></a>
### `tvm.tirx.Ramp`

```python
class tvm.tirx.Ramp(base: Expr, stride: Expr, lanes: Expr, span: Span | None = None)
```

Ramp node.

**Parameters:**

- **base** (tvm.relax.Expr) – The base expression.
- **stride** (tvm.relax.Expr) – The stride of the ramp.
- **lanes** (tvm.relax.Expr) – The lanes of the expression.
- **span** (*Optional*[tvm.ir.Span]) – The location of this expression in the source code.

<a id="tvm.tirx.Broadcast"></a>
### `tvm.tirx.Broadcast`

```python
class tvm.tirx.Broadcast(value: Expr, lanes: Expr, span: Span | None = None)
```

Broadcast node.

**Parameters:**

- **value** (tvm.relax.Expr) – The value of the expression.
- **lanes** (tvm.relax.Expr) – The lanes of the expression.
- **span** (*Optional*[tvm.ir.Span]) – The location of this expression in the source code.

<a id="tvm.tirx.Shuffle"></a>
### `tvm.tirx.Shuffle`

```python
class tvm.tirx.Shuffle(vectors: list[Expr], indices: list[Expr], span: Span | None = None)
```

Shuffle node.

**Parameters:**

- **vectors** (*List*[tvm.relax.Expr]) – The vectors
- **indices** (*List*[tvm.relax.Expr]) – The indices
- **span** (*Optional*[tvm.ir.Span]) – The location of this expression in the source code.

<a id="tvm.tirx.CallEffectKind"></a>
### `tvm.tirx.CallEffectKind`

```python
class tvm.tirx.CallEffectKind
```

Possible kinds of tirx.Call effects.

<a id="tvm.tirx.Let"></a>
### `tvm.tirx.Let`

```python
class tvm.tirx.Let(var: Var, value: Expr, body: Expr, span: Span | None = None)
```

Let node.

**Parameters:**

- **var** (*tirx.Var*) – The variable in the binding.
- **value** (tvm.relax.Expr) – The value in to be bound.
- **body** (tvm.relax.Expr) – The body expression.
- **span** (*Optional*[tvm.ir.Span]) – The location of this expression in the source code.

<a id="tvm.tirx.IterVar"></a>
### `tvm.tirx.IterVar`

```python
class tvm.tirx.IterVar(dom: Range, var: Var | str, iter_type: int, thread_tag: str = '', span: Span | None = None)
```

Represent iteration variable.

IterVar represents axis iterations in the computation.

**Parameters:**

- **dom** (tvm.ir.Range) – The domain of the iteration.
- **var** (*Union*[*tirx.Var*, str]) – The internal variable that is used for iteration.
- **iter\_type** (int) – The iteration type.
- **thread\_tag** (str) – The thread type tag.
- **span** (*Optional*[tvm.ir.Span]) – The location of this expression in the source code.

> **See also**
>
> `te.thread_axis`
> :   Create thread axis IterVar.
>
> `te.reduce_axis`
> :   Create reduce axis IterVar.

<a id="tvm.tirx.IterVar.expr_ty"></a>
#### `tvm.tirx.IterVar.expr_ty`

```python
expr_ty() → PrimType
```

Compile-time type of the iteration variable.

<a id="tvm.tirx.CommReducer"></a>
### `tvm.tirx.CommReducer`

```python
class tvm.tirx.CommReducer(lhs: list[Var], rhs: list[Var], result: list[Expr], identity_element: list[Expr], span: Span | None = None)
```

Commutative reduce operator

**Parameters:**

- **lhs** (*List*[*tirx.Var*]) – The left arguments of the reducer.
- **rhs** (*List*[*tirx.Var*]) – The right arguments of the reducer.
- **result** (*List*[tvm.relax.Expr]) – The reduction results.
- **identity\_element** (*List*[tvm.relax.Expr]) – The identity elements.
- **span** (*Optional*[tvm.ir.Span]) – The location of this expression in the source code.

<a id="tvm.tirx.Stmt"></a>
### `tvm.tirx.Stmt`

```python
class tvm.tirx.Stmt(span)
```

Base class of all the statements.

<a id="tvm.tirx.Bind"></a>
### `tvm.tirx.Bind`

```python
class tvm.tirx.Bind(var: Var, value: Expr, span: Span | None = None)
```

Bind node.

Bind a variable to a value in the enclosing scope.
Bind has no body field.
The bound variable is visible in all subsequent statements
within the same enclosing scope (SeqStmt, ForNode.body, etc.).

**Parameters:**

- **var** (*tirx.Var*) – The variable in the binding.
- **value** (tvm.relax.Expr) – The value to be bound.
- **span** (*Optional*[tvm.ir.Span]) – The location of the stmt in the source code.

<a id="tvm.tirx.AssertStmt"></a>
### `tvm.tirx.AssertStmt`

```python
class tvm.tirx.AssertStmt(kind: StringImm, condition: Expr, message_parts: list | None = None, span: Span | None = None)
```

AssertStmt node.

**Parameters:**

- **kind** ([*StringImm*](#tvm.tirx.StringImm)) – The error kind, e.g. “RuntimeError”, “TypeError”, “ValueError”.
- **condition** (tvm.relax.Expr) – The assert condition.
- **message\_parts** (list[[*StringImm*](#tvm.tirx.StringImm)]) – Error message fragments, concatenated at runtime when assertion fails.
- **span** (tvm.ir.Span | *None*) – The location of the stmt in the source code.

<a id="tvm.tirx.ForKind"></a>
### `tvm.tirx.ForKind`

```python
class tvm.tirx.ForKind(value)
```

The kind of the for loop.

> **Note**
>
> ForKind can change the control flow semantics
> of the loop and need to be considered in all TIR passes.

<a id="tvm.tirx.For"></a>
### `tvm.tirx.For`

```python
class tvm.tirx.For(loop_var: Var, min: Expr, extent: Expr, kind: ForKind, body: Stmt, thread_binding: IterVar | None = None, annotations: Mapping[str, Object] | None = None, step: Expr | None = None, span: Span | None = None)
```

For node.

**Parameters:**

- **loop\_var** (*tirx.Var*) – The loop variable.
- **min** (tvm.relax.Expr) – The beginning value.
- **extent** (tvm.relax.Expr) – The length of the loop.
- **kind** ([*ForKind*](#tvm.tirx.ForKind)) – The type of the for.
- **body** ([*Stmt*](#tvm.tirx.Stmt)) – The body statement.
- **thread\_binding** (*Optional*[[*tirx.IterVar*](#tvm.tirx.IterVar)]) – The thread this loop binds to. Only valid
  if kind is ThreadBinding
- **step** (tvm.relax.Expr) – The loop step. Default to none which
  represent one.
- **annotations** (*Optional*[*Mapping*[str, *Object*]]) – Additional annotation hints.
- **span** (*Optional*[tvm.ir.Span]) – The location of the stmt in the source code.

<a id="tvm.tirx.While"></a>
### `tvm.tirx.While`

```python
class tvm.tirx.While(condition: Expr, body: Stmt, span: Span | None = None)
```

While node.

**Parameters:**

- **condition** (tvm.relax.Expr) – The termination condition.
- **body** ([*Stmt*](#tvm.tirx.Stmt)) – The body statement.
- **span** (*Optional*[tvm.ir.Span]) – The location of the stmt in the source code.

<a id="tvm.tirx.Return"></a>
### `tvm.tirx.Return`

```python
class tvm.tirx.Return(value: Expr, span: Span | None = None)
```

Return node.

**Parameters:**

- **value** (tvm.relax.Expr) – The value to return.
- **span** (*Optional*[tvm.ir.Span]) – The location of this statement in the source code.

<a id="tvm.tirx.Break"></a>
### `tvm.tirx.Break`

```python
class tvm.tirx.Break(span: Span | None = None)
```

Break node.

<a id="tvm.tirx.Continue"></a>
### `tvm.tirx.Continue`

```python
class tvm.tirx.Continue(span: Span | None = None)
```

Continue node.

<a id="tvm.tirx.LetStmt"></a>
### `tvm.tirx.LetStmt`

```python
tvm.tirx.LetStmt
```

alias of [`Bind`](#tvm.tirx.Bind)

<a id="tvm.tirx.BufferStore"></a>
### `tvm.tirx.BufferStore`

```python
class tvm.tirx.BufferStore(buffer: Buffer, value: Expr, indices: list[Expr], predicate: Expr | None = None, span: Span | None = None)
```

Buffer store node.

**Parameters:**

- **buffer** ([*Buffer*](#tvm.tirx.Buffer)) – The buffer.
- **value** (tvm.relax.Expr) – The value we to be stored.
- **indices** (*List*[tvm.relax.Expr]) – The indices location to be stored.
- **predicate** (*Optional*[tvm.relax.Expr]) – A vector mask of boolean values indicating which lanes of a vector are to be
  stored. The number lanes of the mask must be equal to the number of lanes in
  value.
- **span** (*Optional*[tvm.ir.Span]) – The location of the stmt in the source code.

<a id="tvm.tirx.AllocBuffer"></a>
### `tvm.tirx.AllocBuffer`

```python
class tvm.tirx.AllocBuffer(buffer: Buffer, *args, **kwargs)
```

AllocBuffer node.

Allocates a buffer and declares it in scope.

**Parameters:**

- **buffer** ([*Buffer*](#tvm.tirx.Buffer)) – The buffer being allocated and declared.
- **annotations** (*Optional*[dict]) – Additional annotations about the allocation.
- **span** (*Optional*[tvm.ir.Span]) – The location of this AllocBuffer in the source code.

<a id="tvm.tirx.AttrStmt"></a>
### `tvm.tirx.AttrStmt`

```python
class tvm.tirx.AttrStmt(node: Any, attr_key: str, value: Expr, body: Stmt, span: Span | None = None)
```

AttrStmt node.

**Parameters:**

- **node** (*Any*) – The node to annotate the attribute
- **attr\_key** (str) – Attribute type key.
- **value** (tvm.relax.Expr) – The value of the attribute
- **body** ([*Stmt*](#tvm.tirx.Stmt)) – The body statement.
- **span** (*Optional*[tvm.ir.Span]) – The location of the stmt in the source code.

<a id="tvm.tirx.DeclBuffer"></a>
### `tvm.tirx.DeclBuffer`

```python
class tvm.tirx.DeclBuffer(buffer: Buffer, *args, **kwargs)
```

DeclBuffer node.

**Parameters:**

- **buffer** ([*Buffer*](#tvm.tirx.Buffer)) – The buffer being declared.
- **span** (*Optional*[tvm.ir.Span]) – The location of this DeclBuffer in the source code.

<a id="tvm.tirx.SeqStmt"></a>
### `tvm.tirx.SeqStmt`

```python
class tvm.tirx.SeqStmt(seq: list[Stmt], span: Span | None = None)
```

Sequence of statements.

**Parameters:**

- **seq** (*List*[[*Stmt*](#tvm.tirx.Stmt)]) – The statements
- **span** (*Optional*[tvm.ir.Span]) – The location of the stmt in the source code.

<a id="tvm.tirx.IfThenElse"></a>
### `tvm.tirx.IfThenElse`

```python
class tvm.tirx.IfThenElse(condition: Expr, then_case: Stmt, else_case: Stmt | None, span: Span | None = None)
```

IfThenElse node.

**Parameters:**

- **condition** (tvm.relax.Expr) – The expression
- **then\_case** ([*Stmt*](#tvm.tirx.Stmt)) – The statement to execute if condition is true.
- **else\_case** (*Optional*[[*Stmt*](#tvm.tirx.Stmt)]) – The statement to execute if condition is false.
- **span** (*Optional*[tvm.ir.Span]) – The location of the stmt in the source code.

<a id="tvm.tirx.Evaluate"></a>
### `tvm.tirx.Evaluate`

```python
class tvm.tirx.Evaluate(value: Expr, span: Span | None = None)
```

Evaluate node.

**Parameters:**

- **value** (tvm.relax.Expr) – The expression to be evaluated.
- **span** (*Optional*[tvm.ir.Span]) – The location of the stmt in the source code.

<a id="tvm.tirx.stmt_seq"></a>
### `tvm.tirx.stmt_seq`

```python
tvm.tirx.stmt_seq(*args: Expr | Stmt) → SeqStmt
```

Make sequence of statements

**Parameters:**

**\*args** (*Union*[tvm.relax.Expr, [*Stmt*](#tvm.tirx.Stmt)]) – List of statements to be combined as sequence.

**Returns:**

**stmt** – The combined statement.

**Return type:**

[Stmt](#tvm.tirx.Stmt)

<a id="tvm.tirx.stmt_list"></a>
### `tvm.tirx.stmt_list`

```python
tvm.tirx.stmt_list(stmt: Stmt) → list[Stmt]
```

Make list of stmt from blocks.

**Parameters:**

**stmt** ([*Stmt*](#tvm.tirx.Stmt)) – The input statement.

**Returns:**

**stmt\_list** – The unpacked list of statements

**Return type:**

List[[Stmt](#tvm.tirx.Stmt)]

<a id="tvm.tirx.BufferRegion"></a>
### `tvm.tirx.BufferRegion`

```python
class tvm.tirx.BufferRegion(buffer: Buffer, region: list[Range])
```

BufferRegion node.

**Parameters:**

- **buffer** ([*Buffer*](#tvm.tirx.Buffer)) – The buffer of the buffer region
- **region** (*List*[tvm.ir.Range]) – The region array of the buffer region

<a id="tvm.tirx.MatchBufferRegion"></a>
### `tvm.tirx.MatchBufferRegion`

```python
class tvm.tirx.MatchBufferRegion(buffer: Buffer, source: BufferRegion)
```

MatchBufferRegion node.

**Parameters:**

- **buffer** ([*Buffer*](#tvm.tirx.Buffer)) – The target buffer
- **source** ([*BufferRegion*](#tvm.tirx.BufferRegion)) – The region of source buffer

<a id="tvm.tirx.SBlock"></a>
### `tvm.tirx.SBlock`

```python
class tvm.tirx.SBlock(iter_vars: list[IterVar], reads: list[BufferRegion], writes: list[BufferRegion], name_hint: str, body: Stmt, init: Stmt | None = None, alloc_buffers: list[Buffer] | None = None, match_buffers: list[MatchBufferRegion] | None = None, annotations: Mapping[str, Object] | None = None, span: Span | None = None)
```

SBlock node.

**Parameters:**

- **iter\_vars** (*List*[[*IterVar*](#tvm.tirx.IterVar)]) – The block Variable.
- **reads** (*List*[[*BufferRegion*](#tvm.tirx.BufferRegion)]) – The read buffer regions of the block.
- **writes** (*List*[[*BufferRegion*](#tvm.tirx.BufferRegion)]) – The write buffer regions of the block.
- **name\_hint** (str) – the name\_hint of the block.
- **body** ([*Stmt*](#tvm.tirx.Stmt)) – The body of the block.
- **init** (*Optional*[[*Stmt*](#tvm.tirx.Stmt)]) – The init block of the reduction block
- **alloc\_buffers** (*Optional*[list[[*Buffer*](#tvm.tirx.Buffer)]]) – The buffer allocations
- **match\_buffers** (*Optional*[*List*[[*MatchBufferRegion*](#tvm.tirx.MatchBufferRegion)]]) – The subregion buffer match
- **annotations** (*Optional*[*Mapping*[str, *Object*]]) – Additional annotation hints.
- **span** (*Optional*[tvm.ir.Span]) – The location of this block in the source code.

<a id="tvm.tirx.SBlockRealize"></a>
### `tvm.tirx.SBlockRealize`

```python
class tvm.tirx.SBlockRealize(iter_values: list[Expr], predicate: Expr | bool, block: SBlock, span: Span | None = None)
```

SBlockRealize node.

**Parameters:**

- **iter\_values** (*List*[tvm.relax.Expr]) – The binding values of the block var.
- **predicate** (*Union*[tvm.relax.Expr, bool]) – The predicate of the block.
- **block** ([*SBlock*](#tvm.tirx.SBlock)) – The block to realize
- **span** (*Optional*[tvm.ir.Span]) – The location of this block\_realize in the source code.

<a id="tvm.tirx.TilePrimitiveCall"></a>
### `tvm.tirx.TilePrimitiveCall`

```python
class tvm.tirx.TilePrimitiveCall(*args: list[Expr], op: Op | None = None, workspace: dict[str, Buffer] | None = None, config: dict[str, Any] | None = None, dispatch: str | None = None, scope: ExecScope | None = None)
```

TilePrimitiveCall node.

**Parameters:**

- **op** (tvm.ir.Op) – The operator.
- **args** (*List*[tvm.relax.Expr]) – The arguments.
- **workspace** (tvm.ir.Map[str, [*Buffer*](#tvm.tirx.Buffer)]) – The workspace.
- **config** (tvm.ir.Map[str, *ObjectRef*]) – The scheduler/config dictionary.
- **dispatch** (*Optional*[str]) – The explicit variant name to dispatch to.
- **scope** ([*ExecScope*](#tvm.tirx.ExecScope)) – The cooperation scope of this call. Defaults to `thread` (an unscoped call).

<a id="tvm.tirx.TilePrimitiveCall.replace"></a>
#### `tvm.tirx.TilePrimitiveCall.replace`

```python
replace(**changes: Any) → TilePrimitiveCall
```

Return a copy of this call with selected fields replaced.

Every field that is not overridden in `changes` is preserved from
`self` (including `scope`), so rebuilds never silently drop fields.
The returned node is downcast to the registered subclass for `op`.

**Parameters:**

**\*\*changes** (*Any*) – Field overrides; any of `op`, `args`, `workspace`, `config`,
`dispatch`, `scope`.

**Returns:**

**new\_call** – A new call with the requested fields replaced.

**Return type:**

[TilePrimitiveCall](#tvm.tirx.TilePrimitiveCall)

<a id="tvm.tirx.TilePrimitiveCall.with_workspace"></a>
#### `tvm.tirx.TilePrimitiveCall.with_workspace`

```python
with_workspace(workspace: dict[str, Buffer]) → TilePrimitiveCall
```

Return a copy with `workspace` replaced, preserving all other fields.

<a id="tvm.tirx.TilePrimitiveCall.get_private_buffers"></a>
#### `tvm.tirx.TilePrimitiveCall.get_private_buffers`

```python
get_private_buffers(buffer_dict: dict[Any, tuple[Buffer, Stmt | None]], sctx: DispatchContext) → dict[str, Any]
```

Create private (intermediate) buffers needed in this operator.

**Parameters:**

- **buffer\_dict** (*Dict*[*Any*, tvm.relax.Tuple[[*Buffer*](#tvm.tirx.Buffer), *Optional*[[*Stmt*](#tvm.tirx.Stmt)]]]) – A dictionary containing private buffers (and their init stmts) in other operators.
  Key can be anything to reference the buffer.
  This is used to reuse private buffers in other operators (like identity tensor etc.).
  If the buffer is not found in the buffer\_dict, it will be created and added to
  the buffer\_dict.
  If the buffer is found in the buffer\_dict but smaller than required, it will be
  enlarged and updated.
- **sctx** (*DispatchContext*) – The dispatch context.
  This is used to get the target and reuse op dispatch implementations.
- **Returns** – private\_buffer\_refs: Dict[str, Any]
  The references to private buffers created in this operator.
  Key will be the name to add into workspace.
  private buffer can be accessed by buffer\_dict[private\_buffer\_refs[name]]

<a id="tvm.tirx.ScopeIdDefStmt"></a>
### `tvm.tirx.ScopeIdDefStmt`

```python
class tvm.tirx.ScopeIdDefStmt(def_: ScopeIdDef, span: Span | None = None)
```

ScopeIdDefStmt node.

Leaf statement that introduces scope-identifier vars
(`wg_id = Tx.warpgroup_id([N])`, `warp_id = Tx.warp_id_in_wg([4])`,
`lane_id = Tx.lane_id([32])`, …) at the kernel-body top level. The
underlying `ScopeIdDef` carries the def vars, their extents, and
the parent/child scope binding.

Note: the C++ field is named `def` (a Python keyword). Access it
via `getattr(stmt, "def")` or `stmt.__getattribute__("def")` —
the type-annotation alias here is purely for documentation.

**Parameters:**

- **def** ([*ScopeIdDef*](#tvm.tirx.ScopeIdDef)) – The scope-id definition (def vars, extents, scope binding).
- **span** (*Optional*[tvm.ir.Span]) – The location of this statement in the source code.

<a id="tvm.tirx.PrimFunc"></a>
### `tvm.tirx.PrimFunc`

```python
class tvm.tirx.PrimFunc(params, body, ret_type=None, buffer_map=None, attrs=None, span=None)
```

A function declaration expression.

**Parameters:**

- **params** (*List*[*Union*[*tvm.tirx.Var*, [*tvm.tirx.Buffer*](#tvm.tirx.Buffer)]]) – List of input parameters to the function.
- **body** ([*tvm.tirx.Stmt*](#tvm.tirx.Stmt)) – The body of the function.
- **ret\_type** (tvm.ir.Type) – The return type annotation of the function.
- **buffer\_map** (tvm.ir.Map[*tvm.tirx.Var*, [*tvm.tirx.Buffer*](#tvm.tirx.Buffer)]) – The buffer binding map.
- **attrs** (*Optional*[*tvm.Attrs*]) – Attributes of the function, can be None
- **span** (*Optional*[tvm.ir.Span]) – The location of this itervar in the source code.

<a id="tvm.tirx.PrimFunc.with_body"></a>
#### `tvm.tirx.PrimFunc.with_body`

```python
with_body(new_body, span=None)
```

Create a new PrimFunc with the same set signatures but a new body.

**Parameters:**

- **new\_body** ([*Stmt*](#tvm.tirx.Stmt)) – The new body.
- **span** (*Optional*[tvm.ir.Span]) – The location of this itervar in the source code.

**Returns:**

**new\_func** – The created new function.

**Return type:**

[PrimFunc](#tvm.tirx.PrimFunc)

<a id="tvm.tirx.PrimFunc.specialize"></a>
#### `tvm.tirx.PrimFunc.specialize`

```python
specialize(param_map: Mapping[Var, Expr | Buffer])
```

Specialize parameters of PrimFunc

**Parameters:**

**param\_map** (*Mapping*[*tirx.Var*, *Union*[tvm.relax.Expr, [*Buffer*](#tvm.tirx.Buffer)]]) – The mapping from function params to the instance

Examples

We can define a Meta TIR function with symbolic shape:

```python
@T.prim_func(s_tir=True)
def mem_copy(a: T.handle, b: T.handle, m: T.int32, n: T.int32) -> None:
    A = T.match_buffer(a, (m, n), "float32")
    B = T.match_buffer(b, (m, n), "float32")

    for i, j in T.grid(m, n):
        with T.sblock():
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj]
```

Then we can make it specialized with given shapes or buffers.

```python
a, _, m, n = mem_copy.params
func = mem_copy.specialize({a: tirx.decl_buffer((16, 16))})
# or
func = mem_copy.specialize({n: 16, m: 16})
```

The specialized function:

```python
@T.prim_func(s_tir=True)
def mem_copy_16_16(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float32")
    B = T.match_buffer(b, (16, 16), "float32")

    for i, j in T.grid(16, 16):
        with T.sblock():
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj]
```

**Returns:**

**func** – The new function with parameter specialized

**Return type:**

[PrimFunc](#tvm.tirx.PrimFunc)

<a id="tvm.tirx.TensorIntrin"></a>
### `tvm.tirx.TensorIntrin`

```python
class tvm.tirx.TensorIntrin(desc, impl)
```

A tensor intrinsic.

**Parameters:**

- **desc** ([*PrimFunc*](#tvm.tirx.PrimFunc)) – The function to describe the computation.
- **impl** ([*PrimFunc*](#tvm.tirx.PrimFunc)) – The function of the implementation for the execution.

<a id="tvm.tirx.TensorIntrin.register"></a>
#### `tvm.tirx.TensorIntrin.register`

```python
static register(name: str, desc: PrimFunc, impl: PrimFunc, override: bool = False)
```

Register a tensor intrinsic with its name.

**Parameters:**

- **name** (str) – The name of the TensorIntrin to register.
- **desc** ([*PrimFunc*](#tvm.tirx.PrimFunc)) – The function to describe the computation.
- **impl** ([*PrimFunc*](#tvm.tirx.PrimFunc)) – The function of the implementation for the execution.
- **override** (bool) – Whether override existing intrinsic.

<a id="tvm.tirx.TensorIntrin.get"></a>
#### `tvm.tirx.TensorIntrin.get`

```python
static get(name: str, allow_missing: bool = False) → TensorIntrin | None
```

Look up a tensor intrinsic by its name.

**Parameters:**

- **name** (str) – The name of the TensorIntrin to look up.
- **allow\_missing** (bool) – Whether to allow missing tensor intrin. If False, raise an error if the tensor intrin
- **exist.** (*doesn't*)

**Returns:**

**result** – The TensorIntrin with the specified name, or None if not found.

**Return type:**

Optional[[TensorIntrin](#tvm.tirx.TensorIntrin)]

<a id="tvm.tirx.IndexMap"></a>
### `tvm.tirx.IndexMap`

```python
class tvm.tirx.IndexMap(initial_indices, final_indices, inverse_index_map)
```

A mapping from multi-dimensional indices to another set of multi-dimensional indices

**Parameters:**

- **initial\_indices** (*List*[*tirx.Var*]) – Variables representing the indices prior to remapping.
- **final\_indices** (*List*[tvm.relax.Expr]) – Expressions defining the indices after remapping.
- **inverse\_index\_map** (*Union*[*Callable*, *Optional*[[*IndexMap*](#tvm.tirx.IndexMap)]]) – The optional pre-defined inverse index map.
  When this is defined, IndexMap::Inverse will return the pre-defined inverse index map.
  Otherwise, the inverse index map will be computed on the fly.
  It is the user’s responsibility to ensure the correctness of the pre-defined inverse
  index map.

<a id="tvm.tirx.IndexMap.from_func"></a>
#### `tvm.tirx.IndexMap.from_func`

```python
static from_func(mapping_function: Callable, ndim: int | None = None, inverse_index_map: Callable | IndexMap | None = None, *, index_dtype: str = 'int64')
```

Create an index map from a function

**Parameters:**

- **mapping\_function** (*Callable*) – The function to map from source indices to target indices.
  The function should accept tirx.Var parameters and return
  a either a tirx.Expr, or a list of tirx.Expr.
  Returning a tirx.Expr is equivalent to returning a
  list of length 1 containing that tirx.Expr.
- **ndim** (*Optional*[int]) – The dimensionality of the buffer to which this
  transformation should be applied. If mapping\_function uses
  variadic argument \*args, ndim must be specified. If
  mapping\_function does not use variadic arguments, ndim is
  optional.
- **inverse\_index\_map** (*Union*[*Callable*, *Optional*[[*IndexMap*](#tvm.tirx.IndexMap)]]) – The optional pre-defined inverse index map.
  When this is defined, IndexMap::Inverse will return the pre-defined inverse index map.
  Otherwise, the inverse index map will be computed on the fly.
  It is the user’s responsibility to ensure the correctness of the pre-defined inverse
  index map.
- **index\_dtype** (str) – The default index dtype to use for input iters in the mapping function.

**Returns:**

**index\_map** – Returns an IndexMap representing the mapping\_function.

**Return type:**

[IndexMap](#tvm.tirx.IndexMap)

<a id="tvm.tirx.IndexMap.is_equivalent_to"></a>
#### `tvm.tirx.IndexMap.is_equivalent_to`

```python
is_equivalent_to(other_map: IndexMap, analyzer=None) → bool
```

Return if the index maps are equivalent.

**Parameters:**

- **other\_map** ([*IndexMap*](#tvm.tirx.IndexMap)) – The IndexMap to which the comparison should be made.
- **analyzer** (*Optional*[tvm.arith.Analyzer]) – The analyzer to use while comparing the mapped indices. When
  provided, its accumulated bindings and constraints are reused so
  that maps that are only equivalent under those bindings can be
  proven equal.

**Returns:**

**is\_equivalent** – True if the two mappings represent the same
transformation, otherwise False

**Return type:**

bool

<a id="tvm.tirx.IndexMap.map_indices"></a>
#### `tvm.tirx.IndexMap.map_indices`

```python
map_indices(indices: list[Expr], analyzer=None) → list[Expr]
```

Apply the index map to a set of indices

**Parameters:**

- **indices** (*List*[tvm.relax.Expr]) – The indices to be mapped
- **analyzer** (*Optional*[tvm.arith.Analyzer]) – The analyzer to use while simplifying mapped indices.

**Returns:**

**result** – The mapped indices

**Return type:**

List[tvm.relax.Expr]

<a id="tvm.tirx.IndexMap.map_shape"></a>
#### `tvm.tirx.IndexMap.map_shape`

```python
map_shape(shape: list[Expr], analyzer=None) → list[Expr]
```

Apply the index map to a buffer shape

**Parameters:**

- **shape** (*List*[tvm.relax.Expr]) – The buffer shape to be mapped
- **analyzer** (*Optional*[tvm.arith.Analyzer]) – The analyzer to use while simplifying mapped shape expressions.

**Returns:**

**result** – The mapped shape

**Return type:**

List[tvm.relax.Expr]

<a id="tvm.tirx.IndexMap.map_tensor"></a>
#### `tvm.tirx.IndexMap.map_tensor`

```python
map_tensor(arr_src: Tensor) → Tensor
```

Apply thie index map to transform the layout of the input Tensor

**Parameters:**

**arr\_src** (*runtime.Tensor*) – The Tensor to be transformed

**Returns:**

**arr\_dst** – The transformed Tensor

**Return type:**

runtime.Tensor

<a id="tvm.tirx.IndexMap.inverse"></a>
#### `tvm.tirx.IndexMap.inverse`

```python
inverse(shape: list[Range | Expr], analyzer=None) → IndexMap
```

Return the inverse of the map

Throws an error if the function is not bijective.

**Parameters:**

- **shape** (*List*[*Union*[tvm.ir.Range,tvm.relax.Expr]]) – The region over which the inverse should be determined.
  Used for validating that the mapping is bijective over
  this range.
- **analyzer** (*Optional*[tvm.arith.Analyzer]) – The analyzer to use while deriving and validating the inverse.

**Returns:**

**inverse** – The inverse

**Return type:**

[IndexMap](#tvm.tirx.IndexMap)

<a id="tvm.tirx.IndexMap.non_surjective_inverse"></a>
#### `tvm.tirx.IndexMap.non_surjective_inverse`

```python
non_surjective_inverse(shape: list[Range | Expr], analyzer=None) → tuple[IndexMap, Expr]
```

Return the inverse of the map

Can be applied to transformations that introduce padding.

**Parameters:**

- **shape** (*List*[*Union*[tvm.ir.Range,tvm.relax.Expr]]) – The region over which the inverse should be determined.
  Used for determining the predicate.
- **analyzer** (*Optional*[tvm.arith.Analyzer]) – The analyzer to use while deriving the inverse and padding predicate.

**Returns:**

**result** – The inverse, and a predicate for which the inverse maps to
a valid index in the input range.

**Return type:**

tvm.relax.Tuple[[IndexMap](#tvm.tirx.IndexMap), tvm.relax.Expr]

Examples

```python
index_map = IndexMap.from_func(lambda i: [i//4, i%4])
inverse_map, predicate = index_map.non_surjective_inverse([14])
assert inverse_map.is_equivalent_to(IndexMap.from_func(lambda j,k: [4*j + k])
print(predicate) # Prints "(axis0==3) && (axis2 >= 2)"
```

<a id="tvm.tirx.call_packed_lowered"></a>
### `tvm.tirx.call_packed_lowered`

```python
tvm.tirx.call_packed_lowered(*args, span=None)
```

Lowered version of call packed.
The argument to packed function can be Expr or Buffer.
The argument is the corresponding POD type when Expr is presented.
When the argument is Buffer, the corresponding PackedFunc
will receive an TVMArrayHandle whose content is valid during the callback period.
If the PackedFunc is a python callback, then the corresponding argument is Tensor.

**Parameters:**

- **args** (list *of* tvm.relax.Expr *or* *Buffer.*) – Positional arguments.
- **span** (*Optional*[tvm.ir.Span]) – The location of this operator in the source code.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

> **See also**
>
> `te.extern`
> :   Create tensor with extern function call.

<a id="tvm.tirx.call_cpacked_lowered"></a>
### `tvm.tirx.call_cpacked_lowered`

```python
tvm.tirx.call_cpacked_lowered(*args, span=None)
```

Lowered version of call c-packed.
Same as call\_packed, except that the first argument is the function name
(as in call\_extern), and the last argument is the resource handle.

**Parameters:**

- **args** (list *of* tvm.relax.Expr *or* *Buffer.*) – Positional arguments.
- **span** (*Optional*[tvm.ir.Span]) – The location of this operator in the source code.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

> **See also**
>
> `te.extern`
> :   Create tensor with extern function call.

<a id="tvm.tirx.call_tir"></a>
### `tvm.tirx.call_tir`

```python
tvm.tirx.call_tir(global_var: GlobalVar, *args)
```

Performs a call into another PrimFunc in the same IRModule

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.call_packed"></a>
### `tvm.tirx.call_packed`

```python
tvm.tirx.call_packed(*args, span=None)
```

Build expression by call an external packed function.

The argument to packed function can be Expr or Buffer.
The argument is the corresponding POD type when Expr is presented.

When the argument is Buffer, the corresponding PackedFunc
will receive an TVMArrayHandle whose content is valid during the callback period.
If the PackedFunc is a python callback, then the corresponding argument is Tensor.

**Parameters:**

- **args** (list *of* tvm.relax.Expr *or* *Buffer.*) – Positional arguments.
- **span** (*Optional*[tvm.ir.Span]) – The location of this operator in the source code.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

> **See also**
>
> `te.extern`
> :   Create tensor with extern function call.

<a id="tvm.tirx.call_cpacked"></a>
### `tvm.tirx.call_cpacked`

```python
tvm.tirx.call_cpacked(*args, span=None)
```

Build expression by call an external packed function.

Same as call\_packed, except that the first argument is the function name
(as in call\_extern), and the last argument is the resource handle.

**Parameters:**

- **args** (list *of* tvm.relax.Expr *or* *Buffer.*) – Positional arguments.
- **span** (*Optional*[tvm.ir.Span]) – The location of this operator in the source code.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

> **See also**
>
> `te.extern`
> :   Create tensor with extern function call.

<a id="tvm.tirx.call_intrin"></a>
### `tvm.tirx.call_intrin`

```python
tvm.tirx.call_intrin(dtype: str | Type, func_name, *args, attrs=None, span=None)
```

Build expression by calling an intrinsic function.

Intrinsics can be overloaded with multiple data types via
the intrinsic translation rule.

**Parameters:**

- **dtype** (str *or* tvm.ir.Type) – The data type of the result.
- **func\_name** (str) – The intrinsic function name.
- **args** (list) – Positional arguments.
- **attrs** (*Optional*[tvm.ir.Attrs *or* *Dict*[str, *Object*]]) – Additional attributes for the call.
- **span** (*Optional*[tvm.ir.Span]) – The location of this operator in the source code.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.call_pure_extern"></a>
### `tvm.tirx.call_pure_extern`

```python
tvm.tirx.call_pure_extern(dtype, func_name, *args, span=None)
```

Build expression by calling a pure extern function.

**Parameters:**

- **dtype** (str) – The data type of the result.
- **func\_name** (str) – The extern function name.
- **args** (list) – Positional arguments.
- **span** (*Optional*[tvm.ir.Span]) – The location of this operator in the source code.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.call_extern"></a>
### `tvm.tirx.call_extern`

```python
tvm.tirx.call_extern(dtype, func_name, *args, span=None)
```

Build expression by calling a extern function.

**Parameters:**

- **dtype** (str) – The data type of the result.
- **func\_name** (str) – The extern function name.
- **args** (list) – Positional arguments.
- **span** (*Optional*[tvm.ir.Span]) – The location of this operator in the source code.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.call_llvm_intrin"></a>
### `tvm.tirx.call_llvm_intrin`

```python
tvm.tirx.call_llvm_intrin(dtype, name, *args, span=None)
```

Build expression by calling a llvm intrinsic function

**Parameters:**

- **dtype** (str) – The data type of the result.
- **name** (str) – The name of the llvm intrinsic function.
- **args** (list) – Positional arguments.
- **span** (*Optional*[tvm.ir.Span]) – The location of this operator in the source code.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.call_llvm_pure_intrin"></a>
### `tvm.tirx.call_llvm_pure_intrin`

```python
tvm.tirx.call_llvm_pure_intrin(dtype, name, *args, span=None)
```

Build expression by calling a pure llvm intrinsic function

**Parameters:**

- **dtype** (str) – The data type of the result.
- **name** (str) – The name of the llvm intrinsic function.
- **args** (list) – Positional arguments.
- **span** (*Optional*[tvm.ir.Span]) – The location of this operator in the source code.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.all"></a>
### `tvm.tirx.all`

```python
tvm.tirx.all(*args, span=None)
```

**Create a new expression of the intersection of all conditions in the**

arguments

**Parameters:**

- **args** (list) – List of symbolic boolean expressions
- **span** (*Optional*[tvm.ir.Span]) – The location of this operator in the source code.

**Returns:**

**expr** – Expression

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.any"></a>
### `tvm.tirx.any`

```python
tvm.tirx.any(*args, span=None)
```

Create a new experssion of the union of all conditions in the arguments

**Parameters:**

- **args** (list) – List of symbolic boolean expressions
- **span** (*Optional*[tvm.ir.Span]) – The location of this operator in the source code.

**Returns:**

**expr** – Expression

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.min_value"></a>
### `tvm.tirx.min_value`

```python
tvm.tirx.min_value(dtype, span=None)
```

minimum value of dtype

**Parameters:**

- **dtype** (str) – The data type.
- **span** (*Optional*[tvm.ir.Span]) – The location of this operator in the source code.

**Returns:**

**value** – The minimum value of dtype.

**Return type:**

tvm.Expr

<a id="tvm.tirx.max_value"></a>
### `tvm.tirx.max_value`

```python
tvm.tirx.max_value(dtype: str, span: Span | None = None) → Any
```

maximum value of dtype

**Parameters:**

- **dtype** (str) – The data type.
- **span** (*Optional*[tvm.ir.Span]) – The location of this operator in the source code.

**Returns:**

**value** – The maximum value of dtype.

**Return type:**

tvm.Expr

<a id="tvm.tirx.trace"></a>
### `tvm.tirx.trace`

```python
tvm.tirx.trace(args, trace_action='tvm.default_trace_action')
```

Trace tensor data at the runtime.

The trace function allows to trace specific tensor at the
runtime. The tracing value should come as last argument.
The trace action should be specified, by default
tvm.default\_trace\_action is used.

**Parameters:**

- **args** (list *of* tvm.relax.Expr *or* *Buffers.*) – Positional arguments.
- **trace\_action** (*str.*) – The name of the trace action.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

> **See also**
>
> [`tvm.tirx.call_packed`](#tvm.tirx.call_packed)
> :   Creates packed function.

<a id="tvm.tirx.tvm_stack_alloca"></a>
### `tvm.tirx.tvm_stack_alloca`

```python
tvm.tirx.tvm_stack_alloca(dtype_str, num)
```

Return new on stack dtype[num]

**Parameters:**

- **dtype\_str** (str) – The data type of array.
- **num** (int) – The size of array.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.tvm_stack_make_shape"></a>
### `tvm.tirx.tvm_stack_make_shape`

```python
tvm.tirx.tvm_stack_make_shape(*args)
```

Allocate a shape tuple on stack, return the handle

**Parameters:**

**args** (int) – The tuple shape.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.tvm_stack_make_array"></a>
### `tvm.tirx.tvm_stack_make_array`

```python
tvm.tirx.tvm_stack_make_array(data, shape, strides, ndim, arr_dtype, elem_offset)
```

Allocate a Tensor(DLTensor) on stack, return the handle

**Parameters:**

- **data** (tvm.relax.Expr) – The data of array.
- **shape** (tvm.relax.Expr) – The shape of array.
- **strides** (tvm.relax.Expr) – The strides of array.
- **ndim** (tvm.relax.Expr) – The dimensions of array.
- **arr\_dtype** (tvm.relax.Expr) – The data type of array.
- **elem\_offse** (tvm.relax.Expr) – The element offset of array.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.tvm_tuple"></a>
### `tvm.tirx.tvm_tuple`

```python
tvm.tirx.tvm_tuple(*value)
```

Create a tuple structure in value field of AttrStmt

**Parameters:**

**value** (tvm.relax.Expr) – The value in tuple.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.handle_add_byte_offset"></a>
### `tvm.tirx.handle_add_byte_offset`

```python
tvm.tirx.handle_add_byte_offset(handle, offset)
```

Add offset to handle

**Parameters:**

- **handle** (tvm.relax.Expr) – The handle.
- **offset** (int) – The offset.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.tvm_struct_get"></a>
### `tvm.tirx.tvm_struct_get`

```python
tvm.tirx.tvm_struct_get(arr, index, field, dtype)
```

Get struct field value in array

**Parameters:**

- **dtype** (str) – The date type of the result.
- **arr** (*StructType\**) – The array of struct.
- **index** (int) – The index of struct.
- **field** (int) – The field of struct.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.tvm_struct_set"></a>
### `tvm.tirx.tvm_struct_set`

```python
tvm.tirx.tvm_struct_set(arr, index, field, value)
```

Set value in struct field in array

**Parameters:**

- **arr** (*StructType\**) – The array of struct.
- **index** (int) – The index of struct.
- **field** (int) – The field of struct.
- **value** (tvm.relax.Expr) – The value to be set in field.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.address_of"></a>
### `tvm.tirx.address_of`

```python
tvm.tirx.address_of(obj: Buffer | BufferLoad | Var, span: Span | None = None) → Expr
```

Returns the address of a buffer element or addressable variable.

**Parameters:**

- **obj** (*Union*[[*Buffer*](#tvm.tirx.Buffer), [*BufferLoad*](#tvm.tirx.BufferLoad), *tirx.Var*]) – The buffer, buffer load, or addressable variable.
- **span** (*Optional*[tvm.ir.Span]) – The location of this operator in the source code.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.lookup_param"></a>
### `tvm.tirx.lookup_param`

```python
tvm.tirx.lookup_param(param_name, span=None)
```

Returns the param by name

**Parameters:**

- **param\_name** (str) – The name of param.
- **span** (*Optional*[tvm.ir.Span]) – The location of this operator in the source code.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.assume"></a>
### `tvm.tirx.assume`

```python
tvm.tirx.assume(cond=None)
```

Provide a true statement that can be used for simplifications

**Parameters:**

**cond** (tvm.relax.Expr) – The constraint condition.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.undef"></a>
### `tvm.tirx.undef`

```python
tvm.tirx.undef()
```

Returns an initialized but arbitrary value

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.continue_loop"></a>
### `tvm.tirx.continue_loop`

```python
tvm.tirx.continue_loop(span=None)
```

Create a tir intrinsic call to represent continue expression

**Parameters:**

**span** (*Optional*[tvm.ir.Span]) – The location of this operator in the source code.

**Returns:**

**ret** – The continue expression

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.break_loop"></a>
### `tvm.tirx.break_loop`

```python
tvm.tirx.break_loop(span=None)
```

Create a tir intrinsic call to represent break expression

**Parameters:**

**span** (*Optional*[tvm.ir.Span]) – The location of this operator in the source code.

**Returns:**

**ret** – The break expression

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.tvm_thread_allreduce"></a>
### `tvm.tirx.tvm_thread_allreduce`

```python
tvm.tirx.tvm_thread_allreduce(*freduce_args)
```

Perform allreduce inside threadblock.

**Parameters:**

**freduce\_args** (tvm.relax.Expr) – The args.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.type_annotation"></a>
### `tvm.tirx.type_annotation`

```python
tvm.tirx.type_annotation(dtype)
```

Create a type annotation expression

**Parameters:**

**dtype** (tvm.relax.Expr) – The data type.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.tvm_access_ptr"></a>
### `tvm.tirx.tvm_access_ptr`

```python
tvm.tirx.tvm_access_ptr(ptype, data, offset, extent, rw_mask)
```

Get head access address with memory access pattern info

**Parameters:**

- **ptype** (tvm.relax.Expr *or* str) – The data type of pointer. If a `str`, it is wrapped via
  [`type_annotation()`](#tvm.tirx.type_annotation) so that the lowering rule (which reads
  `args[0].dtype()` for the cast type) sees the intended dtype
  instead of `void` from a raw StringImm.
- **data** (*DType\**) – The data of pointer.
- **offset** (int) – The offset of pointer.
- **extent** (int) – The extent of pointer.
- **rw\_mask** (int) – The read write mask.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.ptr_byte_offset"></a>
### `tvm.tirx.ptr_byte_offset`

```python
tvm.tirx.ptr_byte_offset(data, byte_offset, dtype)
```

Cast `data + byte_offset` to `dtype*`.

`byte_offset` is always in bytes. Use this when the source CUDA shape
needs an explicitly typed local pointer derived from a byte-addressed base.

<a id="tvm.tirx.tvm_throw_last_error"></a>
### `tvm.tirx.tvm_throw_last_error`

```python
tvm.tirx.tvm_throw_last_error()
```

Throw TVMGetLastError()

**Returns:**

**ret** – The return expression

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.tvm_load_matrix_sync"></a>
### `tvm.tirx.tvm_load_matrix_sync`

```python
tvm.tirx.tvm_load_matrix_sync(fragment, m, n, k, index, buffer_ptr, stride, layout)
```

TVM intrinsic for tensor core load operators

**Parameters:**

- **fragment** (*tirx.Var*) – The wmma fragment.
- **m** (*UIntImm*) – The shape of wmma fragment.
- **n** (*UIntImm*) – The shape of wmma fragment.
- **k** (*UIntImm*) – The shape of wmma fragment.
- **index** (tvm.relax.Expr) – The fragment index.
- **buffer\_ptr** (tvm.relax.Expr) – The fragment buffer pointer.
- **stride** (tvm.relax.Expr) – The fragment stride.
- **layout** (*Literal*[*"row\_major"*, *"column\_major"*]) – The fragment layout.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.tvm_store_matrix_sync"></a>
### `tvm.tirx.tvm_store_matrix_sync`

```python
tvm.tirx.tvm_store_matrix_sync(fragment, m, n, k, index, buffer_ptr, stride, layout)
```

TVM intrinsic for tensor core store operators

**Parameters:**

- **fragment** (*tirx.Var*) – The wmma fragment.
- **m** (*UIntImm*) – The shape of wmma fragment.
- **n** (*UIntImm*) – The shape of wmma fragment.
- **k** (*UIntImm*) – The shape of wmma fragment.
- **index** (tvm.relax.Expr) – The fragment index.
- **buffer\_ptr** (tvm.relax.Expr) – The fragment buffer pointer.
- **stride** (tvm.relax.Expr) – The fragment stride.
- **layout** (*Literal*[*"row\_major"*, *"column\_major"*]) – The fragment layout.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.tvm_mma_sync"></a>
### `tvm.tirx.tvm_mma_sync`

```python
tvm.tirx.tvm_mma_sync(fragment_d, index_d, fragment_a, index_a, fragment_b, index_b, fragment_c, index_c)
```

TVM intrinsic for tensor core mma\_sync operators

**Parameters:**

- **fragment\_d** (*tirx.Var*) – The wmma fragment\_d.
- **index\_d** (tvm.relax.Expr) – The fragment\_d index.
- **fragment\_a** (*tirx.Var*) – The wmma fragment\_a.
- **index\_a** (tvm.relax.Expr) – The fragment\_a index.
- **fragment\_b** (*tirx.Var*) – The wmma fragment\_b.
- **index\_b** (tvm.relax.Expr) – The fragment\_b index.
- **fragment\_c** (*tirx.Var*) – The wmma fragment\_c.
- **index\_c** (tvm.relax.Expr) – The fragment\_c index.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.tvm_bmma_sync"></a>
### `tvm.tirx.tvm_bmma_sync`

```python
tvm.tirx.tvm_bmma_sync(fragment_d, index_d, fragment_a, index_a, fragment_b, index_b, fragment_c, index_c)
```

TVM intrinsic for tensor core bmma\_sync operators

**Parameters:**

- **fragment\_d** (*tirx.Var*) – The bwmma fragment\_d.
- **index\_d** (tvm.relax.Expr) – The fragment\_d index.
- **fragment\_a** (*tirx.Var*) – The bwmma fragment\_a.
- **index\_a** (tvm.relax.Expr) – The fragment\_a index.
- **fragment\_b** (*tirx.Var*) – The bwmma fragment\_b.
- **index\_b** (tvm.relax.Expr) – The fragment\_b index.
- **fragment\_c** (*tirx.Var*) – The bwmma fragment\_c.
- **index\_c** (tvm.relax.Expr) – The fragment\_c index.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.tvm_fill_fragment"></a>
### `tvm.tirx.tvm_fill_fragment`

```python
tvm.tirx.tvm_fill_fragment(fragment, m, n, k, index, value)
```

TVM intrinsic for tensor core fill\_fragment operators

**Parameters:**

- **fragment** (*tirx.Var*) – The wmma fragment
- **m** (*UIntImm*) – The shape of wmma fragment.
- **n** (*UIntImm*) – The shape of wmma fragment.
- **k** (*UIntImm*) – The shape of wmma fragment.
- **index** (tvm.relax.Expr) – The fragment index.
- **value** (tvm.relax.Expr) – The value to be filled in fragment.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.vectorlow"></a>
### `tvm.tirx.vectorlow`

```python
tvm.tirx.vectorlow(dtype, vec)
```

Get the low level half of the vector

**Parameters:**

- **dtype** (str) – The data type of the result.
- **vec** (list) – The input vector.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.vectorhigh"></a>
### `tvm.tirx.vectorhigh`

```python
tvm.tirx.vectorhigh(dtype, vec)
```

Get the high level half of the vector

**Parameters:**

- **dtype** (str) – The data type of the result.
- **vec** (list) – The input vector.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.vectorcombine"></a>
### `tvm.tirx.vectorcombine`

```python
tvm.tirx.vectorcombine(dtype, vec1, vec2)
```

Concat two vectors

**Parameters:**

- **vec1** (list) – The input vector.
- **vec2** (list) – The input vector.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.infinity"></a>
### `tvm.tirx.infinity`

```python
tvm.tirx.infinity(dtype: str, span: Span | None = None) → Any
```

infinity value of dtype

**Parameters:**

- **dtype** (str) – The data type.
- **span** (*Optional*[tvm.ir.Span]) – The location of this operator in the source code.

**Returns:**

**value** – The infinity value of dtype.

**Return type:**

tvm.Expr

<a id="tvm.tirx.reinterpret"></a>
### `tvm.tirx.reinterpret`

```python
tvm.tirx.reinterpret(dtype, value, span: Span | None = None) → Expr
```

Reinterpret a value as an exact primitive or pointer type.

**Parameters:**

- **dtype** (str *or* tvm.ir.Type) – The data type.
- **value** (tvm.relax.Expr) – The input value.
- **span** (*Optional*[tvm.ir.Span]) – The location of this operator in the source code.

**Returns:**

**value** – The reinterpret cast value of dtype.

**Return type:**

tvm.Expr

<a id="tvm.tirx.exp"></a>
### `tvm.tirx.exp`

```python
tvm.tirx.exp(x)
```

Take exponential of input x.

**Parameters:**

**x** (tvm.relax.Expr) – Input argument.

**Returns:**

**y** – The result.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.exp2"></a>
### `tvm.tirx.exp2`

```python
tvm.tirx.exp2(x)
```

Calculate 2\*\*x

**Parameters:**

**x** (tvm.relax.Expr) – Input argument.

**Returns:**

**y** – The result.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.exp10"></a>
### `tvm.tirx.exp10`

```python
tvm.tirx.exp10(x)
```

Calculate 10\*\*x

**Parameters:**

**x** (tvm.relax.Expr) – Input argument.

**Returns:**

**y** – The result.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.log"></a>
### `tvm.tirx.log`

```python
tvm.tirx.log(x)
```

Take log of input x.

**Parameters:**

**x** (tvm.relax.Expr) – Input argument.

**Returns:**

**y** – The result.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.log2"></a>
### `tvm.tirx.log2`

```python
tvm.tirx.log2(x)
```

Take log2 of input x.

**Parameters:**

**x** (tvm.relax.Expr) – Input argument.

**Returns:**

**y** – The result.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.log10"></a>
### `tvm.tirx.log10`

```python
tvm.tirx.log10(x)
```

Take log10 of input x.

**Parameters:**

**x** (tvm.relax.Expr) – Input argument.

**Returns:**

**y** – The result.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.log1p"></a>
### `tvm.tirx.log1p`

```python
tvm.tirx.log1p(x)
```

Take log(x + 1) with respect to input x.

**Parameters:**

**x** (tvm.relax.Expr) – Input argument.

**Returns:**

**y** – The result.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.ldexp"></a>
### `tvm.tirx.ldexp`

```python
tvm.tirx.ldexp(x1, x2)
```

Returns x1 \* (2 \*\* x2).

**Parameters:**

- **x1** (tvm.relax.Expr) – Input argument.
- **x2** (tvm.relax.Expr) – Input argument.

**Returns:**

**y** – The result.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.clz"></a>
### `tvm.tirx.clz`

```python
tvm.tirx.clz(x)
```

Count leading zero bits of an integer x.

**Parameters:**

**x** (tvm.relax.Expr) – Input 32 or 64 bit integer.
The result is undefined if the input is 0.

**Returns:**

**y** – The result.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.sin"></a>
### `tvm.tirx.sin`

```python
tvm.tirx.sin(x)
```

Take sin of input x.

**Parameters:**

**x** (tvm.relax.Expr) – Input argument.

**Returns:**

**y** – The result.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.sinh"></a>
### `tvm.tirx.sinh`

```python
tvm.tirx.sinh(x)
```

Take sinh of input x.

**Parameters:**

**x** (tvm.relax.Expr) – Input argument.

**Returns:**

**y** – The result.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.asin"></a>
### `tvm.tirx.asin`

```python
tvm.tirx.asin(x)
```

Take asin of input x.

**Parameters:**

**x** (tvm.relax.Expr) – Input argument.

**Returns:**

**y** – The result.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.asinh"></a>
### `tvm.tirx.asinh`

```python
tvm.tirx.asinh(x)
```

Take asinh of input x.

**Parameters:**

**x** (tvm.relax.Expr) – Input argument.

**Returns:**

**y** – The result.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.cos"></a>
### `tvm.tirx.cos`

```python
tvm.tirx.cos(x)
```

Take cos of input x.

**Parameters:**

**x** (tvm.relax.Expr) – Input argument.

**Returns:**

**y** – The result.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.cosh"></a>
### `tvm.tirx.cosh`

```python
tvm.tirx.cosh(x)
```

Take cosh of input x.

**Parameters:**

**x** (tvm.relax.Expr) – Input argument.

**Returns:**

**y** – The result.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.acos"></a>
### `tvm.tirx.acos`

```python
tvm.tirx.acos(x)
```

Take acos of input x.

**Parameters:**

**x** (tvm.relax.Expr) – Input argument.

**Returns:**

**y** – The result.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.acosh"></a>
### `tvm.tirx.acosh`

```python
tvm.tirx.acosh(x)
```

Take acos of input x.

**Parameters:**

**x** (tvm.relax.Expr) – Input argument.

**Returns:**

**y** – The result.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.tan"></a>
### `tvm.tirx.tan`

```python
tvm.tirx.tan(x)
```

Take tan of input x.

**Parameters:**

**x** (tvm.relax.Expr) – Input argument.

**Returns:**

**y** – The result.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.tanh"></a>
### `tvm.tirx.tanh`

```python
tvm.tirx.tanh(x)
```

Take hyperbolic tanh of input x.

**Parameters:**

**x** (tvm.relax.Expr) – Input argument.

**Returns:**

**y** – The result.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.atan"></a>
### `tvm.tirx.atan`

```python
tvm.tirx.atan(x)
```

Take atan of input x.

**Parameters:**

**x** (tvm.relax.Expr) – Input argument.

**Returns:**

**y** – The result.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.atan2"></a>
### `tvm.tirx.atan2`

```python
tvm.tirx.atan2(x1, x2)
```

Take arctan2(x1, x2).

**Parameters:**

- **x1** (tvm.relax.Expr) – Input argument.
- **x2** (tvm.relax.Expr) – Input argument.

**Returns:**

**y** – The result.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.atanh"></a>
### `tvm.tirx.atanh`

```python
tvm.tirx.atanh(x)
```

Take atanh of input x.

**Parameters:**

**x** (tvm.relax.Expr) – Input argument.

**Returns:**

**y** – The result.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.bitwise_and"></a>
### `tvm.tirx.bitwise_and`

```python
tvm.tirx.bitwise_and(x, y, span=None)
```

Take bitwise and of two values

**Parameters:**

- **x** (tvm.relax.Expr) – Left operand
- **y** (tvm.relax.Expr) – Right operand
- **span** (*Optional*[tvm.ir.Span]) – The location of this operator in the source code.

**Returns:**

**res** – The result.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.bitwise_not"></a>
### `tvm.tirx.bitwise_not`

```python
tvm.tirx.bitwise_not(x, span=None)
```

Take bitwise not of input value

**Parameters:**

- **x** (tvm.relax.Expr) – Input operand
- **span** (*Optional*[tvm.ir.Span]) – The location of this operator in the source code.

**Returns:**

**res** – The result.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.bitwise_or"></a>
### `tvm.tirx.bitwise_or`

```python
tvm.tirx.bitwise_or(x, y, span=None)
```

Take bitwise or of two values

**Parameters:**

- **x** (tvm.relax.Expr) – Left operand
- **y** (tvm.relax.Expr) – Right operand
- **span** (*Optional*[tvm.ir.Span]) – The location of this operator in the source code.

**Returns:**

**res** – The result.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.bitwise_xor"></a>
### `tvm.tirx.bitwise_xor`

```python
tvm.tirx.bitwise_xor(x, y, span=None)
```

Take bitwise xor of two values

**Parameters:**

- **x** (tvm.relax.Expr) – Left operand
- **y** (tvm.relax.Expr) – Right operand
- **span** (*Optional*[tvm.ir.Span]) – The location of this operator in the source code.

**Returns:**

**res** – The result.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.erf"></a>
### `tvm.tirx.erf`

```python
tvm.tirx.erf(x)
```

Take gauss error function of the input x.

**Parameters:**

**x** (tvm.relax.Expr) – Input argument.

**Returns:**

**y** – The result.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.sigmoid"></a>
### `tvm.tirx.sigmoid`

```python
tvm.tirx.sigmoid(x)
```

Quick function to get sigmoid

**Parameters:**

**x** (tvm.relax.Expr) – Input argument.

**Returns:**

**y** – The result.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.sqrt"></a>
### `tvm.tirx.sqrt`

```python
tvm.tirx.sqrt(x)
```

Take square root of input x.

**Parameters:**

**x** (tvm.relax.Expr) – Input argument.

**Returns:**

**y** – The result.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.rsqrt"></a>
### `tvm.tirx.rsqrt`

```python
tvm.tirx.rsqrt(x)
```

Take reciprocal of square root of input x.

**Parameters:**

**x** (tvm.relax.Expr) – Input argument.

**Returns:**

**y** – The result.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.floor"></a>
### `tvm.tirx.floor`

```python
tvm.tirx.floor(x: ExprWithOp, span=None)
```

Take floor of float input x.

**Parameters:**

- **x** (tvm.relax.Expr) – Input argument.
- **span** (*Optional*[tvm.ir.Span]) – The location of this operator in the source code.

**Returns:**

**y** – The result.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.ceil"></a>
### `tvm.tirx.ceil`

```python
tvm.tirx.ceil(x, span=None)
```

Take ceil of float input x.

**Parameters:**

- **x** (tvm.relax.Expr) – Input argument.
- **span** (*Optional*[tvm.ir.Span]) – The location of this operator in the source code.

**Returns:**

**y** – The result.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.hypot"></a>
### `tvm.tirx.hypot`

```python
tvm.tirx.hypot(x1, x2)
```

Equivalent to sqrt(x1\*\*2 + x2\*\*2), element-wise.

**Parameters:**

- **x1** (tvm.relax.Expr) – Input argument.
- **x2** (tvm.relax.Expr) – Input argument.

**Returns:**

**y** – The result.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.trunc"></a>
### `tvm.tirx.trunc`

```python
tvm.tirx.trunc(x, span=None)
```

Get truncated value of the input.

The truncated value of the scalar x is the
nearest integer i which is closer to zero than x is.

**Parameters:**

- **x** (tvm.relax.Expr) – Input argument.
- **span** (*Optional*[tvm.ir.Span]) – The location of this operator in the source code.

**Returns:**

**y** – The result.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.abs"></a>
### `tvm.tirx.abs`

```python
tvm.tirx.abs(x, span=None)
```

Get absolute value of the input element-wise.

**Parameters:**

- **x** (tvm.relax.Expr) – Input argument.
- **span** (*Optional*[tvm.ir.Span]) – The location of this operator in the source code.

**Returns:**

**y** – The result.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.round"></a>
### `tvm.tirx.round`

```python
tvm.tirx.round(x, span=None)
```

Round elements of the array to the nearest integer.

**Parameters:**

- **x** (tvm.relax.Expr) – Input argument.
- **span** (*Optional*[tvm.ir.Span]) – The location of this operator in the source code.

**Returns:**

**y** – The result.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.nextafter"></a>
### `tvm.tirx.nextafter`

```python
tvm.tirx.nextafter(x1, x2)
```

Return the next floating-point value after x1 towards x2.

**Parameters:**

- **x1** (tvm.relax.Expr) – Input argument.
- **x2** (tvm.relax.Expr) – Input argument.

**Returns:**

**y** – The result.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.nearbyint"></a>
### `tvm.tirx.nearbyint`

```python
tvm.tirx.nearbyint(x, span=None)
```

Round elements of the array to the nearest integer.
This intrinsic uses llvm.nearbyint instead of llvm.round
which is faster but will results different from te.round.
Notably nearbyint rounds according to the rounding mode,
whereas te.round (llvm.round) ignores that.
For differences between the two see:
<https://en.cppreference.com/w/cpp/numeric/math/round>
<https://en.cppreference.com/w/cpp/numeric/math/nearbyint>

**Parameters:**

- **x** (tvm.relax.Expr) – Input argument.
- **span** (*Optional*[tvm.ir.Span]) – The location of this operator in the source code.

**Returns:**

**y** – The result.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.power"></a>
### `tvm.tirx.power`

```python
tvm.tirx.power(x, y, span=None)
```

x power y

**Parameters:**

- **x** (tvm.relax.Expr) – Input argument.
- **y** (tvm.relax.Expr) – The exponent
- **span** (*Optional*[tvm.ir.Span]) – The location of this operator in the source code.

**Returns:**

**z** – The result.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.pow"></a>
### `tvm.tirx.pow`

```python
tvm.tirx.pow(x, y, span=None)
```

x power y

**Parameters:**

- **x** (tvm.relax.Expr) – Input argument.
- **y** (tvm.relax.Expr) – The exponent
- **span** (*Optional*[tvm.ir.Span]) – The location of this operator in the source code.

**Returns:**

**z** – The result.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.popcount"></a>
### `tvm.tirx.popcount`

```python
tvm.tirx.popcount(x)
```

Count the number of set bits in input x.

**Parameters:**

**x** (tvm.relax.Expr) – Input argument.

**Returns:**

**y** – The result.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.fmod"></a>
### `tvm.tirx.fmod`

```python
tvm.tirx.fmod(x, y)
```

Return the remainder of x divided by y with the same sign as x.

**Parameters:**

- **x** (tvm.relax.Expr) – Input argument.
- **y** (tvm.relax.Expr) – Input argument.

**Returns:**

**z** – The result.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.if_then_else"></a>
### `tvm.tirx.if_then_else`

```python
tvm.tirx.if_then_else(cond, t, f, span=None)
```

Conditional selection expression.

**Parameters:**

- **cond** (tvm.relax.Expr) – The condition
- **t** (tvm.relax.Expr) – The result expression if cond is true.
- **f** (tvm.relax.Expr) – The result expression if cond is false.
- **span** (*Optional*[tvm.ir.Span]) – The location of this operator in the source.

**Returns:**

**result** – The result of conditional expression.

**Return type:**

tvm.ir.Node

> **Note**
>
> Unlike Select, if\_then\_else will not execute
> the branch that does not satisfy the condition.
> You can use it to guard against out of bound access.
> Unlike Select, if\_then\_else cannot be vectorized
> if some lanes in the vector have different conditions.

<a id="tvm.tirx.likely"></a>
### `tvm.tirx.likely`

```python
tvm.tirx.likely(cond, span=None)
```

Mark condition as likely.

**Parameters:**

- **cond** (tvm.relax.Expr) – Input argument.
- **span** (*Optional*[tvm.ir.Span]) – The location of this operator in the source code.

**Returns:**

**y** – The marked expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.isnan"></a>
### `tvm.tirx.isnan`

```python
tvm.tirx.isnan(x, span=None)
```

Check if input value is Nan.

**Parameters:**

- **x** (tvm.relax.Expr) – Input argument.
- **span** (*Optional*[tvm.ir.Span]) – The location of this operator in the source code.

**Returns:**

**y** – The result.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.isnullptr"></a>
### `tvm.tirx.isnullptr`

```python
tvm.tirx.isnullptr(x, span=None)
```

Check if input value is nullptr.

**Parameters:**

- **x** (tvm.relax.Expr) – Input argument.
- **span** (*Optional*[tvm.ir.Span]) – The location of this operator in the source code.

**Returns:**

**y** – The result.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.isfinite"></a>
### `tvm.tirx.isfinite`

```python
tvm.tirx.isfinite(x, span=None)
```

Check if input value is finite.

**Parameters:**

- **x** (tvm.relax.Expr) – Input argument.
- **span** (*Optional*[tvm.ir.Span]) – The location of this operator in the source code.

**Returns:**

**y** – The result.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.isinf"></a>
### `tvm.tirx.isinf`

```python
tvm.tirx.isinf(x, span=None)
```

Check if input value is infinite.

**Parameters:**

- **x** (tvm.relax.Expr) – Input argument.
- **span** (*Optional*[tvm.ir.Span]) – The location of this operator in the source code.

**Returns:**

**y** – The result.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.copysign"></a>
### `tvm.tirx.copysign`

```python
tvm.tirx.copysign(x1, x2)
```

Change the sign of x1 to that of x2, element-wise.

**Parameters:**

- **x1** (tvm.relax.Expr) – Input argument.
- **x2** (tvm.relax.Expr) – Input argument.

**Returns:**

**y** – The result.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.div"></a>
### `tvm.tirx.div`

```python
tvm.tirx.div(a, b, span=None)
```

Compute a / b as in C/C++ semantics.

**Parameters:**

- **a** (tvm.relax.Expr) – The left hand operand, known to be non-negative.
- **b** (tvm.relax.Expr) – The right hand operand, known to be non-negative.
- **span** (*Optional*[tvm.ir.Span]) – The location of this operator in the source.

**Returns:**

**res** – The result expression.

**Return type:**

tvm.relax.Expr

> **Note**
>
> When operands are integers, returns truncdiv(a, b, span).

<a id="tvm.tirx.indexdiv"></a>
### `tvm.tirx.indexdiv`

```python
tvm.tirx.indexdiv(a, b, span=None)
```

Compute floor(a / b) where a and b are non-negative.

**Parameters:**

- **a** (tvm.relax.Expr) – The left hand operand, known to be non-negative.
- **b** (tvm.relax.Expr) – The right hand operand, known to be non-negative.
- **span** (*Optional*[tvm.ir.Span]) – The location of this operator in the source.

**Returns:**

**res** – The result expression.

**Return type:**

tvm.relax.Expr

> **Note**
>
> Use this function to split non-negative indices.
> This function may take advantage of operands’
> non-negativeness.

<a id="tvm.tirx.indexmod"></a>
### `tvm.tirx.indexmod`

```python
tvm.tirx.indexmod(a, b, span=None)
```

Compute the remainder of indexdiv. a and b are non-negative.

**Parameters:**

- **a** (tvm.relax.Expr) – The left hand operand, known to be non-negative.
- **b** (tvm.relax.Expr) – The right hand operand, known to be non-negative.
- **span** (*Optional*[tvm.ir.Span]) – The location of this operator in the source.

**Returns:**

**res** – The result expression.

**Return type:**

tvm.relax.Expr

> **Note**
>
> Use this function to split non-negative indices.
> This function may take advantage of operands’
> non-negativeness.

<a id="tvm.tirx.truncdiv"></a>
### `tvm.tirx.truncdiv`

```python
tvm.tirx.truncdiv(a, b, span=None)
```

Compute the truncdiv of two expressions.

**Parameters:**

- **a** (tvm.relax.Expr) – The left hand operand
- **b** (tvm.relax.Expr) – The right hand operand
- **span** (*Optional*[tvm.ir.Span]) – The location of this operator in the source.

**Returns:**

**res** – The result expression.

**Return type:**

tvm.relax.Expr

> **Note**
>
> This is the default integer division behavior in C.

<a id="tvm.tirx.truncmod"></a>
### `tvm.tirx.truncmod`

```python
tvm.tirx.truncmod(a, b, span=None)
```

Compute the truncmod of two expressions.

**Parameters:**

- **a** (tvm.relax.Expr) – The left hand operand
- **b** (tvm.relax.Expr) – The right hand operand
- **span** (*Optional*[tvm.ir.Span]) – The location of this operator in the source.

**Returns:**

**res** – The result expression.

**Return type:**

tvm.relax.Expr

> **Note**
>
> This is the default integer division behavior in C.

<a id="tvm.tirx.floordiv"></a>
### `tvm.tirx.floordiv`

```python
tvm.tirx.floordiv(a, b, span=None)
```

Compute the floordiv of two expressions.

**Parameters:**

- **a** (tvm.relax.Expr) – The left hand operand
- **b** (tvm.relax.Expr) – The right hand operand
- **span** (*Optional*[tvm.ir.Span]) – The location of this operator in the source.

**Returns:**

**res** – The result expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.floormod"></a>
### `tvm.tirx.floormod`

```python
tvm.tirx.floormod(a, b, span=None)
```

Compute the floormod of two expressions.

**Parameters:**

- **a** (tvm.relax.Expr) – The left hand operand
- **b** (tvm.relax.Expr) – The right hand operand
- **span** (*Optional*[tvm.ir.Span]) – The location of this operator in the source.

**Returns:**

**res** – The result expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.ceildiv"></a>
### `tvm.tirx.ceildiv`

```python
tvm.tirx.ceildiv(lhs, rhs, span=None)
```

Generic ceildiv operator.

**Parameters:**

- **lhs** (object) – The left operand.
- **rhs** (object) – The right operand.
- **span** (*Optional*[tvm.ir.Span]) – The location of this operator in the source.

**Returns:**

**op** – The result Expr of ceildiv operaton.

**Return type:**

tvm.Expr

<a id="tvm.tirx.logaddexp"></a>
### `tvm.tirx.logaddexp`

```python
tvm.tirx.logaddexp(a, b, span=None)
```

Compute the logaddexp of two expressions.

**Parameters:**

- **a** (tvm.relax.Expr) – The left hand operand
- **b** (tvm.relax.Expr) – The right hand operand
- **span** (*Optional*[tvm.ir.Span]) – The location of this operator in the source.

**Returns:**

**res** – The result expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.comm_reducer"></a>
### `tvm.tirx.comm_reducer`

```python
tvm.tirx.comm_reducer(fcombine, fidentity, name='reduce')
```

Create a commutative reducer for reduction.

**Parameters:**

- **fcombine** (*function**(**Expr -> Expr -> Expr**)*) – A binary function which takes two Expr as input to return a Expr.
- **fidentity** (*function**(**str -> Expr**)*) – A function which takes a type string as input to return a const Expr.

**Returns:**

**reducer** – A function which creates a reduce expression over axis.
There are two ways to use it:

1. accept (expr, axis, where) to produce an Reduce Expr on
   specified axis;
2. simply use it with multiple Exprs.

**Return type:**

function

Example

```python
n = te.var("n")
m = te.var("m")
mysum = te.comm_reducer(lambda x, y: x+y,
    lambda t: tvm.tirx.const(0, dtype=t), name="mysum")
A = te.placeholder((n, m), name="A")
k = te.reduce_axis((0, m), name="k")
B = te.compute((n,), lambda i: mysum(A[i, k], axis=k), name="B")
```

<a id="tvm.tirx.min"></a>
### `tvm.tirx.min`

```python
tvm.tirx.min(expr, axis, where=None, init=None, *args)
```

Create a min expression over axis.

**Parameters:**

- **expr** (tvm.relax.Expr) – The source expression.
- **axis** ([*IterVar*](#tvm.tirx.IterVar)) – The reduction IterVar axis
- **where** (*optional*, tvm.relax.Expr) – Filtering predicate of the reduction.

**Returns:**

**value** – The result value.

**Return type:**

tvm.relax.Expr

Example

```python
m = te.var("m")
n = te.var("n")
A = te.placeholder((m, n), name="A")
k = te.reduce_axis((0, n), name="k")

# there are two way to use this min reducer:
# mode 1, accept (expr, axis, where) to produce an Reduce Expr
# tvm.min represents tvm.te.min or tvm.tirx.min.
B = te.compute((m,), lambda i: tvm.min(A[i, k], axis=k), name="B")

# mode 2, simply use it with multiple Exprs:
min_res = tvm.min(m, n)
```

<a id="tvm.tirx.max"></a>
### `tvm.tirx.max`

```python
tvm.tirx.max(expr, axis, where=None, init=None, *args)
```

Create a max expression over axis.

**Parameters:**

- **expr** (tvm.relax.Expr) – The source expression.
- **axis** ([*IterVar*](#tvm.tirx.IterVar)) – The reduction IterVar axis
- **where** (*optional*, tvm.relax.Expr) – Filtering predicate of the reduction.

**Returns:**

**value** – The result value.

**Return type:**

tvm.relax.Expr

Example

```python
m = te.var("m")
n = te.var("n")
A = te.placeholder((m, n), name="A")
k = te.reduce_axis((0, n), name="k")

# there are two way to use this max reducer:
# mode 1, accept (expr, axis, where) to produce an Reduce Expr
# tvm.max represents tvm.te.max or tvm.tirx.max.
B = te.compute((m,), lambda i: tvm.max(A[i, k], axis=k), name="B")

# mode 2, simply use it with multiple Exprs:
max_res = tvm.max(m, n)
```

<a id="tvm.tirx.sum"></a>
### `tvm.tirx.sum`

```python
tvm.tirx.sum(expr, axis, where=None, init=None, *args)
```

Create a sum expression over axis.

**Parameters:**

- **expr** (tvm.relax.Expr) – The source expression.
- **axis** ([*IterVar*](#tvm.tirx.IterVar)) – The reduction IterVar axis
- **where** (*optional*, tvm.relax.Expr) – Filtering predicate of the reduction.

**Returns:**

**value** – The result value.

**Return type:**

tvm.relax.Expr

Example

```python
m = te.var("m")
n = te.var("n")
A = te.placeholder((m, n), name="A")
k = te.reduce_axis((0, n), name="k")

# there are two way to use this sum reducer:
# mode 1, accept (expr, axis, where) to produce an Reduce Expr
# tvm.sum represents tvm.te.sum or tvm.tirx.sum.
B = te.compute((m,), lambda i: tvm.sum(A[i, k], axis=k), name="B")

# mode 2, simply use it with multiple Exprs:
sum_res = tvm.sum(m, n)
```

<a id="tvm.tirx.q_multiply_shift"></a>
### `tvm.tirx.q_multiply_shift`

```python
tvm.tirx.q_multiply_shift(x, y, q, s)
```

Execute a multiplication between two Q-numbers x and y
followed by a right shift s. The mathematical expression is:

> out = round(x\*y\*2^-s)

More about Q-numbers here: <https://en.wikipedia.org/wiki/Q_(number_format>)
The rounding rule is to the nearest value, rounding half up
(i.e., round(x.1) = x and round (x.5) = x+1)

**Parameters:**

- **x** (tvm.relax.Expr) – First Q-number
- **y** (tvm.relax.Expr) – Second Q-number
- **q** (tvm.relax.Expr) – Number of fractional bits in x and y. Needs to be > 0
- **s** (tvm.relax.Expr) – Integer shift

**Returns:**

**y** – The result.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.q_multiply_shift_per_axis"></a>
### `tvm.tirx.q_multiply_shift_per_axis`

```python
tvm.tirx.q_multiply_shift_per_axis(x: Expr, y: Expr, ls: Expr, rs: Expr, q: IntImm, is_lshift_required: IntImm, is_rshift_required: IntImm)
```

Execute a multiplication between two Q-numbers x and y

**Parameters:**

- **x** (tvm.relax.Expr) – First Q-number.
- **y** (tvm.relax.Expr) – Second Q-number.
- **ls** (tvm.relax.Expr) – Integer left shift.
- **rs** (tvm.relax.Expr) – Integer right shift.
- **q** ([*IntImm*](#tvm.tirx.IntImm)) – Number of fractional bits in x and y. Needs to be > 0.
- **is\_lshift\_required** ([*IntImm*](#tvm.tirx.IntImm)) – Whether we need to do left shift or not.
- **is\_rshift\_required** ([*IntImm*](#tvm.tirx.IntImm)) – Whether we need to do right shift or not.

**Returns:**

**z** – The result.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.shift_left"></a>
### `tvm.tirx.shift_left`

```python
tvm.tirx.shift_left(x, y, span=None)
```

Return the result of x left shifted by y bits.

**Parameters:**

- **x** (tvm.relax.Expr) – Input argument.
- **y** (tvm.relax.Expr) – Input argument.

**Returns:**

**z** – The result.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.shift_right"></a>
### `tvm.tirx.shift_right`

```python
tvm.tirx.shift_right(x, y, span=None)
```

Return the result of x right shifted by y bits.

**Parameters:**

- **x** (tvm.relax.Expr) – Input argument.
- **y** (tvm.relax.Expr) – Input argument.

**Returns:**

**z** – The result.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.TVMBackendAllocWorkspace"></a>
### `tvm.tirx.TVMBackendAllocWorkspace`

```python
tvm.tirx.TVMBackendAllocWorkspace(device_type, device_id, nbytes, dtype_code_hint, dtype_bits_hint)
```

Backend function to allocate temporal workspace

**Parameters:**

- **device\_type** (int) – The device type which the space will be allocated.
- **device\_id** (int) – The device id which the space will be allocated.
- **nbytes** (int) – The size of the space requested.
- **dtype\_code\_hint** (int) – The type code of the array elements. Only used in certain backends such as OpenGL.
- **dtype\_bits\_hint** (int) – The type bits of the array elements. Only used in certain backends such as OpenGL.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.TVMBackendFreeWorkspace"></a>
### `tvm.tirx.TVMBackendFreeWorkspace`

```python
tvm.tirx.TVMBackendFreeWorkspace(device_type, device_id, ptr)
```

Backend function to free temporal workspace.

**Parameters:**

- **device\_type** (int) – The device type which the space will be allocated.
- **device\_id** (int) – The device id which the space will be allocated.
- **ptr** (*tirx.Var*) – The result allocated space pointer.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.start_profile_intrinsic"></a>
### `tvm.tirx.start_profile_intrinsic`

```python
tvm.tirx.start_profile_intrinsic(id)
```

Start profile intrinsic.
:param id: The intrinsic id.
:type id: int

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.end_profile_intrinsic"></a>
### `tvm.tirx.end_profile_intrinsic`

```python
tvm.tirx.end_profile_intrinsic(id)
```

End profile intrinsic.
:param id: The intrinsic id.
:type id: int

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.vscale"></a>
### `tvm.tirx.vscale`

```python
tvm.tirx.vscale()
```

Get the target’s vscale value. It will be lowered to llvm.vscale intrinsic
(<https://llvm.org/docs/LangRef.html#llvm-vscale-intrinsic>)
:returns: **call** – tirx.Call to the vscale intrinsic
:rtype: Expr

<a id="tvm.tirx.get_active_lane_mask"></a>
### `tvm.tirx.get_active_lane_mask`

```python
tvm.tirx.get_active_lane_mask(dtype, base, limit)
```

Calculate a predicate mask given an upper bound (limit) and a current value (base).

It will be lowered to the llvm.get.active.lane.mask intrinsic.
(<https://llvm.org/docs/LangRef.html#llvm-get-active-lane-mask-intrinsics>)

**Parameters:**

- **dtype** (str) – The data type of the result.
- **base** (tvm.relax.Expr) – An expression reprsenting the base.
- **limit** (tvm.relax.Expr) – An expression representing the limit.

<a id="tvm.tirx.get_vscale_expr"></a>
### `tvm.tirx.get_vscale_expr`

```python
tvm.tirx.get_vscale_expr(dtype: str | dtype, min_size: int = 128) → Expr
```

Create a datatype dependent scalable expression.

**Parameters:**

- **dtype** (*Union*[str, *tvm\_ffi.DataType*]) – Element data type.
- **min\_size** (int) – The minimum size of the scalable vector in bits.

<a id="tvm.tirx.dp4a"></a>
### `tvm.tirx.dp4a`

```python
tvm.tirx.dp4a(vec1, vec2, acc=0)
```

Dot product of two int8x4 vectors and add an optional accumulator

**Parameters:**

- **vec1** (*int8x4*) – The input vector.
- **vec2** (*int8x4*) – The input vector.
- **acc** (*int32*) – The accumulator.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.ignore_loop_partition"></a>
### `tvm.tirx.ignore_loop_partition`

```python
tvm.tirx.ignore_loop_partition(predicate) → Expr
```

Annotate a predicate not be considered as target condition of loop partition.

**Parameters:**

**predicate** (tvm.relax.Expr) – The annotated predicate expression.

<a id="tvm.tirx.ExecScope"></a>
### `tvm.tirx.ExecScope`

```python
class tvm.tirx.ExecScope(name: str)
```

An execution scope, identified by one of {cluster, cta, warpgroup, warp,
thread}. The ctor FATALs on any other name.

<a id="tvm.tirx.ExecScope.name"></a>
#### `tvm.tirx.ExecScope.name`

```python
property name: str
```

Human-readable name of this scope (derived from `kind`).

<a id="tvm.tirx.ScopeIdDef"></a>
### `tvm.tirx.ScopeIdDef`

```python
class tvm.tirx.ScopeIdDef(def_ids: list[Var], extents: list[Expr] | None, parent: str, cur: str, preferred_extents: list[Expr] | None = None)
```

Definition of scope identifiers with their extents and parent-child relationships.

The constructor accepts `parent` and `cur` as scope-name strings; they
are converted by the FFI into the closed `ScopeBinding` enum and stored
on the `scope` field (an `int` value of that enum).

`extents=None` defers the extent: the value is inferred from sibling
ScopeIdDef relationships at LowerTIRx entry via the verifier’s closure.
Deferred form requires `def_ids` to contain exactly one Var.

<a id="tvm.tirx.TileLayout"></a>
### `tvm.tirx.TileLayout`

```python
class tvm.tirx.TileLayout(spec: _LayoutSpec)
```

A memory layout that tiles data across devices.

<a id="tvm.tirx.TileLayout.from_iters"></a>
#### `tvm.tirx.TileLayout.from_iters`

```python
static from_iters(shard: Sequence[Iter] = (), replica: Sequence[Iter] = (), offset: dict[Axis | str, Expr] | None = None) → TileLayout
```

Construct a TileLayout from pre-built Iter objects.

<a id="tvm.tirx.TileLayout.is_trivial"></a>
#### `tvm.tirx.TileLayout.is_trivial`

```python
is_trivial() → bool
```

Check if the layout is trivial.

<a id="tvm.tirx.TileLayout.group"></a>
#### `tvm.tirx.TileLayout.group`

```python
group(shape: list[Expr]) → tuple[Layout, list[int]]
```

Group the current layout by the given shape.

**Parameters:**

**shape** (*List*[tvm.relax.Expr]) – The shape to group by

**Returns:**

The grouped layout and the separators

**Return type:**

tvm.relax.Tuple[[Layout](#tvm.tirx.Layout), List[int]]

<a id="tvm.tirx.TileLayout.get_scope"></a>
#### `tvm.tirx.TileLayout.get_scope`

```python
get_scope() → tuple[ExecScope, ExecScope] | None
```

Get the scope pair of the layout.

<a id="tvm.tirx.TileLayout.trainium"></a>
#### `tvm.tirx.TileLayout.trainium`

```python
classmethod trainium(annotation: str, shape: tuple[Expr], is_psum: bool = False) → TileLayout
```

Create a TileLayout from an annotation string and a shape.

<a id="tvm.tirx.TileLayout.to_psum"></a>
#### `tvm.tirx.TileLayout.to_psum`

```python
to_psum() → TileLayout
```

Convert the layout to a psum layout.

<a id="tvm.tirx.TileLayout.permute_dims"></a>
#### `tvm.tirx.TileLayout.permute_dims`

```python
permute_dims(perm: list[int]) → TileLayout
```

Permute the dimensions of the layout.

<a id="tvm.tirx.TileLayout.permute_by_groups"></a>
#### `tvm.tirx.TileLayout.permute_by_groups`

```python
permute_by_groups(seps: list[int], perm: list[int]) → TileLayout
```

Permute groups of shard iters defined by `seps`.

`seps` follows the convention of [`group()`](#tvm.tirx.TileLayout.group)’s second return value:
`seps[0] == 0` and group `i` covers shard indices
`[seps[i], seps[i + 1])`. The number of groups is `len(seps) - 1`.

**Parameters:**

- **seps** (list[int]) – Group boundary positions in the shard list.
- **perm** (list[int]) – Permutation of `range(len(seps) - 1)` selecting the new group order.

<a id="tvm.tirx.Layout"></a>
### `tvm.tirx.Layout`

```python
class tvm.tirx.Layout
```

<a id="tvm.tirx.Layout.verify_well_formed"></a>
#### `tvm.tirx.Layout.verify_well_formed`

```python
verify_well_formed() → bool
```

Verify if the layout is well-formed.

**Returns:**

True if the layout is well-formed, False otherwise

**Return type:**

bool

<a id="tvm.tirx.Layout.size"></a>
#### `tvm.tirx.Layout.size`

```python
size(axis_name: str | None = None)
```

Get the size of the layout.

**Parameters:**

**axis\_name** (*Optional*[str]) – The name of the axis to get the size of. If not provided, the default input size will be returned.

<a id="tvm.tirx.Layout.span"></a>
#### `tvm.tirx.Layout.span`

```python
span(axis_name: str | None = None)
```

Get the span of the layout.

**Parameters:**

**axis\_name** (*Optional*[str]) – The name of the axis to get the span of. If not provided, the default span will be returned.

<a id="tvm.tirx.Layout.apply"></a>
#### `tvm.tirx.Layout.apply`

```python
apply(*coord: list[Expr], shape: list[Expr] | None = None) → dict[str, Expr]
```

Apply the layout on the input coordinate and get the mapped output.

Input cases:
- coord is a single element -> will be treated as a 1D coordinate
- coord is a list of elements -> will be treated as a multi-dimensional coordinate
- shape is provided -> turn the coord with shape into a 1D coordinate
- shape is not provided -> use the default shape

**Returns:**

The mapped output (axis name -> value on the axis)

**Return type:**

Dict[str, tvm.relax.Expr]

<a id="tvm.tirx.Layout.apply_to_shape"></a>
#### `tvm.tirx.Layout.apply_to_shape`

```python
apply_to_shape(coord: list[Expr], input_shape: list[Expr]) → list[Expr]
```

Compute the per-shard value that each shard would take if `coord`
were interpreted against `input_shape`.

Tries `self.group(input_shape)` first. On success, each group owns
exactly one `input_shape` entry, so `coord[d]` can be split
*within* that group’s shard extents (bounds stay local to one input
dim — simpler analyzer simplification, no cross-dim complications).

Falls back to `FlattenCoord(coord, input_shape)` + `SplitCoord`
on `self`’s raw shard shape when the group call fails (e.g. when
`input_shape` does not align with the layout’s factor boundaries).

Returns a list of length `len(self.shard)`; each entry is the value
that shard would iterate.

<a id="tvm.tirx.Layout.canonicalize"></a>
#### `tvm.tirx.Layout.canonicalize`

```python
canonicalize() → Layout
```

Canonicalize the layout by simplifying and fusing iterators where possible.

**Returns:**

The canonicalized layout

**Return type:**

[Layout](#tvm.tirx.Layout)

<a id="tvm.tirx.Layout.tile"></a>
#### `tvm.tirx.Layout.tile`

```python
tile(outer: TileLayout, outer_shape: list[Expr], inner_shape: list[Expr]) → TileLayout | ComposeLayout
```

Tile the current layout with an outer layout.

**Parameters:**

- **outer** ([*TileLayout*](#tvm.tirx.TileLayout)) – The outer layout to tile with
- **outer\_shape** (*List*[tvm.relax.Expr]) – The shape of the outer layout
- **inner\_shape** (*List*[tvm.relax.Expr]) – The shape of the inner layout

**Returns:**

The resulting tiled layout

**Return type:**

Union[[TileLayout](#tvm.tirx.TileLayout), [ComposeLayout](#tvm.tirx.ComposeLayout)]

<a id="tvm.tirx.Layout.direct_sum"></a>
#### `tvm.tirx.Layout.direct_sum`

```python
direct_sum(left: TileLayout, left_shape: list[Expr], right_shape: list[Expr]) → TileLayout | ComposeLayout
```

Direct-sum on the tiling domain (unscaled composition): A + B.

This layout is treated as the right addend B grouped by right\_shape.
The left layout is treated as A grouped by left\_shape.
The resulting layout is evaluated over the interleaved domain S\_A ⊗ S\_B,
without span scaling (unlike tiling).

<a id="tvm.tirx.Layout.is_tile_inner"></a>
#### `tvm.tirx.Layout.is_tile_inner`

```python
is_tile_inner(tile_layout: TileLayout | ComposeLayout, tiled_shape: list[Expr], inner_shape: list[Expr]) → TileLayout | None
```

Check if a layout is the inner layout of a tiled layout.

**Parameters:**

- **tile\_layout** (*Union*[[*TileLayout*](#tvm.tirx.TileLayout), [*ComposeLayout*](#tvm.tirx.ComposeLayout)]) – The tiled layout to check
- **tiled\_shape** (*List*[tvm.relax.Expr]) – The shape of the tiled layout
- **inner\_shape** (*List*[tvm.relax.Expr]) – The shape of the inner layout

**Returns:**

The outer layout if it is the inner layout of the tiled layout, None otherwise

**Return type:**

Optional[[TileLayout](#tvm.tirx.TileLayout)]

<a id="tvm.tirx.Layout.is_tile_outer"></a>
#### `tvm.tirx.Layout.is_tile_outer`

```python
is_tile_outer(tile_layout: TileLayout | ComposeLayout, tiled_shape: list[Expr], outer_shape: list[Expr]) → Layout | None
```

Check if a layout is the outer layout of a tiled layout.

**Parameters:**

- **tile\_layout** (*Union*[[*TileLayout*](#tvm.tirx.TileLayout), [*ComposeLayout*](#tvm.tirx.ComposeLayout)]) – The tiled layout to check
- **tiled\_shape** (*List*[tvm.relax.Expr]) – The shape of the tiled layout
- **outer\_shape** (*List*[tvm.relax.Expr]) – The shape of the outer layout

**Returns:**

The inner layout if it is the outer layout of the tiled layout, None otherwise

**Return type:**

Optional[[Layout](#tvm.tirx.Layout)]

<a id="tvm.tirx.Layout.is_direct_sum_right"></a>
#### `tvm.tirx.Layout.is_direct_sum_right`

```python
is_direct_sum_right(sum_layout: TileLayout | ComposeLayout, interleaved_shape: list[Expr], right_shape: list[Expr]) → TileLayout | None
```

Check if this layout is the right addend B in a direct-sum A + B.

Returns the left addend A if recognized, otherwise None.

<a id="tvm.tirx.Layout.is_direct_sum_left"></a>
#### `tvm.tirx.Layout.is_direct_sum_left`

```python
is_direct_sum_left(sum_layout: TileLayout | ComposeLayout, interleaved_shape: list[Expr], left_shape: list[Expr]) → Layout | None
```

Check if this layout is the left addend A in a direct-sum A + B.

Returns the right addend B if recognized, otherwise None.

<a id="tvm.tirx.Layout.slice"></a>
#### `tvm.tirx.Layout.slice`

```python
slice(shape: list[Expr], region: list[tuple[Expr, Expr]]) → Layout | None
```

Slice the layout with a given shape and region.

**Parameters:**

- **shape** (*List*[tvm.relax.Expr]) – The shape of the layout
- **region** (*List*[tvm.relax.Tuple[tvm.relax.Expr, tvm.relax.Expr], tvm.ir.Range]) – The region to slice, each element is (begin, end)

**Returns:**

The sliced layout, or None if slicing is not possible

**Return type:**

Optional[[Layout](#tvm.tirx.Layout)]

<a id="tvm.tirx.Layout.tile_to"></a>
#### `tvm.tirx.Layout.tile_to`

```python
tile_to(to_shape: list[Expr], current_shape: list[Expr]) → Layout
```

Tile the current layout to the given shape.

**Parameters:**

- **to\_shape** (*List*[tvm.relax.Expr]) – The shape to tile to
- **current\_shape** (*List*[tvm.relax.Expr]) – The current shape of the layout

<a id="tvm.tirx.Layout.is_swizzle"></a>
#### `tvm.tirx.Layout.is_swizzle`

```python
is_swizzle() → bool
```

Check if the layout is swizzle.

<a id="tvm.tirx.Layout.is_trivial"></a>
#### `tvm.tirx.Layout.is_trivial`

```python
is_trivial() → bool
```

Check if the layout is trivial.

<a id="tvm.tirx.Layout.is_trainium"></a>
#### `tvm.tirx.Layout.is_trainium`

```python
is_trainium() → bool
```

Check if the layout is trainium layout.

<a id="tvm.tirx.Layout.unpack"></a>
#### `tvm.tirx.Layout.unpack`

```python
unpack(num: int) → Layout
```

Unpack the layout, where a single element in the layout is unpacked into num contiguous elements.

**Parameters:**

**num** (int) – The number of elements to unpack into

**Returns:**

The unpacked layout

**Return type:**

[Layout](#tvm.tirx.Layout)

<a id="tvm.tirx.Layout.broadcast"></a>
#### `tvm.tirx.Layout.broadcast`

```python
broadcast(num: int, position: int = -1, axis: 'Axis' | str = 'm') → Layout
```

Insert a stride-0 broadcast dim of extent `num` at `position`.

`position` follows Python list-insert semantics (negative indices
count from the end; `-1` appends after the last shard dim). The
new dim has stride 0 — accessing along it doesn’t move the byte
offset, so the same physical element is “seen” `num` times.

Useful for layouts where a consumer reads the same SMEM datum
multiple times (e.g. `sf_reuse` over MMA-K steps).

<a id="tvm.tirx.Layout.pack"></a>
#### `tvm.tirx.Layout.pack`

```python
pack(num: int) → Layout
```

Pack the layout, where num contiguous elements in the layout are packed into a single element.

**Parameters:**

**num** (int) – The number of elements to pack into

**Returns:**

The packed layout

**Return type:**

[Layout](#tvm.tirx.Layout)

<a id="tvm.tirx.SwizzleLayout"></a>
### `tvm.tirx.SwizzleLayout`

```python
class tvm.tirx.SwizzleLayout(per_element: int, swizzle_len: int, atom_len: int, swizzle_inner: bool = True)
```

A memory layout that swizzles elements to improve memory access patterns.

<a id="tvm.tirx.ComposeLayout"></a>
### `tvm.tirx.ComposeLayout`

```python
class tvm.tirx.ComposeLayout(layout_A: SwizzleLayout, layout_B: TileLayout)
```

A memory layout that composes 2 layouts.

<a id="tvm.tirx.Predicate"></a>
### `tvm.tirx.Predicate`

```python
class tvm.tirx.Predicate(f_pred: Callable[[...], Expr])
```

A predicate object for TIRX

<a id="tvm.tirx.Predicate.apply"></a>
#### `tvm.tirx.Predicate.apply`

```python
apply(indices: list[Expr]) → Expr
```

Apply the predicate to the given indices

<a id="tvm.tirx.ExprFunctor"></a>
### `tvm.tirx.ExprFunctor`

```python
class tvm.tirx.ExprFunctor
```

An abstract visitor over Expr, with visiting function defined for each Expr type.

<a id="tvm.tirx.ExprFunctor.visit_expr"></a>
#### `tvm.tirx.ExprFunctor.visit_expr`

```python
visit_expr(expr: Expr)
```

Apply the visitor to an expression.

**Parameters:**

**expr** (tvm.relax.Expr) – The expression to be visited.

**Returns:**

**result** – The result of the visit.

**Return type:**

Any

<a id="tvm.tirx.ExprFunctor.visit_var_"></a>
#### `tvm.tirx.ExprFunctor.visit_var_`

```python
visit_var_(op)
```

Default visitor for tirx.Var node.

<a id="tvm.tirx.ExprFunctor.visit_buffer_load_"></a>
#### `tvm.tirx.ExprFunctor.visit_buffer_load_`

```python
visit_buffer_load_(op)
```

Default visitor for BufferLoad node.

<a id="tvm.tirx.ExprFunctor.visit_producer_load_"></a>
#### `tvm.tirx.ExprFunctor.visit_producer_load_`

```python
visit_producer_load_(op)
```

Default visitor for ProducerLoad node.

<a id="tvm.tirx.ExprFunctor.visit_let_"></a>
#### `tvm.tirx.ExprFunctor.visit_let_`

```python
visit_let_(op)
```

Default visitor for Let node.

<a id="tvm.tirx.ExprFunctor.visit_call_"></a>
#### `tvm.tirx.ExprFunctor.visit_call_`

```python
visit_call_(op)
```

Default visitor for tirx.Call node.

<a id="tvm.tirx.ExprFunctor.visit_add_"></a>
#### `tvm.tirx.ExprFunctor.visit_add_`

```python
visit_add_(op)
```

Default visitor for Add node.

<a id="tvm.tirx.ExprFunctor.visit_sub_"></a>
#### `tvm.tirx.ExprFunctor.visit_sub_`

```python
visit_sub_(op)
```

Default visitor for Sub node.

<a id="tvm.tirx.ExprFunctor.visit_mul_"></a>
#### `tvm.tirx.ExprFunctor.visit_mul_`

```python
visit_mul_(op)
```

Default visitor for Mul node.

<a id="tvm.tirx.ExprFunctor.visit_div_"></a>
#### `tvm.tirx.ExprFunctor.visit_div_`

```python
visit_div_(op)
```

Default visitor for Div node.

<a id="tvm.tirx.ExprFunctor.visit_mod_"></a>
#### `tvm.tirx.ExprFunctor.visit_mod_`

```python
visit_mod_(op)
```

Default visitor for Mod node.

<a id="tvm.tirx.ExprFunctor.visit_floordiv_"></a>
#### `tvm.tirx.ExprFunctor.visit_floordiv_`

```python
visit_floordiv_(op)
```

Default visitor for FloorDiv node.

<a id="tvm.tirx.ExprFunctor.visit_floormod_"></a>
#### `tvm.tirx.ExprFunctor.visit_floormod_`

```python
visit_floormod_(op)
```

Default visitor for FloorMod node.

<a id="tvm.tirx.ExprFunctor.visit_min_"></a>
#### `tvm.tirx.ExprFunctor.visit_min_`

```python
visit_min_(op)
```

Default visitor for Min node.

<a id="tvm.tirx.ExprFunctor.visit_max_"></a>
#### `tvm.tirx.ExprFunctor.visit_max_`

```python
visit_max_(op)
```

Default visitor for Max node.

<a id="tvm.tirx.ExprFunctor.visit_eq_"></a>
#### `tvm.tirx.ExprFunctor.visit_eq_`

```python
visit_eq_(op)
```

Default visitor for EQ node.

<a id="tvm.tirx.ExprFunctor.visit_ne_"></a>
#### `tvm.tirx.ExprFunctor.visit_ne_`

```python
visit_ne_(op)
```

Default visitor for NE node.

<a id="tvm.tirx.ExprFunctor.visit_lt_"></a>
#### `tvm.tirx.ExprFunctor.visit_lt_`

```python
visit_lt_(op)
```

Default visitor for LT node.

<a id="tvm.tirx.ExprFunctor.visit_le_"></a>
#### `tvm.tirx.ExprFunctor.visit_le_`

```python
visit_le_(op)
```

Default visitor for LE node.

<a id="tvm.tirx.ExprFunctor.visit_gt_"></a>
#### `tvm.tirx.ExprFunctor.visit_gt_`

```python
visit_gt_(op)
```

Default visitor for GT node.

<a id="tvm.tirx.ExprFunctor.visit_ge_"></a>
#### `tvm.tirx.ExprFunctor.visit_ge_`

```python
visit_ge_(op)
```

Default visitor for GE node.

<a id="tvm.tirx.ExprFunctor.visit_and_"></a>
#### `tvm.tirx.ExprFunctor.visit_and_`

```python
visit_and_(op)
```

Default visitor for And node.

<a id="tvm.tirx.ExprFunctor.visit_or_"></a>
#### `tvm.tirx.ExprFunctor.visit_or_`

```python
visit_or_(op)
```

Default visitor for Or node.

<a id="tvm.tirx.ExprFunctor.visit_reduce_"></a>
#### `tvm.tirx.ExprFunctor.visit_reduce_`

```python
visit_reduce_(op)
```

Default visitor for Reduce node.

<a id="tvm.tirx.ExprFunctor.visit_cast_"></a>
#### `tvm.tirx.ExprFunctor.visit_cast_`

```python
visit_cast_(op)
```

Default visitor for Cast node.

<a id="tvm.tirx.ExprFunctor.visit_not_"></a>
#### `tvm.tirx.ExprFunctor.visit_not_`

```python
visit_not_(op)
```

Default visitor for Not node.

<a id="tvm.tirx.ExprFunctor.visit_select_"></a>
#### `tvm.tirx.ExprFunctor.visit_select_`

```python
visit_select_(op)
```

Default visitor for Select node.

<a id="tvm.tirx.ExprFunctor.visit_ramp_"></a>
#### `tvm.tirx.ExprFunctor.visit_ramp_`

```python
visit_ramp_(op)
```

Default visitor for Ramp node.

<a id="tvm.tirx.ExprFunctor.visit_broadcast_"></a>
#### `tvm.tirx.ExprFunctor.visit_broadcast_`

```python
visit_broadcast_(op)
```

Default visitor for Broadcast node.

<a id="tvm.tirx.ExprFunctor.visit_shuffle_"></a>
#### `tvm.tirx.ExprFunctor.visit_shuffle_`

```python
visit_shuffle_(op)
```

Default visitor for Shuffle node.

<a id="tvm.tirx.ExprFunctor.visit_int_imm_"></a>
#### `tvm.tirx.ExprFunctor.visit_int_imm_`

```python
visit_int_imm_(op)
```

Default visitor for IntImm node.

<a id="tvm.tirx.ExprFunctor.visit_float_imm_"></a>
#### `tvm.tirx.ExprFunctor.visit_float_imm_`

```python
visit_float_imm_(op)
```

Default visitor for FloatImm node.

<a id="tvm.tirx.ExprFunctor.visit_string_imm_"></a>
#### `tvm.tirx.ExprFunctor.visit_string_imm_`

```python
visit_string_imm_(op)
```

Default visitor for StringImm node.

<a id="tvm.tirx.ExprFunctor.visit_expr_default_"></a>
#### `tvm.tirx.ExprFunctor.visit_expr_default_`

```python
visit_expr_default_(op)
```

Default visitor implementation.

<a id="tvm.tirx.PyStmtExprVisitor"></a>
### `tvm.tirx.PyStmtExprVisitor`

```python
class tvm.tirx.PyStmtExprVisitor
```

A Python StmtExprVisitor to define custom visitor for both Stmt and Expr.

Users can customize any of the visit function.

<a id="tvm.tirx.PyStmtExprVisitor.visit_stmt"></a>
#### `tvm.tirx.PyStmtExprVisitor.visit_stmt`

```python
visit_stmt(stmt: Stmt) → None
```

Visit a Stmt.

**Parameters:**

**stmt** ([*Stmt*](#tvm.tirx.Stmt)) – The Stmt to be visited.

<a id="tvm.tirx.PyStmtExprVisitor.visit_expr"></a>
#### `tvm.tirx.PyStmtExprVisitor.visit_expr`

```python
visit_expr(expr: Expr) → None
```

Visit a Expr.

**Parameters:**

**expr** (tvm.relax.Expr) – The Expr to be visited.

<a id="tvm.tirx.PyStmtExprVisitor.visit_attr_stmt_"></a>
#### `tvm.tirx.PyStmtExprVisitor.visit_attr_stmt_`

```python
visit_attr_stmt_(op: AttrStmt) → None
```

Visit AttrStmt.
Users can customize this function to overwrite VisitStmt\_(const AttrStmtNode\* op)
on the C++ side.

**Parameters:**

**op** ([*AttrStmt*](#tvm.tirx.AttrStmt)) – The AttrStmt to be visited.

<a id="tvm.tirx.PyStmtExprVisitor.visit_if_then_else_"></a>
#### `tvm.tirx.PyStmtExprVisitor.visit_if_then_else_`

```python
visit_if_then_else_(op: IfThenElse) → None
```

Visit IfThenElse.
Users can customize this function to overwrite VisitStmt\_(const IfThenElseNode\* op)
on the C++ side.

**Parameters:**

**op** ([*IfThenElse*](#tvm.tirx.IfThenElse)) – The IfThenElse to be visited.

<a id="tvm.tirx.PyStmtExprVisitor.visit_bind_"></a>
#### `tvm.tirx.PyStmtExprVisitor.visit_bind_`

```python
visit_bind_(op: Bind) → None
```

Visit Bind.
Users can customize this function to overwrite VisitStmt\_(const BindNode\* op)
on the C++ side.

**Parameters:**

**op** ([*Bind*](#tvm.tirx.Bind)) – The Bind node to be visited.

<a id="tvm.tirx.PyStmtExprVisitor.visit_for_"></a>
#### `tvm.tirx.PyStmtExprVisitor.visit_for_`

```python
visit_for_(op: For) → None
```

Visit For.
Users can customize this function to overwrite VisitStmt\_(const ForNode\* op)
on the C++ side.

**Parameters:**

**op** ([*For*](#tvm.tirx.For)) – The For to be visited.

<a id="tvm.tirx.PyStmtExprVisitor.visit_while_"></a>
#### `tvm.tirx.PyStmtExprVisitor.visit_while_`

```python
visit_while_(op: While) → None
```

Visit While.
Users can customize this function to overwrite VisitStmt\_(const WhileNode\* op)
on the C++ side.

**Parameters:**

**op** ([*While*](#tvm.tirx.While)) – The While to be visited.

<a id="tvm.tirx.PyStmtExprVisitor.visit_alloc_buffer_"></a>
#### `tvm.tirx.PyStmtExprVisitor.visit_alloc_buffer_`

```python
visit_alloc_buffer_(op: AllocBuffer) → None
```

Visit AllocBuffer.
Users can customize this function to overwrite VisitStmt\_(const AllocBufferNode\* op)
on the C++ side.

**Parameters:**

**op** ([*AllocBuffer*](#tvm.tirx.AllocBuffer)) – The AllocBuffer to be visited.

<a id="tvm.tirx.PyStmtExprVisitor.visit_decl_buffer_"></a>
#### `tvm.tirx.PyStmtExprVisitor.visit_decl_buffer_`

```python
visit_decl_buffer_(op: DeclBuffer) → None
```

Visit DeclBuffer.
Users can customize this function to overwrite VisitStmt\_(const DeclBufferNode\* op)
on the C++ side.

**Parameters:**

**op** ([*DeclBuffer*](#tvm.tirx.DeclBuffer)) – The DeclBuffer to be visited.

<a id="tvm.tirx.PyStmtExprVisitor.visit_buffer_store_"></a>
#### `tvm.tirx.PyStmtExprVisitor.visit_buffer_store_`

```python
visit_buffer_store_(op: BufferStore) → None
```

Visit BufferStore.
Users can customize this function to overwrite VisitStmt\_(const BufferStoreNode\* op)
on the C++ side.

**Parameters:**

**op** ([*BufferStore*](#tvm.tirx.BufferStore)) – The BufferStore to be visited.

<a id="tvm.tirx.PyStmtExprVisitor.visit_assert_stmt_"></a>
#### `tvm.tirx.PyStmtExprVisitor.visit_assert_stmt_`

```python
visit_assert_stmt_(op: AssertStmt) → None
```

Visit AssertStmt.
Users can customize this function to overwrite VisitStmt\_(const AssertStmtNode\* op)
on the C++ side.

**Parameters:**

**op** ([*AssertStmt*](#tvm.tirx.AssertStmt)) – The AssertStmt to be visited.

<a id="tvm.tirx.PyStmtExprVisitor.visit_seq_stmt_"></a>
#### `tvm.tirx.PyStmtExprVisitor.visit_seq_stmt_`

```python
visit_seq_stmt_(op: SeqStmt) → None
```

Visit SeqStmt.
Users can customize this function to overwrite VisitStmt\_(const SeqStmtNode\* op)
on the C++ side.

**Parameters:**

**op** ([*SeqStmt*](#tvm.tirx.SeqStmt)) – The SeqStmt to be visited.

<a id="tvm.tirx.PyStmtExprVisitor.visit_evaluate_"></a>
#### `tvm.tirx.PyStmtExprVisitor.visit_evaluate_`

```python
visit_evaluate_(op: Evaluate) → None
```

Visit Evaluate.
Users can customize this function to overwrite VisitStmt\_(const EvaluateNode\* op)
on the C++ side.

**Parameters:**

**op** ([*Evaluate*](#tvm.tirx.Evaluate)) – The Evaluate to be visited.

<a id="tvm.tirx.PyStmtExprVisitor.visit_sblock_"></a>
#### `tvm.tirx.PyStmtExprVisitor.visit_sblock_`

```python
visit_sblock_(op: SBlock) → None
```

Visit SBlock.
Users can customize this function to overwrite VisitStmt\_(const SBlockNode\* op)
on the C++ side.

**Parameters:**

**op** ([*SBlock*](#tvm.tirx.SBlock)) – The SBlock to be visited.

<a id="tvm.tirx.PyStmtExprVisitor.visit_sblock_realize_"></a>
#### `tvm.tirx.PyStmtExprVisitor.visit_sblock_realize_`

```python
visit_sblock_realize_(op: SBlockRealize) → None
```

Visit BlockRealize.
Users can customize this function to overwrite VisitStmt\_(const SBlockRealizeNode\* op)
on the C++ side.

**Parameters:**

**op** ([*SBlockRealize*](#tvm.tirx.SBlockRealize)) – The BlockRealize to be visited.

<a id="tvm.tirx.PyStmtExprVisitor.visit_var_"></a>
#### `tvm.tirx.PyStmtExprVisitor.visit_var_`

```python
visit_var_(op: Var) → None
```

Visit Var.

Users can customize this function to overwrite VisitVar\_(const VarNode\* op)
on the C++ side.

**Parameters:**

**op** (*tirx.Var*) – The tirx.Var to be visited.

<a id="tvm.tirx.PyStmtExprVisitor.visit_buffer_load_"></a>
#### `tvm.tirx.PyStmtExprVisitor.visit_buffer_load_`

```python
visit_buffer_load_(op: BufferLoad) → None
```

Visit BufferLoad.

Users can customize this function to overwrite VisitBufferLoad\_(const BufferLoadNode\* op)
on the C++ side.

**Parameters:**

**op** ([*BufferLoad*](#tvm.tirx.BufferLoad)) – The BufferLoad to be visited.

<a id="tvm.tirx.PyStmtExprVisitor.visit_producer_load_"></a>
#### `tvm.tirx.PyStmtExprVisitor.visit_producer_load_`

```python
visit_producer_load_(op: ProducerLoad) → None
```

Visit ProducerLoad.

Users can customize this function to overwrite
VisitProducerLoad\_(const ProducerLoadNode\* op) on the C++ side.

**Parameters:**

**op** ([*ProducerLoad*](#tvm.tirx.ProducerLoad)) – The ProducerLoad to be visited.

<a id="tvm.tirx.PyStmtExprVisitor.visit_let_"></a>
#### `tvm.tirx.PyStmtExprVisitor.visit_let_`

```python
visit_let_(op: Let) → None
```

Visit Let.

Users can customize this function to overwrite VisitLet\_(const LetNode\* op)
on the C++ side.

**Parameters:**

**op** ([*Let*](#tvm.tirx.Let)) – The Let to be visited.

<a id="tvm.tirx.PyStmtExprVisitor.visit_call_"></a>
#### `tvm.tirx.PyStmtExprVisitor.visit_call_`

```python
visit_call_(op: Call) → None
```

Visit Call.

Users can customize this function to overwrite VisitCall\_(const CallNode\* op)
on the C++ side.

**Parameters:**

**op** (*tirx.Call*) – The tirx.Call to be visited.

<a id="tvm.tirx.PyStmtExprVisitor.visit_add_"></a>
#### `tvm.tirx.PyStmtExprVisitor.visit_add_`

```python
visit_add_(op: Add) → None
```

Visit Add.

Users can customize this function to overwrite VisitAdd\_(const AddNode\* op)
on the C++ side.

**Parameters:**

**op** ([*Add*](#tvm.tirx.Add)) – The Add to be visited.

<a id="tvm.tirx.PyStmtExprVisitor.visit_sub_"></a>
#### `tvm.tirx.PyStmtExprVisitor.visit_sub_`

```python
visit_sub_(op: Sub) → None
```

Visit Sub.

Users can customize this function to overwrite VisitSub\_(const SubNode\* op)
on the C++ side.

**Parameters:**

**op** ([*Sub*](#tvm.tirx.Sub)) – The Sub to be visited.

<a id="tvm.tirx.PyStmtExprVisitor.visit_mul_"></a>
#### `tvm.tirx.PyStmtExprVisitor.visit_mul_`

```python
visit_mul_(op: Mul) → None
```

Visit Mul.

Users can customize this function to overwrite VisitMul\_(const MulNode\* op)
on the C++ side.

**Parameters:**

**op** ([*Mul*](#tvm.tirx.Mul)) – The Mul to be visited.

<a id="tvm.tirx.PyStmtExprVisitor.visit_div_"></a>
#### `tvm.tirx.PyStmtExprVisitor.visit_div_`

```python
visit_div_(op: Div) → None
```

Visit Div.

Users can customize this function to overwrite VisitDiv\_(const DivNode\* op)
on the C++ side.

**Parameters:**

**op** ([*Div*](#tvm.tirx.Div)) – The Div to be visited.

<a id="tvm.tirx.PyStmtExprVisitor.visit_mod_"></a>
#### `tvm.tirx.PyStmtExprVisitor.visit_mod_`

```python
visit_mod_(op: Mod) → None
```

Visit Mod.

Users can customize this function to overwrite VisitMod\_(const ModNode\* op)
on the C++ side.

**Parameters:**

**op** ([*Mod*](#tvm.tirx.Mod)) – The Mod to be visited.

<a id="tvm.tirx.PyStmtExprVisitor.visit_floor_div_"></a>
#### `tvm.tirx.PyStmtExprVisitor.visit_floor_div_`

```python
visit_floor_div_(op: FloorDiv) → None
```

Visit FloorDiv.

Users can customize this function to overwrite VisitFloorDiv\_(const FloorDivNode\* op)
on the C++ side.

**Parameters:**

**op** ([*FloorDiv*](#tvm.tirx.FloorDiv)) – The FloorDiv to be visited.

<a id="tvm.tirx.PyStmtExprVisitor.visit_floor_mod_"></a>
#### `tvm.tirx.PyStmtExprVisitor.visit_floor_mod_`

```python
visit_floor_mod_(op: FloorMod) → None
```

Visit FloorMod.

Users can customize this function to overwrite VisitFloorMod\_(const FloorModNode\* op)
on the C++ side.

**Parameters:**

**op** ([*FloorMod*](#tvm.tirx.FloorMod)) – The FloorMod to be visited.

<a id="tvm.tirx.PyStmtExprVisitor.visit_min_"></a>
#### `tvm.tirx.PyStmtExprVisitor.visit_min_`

```python
visit_min_(op: Min) → None
```

Visit Min.

Users can customize this function to overwrite VisitMin\_(const MinNode\* op)
on the C++ side.

**Parameters:**

**op** ([*Min*](#tvm.tirx.Min)) – The Min to be visited.

<a id="tvm.tirx.PyStmtExprVisitor.visit_max_"></a>
#### `tvm.tirx.PyStmtExprVisitor.visit_max_`

```python
visit_max_(op: Max) → None
```

Visit Max.

Users can customize this function to overwrite VisitMax\_(const MaxNode\* op)
on the C++ side.

**Parameters:**

**op** ([*Max*](#tvm.tirx.Max)) – The Max to be visited.

<a id="tvm.tirx.PyStmtExprVisitor.visit_eq_"></a>
#### `tvm.tirx.PyStmtExprVisitor.visit_eq_`

```python
visit_eq_(op: EQ) → None
```

Visit EQ.

Users can customize this function to overwrite VisitEQ\_(const EQNode\* op)
on the C++ side.

**Parameters:**

**op** ([*EQ*](#tvm.tirx.EQ)) – The EQ to be visited.

<a id="tvm.tirx.PyStmtExprVisitor.visit_ne_"></a>
#### `tvm.tirx.PyStmtExprVisitor.visit_ne_`

```python
visit_ne_(op: NE) → None
```

Visit NE.

Users can customize this function to overwrite VisitNE\_(const NENode\* op)
on the C++ side.

**Parameters:**

**op** ([*NE*](#tvm.tirx.NE)) – The NE to be visited.

<a id="tvm.tirx.PyStmtExprVisitor.visit_lt_"></a>
#### `tvm.tirx.PyStmtExprVisitor.visit_lt_`

```python
visit_lt_(op: LT) → None
```

Visit LT.

Users can customize this function to overwrite VisitLT\_(const LTNode\* op)
on the C++ side.

**Parameters:**

**op** ([*LT*](#tvm.tirx.LT)) – The LT to be visited.

<a id="tvm.tirx.PyStmtExprVisitor.visit_le_"></a>
#### `tvm.tirx.PyStmtExprVisitor.visit_le_`

```python
visit_le_(op: LE) → None
```

Visit LE.

Users can customize this function to overwrite VisitLE\_(const LENode\* op)
on the C++ side.

**Parameters:**

**op** ([*LE*](#tvm.tirx.LE)) – The LE to be visited.

<a id="tvm.tirx.PyStmtExprVisitor.visit_gt_"></a>
#### `tvm.tirx.PyStmtExprVisitor.visit_gt_`

```python
visit_gt_(op: GT) → None
```

Visit GT.

Users can customize this function to overwrite VisitGT\_(const GTNode\* op)
on the C++ side.

**Parameters:**

**op** ([*GT*](#tvm.tirx.GT)) – The GT to be visited.

<a id="tvm.tirx.PyStmtExprVisitor.visit_ge_"></a>
#### `tvm.tirx.PyStmtExprVisitor.visit_ge_`

```python
visit_ge_(op: GE) → None
```

Visit GE.

Users can customize this function to overwrite VisitGE\_(const GENode\* op)
on the C++ side.

**Parameters:**

**op** ([*GE*](#tvm.tirx.GE)) – The GE to be visited.

<a id="tvm.tirx.PyStmtExprVisitor.visit_and_"></a>
#### `tvm.tirx.PyStmtExprVisitor.visit_and_`

```python
visit_and_(op: And) → None
```

Visit And.

Users can customize this function to overwrite VisitAnd\_(const AndNode\* op)
on the C++ side.

**Parameters:**

**op** ([*And*](#tvm.tirx.And)) – The And to be visited.

<a id="tvm.tirx.PyStmtExprVisitor.visit_or_"></a>
#### `tvm.tirx.PyStmtExprVisitor.visit_or_`

```python
visit_or_(op: Or) → None
```

Visit Or.

Users can customize this function to overwrite VisitOr\_(const OrNode\* op)
on the C++ side.

**Parameters:**

**op** ([*Or*](#tvm.tirx.Or)) – The Or to be visited.

<a id="tvm.tirx.PyStmtExprVisitor.visit_reduce_"></a>
#### `tvm.tirx.PyStmtExprVisitor.visit_reduce_`

```python
visit_reduce_(op: Reduce) → None
```

Visit Reduce.

Users can customize this function to overwrite VisitReduce\_(const ReduceNode\* op)
on the C++ side.

**Parameters:**

**op** ([*Reduce*](#tvm.tirx.Reduce)) – The Reduce to be visited.

<a id="tvm.tirx.PyStmtExprVisitor.visit_cast_"></a>
#### `tvm.tirx.PyStmtExprVisitor.visit_cast_`

```python
visit_cast_(op: Cast) → None
```

Visit Cast.

Users can customize this function to overwrite VisitCast\_(const CastNode\* op)
on the C++ side.

**Parameters:**

**op** ([*Cast*](#tvm.tirx.Cast)) – The Cast to be visited.

<a id="tvm.tirx.PyStmtExprVisitor.visit_not_"></a>
#### `tvm.tirx.PyStmtExprVisitor.visit_not_`

```python
visit_not_(op: Not) → None
```

Visit Not.

Users can customize this function to overwrite VisitNot\_(const NotNode\* op)
on the C++ side.

**Parameters:**

**op** ([*Not*](#tvm.tirx.Not)) – The Not to be visited.

<a id="tvm.tirx.PyStmtExprVisitor.visit_select_"></a>
#### `tvm.tirx.PyStmtExprVisitor.visit_select_`

```python
visit_select_(op: Select) → None
```

Visit Select.

Users can customize this function to overwrite VisitSelect\_(const SelectNode\* op)
on the C++ side.

**Parameters:**

**op** ([*Select*](#tvm.tirx.Select)) – The Select to be visited.

<a id="tvm.tirx.PyStmtExprVisitor.visit_ramp_"></a>
#### `tvm.tirx.PyStmtExprVisitor.visit_ramp_`

```python
visit_ramp_(op: Ramp) → None
```

Visit Ramp.

Users can customize this function to overwrite VisitRamp\_(const RampNode\* op)
on the C++ side.

**Parameters:**

**op** ([*Ramp*](#tvm.tirx.Ramp)) – The Ramp to be visited.

<a id="tvm.tirx.PyStmtExprVisitor.visit_broadcast_"></a>
#### `tvm.tirx.PyStmtExprVisitor.visit_broadcast_`

```python
visit_broadcast_(op: Broadcast) → None
```

Visit Broadcast.

Users can customize this function to overwrite VisitBroadcast\_(const BroadcastNode\* op)
on the C++ side.

**Parameters:**

**op** ([*Broadcast*](#tvm.tirx.Broadcast)) – The Broadcast to be visited.

<a id="tvm.tirx.PyStmtExprVisitor.visit_shuffle_"></a>
#### `tvm.tirx.PyStmtExprVisitor.visit_shuffle_`

```python
visit_shuffle_(op: Shuffle) → None
```

Visit Shuffle.

Users can customize this function to overwrite VisitShuffle\_(const ShuffleNode\* op)
on the C++ side.

**Parameters:**

**op** ([*Shuffle*](#tvm.tirx.Shuffle)) – The Shuffle to be visited.

<a id="tvm.tirx.PyStmtExprVisitor.visit_int_imm_"></a>
#### `tvm.tirx.PyStmtExprVisitor.visit_int_imm_`

```python
visit_int_imm_(op: IntImm) → None
```

Visit IntImm.

Users can customize this function to overwrite VisitIntImm\_(const IntImmNode\* op)
on the C++ side.

**Parameters:**

**op** ([*IntImm*](#tvm.tirx.IntImm)) – The IntImm to be visited.

<a id="tvm.tirx.PyStmtExprVisitor.visit_float_imm_"></a>
#### `tvm.tirx.PyStmtExprVisitor.visit_float_imm_`

```python
visit_float_imm_(op: FloatImm) → None
```

Visit FloatImm.

Users can customize this function to overwrite VisitFloatImm\_(const FloatImmNode\* op)
on the C++ side.

**Parameters:**

**op** ([*FloatImm*](#tvm.tirx.FloatImm)) – The FloatImm to be visited.

<a id="tvm.tirx.PyStmtExprVisitor.visit_string_imm_"></a>
#### `tvm.tirx.PyStmtExprVisitor.visit_string_imm_`

```python
visit_string_imm_(op: StringImm) → None
```

Visit StringImm.

Users can customize this function to overwrite VisitStringImm\_(const StringImmNode\* op)
on the C++ side.

**Parameters:**

**op** ([*StringImm*](#tvm.tirx.StringImm)) – The StringImm to be visited.

<a id="tvm.tirx.PyStmtExprMutator"></a>
### `tvm.tirx.PyStmtExprMutator`

```python
class tvm.tirx.PyStmtExprMutator
```

A Python StmtExprMutator to define custom mutator for both Stmt and Expr.

Users can customize any of the visit function.

<a id="tvm.tirx.PyStmtExprMutator.visit_expr"></a>
#### `tvm.tirx.PyStmtExprMutator.visit_expr`

```python
visit_expr(expr: Expr) → Expr
```

Visit Expr.
Users can customize this function to overwrite VisitExpr(const Expr& expr)
on the C++ side.

**Parameters:**

**expr** (tvm.relax.Expr) – The Expr to be visited.

**Returns:**

**result** – The mutated Expr.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.PyStmtExprMutator.visit_stmt"></a>
#### `tvm.tirx.PyStmtExprMutator.visit_stmt`

```python
visit_stmt(stmt: Stmt) → Stmt
```

Visit Stmt.
Users can customize this function to overwrite VisitStmt(const Stmt& stmt)
on the C++ side.

**Parameters:**

**stmt** ([*Stmt*](#tvm.tirx.Stmt)) – The Stmt to be visited.

**Returns:**

**result** – The mutated Stmt.

**Return type:**

[Stmt](#tvm.tirx.Stmt)

<a id="tvm.tirx.PyStmtExprMutator.visit_attr_stmt_"></a>
#### `tvm.tirx.PyStmtExprMutator.visit_attr_stmt_`

```python
visit_attr_stmt_(op: AttrStmt) → Stmt
```

Visit AttrStmt.
Users can customize this function to overwrite VisitStmt\_(const AttrStmtNode\* op)
on the C++ side.

**Parameters:**

**op** ([*AttrStmt*](#tvm.tirx.AttrStmt)) – The AttrStmt to be visited.

**Returns:**

**result** – The mutated Stmt.

**Return type:**

[Stmt](#tvm.tirx.Stmt)

<a id="tvm.tirx.PyStmtExprMutator.visit_if_then_else_"></a>
#### `tvm.tirx.PyStmtExprMutator.visit_if_then_else_`

```python
visit_if_then_else_(op: IfThenElse) → Stmt
```

Visit IfThenElse.
Users can customize this function to overwrite VisitStmt\_(const IfThenElseNode\* op)
on the C++ side.

**Parameters:**

**op** ([*IfThenElse*](#tvm.tirx.IfThenElse)) – The IfThenElse to be visited.

**Returns:**

**result** – The mutated Stmt.

**Return type:**

[Stmt](#tvm.tirx.Stmt)

<a id="tvm.tirx.PyStmtExprMutator.visit_bind_"></a>
#### `tvm.tirx.PyStmtExprMutator.visit_bind_`

```python
visit_bind_(op: Bind) → Stmt
```

Visit Bind.
Users can customize this function to overwrite VisitStmt\_(const BindNode\* op)
on the C++ side.

**Parameters:**

**op** ([*Bind*](#tvm.tirx.Bind)) – The Bind node to be visited.

**Returns:**

**result** – The mutated Stmt.

**Return type:**

[Stmt](#tvm.tirx.Stmt)

<a id="tvm.tirx.PyStmtExprMutator.visit_for_"></a>
#### `tvm.tirx.PyStmtExprMutator.visit_for_`

```python
visit_for_(op: For) → Stmt
```

Visit For.
Users can customize this function to overwrite VisitStmt\_(const ForNode\* op)
on the C++ side.

**Parameters:**

**op** ([*For*](#tvm.tirx.For)) – The For to be visited.

**Returns:**

**result** – The mutated Stmt.

**Return type:**

[Stmt](#tvm.tirx.Stmt)

<a id="tvm.tirx.PyStmtExprMutator.visit_while_"></a>
#### `tvm.tirx.PyStmtExprMutator.visit_while_`

```python
visit_while_(op: While) → Stmt
```

Visit While.
Users can customize this function to overwrite VisitStmt\_(const WhileNode\* op)
on the C++ side.

**Parameters:**

**op** ([*While*](#tvm.tirx.While)) – The While to be visited.

**Returns:**

**result** – The mutated Stmt.

**Return type:**

[Stmt](#tvm.tirx.Stmt)

<a id="tvm.tirx.PyStmtExprMutator.visit_alloc_buffer_"></a>
#### `tvm.tirx.PyStmtExprMutator.visit_alloc_buffer_`

```python
visit_alloc_buffer_(op: AllocBuffer) → Stmt
```

Visit AllocBuffer.
Users can customize this function to overwrite VisitStmt\_(const AllocBufferNode\* op)
on the C++ side.

**Parameters:**

**op** ([*AllocBuffer*](#tvm.tirx.AllocBuffer)) – The AllocBuffer to be visited.

**Returns:**

**result** – The mutated Stmt.

**Return type:**

[Stmt](#tvm.tirx.Stmt)

<a id="tvm.tirx.PyStmtExprMutator.visit_decl_buffer_"></a>
#### `tvm.tirx.PyStmtExprMutator.visit_decl_buffer_`

```python
visit_decl_buffer_(op: DeclBuffer) → Stmt
```

Visit DeclBuffer.
Users can customize this function to overwrite VisitStmt\_(const DeclBufferNode\* op)
on the C++ side.

**Parameters:**

**op** ([*DeclBuffer*](#tvm.tirx.DeclBuffer)) – The DeclBuffer to be visited.

**Returns:**

**result** – The mutated Stmt.

**Return type:**

[Stmt](#tvm.tirx.Stmt)

<a id="tvm.tirx.PyStmtExprMutator.visit_buffer_store_"></a>
#### `tvm.tirx.PyStmtExprMutator.visit_buffer_store_`

```python
visit_buffer_store_(op: BufferStore) → Stmt
```

Visit BufferStore.
Users can customize this function to overwrite VisitStmt\_(const BufferStoreNode\* op)
on the C++ side.

**Parameters:**

**op** ([*BufferStore*](#tvm.tirx.BufferStore)) – The BufferStore to be visited.

**Returns:**

**result** – The mutated Stmt.

**Return type:**

[Stmt](#tvm.tirx.Stmt)

<a id="tvm.tirx.PyStmtExprMutator.visit_assert_stmt_"></a>
#### `tvm.tirx.PyStmtExprMutator.visit_assert_stmt_`

```python
visit_assert_stmt_(op: AssertStmt) → Stmt
```

Visit AssertStmt.
Users can customize this function to overwrite VisitStmt\_(const AssertStmtNode\* op)
on the C++ side.

**Parameters:**

**op** ([*AssertStmt*](#tvm.tirx.AssertStmt)) – The AssertStmt to be visited.

**Returns:**

**result** – The mutated Stmt.

**Return type:**

[Stmt](#tvm.tirx.Stmt)

<a id="tvm.tirx.PyStmtExprMutator.visit_seq_stmt_"></a>
#### `tvm.tirx.PyStmtExprMutator.visit_seq_stmt_`

```python
visit_seq_stmt_(op: SeqStmt) → Stmt
```

Visit SeqStmt.
Users can customize this function to overwrite VisitStmt\_(const SeqStmtNode\* op)
on the C++ side.

**Parameters:**

**op** ([*SeqStmt*](#tvm.tirx.SeqStmt)) – The SeqStmt to be visited.

**Returns:**

**result** – The mutated Stmt.

**Return type:**

[Stmt](#tvm.tirx.Stmt)

<a id="tvm.tirx.PyStmtExprMutator.visit_evaluate_"></a>
#### `tvm.tirx.PyStmtExprMutator.visit_evaluate_`

```python
visit_evaluate_(op: Evaluate) → Stmt
```

Visit Evaluate.
Users can customize this function to overwrite VisitStmt\_(const EvaluateNode\* op)
on the C++ side.

**Parameters:**

**op** ([*Evaluate*](#tvm.tirx.Evaluate)) – The Evaluate to be visited.

**Returns:**

**result** – The mutated Stmt.

**Return type:**

[Stmt](#tvm.tirx.Stmt)

<a id="tvm.tirx.PyStmtExprMutator.visit_sblock_"></a>
#### `tvm.tirx.PyStmtExprMutator.visit_sblock_`

```python
visit_sblock_(op: SBlock) → Stmt
```

Visit SBlock.
Users can customize this function to overwrite VisitStmt\_(const SBlockNode\* op)
on the C++ side.

**Parameters:**

**op** ([*SBlock*](#tvm.tirx.SBlock)) – The SBlock to be visited.

**Returns:**

**result** – The mutated Stmt.

**Return type:**

[Stmt](#tvm.tirx.Stmt)

<a id="tvm.tirx.PyStmtExprMutator.visit_sblock_realize_"></a>
#### `tvm.tirx.PyStmtExprMutator.visit_sblock_realize_`

```python
visit_sblock_realize_(op: SBlockRealize) → Stmt
```

Visit BlockRealize.
Users can customize this function to overwrite VisitStmt\_(const SBlockRealizeNode\* op)
on the C++ side.

**Parameters:**

**op** ([*SBlockRealize*](#tvm.tirx.SBlockRealize)) – The SBlockRealize to be visited.

**Returns:**

**result** – The mutated Stmt.

**Return type:**

[Stmt](#tvm.tirx.Stmt)

<a id="tvm.tirx.PyStmtExprMutator.visit_var_"></a>
#### `tvm.tirx.PyStmtExprMutator.visit_var_`

```python
visit_var_(op: Var) → Expr
```

Visit Var.

Users can customize this function to overwrite VisitVar\_(const VarNode\* op)
on the C++ side.

**Parameters:**

**op** (*tirx.Var*) – The tirx.Var to be visited.

**Returns:**

**result** – The mutated Expr.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.PyStmtExprMutator.visit_buffer_load_"></a>
#### `tvm.tirx.PyStmtExprMutator.visit_buffer_load_`

```python
visit_buffer_load_(op: BufferLoad) → Expr
```

Visit BufferLoad.

Users can customize this function to overwrite VisitBufferLoad\_(const BufferLoadNode\* op)
on the C++ side.

**Parameters:**

**op** ([*BufferLoad*](#tvm.tirx.BufferLoad)) – The BufferLoad to be visited.

**Returns:**

**result** – The mutated Expr.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.PyStmtExprMutator.visit_producer_load_"></a>
#### `tvm.tirx.PyStmtExprMutator.visit_producer_load_`

```python
visit_producer_load_(op: ProducerLoad) → Expr
```

Visit ProducerLoad.

Users can customize this function to overwrite
VisitProducerLoad\_(const ProducerLoadNode\* op) on the C++ side.

**Parameters:**

**op** ([*ProducerLoad*](#tvm.tirx.ProducerLoad)) – The ProducerLoad to be visited.

**Returns:**

**result** – The mutated Expr.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.PyStmtExprMutator.visit_let_"></a>
#### `tvm.tirx.PyStmtExprMutator.visit_let_`

```python
visit_let_(op: Let) → Expr
```

Visit Let.

Users can customize this function to overwrite VisitLet\_(const LetNode\* op)
on the C++ side.

**Parameters:**

**op** ([*Let*](#tvm.tirx.Let)) – The Let to be visited.

**Returns:**

**result** – The mutated Expr.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.PyStmtExprMutator.visit_call_"></a>
#### `tvm.tirx.PyStmtExprMutator.visit_call_`

```python
visit_call_(op: Call) → Expr
```

Visit Call.

Users can customize this function to overwrite VisitCall\_(const CallNode\* op)
on the C++ side.

**Parameters:**

**op** (*tirx.Call*) – The tirx.Call to be visited.

**Returns:**

**result** – The mutated Expr.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.PyStmtExprMutator.visit_add_"></a>
#### `tvm.tirx.PyStmtExprMutator.visit_add_`

```python
visit_add_(op: Add) → Expr
```

Visit Add.

Users can customize this function to overwrite VisitAdd\_(const AddNode\* op)
on the C++ side.

**Parameters:**

**op** ([*Add*](#tvm.tirx.Add)) – The Add to be visited.

**Returns:**

**result** – The mutated Expr.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.PyStmtExprMutator.visit_sub_"></a>
#### `tvm.tirx.PyStmtExprMutator.visit_sub_`

```python
visit_sub_(op: Sub) → Expr
```

Visit Sub.

Users can customize this function to overwrite VisitSub\_(const SubNode\* op)
on the C++ side.

**Parameters:**

**op** ([*Sub*](#tvm.tirx.Sub)) – The Sub to be visited.

**Returns:**

**result** – The mutated Expr.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.PyStmtExprMutator.visit_mul_"></a>
#### `tvm.tirx.PyStmtExprMutator.visit_mul_`

```python
visit_mul_(op: Mul) → Expr
```

Visit Mul.

Users can customize this function to overwrite VisitMul\_(const MulNode\* op)
on the C++ side.

**Parameters:**

**op** ([*Mul*](#tvm.tirx.Mul)) – The Mul to be visited.

**Returns:**

**result** – The mutated Expr.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.PyStmtExprMutator.visit_div_"></a>
#### `tvm.tirx.PyStmtExprMutator.visit_div_`

```python
visit_div_(op: Div) → Expr
```

Visit Div.

Users can customize this function to overwrite VisitDiv\_(const DivNode\* op)
on the C++ side.

**Parameters:**

**op** ([*Div*](#tvm.tirx.Div)) – The Div to be visited.

**Returns:**

**result** – The mutated Expr.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.PyStmtExprMutator.visit_mod_"></a>
#### `tvm.tirx.PyStmtExprMutator.visit_mod_`

```python
visit_mod_(op: Mod) → Expr
```

Visit Mod.

Users can customize this function to overwrite VisitMod\_(const ModNode\* op)
on the C++ side.

**Parameters:**

**op** ([*Mod*](#tvm.tirx.Mod)) – The Mod to be visited.

**Returns:**

**result** – The mutated Expr.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.PyStmtExprMutator.visit_floor_div_"></a>
#### `tvm.tirx.PyStmtExprMutator.visit_floor_div_`

```python
visit_floor_div_(op: FloorDiv) → Expr
```

Visit FloorDiv.

Users can customize this function to overwrite VisitFloorDiv\_(const FloorDivNode\* op)
on the C++ side.

**Parameters:**

**op** ([*FloorDiv*](#tvm.tirx.FloorDiv)) – The FloorDiv to be visited.

**Returns:**

**result** – The mutated Expr.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.PyStmtExprMutator.visit_floor_mod_"></a>
#### `tvm.tirx.PyStmtExprMutator.visit_floor_mod_`

```python
visit_floor_mod_(op: FloorMod) → Expr
```

Visit FloorMod.

Users can customize this function to overwrite VisitFloorMod\_(const FloorModNode\* op)
on the C++ side.

**Parameters:**

**op** ([*FloorMod*](#tvm.tirx.FloorMod)) – The FloorMod to be visited.

**Returns:**

**result** – The mutated Expr.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.PyStmtExprMutator.visit_min_"></a>
#### `tvm.tirx.PyStmtExprMutator.visit_min_`

```python
visit_min_(op: Min) → Expr
```

Visit Min.

Users can customize this function to overwrite VisitMin\_(const MinNode\* op)
on the C++ side.

**Parameters:**

**op** ([*Min*](#tvm.tirx.Min)) – The Min to be visited.

**Returns:**

**result** – The mutated Expr.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.PyStmtExprMutator.visit_max_"></a>
#### `tvm.tirx.PyStmtExprMutator.visit_max_`

```python
visit_max_(op: Max) → Expr
```

Visit Max.

Users can customize this function to overwrite VisitMax\_(const MaxNode\* op)
on the C++ side.

**Parameters:**

**op** ([*Max*](#tvm.tirx.Max)) – The Max to be visited.

**Returns:**

**result** – The mutated Expr.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.PyStmtExprMutator.visit_eq_"></a>
#### `tvm.tirx.PyStmtExprMutator.visit_eq_`

```python
visit_eq_(op: EQ) → Expr
```

Visit EQ.

Users can customize this function to overwrite VisitEQ\_(const EQNode\* op)
on the C++ side.

**Parameters:**

**op** ([*EQ*](#tvm.tirx.EQ)) – The EQ to be visited.

**Returns:**

**result** – The mutated Expr.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.PyStmtExprMutator.visit_ne_"></a>
#### `tvm.tirx.PyStmtExprMutator.visit_ne_`

```python
visit_ne_(op: NE) → Expr
```

Visit NE.

Users can customize this function to overwrite VisitNE\_(const NENode\* op)
on the C++ side.

**Parameters:**

**op** ([*NE*](#tvm.tirx.NE)) – The NE to be visited.

**Returns:**

**result** – The mutated Expr.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.PyStmtExprMutator.visit_lt_"></a>
#### `tvm.tirx.PyStmtExprMutator.visit_lt_`

```python
visit_lt_(op: LT) → Expr
```

Visit LT.

Users can customize this function to overwrite VisitLT\_(const LTNode\* op)
on the C++ side.

**Parameters:**

**op** ([*LT*](#tvm.tirx.LT)) – The LT to be visited.

**Returns:**

**result** – The mutated Expr.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.PyStmtExprMutator.visit_le_"></a>
#### `tvm.tirx.PyStmtExprMutator.visit_le_`

```python
visit_le_(op: LE) → Expr
```

Visit LE.

Users can customize this function to overwrite VisitLE\_(const LENode\* op)
on the C++ side.

**Parameters:**

**op** ([*LE*](#tvm.tirx.LE)) – The LE to be visited.

**Returns:**

**result** – The mutated Expr.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.PyStmtExprMutator.visit_gt_"></a>
#### `tvm.tirx.PyStmtExprMutator.visit_gt_`

```python
visit_gt_(op: GT) → Expr
```

Visit GT.

Users can customize this function to overwrite VisitGT\_(const GTNode\* op)
on the C++ side.

**Parameters:**

**op** ([*GT*](#tvm.tirx.GT)) – The GT to be visited.

**Returns:**

**result** – The mutated Expr.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.PyStmtExprMutator.visit_ge_"></a>
#### `tvm.tirx.PyStmtExprMutator.visit_ge_`

```python
visit_ge_(op: GE) → Expr
```

Visit GE.

Users can customize this function to overwrite VisitGE\_(const GENode\* op)
on the C++ side.

**Parameters:**

**op** ([*GE*](#tvm.tirx.GE)) – The GE to be visited.

**Returns:**

**result** – The mutated Expr.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.PyStmtExprMutator.visit_and_"></a>
#### `tvm.tirx.PyStmtExprMutator.visit_and_`

```python
visit_and_(op: And) → Expr
```

Visit And.

Users can customize this function to overwrite VisitAnd\_(const AndNode\* op)
on the C++ side.

**Parameters:**

**op** ([*And*](#tvm.tirx.And)) – The And to be visited.

**Returns:**

**result** – The mutated Expr.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.PyStmtExprMutator.visit_or_"></a>
#### `tvm.tirx.PyStmtExprMutator.visit_or_`

```python
visit_or_(op: Or) → Expr
```

Visit Or.

Users can customize this function to overwrite VisitOr\_(const OrNode\* op)
on the C++ side.

**Parameters:**

**op** ([*Or*](#tvm.tirx.Or)) – The Or to be visited.

**Returns:**

**result** – The mutated Expr.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.PyStmtExprMutator.visit_reduce_"></a>
#### `tvm.tirx.PyStmtExprMutator.visit_reduce_`

```python
visit_reduce_(op: Reduce) → Expr
```

Visit Reduce.

Users can customize this function to overwrite VisitReduce\_(const ReduceNode\* op)
on the C++ side.

**Parameters:**

**op** ([*Reduce*](#tvm.tirx.Reduce)) – The Reduce to be visited.

**Returns:**

**result** – The mutated Expr.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.PyStmtExprMutator.visit_cast_"></a>
#### `tvm.tirx.PyStmtExprMutator.visit_cast_`

```python
visit_cast_(op: Cast) → Expr
```

Visit Cast.

Users can customize this function to overwrite VisitCast\_(const CastNode\* op)
on the C++ side.

**Parameters:**

**op** ([*Cast*](#tvm.tirx.Cast)) – The Cast to be visited.

**Returns:**

**result** – The mutated Expr.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.PyStmtExprMutator.visit_not_"></a>
#### `tvm.tirx.PyStmtExprMutator.visit_not_`

```python
visit_not_(op: Not) → Expr
```

Visit Not.

Users can customize this function to overwrite VisitNot\_(const NotNode\* op)
on the C++ side.

**Parameters:**

**op** ([*Not*](#tvm.tirx.Not)) – The Not to be visited.

**Returns:**

**result** – The mutated Expr.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.PyStmtExprMutator.visit_select_"></a>
#### `tvm.tirx.PyStmtExprMutator.visit_select_`

```python
visit_select_(op: Select) → Expr
```

Visit Select.

Users can customize this function to overwrite VisitSelect\_(const SelectNode\* op)
on the C++ side.

**Parameters:**

**op** ([*Select*](#tvm.tirx.Select)) – The Select to be visited.

**Returns:**

**result** – The mutated Expr.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.PyStmtExprMutator.visit_ramp_"></a>
#### `tvm.tirx.PyStmtExprMutator.visit_ramp_`

```python
visit_ramp_(op: Ramp) → Expr
```

Visit Ramp.

Users can customize this function to overwrite VisitRamp\_(const RampNode\* op)
on the C++ side.

**Parameters:**

**op** ([*Ramp*](#tvm.tirx.Ramp)) – The Ramp to be visited.

**Returns:**

**result** – The mutated Expr.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.PyStmtExprMutator.visit_broadcast_"></a>
#### `tvm.tirx.PyStmtExprMutator.visit_broadcast_`

```python
visit_broadcast_(op: Broadcast) → Expr
```

Visit Broadcast.

Users can customize this function to overwrite VisitBroadcast\_(const BroadcastNode\* op)
on the C++ side.

**Parameters:**

**op** ([*Broadcast*](#tvm.tirx.Broadcast)) – The Broadcast to be visited.

**Returns:**

**result** – The mutated Expr.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.PyStmtExprMutator.visit_shuffle_"></a>
#### `tvm.tirx.PyStmtExprMutator.visit_shuffle_`

```python
visit_shuffle_(op: Shuffle) → Expr
```

Visit Shuffle.

Users can customize this function to overwrite VisitShuffle\_(const ShuffleNode\* op)
on the C++ side.

**Parameters:**

**op** ([*Shuffle*](#tvm.tirx.Shuffle)) – The Shuffle to be visited.

**Returns:**

**result** – The mutated Expr.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.PyStmtExprMutator.visit_int_imm_"></a>
#### `tvm.tirx.PyStmtExprMutator.visit_int_imm_`

```python
visit_int_imm_(op: IntImm) → Expr
```

Visit IntImm.

Users can customize this function to overwrite VisitIntImm\_(const IntImmNode\* op)
on the C++ side.

**Parameters:**

**op** ([*IntImm*](#tvm.tirx.IntImm)) – The IntImm to be visited.

**Returns:**

**result** – The mutated Expr.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.PyStmtExprMutator.visit_float_imm_"></a>
#### `tvm.tirx.PyStmtExprMutator.visit_float_imm_`

```python
visit_float_imm_(op: FloatImm) → Expr
```

Visit FloatImm.

Users can customize this function to overwrite VisitFloatImm\_(const FloatImmNode\* op)
on the C++ side.

**Parameters:**

**op** ([*FloatImm*](#tvm.tirx.FloatImm)) – The FloatImm to be visited.

**Returns:**

**result** – The mutated Expr.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.PyStmtExprMutator.visit_string_imm_"></a>
#### `tvm.tirx.PyStmtExprMutator.visit_string_imm_`

```python
visit_string_imm_(op: StringImm) → Expr
```

Visit StringImm.

Users can customize this function to overwrite VisitStringImm\_(const StringImmNode\* op)
on the C++ side.

**Parameters:**

**op** ([*StringImm*](#tvm.tirx.StringImm)) – The StringImm to be visited.

**Returns:**

**result** – The mutated Expr.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.build"></a>
### `tvm.tirx.build`

```python
tvm.tirx.build(mod: PrimFunc | IRModule, target: str | Target | None = None, pipeline: None | str | Pass = 'default')
```

Build a function with a signature, generating code for devices
coupled with target information.

**Parameters:**

- **mod** (*Union*[[*PrimFunc*](#tvm.tirx.PrimFunc), tvm.ir.IRModule]) – The input to be built.
- **target** (*Optional*[*Union*[str, tvm.target.Target]]) – The target for compilation.
- **pipeline** (*Union*[*None*, str, *tvm.transform.Pass*]) – The pipeline to use for compilation.

**Returns:**

A module combining both host and device code.

**Return type:**

tvm.runtime.Module

<a id="tvm.tirx.get_tir_pipeline"></a>
### `tvm.tirx.get_tir_pipeline`

```python
tvm.tirx.get_tir_pipeline(name: str | None = None, **kwargs) → Pass
```

Get pre-build pipeline by name

**Parameters:**

**name** (*Optional*[str]) – Name of the pipeline

<a id="tvm.tirx.get_default_tir_pipeline"></a>
### `tvm.tirx.get_default_tir_pipeline`

```python
tvm.tirx.get_default_tir_pipeline(target: Target) → Pass
```

Get the default TIR pipeline for the given target.

<a id="tvm.tirx.register_tir_pipeline"></a>
### `tvm.tirx.register_tir_pipeline`

```python
tvm.tirx.register_tir_pipeline(name: str, pipeline_factory) → None
```

Register a named TIR pipeline factory.
