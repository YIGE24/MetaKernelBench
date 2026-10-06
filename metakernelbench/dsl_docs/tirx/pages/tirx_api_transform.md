<!-- source: https://tvm.apache.org/docs/tirx/api/transform.html -->

<a id="module-tvm.tirx.transform"></a>
# tvm.tirx.transform

Namespace of all TIR transformations

<a id="tvm.tirx.transform.prim_func_pass"></a>
### `tvm.tirx.transform.prim_func_pass`

```python
tvm.tirx.transform.prim_func_pass(pass_func=None, opt_level: int | None = None, name: str | None = None, required: list[str] | None = None, traceable=False) → Callable | PrimFuncPass
```

Decorate a function pass.

This function returns a callback when pass\_func
is provided. Otherwise, it returns the created function pass using the
given optimization function.

**Parameters:**

- **pass\_func** (*Optional*[*Callable*[*([tvm.tirx.PrimFunc*](tirx_api_tirx.md#tvm.tirx.PrimFunc), tvm.ir.IRModule, tvm.ir.transform.PassContext*)* *-> tvm.tirx.PrimFunc*]]) – The transformation function or class.
- **opt\_level** (int) – The optimization level of this module pass.
- **name** (*Optional*[str]) – The name of the function pass. The name could be empty. In this case, the
  name of the optimization function will be used as the pass name.
- **required** (*Optional*[*List*[str]]) – The list of passes that the function pass is dependent on.

**Returns:**

**create\_function\_pass** – A decorator will be returned if pass\_func is not provided,
otherwise return the decorated result.
The returned decorator has two behaviors depending on the input:
A new FunctionPass will be returned when we decorate a pass function.
A new FunctionPass class will be returned when we decorate a class type.

**Return type:**

Union[Callable, tvm.relax.transform.FunctionPass]

Examples

The following code block decorates a function pass class.

```python
@tvm.tirx.transform.prim_func_pass(opt_level=1)
class TestReplaceFunc:
    def __init__(self, new_func):
        self.new_func = new_func

    def transform_function(self, func, mod, ctx):
        # just for demo purposes
        # transform func to new_func
        return self.new_func
```

The following code creates a function pass by decorating
a user defined transform function.

```python
@tvm.tirx.transform.prim_func_pass(opt_level=2)
def transform(func, mod, ctx):
    # my transformations here.
    return func

function_pass = transform
assert isinstance(function_pass, transform.FunctionPass)
assert function_pass.info.opt_level == 2

# Given a module m, the optimization could be invoked as the following:
updated_mod = function_pass(m)
# Now constant folding should have been applied to every function in
# the provided module m. And the updated module will be returned.
```

<a id="tvm.tirx.transform.PrimFuncPass"></a>
### `tvm.tirx.transform.PrimFuncPass`

```python
class tvm.tirx.transform.PrimFuncPass(pass_info)
```

A pass that works on each [`tvm.tirx.PrimFunc()`](tirx_api_tirx.md#tvm.tirx.PrimFunc) in a module. A function
pass class should be created through py:func:tvm.tirx.transform.function\_pass.

<a id="tvm.tirx.transform.AnnotateEntryFunc"></a>
### `tvm.tirx.transform.AnnotateEntryFunc`

```python
tvm.tirx.transform.AnnotateEntryFunc()
```

Set a PrimFunc as the entry point if it is only function in IRModule.

**Returns:**

**fpass** – The result pass

**Return type:**

tvm.transform.Pass

<a id="tvm.tirx.transform.Apply"></a>
### `tvm.tirx.transform.Apply`

```python
tvm.tirx.transform.Apply(ftransform)
```

Apply ftransform to each function in the Module.

This function is a thin wrapper around tvm.tirx.transform.prim\_func\_pass

**Parameters:**

**ftransform** (*tvm.tirx.PrimFunc -> tvm.tirx.PrimFunc*) – The transformation pass.

**Returns:**

**fpass** – The result pass

**Return type:**

tvm.transform.Pass

<a id="tvm.tirx.transform.BF16ComputeLegalize"></a>
### `tvm.tirx.transform.BF16ComputeLegalize`

```python
tvm.tirx.transform.BF16ComputeLegalize()
```

Legalize bf16 compute Ops.

**Returns:**

**fpass** – The result pass

**Return type:**

tvm.transform.Pass

<a id="tvm.tirx.transform.BF16StorageLegalize"></a>
### `tvm.tirx.transform.BF16StorageLegalize`

```python
tvm.tirx.transform.BF16StorageLegalize()
```

Legalize bf16 storage types to u16.

**Returns:**

**fpass** – The result pass

**Return type:**

tvm.transform.Pass

<a id="tvm.tirx.transform.BindTarget"></a>
### `tvm.tirx.transform.BindTarget`

```python
tvm.tirx.transform.BindTarget(target)
```

Annotate a PrimFunc with a given target.
:param target: target
:type target: tvm.target.Target

**Returns:**

**fpass** – The result pass

**Return type:**

tvm.transform.Pass

<a id="tvm.tirx.transform.CommonSubexprElim"></a>
### `tvm.tirx.transform.CommonSubexprElim`

```python
tvm.tirx.transform.CommonSubexprElim()
```

Replace redundant computations by new variables.

**Returns:**

**fpass** – The result pass

**Return type:**

tvm.transform.Pass

<a id="tvm.tirx.transform.ConvertSSA"></a>
### `tvm.tirx.transform.ConvertSSA`

```python
tvm.tirx.transform.ConvertSSA()
```

Convert an IRModule to be SSA form.

This pass handles cases where the same tirx.Var appears in
multiple functions within the same module. For example, after
extracting a fragment from one function into another, where the
same tirx.Var may be defined both as within the body of the
original function, and as a parameter within the hoisted function.

**Returns:**

**fpass** – The result pass

**Return type:**

tvm.transform.Pass

<a id="tvm.tirx.transform.FP8ComputeLegalize"></a>
### `tvm.tirx.transform.FP8ComputeLegalize`

```python
tvm.tirx.transform.FP8ComputeLegalize(promote_dtype: str = 'float32')
```

Legalize fp8 compute Ops.

**Parameters:**

**promote\_dtype** (str) – The data type we promote fp8 to, options: float16/float32.

**Returns:**

**fpass** – The result pass

**Return type:**

tvm.transform.Pass

<a id="tvm.tirx.transform.FP8StorageLegalize"></a>
### `tvm.tirx.transform.FP8StorageLegalize`

```python
tvm.tirx.transform.FP8StorageLegalize()
```

Legalize fp8 storage types to u8.

**Returns:**

**fpass** – The result pass

**Return type:**

tvm.transform.Pass

<a id="tvm.tirx.transform.Filter"></a>
### `tvm.tirx.transform.Filter`

```python
tvm.tirx.transform.Filter(fcond: Callable)
```

Filter out PrimFuncs that does not satisfy the given condition.
fcond should be a function that takes a primfunc and returns boolean.

**Returns:**

**fpass** – The result pass

**Return type:**

tvm.transform.Pass

<a id="tvm.tirx.transform.FlattenBuffer"></a>
### `tvm.tirx.transform.FlattenBuffer`

```python
tvm.tirx.transform.FlattenBuffer()
```

Flatten the multi-dimensional BufferLoad and BufferStore to single dimensional
BufferLoad/BufferStore for the TIR not contains opaque block.

**Returns:**

**fpass** – The result pass

**Return type:**

tvm.transform.Pass

<a id="tvm.tirx.transform.ForceNarrowIndexToInt32"></a>
### `tvm.tirx.transform.ForceNarrowIndexToInt32`

```python
tvm.tirx.transform.ForceNarrowIndexToInt32()
```

Force narrow down indexing expressions and integer buffers to int32 dtype.

**Returns:**

**fpass** – The result pass

**Return type:**

tvm.transform.Pass

> **Note**
>
> This pass should not be used in default cases.

<a id="tvm.tirx.transform.HoistExpressionConfig"></a>
### `tvm.tirx.transform.HoistExpressionConfig`

```python
class tvm.tirx.transform.HoistExpressionConfig(hoisted_conditionals=<MISSING>, hoisted_let_bindings=<MISSING>)
```

Config for hoist expression pass

<a id="tvm.tirx.transform.HoistExpressionConfig.hoisted_conditionals"></a>
#### `tvm.tirx.transform.HoistExpressionConfig.hoisted_conditionals`

```python
property hoisted_conditionals
```

Bitflags for the types of boolean expressions to hoist

<a id="tvm.tirx.transform.HoistExpressionConfig.hoisted_let_bindings"></a>
#### `tvm.tirx.transform.HoistExpressionConfig.hoisted_let_bindings`

```python
property hoisted_let_bindings
```

Bitflags for the types of let bindings to hoist

<a id="tvm.tirx.transform.HoistIfThenElseConfig"></a>
### `tvm.tirx.transform.HoistIfThenElseConfig`

```python
class tvm.tirx.transform.HoistIfThenElseConfig(support_block_scope_hoisting=<MISSING>)
```

Config for hoist if then else pass

<a id="tvm.tirx.transform.HoistIfThenElseConfig.support_block_scope_hoisting"></a>
#### `tvm.tirx.transform.HoistIfThenElseConfig.support_block_scope_hoisting`

```python
property support_block_scope_hoisting
```

Hoist if cond with block scope variables

<a id="tvm.tirx.transform.HoistedConditionals"></a>
### `tvm.tirx.transform.HoistedConditionals`

```python
class tvm.tirx.transform.HoistedConditionals(value)
```

Flags for use in HoistExpressionConfig.conditional\_types

Each bitflag represents a type of expression that should be
hoisted to the outermost loop possible.

<a id="tvm.tirx.transform.HoistedConditionals.Never"></a>
#### `tvm.tirx.transform.HoistedConditionals.Never`

```python
Never = 0
```

No hoisting of conditionals

<a id="tvm.tirx.transform.HoistedConditionals.IfElseStmt"></a>
#### `tvm.tirx.transform.HoistedConditionals.IfElseStmt`

```python
IfElseStmt = 1
```

If set, look for hoist candidates in IfElseStmt

<a id="tvm.tirx.transform.HoistedConditionals.IfElseExpr"></a>
#### `tvm.tirx.transform.HoistedConditionals.IfElseExpr`

```python
IfElseExpr = 2
```

If set, look for hoist candidates in tirx.if\_then\_else

<a id="tvm.tirx.transform.HoistedConditionals.BooleanExpression"></a>
#### `tvm.tirx.transform.HoistedConditionals.BooleanExpression`

```python
BooleanExpression = 4
```

If set, look for hoist candidates in all boolean expressions

<a id="tvm.tirx.transform.HoistedConditionals.UsingBlockVar"></a>
#### `tvm.tirx.transform.HoistedConditionals.UsingBlockVar`

```python
UsingBlockVar = 8
```

If set, allow hoisting of conditionals that use a block variable (e.g. threadIdx.x)

<a id="tvm.tirx.transform.HoistedConditionals.All"></a>
#### `tvm.tirx.transform.HoistedConditionals.All`

```python
All = 15
```

Enable all hoisting of conditionals

<a id="tvm.tirx.transform.HoistedLetBindings"></a>
### `tvm.tirx.transform.HoistedLetBindings`

```python
class tvm.tirx.transform.HoistedLetBindings(value)
```

Flags for use in HoistExpressionConfig.let\_binding\_types

Each bitflag represents a type of let binding expression that should be
hoisted to the outermost loop possible.

<a id="tvm.tirx.transform.HoistedLetBindings.Never"></a>
#### `tvm.tirx.transform.HoistedLetBindings.Never`

```python
Never = 0
```

No hoisting of let bindings

<a id="tvm.tirx.transform.HoistedLetBindings.RequiredByConditional"></a>
#### `tvm.tirx.transform.HoistedLetBindings.RequiredByConditional`

```python
RequiredByConditional = 1
```

Bindings that are used by a hoisted conditional

<a id="tvm.tirx.transform.HoistedLetBindings.Bind"></a>
#### `tvm.tirx.transform.HoistedLetBindings.Bind`

```python
Bind = 2
```

Bindings occurring in Bind nodes

<a id="tvm.tirx.transform.HoistedLetBindings.LetExpr"></a>
#### `tvm.tirx.transform.HoistedLetBindings.LetExpr`

```python
LetExpr = 4
```

Bindings occurring in Let expressions

<a id="tvm.tirx.transform.HoistedLetBindings.All"></a>
#### `tvm.tirx.transform.HoistedLetBindings.All`

```python
All = 7
```

Enable all hoisting of let bindings

<a id="tvm.tirx.transform.InlinePrivateFunctions"></a>
### `tvm.tirx.transform.InlinePrivateFunctions`

```python
tvm.tirx.transform.InlinePrivateFunctions()
```

Inline calls to private functions

**Returns:**

**fpass** – The result pass

**Return type:**

tvm.transform.Pass

<a id="tvm.tirx.transform.LowerIntrin"></a>
### `tvm.tirx.transform.LowerIntrin`

```python
tvm.tirx.transform.LowerIntrin()
```

Lower target specific intrinsic calls.

**Returns:**

**fpass** – The result pass

**Return type:**

tvm.transform.Pass

<a id="tvm.tirx.transform.LowerTIRx"></a>
### `tvm.tirx.transform.LowerTIRx`

```python
tvm.tirx.transform.LowerTIRx()
```

Lower TIR to a lower-level IR.

**Returns:**

**fpass** – The result pass

**Return type:**

tvm.transform.Pass

<a id="tvm.tirx.transform.LowerTIRxOpaque"></a>
### `tvm.tirx.transform.LowerTIRxOpaque`

```python
tvm.tirx.transform.LowerTIRxOpaque()
```

Lower opaque constructs in TIRX programs.

Handles AllocBuffer lowering, For(thread\_binding) to AttrStmt(thread\_extent)
conversion, unit loop elimination, and pragma annotation handling.
This is the tirx-specific counterpart of s\_tir.LowerOpaqueBlock,
without any SBlock/SBlockRealize handling.

**Returns:**

**fpass** – The result pass

**Return type:**

tvm.transform.Pass

<a id="tvm.tirx.transform.LowerTVMBuiltin"></a>
### `tvm.tirx.transform.LowerTVMBuiltin`

```python
tvm.tirx.transform.LowerTVMBuiltin()
```

Lower tvm builtin intrinsics.

**Returns:**

**fpass** – The result pass

**Return type:**

tvm.transform.Pass

<a id="tvm.tirx.transform.LowerWarpMemory"></a>
### `tvm.tirx.transform.LowerWarpMemory`

```python
tvm.tirx.transform.LowerWarpMemory()
```

Lower warp memory access to low-level device related function calls.

**Returns:**

**fpass** – The result pass

**Return type:**

tvm.transform.Pass

<a id="tvm.tirx.transform.MakePackedAPI"></a>
### `tvm.tirx.transform.MakePackedAPI`

```python
tvm.tirx.transform.MakePackedAPI()
```

Transform the PrimFuncs in the module to a packed func API.

Prior to this pass, the PrimFunc may have Buffer arguments defined
in the PrimFuncNode::buffer\_map. This pass consumes the
buffer\_map, using it to generate arguments that implement
the packed based TVM FFI API.

For static shapes, the BufferNode::shape, BufferNode::strides,
and BufferNode::elem\_offset member variables are used to
generate runtime checks on the corresponding member variables in
the user-provided DLTensor\* or tvm.runtime.tensor argument. (e.g. A
PrimFunc that accepts a buffer of shape [16,32] validates that
the DLTensor::shape array is [16,32].)

For dynamic Buffers, in which one or more of these BufferNode member
variables use tirx.Var that are not defined by other PrimFunc
parameters, these are instead used to define the variables based on
the corresponding DLTensor members. (e.g. A PrimFunc that accepts a
buffer of shape [tirx.Var(“n”, “int64”), tirx.Var(“m”, “int64”)],
when passed a DLTensor of shape [16, 32], will define n = 16 and
m = 32, based on the argument’s shape.

**Returns:**

**fpass** – The result pass

**Return type:**

tvm.transform.Pass

<a id="tvm.tirx.transform.NarrowDataType"></a>
### `tvm.tirx.transform.NarrowDataType`

```python
tvm.tirx.transform.NarrowDataType(target_bits: int)
```

Narrow down Expr datatype in stmt to target\_bits.

**Parameters:**

**target\_bits** (int) – The target bit configuration.

**Returns:**

**fpass** – The result pass

**Return type:**

tvm.transform.Pass

> **Note**
>
> Run this pass after FlattenBuffer.

<a id="tvm.tirx.transform.PointerValueTypeRewrite"></a>
### `tvm.tirx.transform.PointerValueTypeRewrite`

```python
tvm.tirx.transform.PointerValueTypeRewrite()
```

Rewrite the pointer content type of arguments, as well as Alloc internal to the function to use
the most frequently accessed type for load/store to avoid pointer casting in backend when
possible.

**Returns:**

**fpass** – The result pass

**Return type:**

tvm.transform.Pass

<a id="tvm.tirx.transform.RemoveAssume"></a>
### `tvm.tirx.transform.RemoveAssume`

```python
tvm.tirx.transform.RemoveAssume()
```

Remove all instances of builtin::assume

**Returns:**

**fpass** – The result pass

**Return type:**

tvm.transform.Pass

<a id="tvm.tirx.transform.RemoveNoOp"></a>
### `tvm.tirx.transform.RemoveNoOp`

```python
tvm.tirx.transform.RemoveNoOp()
```

Remove No Op from the Stmt.

**Returns:**

**fpass** – The result pass

**Return type:**

tvm.transform.Pass

<a id="tvm.tirx.transform.RemoveNoOpConfig"></a>
### `tvm.tirx.transform.RemoveNoOpConfig`

```python
class tvm.tirx.transform.RemoveNoOpConfig(max_simplification_steps=<MISSING>, ignore_profiler_call=<MISSING>)
```

Config for remove no op pass

<a id="tvm.tirx.transform.RemoveNoOpConfig.ignore_profiler_call"></a>
#### `tvm.tirx.transform.RemoveNoOpConfig.ignore_profiler_call`

```python
property ignore_profiler_call
```

If true, profiler calls are rendered as no-ops.

<a id="tvm.tirx.transform.RemoveNoOpConfig.max_simplification_steps"></a>
#### `tvm.tirx.transform.RemoveNoOpConfig.max_simplification_steps`

```python
property max_simplification_steps
```

If non-zero, RewriteSimplifier will throw an error after the number of steps specified. For use in debug and testing purposes.

<a id="tvm.tirx.transform.SkipAssert"></a>
### `tvm.tirx.transform.SkipAssert`

```python
tvm.tirx.transform.SkipAssert()
```

Skip assert stmt.

**Returns:**

**fpass** – The result pass

**Return type:**

tvm.transform.Pass

<a id="tvm.tirx.transform.SplitHostDevice"></a>
### `tvm.tirx.transform.SplitHostDevice`

```python
tvm.tirx.transform.SplitHostDevice()
```

Annotate, split, and lower host/device functions.

This pass first annotates device regions within host functions,
then splits them into host and device-side PrimFuncs, and finally
lowers host-to-device calls into the device kernel launch ABI.

**Returns:**

**fpass** – The result pass

**Return type:**

tvm.transform.Pass

<a id="tvm.tirx.transform.StmtSimplify"></a>
### `tvm.tirx.transform.StmtSimplify`

```python
tvm.tirx.transform.StmtSimplify()
```

Run statement-level arithmetic simplifications on the TIR PrimFunc.

**Returns:**

**fpass** – The result pass

**Return type:**

tvm.transform.Pass

<a id="tvm.tirx.transform.StmtSimplifyConfig"></a>
### `tvm.tirx.transform.StmtSimplifyConfig`

```python
class tvm.tirx.transform.StmtSimplifyConfig(transitively_prove_inequalities=<MISSING>, convert_boolean_to_and_of_ors=<MISSING>, apply_constraints_to_boolean_branches=<MISSING>)
```

Config for stmt simplify pass

<a id="tvm.tirx.transform.StmtSimplifyConfig.apply_constraints_to_boolean_branches"></a>
#### `tvm.tirx.transform.StmtSimplifyConfig.apply_constraints_to_boolean_branches`

```python
property apply_constraints_to_boolean_branches
```

If true, simplify each branch of AND/OR under constraints provided by the other branch

<a id="tvm.tirx.transform.StmtSimplifyConfig.convert_boolean_to_and_of_ors"></a>
#### `tvm.tirx.transform.StmtSimplifyConfig.convert_boolean_to_and_of_ors`

```python
property convert_boolean_to_and_of_ors
```

If true, simplify conditionals into an AND of ORs

<a id="tvm.tirx.transform.StmtSimplifyConfig.transitively_prove_inequalities"></a>
#### `tvm.tirx.transform.StmtSimplifyConfig.transitively_prove_inequalities`

```python
property transitively_prove_inequalities
```

If true, simplify conditionals with transitive combinations of scoped constraints

<a id="tvm.tirx.transform.StorageRewrite"></a>
### `tvm.tirx.transform.StorageRewrite`

```python
tvm.tirx.transform.StorageRewrite()
```

Rewrite storage allocation pattern.

Moves the allocation to outer most possible scope.
Trying to share space between allocations to make
a static allocation plan when possible.

**Returns:**

**fpass** – The result pass

**Return type:**

tvm.transform.Pass

<a id="tvm.tirx.transform.TilePrimitiveDispatch"></a>
### `tvm.tirx.transform.TilePrimitiveDispatch`

```python
tvm.tirx.transform.TilePrimitiveDispatch()
```

Lower TIRx tile primitive calls through the active backend dispatch table.

**Returns:**

**fpass** – The result pass

**Return type:**

tvm.transform.Pass

<a id="tvm.tirx.transform.UnrollLoop"></a>
### `tvm.tirx.transform.UnrollLoop`

```python
tvm.tirx.transform.UnrollLoop()
```

Unroll the constant loop marked by unroll.

This pass also automatically attach pragma unroll tag to loops which meets the standard.

**Returns:**

**fpass** – The result pass

**Return type:**

tvm.transform.Pass

<a id="tvm.tirx.transform.UnrollLoopConfig"></a>
### `tvm.tirx.transform.UnrollLoopConfig`

```python
class tvm.tirx.transform.UnrollLoopConfig(auto_max_step=<MISSING>, auto_max_depth=<MISSING>, auto_max_extent=<MISSING>, explicit_unroll=<MISSING>, unroll_local_access=<MISSING>)
```

Config for unroll loop pass

<a id="tvm.tirx.transform.UnrollLoopConfig.auto_max_depth"></a>
#### `tvm.tirx.transform.UnrollLoopConfig.auto_max_depth`

```python
property auto_max_depth
```

The maximum nested level of loops that can be automatically unrolled.

<a id="tvm.tirx.transform.UnrollLoopConfig.auto_max_extent"></a>
#### `tvm.tirx.transform.UnrollLoopConfig.auto_max_extent`

```python
property auto_max_extent
```

The maximum extent` of loop that will be unrolled.

<a id="tvm.tirx.transform.UnrollLoopConfig.auto_max_step"></a>
#### `tvm.tirx.transform.UnrollLoopConfig.auto_max_step`

```python
property auto_max_step
```

Threshold of number of steps in the loop to be automatically unrolled

<a id="tvm.tirx.transform.UnrollLoopConfig.explicit_unroll"></a>
#### `tvm.tirx.transform.UnrollLoopConfig.explicit_unroll`

```python
property explicit_unroll
```

Whether to explicitly unroll the loop instead of setting a pragma

<a id="tvm.tirx.transform.UnrollLoopConfig.unroll_local_access"></a>
#### `tvm.tirx.transform.UnrollLoopConfig.unroll_local_access`

```python
property unroll_local_access
```

Whether to always unroll local access

<a id="tvm.tirx.transform.VectorizeLoop"></a>
### `tvm.tirx.transform.VectorizeLoop`

```python
tvm.tirx.transform.VectorizeLoop(enable_vectorize: bool = True)
```

Lower vectorization loops.

**Parameters:**

**enable\_vectorize** (bool) – Whether vectorization is enabled.
Will lower to scalar loop when it is turned off.

**Returns:**

**fpass** – The result pass

**Return type:**

tvm.transform.Pass

<a id="tvm.tirx.transform.VerifyMemory"></a>
### `tvm.tirx.transform.VerifyMemory`

```python
tvm.tirx.transform.VerifyMemory()
```

Verify if func contains illegal host side direct memory access.

**Returns:**

**fpass** – The result pass

**Return type:**

tvm.transform.Pass
