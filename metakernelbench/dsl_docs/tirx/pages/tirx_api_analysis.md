<!-- source: https://tvm.apache.org/docs/tirx/api/analysis.html -->

<a id="module-tvm.tirx.analysis.analysis"></a>
# tvm.tirx.analysis

Wrapping existing analysis utils.

<a id="tvm.tirx.analysis.analysis.expr_deep_equal"></a>
### `tvm.tirx.analysis.analysis.expr_deep_equal`

```python
tvm.tirx.analysis.analysis.expr_deep_equal(lhs: Expr, rhs: Expr) → bool
```

Deeply compare two nested expressions.

**Parameters:**

- **lhs** (tvm.relax.Expr) – The left operand.
- **rhs** (tvm.relax.Expr) – The right operand.

**Returns:**

**result** – The comparison result

**Return type:**

bool

> **Note**
>
> This function does not remap variable bindings, it will not
> return true for (let x = 1 in x + 1) vs (let y = 1 in y + 1), unless x.same\_as(y).
> Use py:func:tvm\_ffi.structural\_equal to handle structural variable remapping.
>
> Due to the restriction of not remapping variables, this function can run
> faster than StructuralEqual and can be used as a utility function during arithmetic
> simplifications.
>
> Always consider py:func:tvm\_ffi.structural\_equal first, which handles
> the structural remapping.
> **See also**
>
> `tvm_ffi.structural_equal`

<a id="tvm.tirx.analysis.analysis.verify_ssa"></a>
### `tvm.tirx.analysis.analysis.verify_ssa`

```python
tvm.tirx.analysis.analysis.verify_ssa(func: PrimFunc) → bool
```

Verify if the func is in SSA form.

**Parameters:**

**func** ([*tvm.tirx.PrimFunc*](tirx_api_tirx.md#tvm.tirx.PrimFunc)) – The module to be verified.

**Returns:**

**result** – The result of verification.

**Return type:**

bool

<a id="tvm.tirx.analysis.analysis.verify_memory"></a>
### `tvm.tirx.analysis.analysis.verify_memory`

```python
tvm.tirx.analysis.analysis.verify_memory(func: PrimFunc) → bool
```

Verify if func contains illegal host side direct memory access.

**Parameters:**

**func** ([*tvm.tirx.PrimFunc*](tirx_api_tirx.md#tvm.tirx.PrimFunc)) – The module to be verified.

**Returns:**

**result** – The result of verification.

**Return type:**

bool

<a id="tvm.tirx.analysis.analysis.undefined_vars"></a>
### `tvm.tirx.analysis.analysis.undefined_vars`

```python
tvm.tirx.analysis.analysis.undefined_vars(node: Stmt | Expr, defs: list[Var] | None = None) → list[Var]
```

Find undefined vars in a TIR statement or expression.

**Parameters:**

- **node** (*Union*[[*Stmt*](tirx_api_tirx.md#tvm.tirx.Stmt), tvm.relax.Expr]) – The TIR statement or expression to be checked.
- **defs** (*Optional*[*List*[*tirx.Var*]]) – The vars that is defined

**Returns:**

**result** – The undefined vars.

**Return type:**

List[tirx.Var]

<a id="tvm.tirx.analysis.analysis.verify_well_formed"></a>
### `tvm.tirx.analysis.analysis.verify_well_formed`

```python
tvm.tirx.analysis.analysis.verify_well_formed(obj: PrimFunc | IRModule, assert_mode: bool = True) → bool
```

**Verify if the given TIR is well-formed. The verification includes:**

- Check if expressions not contain vars that is defined outside the block.

**Parameters:**

- **obj** (*Union*[[*tvm.tirx.PrimFunc*](tirx_api_tirx.md#tvm.tirx.PrimFunc), tvm.ir.IRModule]) – The function or module to be verified.
- **assert\_mode** (bool) – The indicator if it raises an error when the function is not well-formed.

**Returns:**

**result** – Whether it is a well-formed TIR function.

**Return type:**

bool

<a id="tvm.tirx.analysis.analysis.verify_tirx_well_formed"></a>
### `tvm.tirx.analysis.analysis.verify_tirx_well_formed`

```python
tvm.tirx.analysis.analysis.verify_tirx_well_formed(obj: PrimFunc | IRModule, assert_mode: bool = True, device_func: bool = False) → bool
```

Verify if the given TIRX is well-formed.

**Parameters:**

- **obj** (*Union*[[*tvm.tirx.PrimFunc*](tirx_api_tirx.md#tvm.tirx.PrimFunc), tvm.ir.IRModule]) – The function or module to be verified.
- **assert\_mode** (bool) – The indicator if it raises an error when the function is not well-formed.
- **device\_func** (bool) – The indicator if it is a device function.

**Returns:**

**result** – Whether it is a well-formed TIRX function.

**Return type:**

bool
