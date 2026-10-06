<!-- source: https://tvm.apache.org/docs/tirx/api/stmt_functor.html -->

<a id="module-tvm.tirx.stmt_functor"></a>
# tvm.tirx.stmt_functor

Statement functor utilities for IR transformations

<a id="tvm.tirx.stmt_functor.StmtFunctor"></a>
### `tvm.tirx.stmt_functor.StmtFunctor`

```python
class tvm.tirx.stmt_functor.StmtFunctor
```

An abstract visitor over Statement, with visiting functions defined for each Stmt type.

<a id="tvm.tirx.stmt_functor.StmtFunctor.visit_stmt"></a>
#### `tvm.tirx.stmt_functor.StmtFunctor.visit_stmt`

```python
visit_stmt(stmt)
```

Apply the visitor to a statement.

**Parameters:**

**stmt** ([*tvm.tirx.Stmt*](tirx_api_tirx.md#tvm.tirx.Stmt)) – The statement to be visited.

**Returns:**

**result** – The result of the visit.

**Return type:**

Any

<a id="tvm.tirx.stmt_functor.StmtFunctor.visit_stmt_default_"></a>
#### `tvm.tirx.stmt_functor.StmtFunctor.visit_stmt_default_`

```python
visit_stmt_default_(op)
```

Default visitor implementation for statements.

<a id="tvm.tirx.stmt_functor.StmtFunctor.visit_bind_"></a>
#### `tvm.tirx.stmt_functor.StmtFunctor.visit_bind_`

```python
visit_bind_(op)
```

Visitor for Bind nodes.

<a id="tvm.tirx.stmt_functor.StmtFunctor.visit_attr_"></a>
#### `tvm.tirx.stmt_functor.StmtFunctor.visit_attr_`

```python
visit_attr_(op)
```

Visitor for AttrStmt nodes.

<a id="tvm.tirx.stmt_functor.StmtFunctor.visit_if_then_else_"></a>
#### `tvm.tirx.stmt_functor.StmtFunctor.visit_if_then_else_`

```python
visit_if_then_else_(op)
```

Visitor for IfThenElse nodes.

<a id="tvm.tirx.stmt_functor.StmtFunctor.visit_for_"></a>
#### `tvm.tirx.stmt_functor.StmtFunctor.visit_for_`

```python
visit_for_(op)
```

Visitor for For nodes.

<a id="tvm.tirx.stmt_functor.StmtFunctor.visit_while_"></a>
#### `tvm.tirx.stmt_functor.StmtFunctor.visit_while_`

```python
visit_while_(op)
```

Visitor for While nodes.

<a id="tvm.tirx.stmt_functor.StmtFunctor.visit_return_"></a>
#### `tvm.tirx.stmt_functor.StmtFunctor.visit_return_`

```python
visit_return_(op)
```

Visitor for Return nodes.

<a id="tvm.tirx.stmt_functor.StmtFunctor.visit_break_"></a>
#### `tvm.tirx.stmt_functor.StmtFunctor.visit_break_`

```python
visit_break_(op)
```

Visitor for Break nodes.

<a id="tvm.tirx.stmt_functor.StmtFunctor.visit_continue_"></a>
#### `tvm.tirx.stmt_functor.StmtFunctor.visit_continue_`

```python
visit_continue_(op)
```

Visitor for Continue nodes.

<a id="tvm.tirx.stmt_functor.StmtFunctor.visit_allocate_"></a>
#### `tvm.tirx.stmt_functor.StmtFunctor.visit_allocate_`

```python
visit_allocate_(op)
```

Visitor for Allocate nodes.

<a id="tvm.tirx.stmt_functor.StmtFunctor.visit_allocate_const_"></a>
#### `tvm.tirx.stmt_functor.StmtFunctor.visit_allocate_const_`

```python
visit_allocate_const_(op)
```

Visitor for AllocateConst nodes.

<a id="tvm.tirx.stmt_functor.StmtFunctor.visit_decl_buffer_"></a>
#### `tvm.tirx.stmt_functor.StmtFunctor.visit_decl_buffer_`

```python
visit_decl_buffer_(op)
```

Visitor for DeclBuffer nodes.

<a id="tvm.tirx.stmt_functor.StmtFunctor.visit_buffer_store_"></a>
#### `tvm.tirx.stmt_functor.StmtFunctor.visit_buffer_store_`

```python
visit_buffer_store_(op)
```

Visitor for BufferStore nodes.

<a id="tvm.tirx.stmt_functor.StmtFunctor.visit_buffer_realize_"></a>
#### `tvm.tirx.stmt_functor.StmtFunctor.visit_buffer_realize_`

```python
visit_buffer_realize_(op)
```

Visitor for BufferRealize nodes.

<a id="tvm.tirx.stmt_functor.StmtFunctor.visit_assert_"></a>
#### `tvm.tirx.stmt_functor.StmtFunctor.visit_assert_`

```python
visit_assert_(op)
```

Visitor for AssertStmt nodes.

<a id="tvm.tirx.stmt_functor.StmtFunctor.visit_producer_store_"></a>
#### `tvm.tirx.stmt_functor.StmtFunctor.visit_producer_store_`

```python
visit_producer_store_(op)
```

Visitor for ProducerStore nodes.

<a id="tvm.tirx.stmt_functor.StmtFunctor.visit_producer_realize_"></a>
#### `tvm.tirx.stmt_functor.StmtFunctor.visit_producer_realize_`

```python
visit_producer_realize_(op)
```

Visitor for ProducerRealize nodes.

<a id="tvm.tirx.stmt_functor.StmtFunctor.visit_prefetch_"></a>
#### `tvm.tirx.stmt_functor.StmtFunctor.visit_prefetch_`

```python
visit_prefetch_(op)
```

Visitor for Prefetch nodes.

<a id="tvm.tirx.stmt_functor.StmtFunctor.visit_seqstmt_"></a>
#### `tvm.tirx.stmt_functor.StmtFunctor.visit_seqstmt_`

```python
visit_seqstmt_(op)
```

Visitor for SeqStmt nodes.

<a id="tvm.tirx.stmt_functor.StmtFunctor.visit_evaluate_"></a>
#### `tvm.tirx.stmt_functor.StmtFunctor.visit_evaluate_`

```python
visit_evaluate_(op)
```

Visitor for Evaluate nodes.

<a id="tvm.tirx.stmt_functor.StmtFunctor.visit_block_"></a>
#### `tvm.tirx.stmt_functor.StmtFunctor.visit_block_`

```python
visit_block_(op)
```

Visitor for Block nodes.

<a id="tvm.tirx.stmt_functor.StmtFunctor.visit_block_realize_"></a>
#### `tvm.tirx.stmt_functor.StmtFunctor.visit_block_realize_`

```python
visit_block_realize_(op)
```

Visitor for BlockRealize nodes.

<a id="tvm.tirx.stmt_functor.StmtFunctor.visit_scope_id_def_stmt_"></a>
#### `tvm.tirx.stmt_functor.StmtFunctor.visit_scope_id_def_stmt_`

```python
visit_scope_id_def_stmt_(op)
```

Visitor for ScopeIdDefStmt nodes.

<a id="tvm.tirx.stmt_functor.StmtFunctor.visit_op_call_"></a>
#### `tvm.tirx.stmt_functor.StmtFunctor.visit_op_call_`

```python
visit_op_call_(op)
```

Visitor for TilePrimitiveCall nodes.

<a id="tvm.tirx.stmt_functor.StmtFunctor.visit_buffer_region_"></a>
#### `tvm.tirx.stmt_functor.StmtFunctor.visit_buffer_region_`

```python
visit_buffer_region_(op)
```

Visitor for BufferRegion nodes.

<a id="tvm.tirx.stmt_functor.StmtFunctor.visit_alloc_buffer_"></a>
#### `tvm.tirx.stmt_functor.StmtFunctor.visit_alloc_buffer_`

```python
visit_alloc_buffer_(op)
```

Visitor for AllocBuffer nodes.

<a id="tvm.tirx.stmt_functor.StmtVisitor"></a>
### `tvm.tirx.stmt_functor.StmtVisitor`

```python
class tvm.tirx.stmt_functor.StmtVisitor
```

A visitor over Stmt.

This is a visitor that recursively traverses a statement. Subclasses can
override the visit methods to customize the behavior.

<a id="tvm.tirx.stmt_functor.StmtVisitor.visit_expr"></a>
#### `tvm.tirx.stmt_functor.StmtVisitor.visit_expr`

```python
visit_expr(expr)
```

Visit expressions that occur in a statement.

This method can be overridden to implement expression
traversal in a statement visitor.

**Parameters:**

**expr** (tvm.relax.Expr) – The expression to be visited.

<a id="tvm.tirx.stmt_functor.StmtVisitor.visit_bind_"></a>
#### `tvm.tirx.stmt_functor.StmtVisitor.visit_bind_`

```python
visit_bind_(op)
```

Visitor implementation for Bind.

<a id="tvm.tirx.stmt_functor.StmtVisitor.visit_attr_"></a>
#### `tvm.tirx.stmt_functor.StmtVisitor.visit_attr_`

```python
visit_attr_(op)
```

Visitor implementation for AttrStmt.

<a id="tvm.tirx.stmt_functor.StmtVisitor.visit_if_then_else_"></a>
#### `tvm.tirx.stmt_functor.StmtVisitor.visit_if_then_else_`

```python
visit_if_then_else_(op)
```

Visitor implementation for IfThenElse.

<a id="tvm.tirx.stmt_functor.StmtVisitor.visit_for_"></a>
#### `tvm.tirx.stmt_functor.StmtVisitor.visit_for_`

```python
visit_for_(op)
```

Visitor implementation for For.

<a id="tvm.tirx.stmt_functor.StmtVisitor.visit_while_"></a>
#### `tvm.tirx.stmt_functor.StmtVisitor.visit_while_`

```python
visit_while_(op)
```

Visitor implementation for While.

<a id="tvm.tirx.stmt_functor.StmtVisitor.visit_return_"></a>
#### `tvm.tirx.stmt_functor.StmtVisitor.visit_return_`

```python
visit_return_(op)
```

Visitor implementation for Return.

<a id="tvm.tirx.stmt_functor.StmtVisitor.visit_break_"></a>
#### `tvm.tirx.stmt_functor.StmtVisitor.visit_break_`

```python
visit_break_(op)
```

Visitor implementation for Break.

<a id="tvm.tirx.stmt_functor.StmtVisitor.visit_continue_"></a>
#### `tvm.tirx.stmt_functor.StmtVisitor.visit_continue_`

```python
visit_continue_(op)
```

Visitor implementation for Continue.

<a id="tvm.tirx.stmt_functor.StmtVisitor.visit_allocate_"></a>
#### `tvm.tirx.stmt_functor.StmtVisitor.visit_allocate_`

```python
visit_allocate_(op)
```

Visitor implementation for Allocate.

<a id="tvm.tirx.stmt_functor.StmtVisitor.visit_allocate_const_"></a>
#### `tvm.tirx.stmt_functor.StmtVisitor.visit_allocate_const_`

```python
visit_allocate_const_(op)
```

Visitor implementation for AllocateConst.

<a id="tvm.tirx.stmt_functor.StmtVisitor.visit_decl_buffer_"></a>
#### `tvm.tirx.stmt_functor.StmtVisitor.visit_decl_buffer_`

```python
visit_decl_buffer_(op)
```

Visitor implementation for DeclBuffer.

<a id="tvm.tirx.stmt_functor.StmtVisitor.visit_buffer_store_"></a>
#### `tvm.tirx.stmt_functor.StmtVisitor.visit_buffer_store_`

```python
visit_buffer_store_(op)
```

Visitor implementation for BufferStore.

<a id="tvm.tirx.stmt_functor.StmtVisitor.visit_assert_"></a>
#### `tvm.tirx.stmt_functor.StmtVisitor.visit_assert_`

```python
visit_assert_(op)
```

Visitor implementation for AssertStmt.

<a id="tvm.tirx.stmt_functor.StmtVisitor.visit_seqstmt_"></a>
#### `tvm.tirx.stmt_functor.StmtVisitor.visit_seqstmt_`

```python
visit_seqstmt_(op)
```

Visitor implementation for SeqStmt.

<a id="tvm.tirx.stmt_functor.StmtVisitor.visit_evaluate_"></a>
#### `tvm.tirx.stmt_functor.StmtVisitor.visit_evaluate_`

```python
visit_evaluate_(op)
```

Visitor implementation for Evaluate.

<a id="tvm.tirx.stmt_functor.StmtVisitor.visit_block_"></a>
#### `tvm.tirx.stmt_functor.StmtVisitor.visit_block_`

```python
visit_block_(op)
```

Visitor implementation for Block.

<a id="tvm.tirx.stmt_functor.StmtVisitor.visit_block_realize_"></a>
#### `tvm.tirx.stmt_functor.StmtVisitor.visit_block_realize_`

```python
visit_block_realize_(op)
```

Visitor implementation for BlockRealize.

<a id="tvm.tirx.stmt_functor.StmtVisitor.visit_scope_id_def_stmt_"></a>
#### `tvm.tirx.stmt_functor.StmtVisitor.visit_scope_id_def_stmt_`

```python
visit_scope_id_def_stmt_(op)
```

Visitor implementation for ScopeIdDefStmt.

Mirrors the C++ visitor: walk extents and preferred\_extents via
`visit_expr`; there is no body to recurse into (the def vars
themselves are leaves the visitor doesn’t otherwise inspect).

<a id="tvm.tirx.stmt_functor.StmtVisitor.visit_op_call_"></a>
#### `tvm.tirx.stmt_functor.StmtVisitor.visit_op_call_`

```python
visit_op_call_(op)
```

Visitor implementation for TilePrimitiveCall.

<a id="tvm.tirx.stmt_functor.StmtVisitor.visit_buffer_region_"></a>
#### `tvm.tirx.stmt_functor.StmtVisitor.visit_buffer_region_`

```python
visit_buffer_region_(op)
```

Visitor implementation for BufferRegion.

<a id="tvm.tirx.stmt_functor.StmtVisitor.visit_alloc_buffer_"></a>
#### `tvm.tirx.stmt_functor.StmtVisitor.visit_alloc_buffer_`

```python
visit_alloc_buffer_(op)
```

Visitor implementation for AllocBuffer.

<a id="tvm.tirx.stmt_functor.StmtMutator"></a>
### `tvm.tirx.stmt_functor.StmtMutator`

```python
class tvm.tirx.stmt_functor.StmtMutator
```

A mutator over Stmt.

This is a mutator that recursively transforms a statement. Subclasses can
override the visit methods to customize the behavior.

<a id="tvm.tirx.stmt_functor.StmtMutator.visit_expr"></a>
#### `tvm.tirx.stmt_functor.StmtMutator.visit_expr`

```python
visit_expr(expr)
```

Visit and mutate expressions that occur in a statement.

This method can be overridden to implement expression
mutation in a statement mutator.

**Parameters:**

**expr** (tvm.relax.Expr) – The expression to be visited.

**Returns:**

**result** – The mutated expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.stmt_functor.StmtMutator.visit_bind_"></a>
#### `tvm.tirx.stmt_functor.StmtMutator.visit_bind_`

```python
visit_bind_(op)
```

Mutator implementation for Bind.

<a id="tvm.tirx.stmt_functor.StmtMutator.visit_attr_"></a>
#### `tvm.tirx.stmt_functor.StmtMutator.visit_attr_`

```python
visit_attr_(op)
```

Mutator implementation for AttrStmt.

<a id="tvm.tirx.stmt_functor.StmtMutator.visit_if_then_else_"></a>
#### `tvm.tirx.stmt_functor.StmtMutator.visit_if_then_else_`

```python
visit_if_then_else_(op)
```

Mutator implementation for IfThenElse.

<a id="tvm.tirx.stmt_functor.StmtMutator.visit_for_"></a>
#### `tvm.tirx.stmt_functor.StmtMutator.visit_for_`

```python
visit_for_(op)
```

Mutator implementation for For.

<a id="tvm.tirx.stmt_functor.StmtMutator.visit_while_"></a>
#### `tvm.tirx.stmt_functor.StmtMutator.visit_while_`

```python
visit_while_(op)
```

Mutator implementation for While.

<a id="tvm.tirx.stmt_functor.StmtMutator.visit_return_"></a>
#### `tvm.tirx.stmt_functor.StmtMutator.visit_return_`

```python
visit_return_(op)
```

Mutator implementation for Return.

<a id="tvm.tirx.stmt_functor.StmtMutator.visit_break_"></a>
#### `tvm.tirx.stmt_functor.StmtMutator.visit_break_`

```python
visit_break_(op)
```

Mutator implementation for Break.

<a id="tvm.tirx.stmt_functor.StmtMutator.visit_continue_"></a>
#### `tvm.tirx.stmt_functor.StmtMutator.visit_continue_`

```python
visit_continue_(op)
```

Mutator implementation for Continue.

<a id="tvm.tirx.stmt_functor.StmtMutator.visit_allocate_"></a>
#### `tvm.tirx.stmt_functor.StmtMutator.visit_allocate_`

```python
visit_allocate_(op)
```

Mutator implementation for Allocate.

<a id="tvm.tirx.stmt_functor.StmtMutator.visit_allocate_const_"></a>
#### `tvm.tirx.stmt_functor.StmtMutator.visit_allocate_const_`

```python
visit_allocate_const_(op)
```

Mutator implementation for AllocateConst.

<a id="tvm.tirx.stmt_functor.StmtMutator.visit_decl_buffer_"></a>
#### `tvm.tirx.stmt_functor.StmtMutator.visit_decl_buffer_`

```python
visit_decl_buffer_(op)
```

Mutator implementation for DeclBuffer.

<a id="tvm.tirx.stmt_functor.StmtMutator.visit_buffer_store_"></a>
#### `tvm.tirx.stmt_functor.StmtMutator.visit_buffer_store_`

```python
visit_buffer_store_(op)
```

Mutator implementation for BufferStore.

<a id="tvm.tirx.stmt_functor.StmtMutator.visit_buffer_realize_"></a>
#### `tvm.tirx.stmt_functor.StmtMutator.visit_buffer_realize_`

```python
visit_buffer_realize_(op)
```

Mutator implementation for BufferRealize.

<a id="tvm.tirx.stmt_functor.StmtMutator.visit_assert_"></a>
#### `tvm.tirx.stmt_functor.StmtMutator.visit_assert_`

```python
visit_assert_(op)
```

Mutator implementation for AssertStmt.

<a id="tvm.tirx.stmt_functor.StmtMutator.visit_producer_store_"></a>
#### `tvm.tirx.stmt_functor.StmtMutator.visit_producer_store_`

```python
visit_producer_store_(op)
```

Mutator implementation for ProducerStore.

<a id="tvm.tirx.stmt_functor.StmtMutator.visit_producer_realize_"></a>
#### `tvm.tirx.stmt_functor.StmtMutator.visit_producer_realize_`

```python
visit_producer_realize_(op)
```

Mutator implementation for ProducerRealize.

<a id="tvm.tirx.stmt_functor.StmtMutator.visit_prefetch_"></a>
#### `tvm.tirx.stmt_functor.StmtMutator.visit_prefetch_`

```python
visit_prefetch_(op)
```

Mutator implementation for Prefetch.

<a id="tvm.tirx.stmt_functor.StmtMutator.visit_seqstmt_"></a>
#### `tvm.tirx.stmt_functor.StmtMutator.visit_seqstmt_`

```python
visit_seqstmt_(op)
```

Mutator implementation for SeqStmt.

<a id="tvm.tirx.stmt_functor.StmtMutator.visit_evaluate_"></a>
#### `tvm.tirx.stmt_functor.StmtMutator.visit_evaluate_`

```python
visit_evaluate_(op)
```

Mutator implementation for Evaluate.

<a id="tvm.tirx.stmt_functor.StmtMutator.visit_block_"></a>
#### `tvm.tirx.stmt_functor.StmtMutator.visit_block_`

```python
visit_block_(op)
```

Mutator implementation for Block.

<a id="tvm.tirx.stmt_functor.StmtMutator.visit_block_realize_"></a>
#### `tvm.tirx.stmt_functor.StmtMutator.visit_block_realize_`

```python
visit_block_realize_(op)
```

Mutator implementation for BlockRealize.

<a id="tvm.tirx.stmt_functor.StmtMutator.visit_scope_id_def_stmt_"></a>
#### `tvm.tirx.stmt_functor.StmtMutator.visit_scope_id_def_stmt_`

```python
visit_scope_id_def_stmt_(op)
```

Mutator implementation for ScopeIdDefStmt.

Mirrors the C++ mutator: rewrite `extents` and
`preferred_extents` via `visit_expr`. Deferred-extent defs
(extents is None) and unchanged extents pass through.

<a id="tvm.tirx.stmt_functor.StmtMutator.visit_op_call_"></a>
#### `tvm.tirx.stmt_functor.StmtMutator.visit_op_call_`

```python
visit_op_call_(op)
```

Mutator implementation for TilePrimitiveCall.

<a id="tvm.tirx.stmt_functor.StmtMutator.visit_buffer_region_"></a>
#### `tvm.tirx.stmt_functor.StmtMutator.visit_buffer_region_`

```python
visit_buffer_region_(op)
```

Mutator implementation for BufferRegion.

<a id="tvm.tirx.stmt_functor.StmtMutator.visit_alloc_buffer_"></a>
#### `tvm.tirx.stmt_functor.StmtMutator.visit_alloc_buffer_`

```python
visit_alloc_buffer_(op)
```

Mutator implementation for AllocBuffer.

<a id="tvm.tirx.stmt_functor.StmtExprVisitor"></a>
### `tvm.tirx.stmt_functor.StmtExprVisitor`

```python
class tvm.tirx.stmt_functor.StmtExprVisitor
```

A visitor over both statements and expressions.

This class inherits from both StmtVisitor and ExprVisitor to recursively visit
both statements and expressions.

<a id="tvm.tirx.stmt_functor.StmtExprVisitor.visit_expr"></a>
#### `tvm.tirx.stmt_functor.StmtExprVisitor.visit_expr`

```python
visit_expr(expr)
```

Visit an expression used in a statement.

**Parameters:**

**expr** (tvm.relax.Expr) – The expression to be visited.

<a id="tvm.tirx.stmt_functor.StmtExprMutator"></a>
### `tvm.tirx.stmt_functor.StmtExprMutator`

```python
class tvm.tirx.stmt_functor.StmtExprMutator
```

A mutator over both statements and expressions.

This class inherits from both StmtMutator and ExprMutator to recursively transform
both statements and expressions.

<a id="tvm.tirx.stmt_functor.StmtExprMutator.visit_expr"></a>
#### `tvm.tirx.stmt_functor.StmtExprMutator.visit_expr`

```python
visit_expr(expr)
```

Mutate an expression used in a statement.

**Parameters:**

**expr** (tvm.relax.Expr) – The expression to be mutated.

**Returns:**

**result** – The mutated expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.tirx.stmt_functor.ir_transform"></a>
### `tvm.tirx.stmt_functor.ir_transform`

```python
tvm.tirx.stmt_functor.ir_transform(stmt, preorder, postorder, only_enable=None)
```

Recursively visit and transform ir nodes in post DFS order.

**Parameters:**

- **stmt** ([*tvm.tirx.Stmt*](tirx_api_tirx.md#tvm.tirx.Stmt)) – The input to be transformed.
- **preorder** (*function*) – The function called in before recursive mutation
  If preorder returns None, then the transform will proceed to recursive call.
  If preorder returns a not None tvm.tirx.Stmt/Expr, the transformer will simply return it and
  won’t do further recursion.
- **postorder** (*function*) – The function called after recursive mutation.
- **only\_enable** (*Optional*[*List*[str]]) – List of types that we only enable.

**Returns:**

**result** – The result.

**Return type:**

[tvm.tirx.Stmt](tirx_api_tirx.md#tvm.tirx.Stmt)

<a id="tvm.tirx.stmt_functor.post_order_visit"></a>
### `tvm.tirx.stmt_functor.post_order_visit`

```python
tvm.tirx.stmt_functor.post_order_visit(node, fvisit)
```

**Recursively visit a statement or expression in post DFS order, applying fvisit.**

Each node is guaranteed to be visited only once.

**Parameters:**

- **node** ([*tvm.tirx.Stmt*](tirx_api_tirx.md#tvm.tirx.Stmt) *or* *tvm.ir.Expr*) – The statement or expression to visit.
- **fvisit** (*function*) – The visitor function.

<a id="tvm.tirx.stmt_functor.pre_order_visit"></a>
### `tvm.tirx.stmt_functor.pre_order_visit`

```python
tvm.tirx.stmt_functor.pre_order_visit(node, fvisit)
```

**Recursively visit a statement or expression in pre-order, applying fvisit.**

If fvisit returns False, it won’t visit the children of the node.

**Parameters:**

- **node** ([*tvm.tirx.Stmt*](tirx_api_tirx.md#tvm.tirx.Stmt) *or* *tvm.ir.Expr*) – The statement or expression to visit.
- **fvisit** (*function* *of* *the signature Object -> bool*) – The visitor function.

<a id="tvm.tirx.stmt_functor.substitute"></a>
### `tvm.tirx.stmt_functor.substitute`

```python
tvm.tirx.stmt_functor.substitute(node, vmap)
```

Substitute the var specified by vmap.

**Parameters:**

- **node** (*ObjectRef*) – The input.
- **vmap** (*Dict*[*tirx.Var*, tvm.relax.Expr]) – The variable mapping.

**Returns:**

**result** – The result.

**Return type:**

[tvm.tirx.Stmt](tirx_api_tirx.md#tvm.tirx.Stmt)

<a id="tvm.tirx.stmt_functor.renew_defs"></a>
### `tvm.tirx.stmt_functor.renew_defs`

```python
tvm.tirx.stmt_functor.renew_defs(func: PrimFunc)
```

Re-generate the definition nodes for a TIR, including VarDef, BufferDef.
This pass works as a simple DeepCopy to duplicate a function with different Vars and
Buffers but the same behavior

**Parameters:**

**func** ([*PrimFunc*](tirx_api_tirx.md#tvm.tirx.PrimFunc)) – The input function

**Returns:**

**result** – The new generated func.

**Return type:**

[PrimFunc](tirx_api_tirx.md#tvm.tirx.PrimFunc)
