<!-- source: https://tvm.apache.org/docs/tirx/tile_primitives.html -->

<a id="tile-primitives"></a>
# Tile Primitives
> **Note**
>
> This page documents the tile-primitive surface and dispatch as it exists in
> the source today; signatures and variants may change.

Tile primitives are the dispatchable, hardware-level operations a TIRx kernel
issues — data movement (`copy`, `copy_async`), matrix multiply (`gemm`,
`gemm_async`), reductions, elementwise math, and a few fused/compose forms.
A primitive call is recorded as an **unresolved** `TilePrimitiveCall` IR node;
the compiler later *dispatches* it — selecting a concrete lowering from the
primitive, the execution scope, the operand layouts, the target, and an optional
explicit hint — and replaces it with native IR (loops, address arithmetic,
synchronization, and backend intrinsics).

<a id="calling-convention"></a>
## Calling convention

Tile primitives are called in TVMScript via `from tvm.script import tirx as T`,
on the injected `Tx` namespace. The namespace prefix selects the **cooperation
scope**:

- `Tx.<name>(...)` — unqualified, runs at **thread** scope.
- `Tx.warp.<name>` / `Tx.wg.<name>` (alias `Tx.warpgroup`) / `Tx.cta.<name>`
  / `Tx.cluster.<name>` / `Tx.thread.<name>` — bind a wider scope.

Every primitive also accepts, besides its operands: `scope` (usually set by the
namespace), `workspace: dict[str, Buffer] | None`, `dispatch: str | None`
(force a named lowering variant), and `**kwargs` collected into a `config`
dict that tunes the chosen lowering. Operands are `Buffer` / `BufferRegion`
values, each carrying a [TileLayout](tirx_layout.md) that dispatch reads.

Wiring (three layers): the authoritative op list is the C++ registry
(`src/tirx/op/tirx.cc`, 29 ops named `tirx.tile.<name>`); the IR wrapper
classes are in `python/tvm/tirx/operator/tile_primitive/ops.py`; the
user-facing `Tx.*` builders are in `python/tvm/tirx/script/builder/tirx.py`.

<a id="primitive-catalog"></a>
## Primitive catalog

The 29 primitives, grouped. Signatures show the operands plus the common
`workspace`/`dispatch`/`scope`/`**kwargs` tail (abbreviated `...`).

<a id="data-movement"></a>
### Data movement
```
copy(dst, src, ...)            # synchronous element copy src -> dst
copy_async(dst, src, ...)      # asynchronous copy (caller commits/waits)
permute_layout(dst, src, ...)  # rearrange under a different layout (may alias)
```

<a id="matrix-multiply"></a>
### Matrix multiply
```
gemm(D, A, B, C, transpose_A=False, transpose_B=False,
     alpha=1.0, beta=0.0, ...)          # D = alpha*A*B + beta*C (register mma)
gemm_async(C, A, B, SFA=None, SFB=None,
           transA=False, transB=False, accum=False, ...)  # async / block-scaled
```

<a id="fill-memset-zero"></a>
### Fill / memset / zero
```
fill(dst, value, ...)        # fill region with a scalar
memset(dst, value, ...)      # set all elements to a value
zero(dst, src=None, ...)     # zero out (in place if src omitted)
```

<a id="cast-and-elementwise"></a>
### Cast and elementwise
```
cast(dst, src=None, ...)                         # dtype cast (buffer form)
sqrt / exp / exp2(dst, src=None, bias=None, scale=None, ...)
reciprocal(dst, src=None, ...)                   # dst = 1/src
silu(dst, src, ...)                              # dst = src*sigmoid(src)
add / sub / mul / fdiv(dst, src1, src2, ...)     # element-wise arithmetic
maximum / minimum(dst, src1, src2, ...)          # element-wise max / min
fma(dst, src, scale, bias, ...)                  # dst = src*scale + bias
select(dst, true_value, false_value, pred, scope=None)  # dst = pred ? t : f
```

<a id="reductions"></a>
### Reductions
```
sum / max / min(dst, src, axes=-1, accum=False, ...)   # reduce over axes
```

<a id="fused-compose"></a>
### Fused / compose
```
binary_reduce(...)   # binary op then reduce, fused
unary_reduce(...)    # unary (with bias/scale) then reduce
binary_chain(...)    # chain two binary ops
reduce_negate(...)   # reduce then negate
compose_op(...)      # frame/context manager to group primitives
```

<a id="dispatch-config"></a>
## Dispatch config

A call is materialized as a `TilePrimitiveCall` node whose fields carry
everything dispatch needs (`python/tvm/tirx/stmt.py`):

| Field | Type | Meaning |
| --- | --- | --- |
| `op` | `tvm.Op` | primitive identity, e.g. `tirx.tile.copy_async` |
| `args` | `Array` | operands (regions / scalars), in the order shown above |
| `workspace` | `Map[str, Buffer]` | pre-allocated scratch buffers |
| `config` | `Map[str, Any]` | open-ended tuning bag (table below) |
| `dispatch` | `Optional[str]` | forced variant name; `None` = auto-select |
| `scope` | `ExecScope` | cooperation scope (default `thread`) |

`config` has **no central schema** — each key is read only by the dispatch
variant(s) that need it (via `config.get(...)`); a key meant for another
primitive is simply ignored. Only `dispatch` is generic. The keys observed in
the CUDA backend, by consumer:

| Key | Used by | Type / values | Meaning |
| --- | --- | --- | --- |
| `dispatch` | any primitive | variant name (str) | force a lowering variant (also settable via the `dispatch=` kwarg) |
| `vec_len` | `copy` / `copy_async` (vectorized variants) | int | None | vectorization width for the copy |
| `mbar` | `copy_async`: `tma` (g2s), `dsmem` | mbarrier handle | completion barrier |
| `cta_group` | `copy_async`: `tma`, `smem→tmem`; `gemm_async`: `tcgen05` | `1` | `2` | CTA-group; `2` routes completion to the cluster |
| `cta_mask` | `copy_async`: `tma` (g2s) | int | None | multicast CTA mask |
| `cache_hint` | `copy_async`: `tma` | `"evict_normal"` | `""` | L2 cache eviction hint |
| `oob` | `copy_async`: `tma` | `"zero"` | `"nan"` | None | out-of-bounds fill policy (`nan` is float-only) |
| `use_tma_reduce` | `copy_async`: `tma` (s2g) | str (e.g. `"add"`) | None | TMA store-with-reduction mode |
| `prefetch_tensormap` | `copy_async`: `tma` | bool | prefetch the tensor map at kernel entry |
| `remote_cta_id` | `copy_async`: `dsmem` | int | PrimExpr | target CTA for a cross-CTA shared→shared copy |
| `descI` | `gemm_async`: `tcgen05` | uint32 | None | pre-encoded MMA instruction descriptor |
| `thread_reduce` | `reduction`: `local` (warp scope) | bool | per-thread shuffle reduction |
| `rounding_mode` | `elementwise`: binary ops | `"rn"` | `"rz"` | … | FP rounding mode for the packed form |

Three dispatch inputs are **implicit**, not config keys: the **execution scope**
(set by the namespace, then refined against the active thread set tracked through
control flow into `inter`/`intra` maps and a `scope_kind`), the **operand
layouts** (each `Buffer.layout`), and the **target** (the dispatch table is
keyed by its kind, e.g. `"cuda"`).

<a id="dispatch-mechanism"></a>
## Dispatch mechanism

<a id="pipeline"></a>
### Pipeline

Dispatch runs in the `tirx.TilePrimitiveDispatch` pass — the sole pass inside
`LowerTIRx()`, the first stage of the compile pipeline. The C++ mutator
`TilePrimitiveDispatcher` walks the IR and, per call:

1. resolves the `(inter, intra)` execution split for the call’s scope from the
   active set tracked through control flow (`if wg_id == ...`, `warp_id`,
   `T.ptx.elect_sync()`);
2. builds a `DispatchContext` carrying `target`, scope, launch params, value
   ranges, and the encoded `inter`/`intra` + `scope_kind`;
3. invokes the global FFI hook `tirx.f_op_dispatcher` (Python) with the call
   and context, which returns a `PrimFunc`;
4. splices that `PrimFunc` body in place of the call and drains side-effect
   callbacks (private allocs, device/host init statements).

If any `TilePrimitiveCall` survives lowering, a verifier makes it a fatal error.

<a id="selection-run-dispatch"></a>
### Selection ( run_dispatch )

The Python dispatcher holds a table `_DISPATCH_TABLE` keyed by
`(Op, target_kind)`. Each entry is a list of *cases*, registered by backends
via `@register_dispatch(op_name, target_kind, variant=..., priority=...,
when=[preds])`. `run_dispatch(op_call, sctx)`:

1. `key = (op_call.op, sctx.target.kind.name)`; look up cases. None → error.
2. If `op_call.dispatch` is set, **filter** to that variant (error if unknown).
3. Sort cases by `(-priority, variant)` — highest priority first.
4. For each case, evaluate its predicates; if any fails, record the reason and
   continue. If all pass, run the impl; on success return its `PrimFunc`.
5. An impl may still **decline** by raising `DispatchFail` (e.g. a hardware
   constraint found while emitting) — the search continues.
6. If every variant is rejected, raise a `RuntimeError` listing each variant’s
   rejection reason.

So dispatch is **keyed by (primitive, target)**, then a **priority-ordered,
predicate-guarded** case list, with an optional `dispatch=` override.

Two recurring predicate helpers: `validate_copy_op` (both operands have a
layout, equal dtype, equal non-unit extents) and `_all_threads_active` (the
exec scope is full — `laneid` spans 32, etc., none of it narrowed by an
enclosing `if`), so a partial-warp copy is rejected rather than mis-lowered.

<a id="dispatch-by-primitive"></a>
## Dispatch by primitive

Each page below documents one primitive’s dispatch in detail — the variants, how
each is selected, the algorithm it runs, the IR it emits, and when it declines.

- [copy](tirx_tile_primitives_copy.md)
- [copy\_async](tirx_tile_primitives_copy_async.md)
- [gemm](tirx_tile_primitives_gemm.md)
- [gemm\_async](tirx_tile_primitives_gemm_async.md)
- [elementwise](tirx_tile_primitives_elementwise.md)
- [reduction](tirx_tile_primitives_reduction.md)
- [permute\_layout](tirx_tile_primitives_permute_layout.md)

<a id="see-also"></a>
## See also

- [Tensor Layout](tirx_layout.md) — the `TileLayout` model dispatch reads from operands.
- [Overview](tirx_overview.md) — execution scope, tensor layout, and tile primitive dispatch
  as the three core constructs.
