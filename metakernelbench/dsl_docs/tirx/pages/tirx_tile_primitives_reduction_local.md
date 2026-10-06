<!-- source: https://tvm.apache.org/docs/tirx/tile_primitives/reduction/local.html -->

<a id="reduction-local"></a>
# reduction → local

The `local` variant lowers a reduction (`sum` / `max` / `min`) when **both
source and destination are register** (`local`) buffers. At thread scope it is a
plain sequential reduction over each thread’s own elements; at warp scope, if the
destination layout carries a `laneid` replica, it also folds across lanes with a
`__shfl_xor` tree. Source:
`python/tvm/backend/cuda/operator/tile_primitive/reduction/local.py`.

<a id="what-it-accepts"></a>
## What it accepts
```python
@register_dispatch(op_name, "cuda", variant="local", priority=10, when=[
    predicate("storage_scope", _match_reduction_storage_scope, expected_scope=["local"]),
    predicate("local_valid", validate_reduction_local),
])
```

| Property | Requirement |
| --- | --- |
| target / priority | `cuda`; priority `10` |
| operand scope | src **and** dst in `local` (registers), equal dtype |
| exec scope | `thread` (always valid — pure thread-local); `warp` / `warpgroup` require a valid (non-swizzled) `TileLayout`; `warp` may additionally cross-lane reduce when `thread_reduce` and a `laneid` shard→replica pattern are present |
| shape | dst spatial dims match src; reduced dims have `local_extent == 1` on dst |

<a id="demonstration-program"></a>
## Demonstration program

A single thread reduces a 4-element `float32` register vector to a scalar
(thread-wise path, from `test_reduction.py`):

```python
@T.prim_func
def test_func(A_ptr: T.handle, B_ptr: T.handle):
    A = T.match_buffer(A_ptr, [4], "float32", layout=TileLayout(S[(4,)]))
    B = T.match_buffer(B_ptr, [1], "float32", layout=TileLayout(S[(1,)]))
    T.device_entry(); T.cta_id([1]); T.thread_id([1])
    A_local = T.alloc_buffer([4], "float32", scope="local")
    B_local = T.alloc_buffer([1], "float32", scope="local")
    for i in T.serial(4): A_local[i] = A[i]
    Tx.sum(B_local, A_local, accum=False)     # reduction local dispatch
    B[0] = B_local[0]
```

(4 < 8 elements, so this stays on `local` rather than the
[reduction → sm100\_packed](tirx_tile_primitives_reduction_sm100_packed.md) fast path.)

<a id="algorithm"></a>
## Algorithm

**Thread-wise** (`_emit_reduction_local_thread_wise`): a spatial loop over the
output positions, each initialized to the op’s identity (unless `accum`), then a
reduction loop accumulating the source — no cross-thread communication:

```python
for spa in range(spatial_len):
    if not accum: dst[spa] = identity
    for red in range(reduction_len):
        dst[spa] = op(dst[spa], src[spa, red])
```

**Warp-shuffle** (`_gen_warp_shuffle_reduce`): when the dst layout has a
`laneid` replica, each lane first reduces its own elements, then
`T.cuda.warp_reduce` folds across lanes — a `__shfl_xor` tree over the **full**
`0xFFFFFFFF` mask. (This differs from [reduction → shared](tirx_tile_primitives_reduction_shared.md), which uses explicit
`tvm_warp_shuffle_xor` steps over `__activemask()` at the *group* width.)

<a id="generated-tirx-ir"></a>
## Generated TIRx IR

For the 4-element thread reduction:

```python
for spa in range(1):
    for red in range(4):
        dst[...] = dst[...] + src[...]        # op = sum
```

<a id="generated-cuda"></a>
## Generated CUDA
```c
for (int red = 0; red < 4; ++red)
  B_local_ptr[0] = B_local_ptr[0] + A_local_ptr[red];
```

(Verified on `sm_100a` — `B == sum(A)`.)

<a id="how-inputs-change-the-algorithm"></a>
## How inputs change the algorithm

| input | effect |
| --- | --- |
| op | `sum` → `+`, `max` → `max`, `min` → `min` (and the identity) |
| exec scope | `thread` → sequential; `warp` with a `laneid` replica → adds a `__shfl_xor` cross-lane tree |
| axes / shape | set the spatial vs reduction loop extents |
| accum | `True` reuses the old dst value instead of the identity |
