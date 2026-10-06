<!-- source: https://tvm.apache.org/docs/tirx/tile_primitives/reduction/shared.html -->

<a id="reduction-shared"></a>
# reduction → shared

The `shared` variant lowers a reduction (`sum` / `max` / `min`) when
**both source and destination are shared** memory. At CTA / warpgroup / warp scope
it partitions the threads into groups — one group per output position — has each
thread gather a chunk of the reduction axis, then folds the group with an adaptive
`__shfl_xor` tree. Source:
`python/tvm/backend/cuda/operator/tile_primitive/reduction/shared.py`.

<a id="what-it-accepts"></a>
## What it accepts
```python
@register_dispatch(op_name, "cuda", variant="shared", priority=10, when=[
    predicate("storage_scope", _match_reduction_storage_scope, expected_scope=["shared*"]),
    predicate("shared_valid", validate_reduction_shared),
])
```

| Property | Requirement |
| --- | --- |
| target / priority | `cuda`; priority `10` |
| operand scope | src **and** dst in `shared*`, equal dtype |
| exec scope | `cta` / `warpgroup` / `warp` (shuffle tree) or `thread` (sequential) |
| thread binding | `threadIdx.x` present and **1-D** (no `threadIdx.y` / `z`) |
| shape | `dst` size equals the source’s spatial extent (product of the non-reduced dims) |

<a id="demonstration-program"></a>
## Demonstration program

A 32-thread CTA reduces each row of a `4×8` `float32` shared tile (reduce
axis `-1`) to a `4`-vector (from `test_reduction.py`):

```python
@T.prim_func
def test_reduction(A_ptr: T.handle, B_ptr: T.handle):
    A = T.match_buffer(A_ptr, (4, 8), "float32", layout=TileLayout(S[(4, 8)]))
    B = T.match_buffer(B_ptr, (4,),  "float32", layout=TileLayout(S[(4,)]))
    T.device_entry(); T.cta_id([1]); T.thread_id([32])
    A_smem = T.alloc_buffer((4, 8), "float32", scope="shared", layout=TileLayout(S[(4, 8)]))
    B_smem = T.alloc_buffer((4,),  "float32", scope="shared", layout=TileLayout(S[(4,)]))
    Tx.cta.copy(A_smem, A); T.cuda.cta_sync()
    Tx.cta.sum(B_smem, A_smem, axes=(-1,), accum=False)   # reduction shared dispatch
    T.cuda.cta_sync()
    Tx.cta.copy(B, B_smem)
```

<a id="algorithm"></a>
## Algorithm

**1. Choose the group size.** `group_size = min(next_power_of_2(reduction_len),
32, thread_cnt)` — here `reduction_len = 8` ⇒ `group_size = 8`. Each group of 8
lanes cooperatively reduces one row; the CTA processes the 4 rows in parallel.

**2. Gather + shuffle tree.** Each lane loads its slice of the reduction axis into a
register, then `log2(group_size)` `shfl_xor` steps (masks `1, 2, 4`) fold the
group; lane 0 of each group writes the result, followed by a barrier:

```python
mask = T.tvm_warp_activemask()
for i in range(n_shuffles):                       # n_shuffles = log2(group_size)
    thread_data[0] = op(thread_data[0],
                        T.tvm_warp_shuffle_xor(mask, thread_data[0], 1 << i, group_size, 32))
```

(`warp` uses `warp_sync`; `warpgroup` `warpgroup_sync(8)`; `cta`
`cta_sync`. Thread scope is instead the sequential loop of [reduction → local](tirx_tile_primitives_reduction_local.md).)

<a id="generated-tirx-ir"></a>
## Generated TIRx IR
```python
thread_data[0] = thread_data[0] + T.tvm_warp_shuffle_xor(T.tvm_warp_activemask(), thread_data[0], 1, 8, 32)
thread_data[0] = thread_data[0] + T.tvm_warp_shuffle_xor(T.tvm_warp_activemask(), thread_data[0], 2, 8, 32)
thread_data[0] = thread_data[0] + T.tvm_warp_shuffle_xor(T.tvm_warp_activemask(), thread_data[0], 4, 8, 32)
```

<a id="generated-cuda"></a>
## Generated CUDA
```c
thread_data_ptr[0] = thread_data_ptr[0] + __shfl_xor_sync(__activemask(), thread_data_ptr[0], 1, 8);
thread_data_ptr[0] = thread_data_ptr[0] + __shfl_xor_sync(__activemask(), thread_data_ptr[0], 2, 8);
thread_data_ptr[0] = thread_data_ptr[0] + __shfl_xor_sync(__activemask(), thread_data_ptr[0], 4, 8);
```

(Verified on `sm_100a` — each `B[r] == sum(A[r, :])`. The shuffle width `8`
is the group size, not the full warp.)

<a id="how-inputs-change-the-algorithm"></a>
## How inputs change the algorithm

| input | effect |
| --- | --- |
| op | `sum` → `+` shuffle tree; `max` / `min` → the corresponding combine |
| reduction length / thread count | set `group_size = min(next_pow2(reduction_len), 32, thread_cnt)` and hence the number of shuffle steps |
| exec scope | `cta` / `warpgroup` / `warp` → shuffle tree (different sync); `thread` → sequential loop |
| accum | `True` combines the reduced value with the old dst |
