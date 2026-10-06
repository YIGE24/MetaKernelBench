<!-- source: https://tvm.apache.org/docs/tirx/native_basics/cuda/threads_sync.html -->

<a id="cuda-c-ptx-intrinsics"></a>
# CUDA C++/PTX intrinsics

When no tile primitive covers what you need, two escape hatches reach the hardware
directly: **call a backend intrinsic** (the `T.cuda.*` / `T.ptx.*` namespaces
from `tvm.backend.cuda`), or **inline raw CUDA** source.

<a id="calling-backend-intrinsics"></a>
## Calling backend intrinsics

`T.cuda.*` and `T.ptx.*` expose the CUDA backend’s device intrinsics directly —
synchronization, mbarriers, reductions, and the PTX data-movement / MMA families:

```python
T.cuda.cta_sync()                    # block barrier (__syncthreads)
T.cuda.warp_sync()                   # __syncwarp
T.cuda.warpgroup_sync(8)             # warpgroup barrier
T.cuda.cta_sum(val, num_warps, scratch.ptr_to([0]))   # block-level reduction

bar = T.alloc_shared((1,), "uint64")
T.ptx.mbarrier.init(bar.data, 1)     # mbarrier for async completion
T.ptx.mbarrier.try_wait(bar.data, phase)
```

A complete, runnable example — a warp all-reduce via `T.tvm_warp_shuffle_xor`:

```python
@T.prim_func
def warp_reduce(A_ptr: T.handle):
    A = T.match_buffer(A_ptr, (32,), "float32", align=16)
    T.device_entry()
    cta_id = T.cta_id([1]); warp_id = T.warp_id([1]); lane_id = T.lane_id([32])
    v = T.alloc_local((1,), "float32"); i = T.alloc_local((1,), "int32")
    v[0] = T.float32(31 - lane_id)
    i[0] = 16
    while i[0] >= 1:
        v[0] += T.tvm_warp_shuffle_xor(0xFFFFFFFF, v[0], i[0], 32, 32)
        i[0] = i[0] // 2
    A[lane_id] = v[0]
```

The shuffle lowers straight to `__shfl_xor_sync`:

```c
v_ptr[0] = v_ptr[0] + __shfl_xor_sync(0xFFFFFFFF, v_ptr[0], i_ptr[0], 32);
```

Other families under `T.ptx.*` / `T.cuda.*`: `cp_async` (LDGSTS),
`cp_async.bulk.tensor` (TMA), `ldmatrix` / `stmatrix`, `tcgen05.*`
(Blackwell MMA), `atomic_add`, `fence` … See [tvm.backend.cuda](tirx_api_backend.md) for the
full `tvm.backend.cuda` reference.

<a id="inlining-raw-cuda"></a>
## Inlining raw CUDA

For something with no intrinsic at all, inject a `__device__` function from a
source string with `T.cuda.func_call(name, *args, source_code=..., return_type=...)`:

```python
SRC = r"""
__device__ __forceinline__ float my_relu(float x) { return x > 0.f ? x : 0.f; }
"""

@T.prim_func
def k(A_ptr: T.handle, B_ptr: T.handle):
    A = T.match_buffer(A_ptr, (256,), "float32")
    B = T.match_buffer(B_ptr, (256,), "float32")
    T.device_entry(); bx = T.cta_id([1]); tx = T.thread_id([256])
    B[tx] = T.cuda.func_call("my_relu", A[tx], source_code=SRC, return_type="float32")
```

The source is emitted verbatim and the call is wired in:

```c
__device__ __forceinline__ float my_relu(float x) { return x > 0.f ? x : 0.f; }
// ...
B_ptr[tx] = my_relu(A_ptr[tx]);
```
