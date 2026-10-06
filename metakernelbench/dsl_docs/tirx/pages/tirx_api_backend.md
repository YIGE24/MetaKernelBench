<!-- source: https://tvm.apache.org/docs/tirx/api/backend.html -->

<a id="tvm-backend-cuda"></a>
# tvm.backend.cuda

The CUDA backend — the tile-primitive dispatch, intrinsic builders, the `T.cuda`
/ `T.ptx` script namespaces, and the shared/tensor-memory pools — lives under
`tvm.backend.cuda`, separate from the TIRx frontend (`tvm.tirx`). Other
backends sit alongside it (`tvm.backend.rocm` and so on).

<a id="id1"></a>
## tvm.backend.cuda

CUDA-owned TIRx modules.

<a id="tvm.backend.cuda.register_backend"></a>
### `tvm.backend.cuda.register_backend`

```python
tvm.backend.cuda.register_backend()
```

Register CUDA-owned Python semantics.

<a id="tvm.backend.cuda.script_namespace"></a>
### `tvm.backend.cuda.script_namespace`

```python
tvm.backend.cuda.script_namespace(**kwargs)
```

Return the CUDA TVMScript namespace object.

<a id="tvm.backend.cuda.script_namespaces"></a>
### `tvm.backend.cuda.script_namespaces`

```python
tvm.backend.cuda.script_namespaces(**_)
```

Return CUDA-owned TVMScript namespaces.

<a id="module-tvm.backend.cuda.lang"></a>
## tvm.backend.cuda.lang

CUDA-specific TIRx language helpers.

<a id="module-tvm.backend.cuda.op"></a>
## tvm.backend.cuda.op

CUDA, PTX, and NVSHMEM TIR intrinsic builders.

<a id="tvm.backend.cuda.op.is_prim_expr"></a>
### `tvm.backend.cuda.op.is_prim_expr`

```python
tvm.backend.cuda.op.is_prim_expr(value: object) → bool
```

Return whether an expression has a primitive result type.

<a id="tvm.backend.cuda.op.const"></a>
### `tvm.backend.cuda.op.const`

```python
tvm.backend.cuda.op.const(value, dtype=None, span=None)
```

construct a constant

**Parameters:**

- **value** (*number*) – The content of the constant number.
- **dtype** (str *or* *None*, *optional*) – The data type.
- **span** (*Optional*[tvm.ir.Span]) – The location of the constant value in the source.

**Returns:**

**const\_val** – The result expression.

**Return type:**

tvm.Expr

<a id="tvm.backend.cuda.op.bitwise_and"></a>
### `tvm.backend.cuda.op.bitwise_and`

```python
tvm.backend.cuda.op.bitwise_and(x, y, span=None)
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

<a id="tvm.backend.cuda.op.call_intrin"></a>
### `tvm.backend.cuda.op.call_intrin`

```python
tvm.backend.cuda.op.call_intrin(dtype: str | Type, func_name, *args, attrs=None, span=None)
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

<a id="tvm.backend.cuda.op.tvm_access_ptr"></a>
### `tvm.backend.cuda.op.tvm_access_ptr`

```python
tvm.backend.cuda.op.tvm_access_ptr(ptype, data, offset, extent, rw_mask)
```

Get head access address with memory access pattern info

**Parameters:**

- **ptype** (tvm.relax.Expr *or* str) – The data type of pointer. If a `str`, it is wrapped via
  `type_annotation()` so that the lowering rule (which reads
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

<a id="tvm.backend.cuda.op.cuda_func_call"></a>
### `tvm.backend.cuda.op.cuda_func_call`

```python
tvm.backend.cuda.op.cuda_func_call(func_name, *args, source_code, return_type='void')
```

TVM intrinsic to call a CUDA function. Source code is provided as a string.

**Parameters:**

- **func\_name** (str) – The name of the CUDA function.
- **args** (tvm.relax.Expr) – The arguments to the CUDA function.
- **source\_code** (str) – The source code of the CUDA function.
- **return\_type** (str) – The return type of the CUDA function.

<a id="tvm.backend.cuda.op.cuda_warp_reduce"></a>
### `tvm.backend.cuda.op.cuda_warp_reduce`

```python
tvm.backend.cuda.op.cuda_warp_reduce(value, op, width=32)
```

Warp-level butterfly shuffle-XOR reduction.

Reduces `value` across `width` adjacent lanes using the specified
operation. Codegen emits `log2(width)` steps of
`__shfl_xor_sync(0xFFFFFFFF, val, mask)` with descending XOR masks.

**Parameters:**

- **value** (tvm.relax.Expr) – The per-thread scalar value to reduce.
- **op** (str) – Reduction operation: `"sum"`, `"max"`, or `"min"`.
- **width** (int) – Number of lanes participating in each reduction group.
  Must be a power of two in [2, 32]. Defaults to 32 (full warp).

**Returns:**

**call** – The reduced value (same dtype as *value*).

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.cuda_warp_sum"></a>
### `tvm.backend.cuda.op.cuda_warp_sum`

```python
tvm.backend.cuda.op.cuda_warp_sum(value, width=32)
```

Convenience wrapper: `cuda_warp_reduce(value, "sum", width)`.

<a id="tvm.backend.cuda.op.cuda_warp_max"></a>
### `tvm.backend.cuda.op.cuda_warp_max`

```python
tvm.backend.cuda.op.cuda_warp_max(value, width=32)
```

Convenience wrapper: `cuda_warp_reduce(value, "max", width)`.

<a id="tvm.backend.cuda.op.cuda_warp_min"></a>
### `tvm.backend.cuda.op.cuda_warp_min`

```python
tvm.backend.cuda.op.cuda_warp_min(value, width=32)
```

Convenience wrapper: `cuda_warp_reduce(value, "min", width)`.

<a id="tvm.backend.cuda.op.cuda_cta_reduce"></a>
### `tvm.backend.cuda.op.cuda_cta_reduce`

```python
tvm.backend.cuda.op.cuda_cta_reduce(value, op, num_warps, scratch)
```

CTA-wide reduction via warp shuffle + shared memory.

Two-step reduction: (1) intra-warp shuffle reduction, (2) warp-0
collects per-warp partials from `scratch`, reduces, broadcasts via
`__syncthreads()`. All CTA threads must participate.

**Parameters:**

- **value** (tvm.relax.Expr) – Per-thread scalar value to reduce.
- **op** (str) – Reduction operation: `"sum"`, `"max"`, or `"min"`.
- **num\_warps** (int) – Number of warps in the CTA. Must be a power of two in [1, 32].
- **scratch** (tvm.ir.Var) – Data pointer to shared-memory scratch space (>= num\_warps elements).

**Returns:**

**call** – The reduced value broadcast to all threads (same dtype as *value*).

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.cuda_cta_sum"></a>
### `tvm.backend.cuda.op.cuda_cta_sum`

```python
tvm.backend.cuda.op.cuda_cta_sum(value, num_warps, scratch)
```

Convenience wrapper: `cuda_cta_reduce(value, "sum", num_warps, scratch)`.

<a id="tvm.backend.cuda.op.cuda_cta_max"></a>
### `tvm.backend.cuda.op.cuda_cta_max`

```python
tvm.backend.cuda.op.cuda_cta_max(value, num_warps, scratch)
```

Convenience wrapper: `cuda_cta_reduce(value, "max", num_warps, scratch)`.

<a id="tvm.backend.cuda.op.cuda_cta_min"></a>
### `tvm.backend.cuda.op.cuda_cta_min`

```python
tvm.backend.cuda.op.cuda_cta_min(value, num_warps, scratch)
```

Convenience wrapper: `cuda_cta_reduce(value, "min", num_warps, scratch)`.

<a id="tvm.backend.cuda.op.cuda_warp_sync"></a>
### `tvm.backend.cuda.op.cuda_warp_sync`

```python
tvm.backend.cuda.op.cuda_warp_sync()
```

TVM intrinsic to synchronize threads within the current warp.

This lowers to a CUDA \_\_syncwarp() call.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.cuda_cta_sync"></a>
### `tvm.backend.cuda.op.cuda_cta_sync`

```python
tvm.backend.cuda.op.cuda_cta_sync()
```

TVM intrinsic to call CUDA syncthreads (block-wide barrier)

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.cuda_grid_sync"></a>
### `tvm.backend.cuda.op.cuda_grid_sync`

```python
tvm.backend.cuda.op.cuda_grid_sync()
```

TVM intrinsic to call CUDA grid-wide sync (cooperative groups)

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.cuda_cluster_sync"></a>
### `tvm.backend.cuda.op.cuda_cluster_sync`

```python
tvm.backend.cuda.op.cuda_cluster_sync()
```

TVM intrinsic to call CUDA cluster-wide barrier sync

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.cuda_thread_rank"></a>
### `tvm.backend.cuda.op.cuda_thread_rank`

```python
tvm.backend.cuda.op.cuda_thread_rank()
```

TVM intrinsic that returns `cooperative_groups::thread_rank()`
for the enclosing CTA – the linear thread index within the block.

Useful for building “single thread of CTA” predicates without
referencing user-declared scope\_id vars. For example, the idiomatic
mbarrier.init leader predicate is:

```
T.cuda.thread_rank() == 0
```

**Returns:**

**call** – The call expression (`int32`).

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.cuda_half2float"></a>
### `tvm.backend.cuda.op.cuda_half2float`

```python
tvm.backend.cuda.op.cuda_half2float(src)
```

TVM intrinsic to convert half to float

**Parameters:**

**src** (tvm.relax.Expr) – Source pointer.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.cuda_bfloat162float"></a>
### `tvm.backend.cuda.op.cuda_bfloat162float`

```python
tvm.backend.cuda.op.cuda_bfloat162float(src)
```

TVM intrinsic to convert bfloat16 to float

**Parameters:**

**src** (tvm.relax.Expr) – Source pointer.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.cuda_float22half2"></a>
### `tvm.backend.cuda.op.cuda_float22half2`

```python
tvm.backend.cuda.op.cuda_float22half2(dst, src)
```

TVM intrinsic to convert float2 to half2 with rounding

**Parameters:**

- **dst** (tvm.relax.Expr) – Destination pointer.
- **src** (tvm.relax.Expr) – Source pointer.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.cuda_trap_when_assert_failed"></a>
### `tvm.backend.cuda.op.cuda_trap_when_assert_failed`

```python
tvm.backend.cuda.op.cuda_trap_when_assert_failed(cond)
```

TVM intrinsic to trap when assertion failed (cond == false)

**Parameters:**

**cond** (tvm.relax.Expr) – Condition to check.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.cuda_runtime_instr_desc"></a>
### `tvm.backend.cuda.op.cuda_runtime_instr_desc`

```python
tvm.backend.cuda.op.cuda_runtime_instr_desc(desc, sf_id)
```

TVM intrinsic to update runtime instruction descriptor

**Parameters:**

- **desc** (tvm.relax.Expr) – Pointer to the descriptor (uint32\*).
- **sf\_id** (tvm.relax.Expr) – The subfragment id.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.cuda_half8tofloat8"></a>
### `tvm.backend.cuda.op.cuda_half8tofloat8`

```python
tvm.backend.cuda.op.cuda_half8tofloat8(src_addr, dst_addr)
```

TVM intrinsic to convert 8 half2s to 8 float2s

**Parameters:**

- **src\_addr** (tvm.relax.Expr) – Source pointer.
- **dst\_addr** (tvm.relax.Expr) – Destination pointer.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.cuda_float8tohalf8"></a>
### `tvm.backend.cuda.op.cuda_float8tohalf8`

```python
tvm.backend.cuda.op.cuda_float8tohalf8(src_addr, dst_addr)
```

TVM intrinsic to convert 8 float2s to 8 half2s

**Parameters:**

- **src\_addr** (tvm.relax.Expr) – Source pointer.
- **dst\_addr** (tvm.relax.Expr) – Destination pointer.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.ptx_mma_sp"></a>
### `tvm.backend.cuda.op.ptx_mma_sp`

```python
tvm.backend.cuda.op.ptx_mma_sp(dtype, shape, A_layout, B_layout, A_dtype, B_dtype, C_dtype, multiplicand_a, a_index, multiplicand_b, b_index, accumulator, c_index, metadata, meta_index, sparse_selector, saturate)
```

TVM intrinsic for sparse tensor core ptx instructions
<https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-for-sparse-mma>

**Parameters:**

- **dtype** (str) – The data type of the result.
- **shape** (str) – The shape of mma fragment.
- **A\_layout** (*Literal*[*"row"*, *"col"*]) – The layout of multiplicand fragment A.
- **B\_layout** (*Literal*[*"row"*, *"col"*]) – The layout of multiplicand fragment B.
- **A\_dtype** (str) – The data type of multiplicand fragment A.
- **B\_dtype** (str) – The data type of multiplicand fragment B.
- **C\_dtype** (str) – The data type of multiplicand fragment C.
- **multiplicand\_a** (tvm.ir.Var) – The multiplicand fragment A variable.
- **a\_index** (tvm.relax.Expr) – The index of multiplicand fragment A.
- **multiplicand\_b** (tvm.ir.Var) – The multiplicand fragment B variable.
- **b\_index** (tvm.relax.Expr) – The index of multiplicand fragment B.
- **accumulator** (tvm.ir.Var) – The accumulator fragment C variable.
- **c\_index** (tvm.relax.Expr) – The index of accumulator fragment C.
- **metadata** (tvm.relax.Expr) – The metadata of operand.
- **meta\_index** (tvm.relax.Expr) – The metadata index of operand.
- **sparse\_selector** (tvm.relax.Expr) – The sparse selector indicating the thread that stores the metadata.
- **saturate** (bool) – The optional saturation at the output.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.ptx_cp_async_bulk"></a>
### `tvm.backend.cuda.op.ptx_cp_async_bulk`

```python
tvm.backend.cuda.op.ptx_cp_async_bulk(dtype, shared_ptr, shared_offset, global_ptr, global_offset, bytes, barrier_id)
```

TVM intrinsic for ptx async copy from global to shared memory using cp.async.bulk
<https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk>

**Parameters:**

- **dtype** (str) – The data type of the result.
- **shared\_ptr** (tvm.ir.Var) – The shared memory pointer variable.
- **shared\_offset** (tvm.relax.Expr) – The offset of shared memory pointer.
- **global\_ptr** (tvm.ir.Var) – The global memory pointer variable.
- **global\_offset** (tvm.relax.Expr) – The offset of global memory pointer.
- **bytes** (int) – The data size to copy.
- **barrier\_id** (int) – The ID of the barrier shared memory pointer.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.ptx_cp_async_bulk_shared_to_cluster"></a>
### `tvm.backend.cuda.op.ptx_cp_async_bulk_shared_to_cluster`

```python
tvm.backend.cuda.op.ptx_cp_async_bulk_shared_to_cluster(dst_ptr, src_ptr, size, mbar)
```

PTX cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete\_tx::bytes

Asynchronous bulk copy from executing CTA’s shared memory to a remote
CTA’s shared memory within the same cluster.

**Parameters:**

- **dst\_ptr** (tvm.relax.Expr) – Destination pointer in shared::cluster address space (remote CTA).
- **src\_ptr** (tvm.relax.Expr) – Source pointer in shared::cta address space (local CTA).
- **size** (tvm.relax.Expr) – Number of bytes to copy (must be multiple of 16).
- **mbar** (tvm.relax.Expr) – Mbarrier address in shared::cluster space for completion signaling,
  usually produced by `T.ptx.map_shared_rank`.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.ptx_cp_async_mbarrier_arrive"></a>
### `tvm.backend.cuda.op.ptx_cp_async_mbarrier_arrive`

```python
tvm.backend.cuda.op.ptx_cp_async_mbarrier_arrive(barrier_id)
```

TVM intrinsic for ptx async copy barrier using cp.async.mbarrier.arrive
<https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-cp-async-mbarrier-arrive>

**Parameters:**

**barrier\_id** (int) – The ID of the barrier shared memory pointer.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.ptx_fence"></a>
### `tvm.backend.cuda.op.ptx_fence`

```python
tvm.backend.cuda.op.ptx_fence(sem: str, scope: str)
```

TVM intrinsic for PTX fence instruction.

Generates: fence.{sem}.{scope};

**Parameters:**

- **sem** (str) – The semantics of the fence. One of “sc”, “acq\_rel”.
- **scope** (str) – The scope of the fence. One of “cta”, “cluster”, “gpu”, “sys”.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.ptx_fence_proxy_async"></a>
### `tvm.backend.cuda.op.ptx_fence_proxy_async`

```python
tvm.backend.cuda.op.ptx_fence_proxy_async(space: str = '')
```

TVM intrinsic for PTX fence.proxy.async instruction.

Generates: fence.proxy.async[.{space}];

**Parameters:**

**space** (str) – The address space qualifier. One of “”, “global”, “shared::cta”, “shared::cluster”.
Empty string means no qualifier.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.ptx_mbarrier_init"></a>
### `tvm.backend.cuda.op.ptx_mbarrier_init`

```python
tvm.backend.cuda.op.ptx_mbarrier_init(bar, thread_count)
```

TVM intrinsic to call mbarrier.init.shared::cta.b64

**Parameters:**

- **bar** (tvm.ir.Var) – The pointer to barrier variable.
- **thread\_count** (int) – The number of threads expected to arrive at the barrier.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.ptx_mbarrier_arrive"></a>
### `tvm.backend.cuda.op.ptx_mbarrier_arrive`

```python
tvm.backend.cuda.op.ptx_mbarrier_arrive(bar, cta_id=None, pred=None, count=None)
```

**TVM intrinsic to call**

mbarrier.arrive.shared::cta.b64

**or**

@p mapa.shared::cluster.u32
@p mbarrier.arrive.shared::cluster.b64 [, count]

**Parameters:**

- **bar** (tvm.ir.Var) – The pointer to barrier variable.
- **cta\_id** (*Optional*[tvm.relax.Expr]) – The cta id.
- **pred** (*Optional*[tvm.relax.Expr]) – The predicate to guard the operation.
- **count** (*Optional*[tvm.relax.Expr]) – Explicit arrival count operand for the cross-CTA (cluster) form. When
  `None` the implicit count-of-1 form is emitted; when given, emits
  `mbarrier.arrive.shared::cluster.b64 _, [addr], count`.

<a id="tvm.backend.cuda.op.ptx_mbarrier_arrive_cluster_count"></a>
### `tvm.backend.cuda.op.ptx_mbarrier_arrive_cluster_count`

```python
tvm.backend.cuda.op.ptx_mbarrier_arrive_cluster_count(bar, cta_id, count)
```

Cross-CTA `mbarrier.arrive` on CTA `cta_id` with an explicit count.

Convenience for an already-elected thread: emits
`@p mapa.shared::cluster.u32` + `@p mbarrier.arrive.shared::cluster.b64 _,
[addr], count` with the guard defaulted to 1.

<a id="tvm.backend.cuda.op.ptx_mbarrier_arrive_expect_tx"></a>
### `tvm.backend.cuda.op.ptx_mbarrier_arrive_expect_tx`

```python
tvm.backend.cuda.op.ptx_mbarrier_arrive_expect_tx(bar, byte_count, cta_id=None, pred=None)
```

**TVM intrinsic to call**

mbarrier.arrive\_expect\_tx.shared::cta.b64

**or**

@p mapa.shared::cluster.u32
@p mbarrier.arrive\_expect\_tx.shared::cluster.b64

**Parameters:**

- **bar** (tvm.ir.Var) – The pointer to barrier variable.
- **byte\_count** (int) – Increases the tx count of the mbarrier object to track completion of
  addtional async transactions.
- **cta\_id** (*Optional*[tvm.relax.Expr]) – The cta id.
- **pred** (*Optional*[tvm.relax.Expr]) – The predicate to guard the operation.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.ptx_mbarrier_try_wait"></a>
### `tvm.backend.cuda.op.ptx_mbarrier_try_wait`

```python
tvm.backend.cuda.op.ptx_mbarrier_try_wait(bar, phase)
```

TVM intrinsic to call mbarrier.try\_wait.parity repeatedly until it returns true

**Parameters:**

- **bar** (tvm.ir.Var) – The pointer to barrier variable.
- **phase** (int) – The phase of the barrier.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.ptx_mbarrier_try_wait_acquire_cluster"></a>
### `tvm.backend.cuda.op.ptx_mbarrier_try_wait_acquire_cluster`

```python
tvm.backend.cuda.op.ptx_mbarrier_try_wait_acquire_cluster(bar, phase)
```

`mbarrier.try_wait.parity.acquire.cluster` retry loop.

Cluster-scope acquire wait — used to wait on a barrier that a remote CTA in
the cluster arrives on (a group cluster wait).

**Parameters:**

- **bar** (tvm.ir.Var) – The pointer to barrier variable.
- **phase** (int) – The phase of the barrier.

<a id="tvm.backend.cuda.op.ptx_mbarrier_try_wait_once"></a>
### `tvm.backend.cuda.op.ptx_mbarrier_try_wait_once`

```python
tvm.backend.cuda.op.ptx_mbarrier_try_wait_once(bar, phase, ticks)
```

TVM intrinsic for one-shot non-blocking `mbarrier.try_wait.parity`.

Returns `1` if the requested parity has been reached and `0` otherwise.
This is intended for bounded debug waits; production waits should use
[`ptx_mbarrier_try_wait()`](#tvm.backend.cuda.op.ptx_mbarrier_try_wait).

<a id="tvm.backend.cuda.op.ptx_bar_arrive"></a>
### `tvm.backend.cuda.op.ptx_bar_arrive`

```python
tvm.backend.cuda.op.ptx_bar_arrive(name_bar_id, thread_count)
```

TVM intrinsic to call bar.arrive a, b

**Parameters:**

- **name\_bar\_id** (int) – The ID of the named barrier.
- **thread\_count** (int) – The number of threads expected to arrive at the barrier.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.ptx_bar_sync"></a>
### `tvm.backend.cuda.op.ptx_bar_sync`

```python
tvm.backend.cuda.op.ptx_bar_sync(name_bar_id, thread_count)
```

TVM intrinsic to call bar.sync a, {b}

**Parameters:**

- **name\_bar\_id** (int) – The ID of the named barrier.
- **thread\_count** (int) – The number of threads expected to arrive at the barrier.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.ptx_cp_async"></a>
### `tvm.backend.cuda.op.ptx_cp_async`

```python
tvm.backend.cuda.op.ptx_cp_async(dst_ptr, src_ptr, cp_size, *, cache_hint='', cache_policy=None, prefetch_size=-1, predicate=-1, fill_mode='')
```

TVM intrinsic for ptx async copy from global to shared memory using cp.async
<https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async>

Dispatches to one of three PTX-form-aligned ops:

- `ptx_cp_async_src_size` for `fill_mode == "zero"` (zero-fill via
  `src_size = pred ? cp_size : 0`).
- `ptx_cp_async_ignore_src` for a non-empty `predicate` with no
  fill\_mode (`setp+@p` guards the asm).
- `ptx_cp_async_plain` for the no-predicate / no-fill\_mode case.

**Parameters:**

- **shared\_ptr** (tvm.relax.Expr) – The pointer to the shared memory.
- **global\_ptr** (tvm.relax.Expr) – The pointer to the global memory.
- **cp\_size** (int) – The data size to copy.
- **cache\_hint** (str[*"evict\_last"*, *"evict\_first"*, *"evict\_normal"*, *""*]) – The cache hint.
- **prefetch\_size** (int[*-1*, *64*, *128*, *256*]) – The prefetch size.
- **predicate** (tvm.relax.Expr) – The predicate to guard the operation.
- **fill\_mode** (str[*"zero"*, *""*]) – The fill mode.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.ptx_cp_async_legacy"></a>
### `tvm.backend.cuda.op.ptx_cp_async_legacy`

```python
tvm.backend.cuda.op.ptx_cp_async_legacy(*all_args)
```

Legacy `ptx_cp_async` API taking explicit src/dst offsets.

Signature: `(dst_ptr, dst_offset, src_ptr, src_offset, cp_size)`.
Offsets are folded into the pointers via `tvm_access_ptr` then
dispatched to fork-native [`ptx_cp_async()`](#tvm.backend.cuda.op.ptx_cp_async).

`T.ptx.cp_async_legacy` runs through `_dtype_forward` which
prepends a `dtype=` kwarg as a leading positional. The dtype names
the *element* type of the buffer (offsets are in elements of that
dtype, not bytes), so this function accepts either 5 or 6 positional
args.

<a id="tvm.backend.cuda.op.ptx_cp_async_commit_group"></a>
### `tvm.backend.cuda.op.ptx_cp_async_commit_group`

```python
tvm.backend.cuda.op.ptx_cp_async_commit_group()
```

TVM intrinsic for ptx async copy commit
<https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-commit-group>

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.ptx_cp_async_wait_group"></a>
### `tvm.backend.cuda.op.ptx_cp_async_wait_group`

```python
tvm.backend.cuda.op.ptx_cp_async_wait_group(num=0)
```

TVM intrinsic for ptx async copy wait
<https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-wait-group>

**Parameters:**

**num** (int, *optional*) – The number of the most recent uncommitted pending cp.async groups to wait.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.ptx_cp_async_bulk_tensor_global_to_cluster"></a>
### `tvm.backend.cuda.op.ptx_cp_async_bulk_tensor_global_to_cluster`

```python
tvm.backend.cuda.op.ptx_cp_async_bulk_tensor_global_to_cluster(dim, dst_ptr, bar, tensormap_addr, cta_mask, cta_group, cache_hint, *coords, cache_policy=None)
```

TVM intrinsic to call cp.async.bulk.tensor.dim.shared::cluster.global.tile.mbarrier::complete\_tx::bytes

**Parameters:**

- **dim** (int) – The dimension of the source tensor.
- **dst\_ptr** (tvm.relax.Expr) – The destination pointer to the shared memory.
- **bar** (tvm.relax.Expr) – The pointer to mbarrier variable.
- **tensormap\_addr** (tvm.relax.Expr) – The generic address of the tensor map object.
- **cta\_mask** (int) – The mask of the cta for multicast.
- **cta\_group** (int) – Must be either 1 or 2.
  If set to 1, mbarrier must be in the shared memory of the same CTA
  as the shared memory destination. If set to 2, mbarrier can be in
  shared memory of either the same CTA as the shared memory destination
  or the shared memory of the peer CTA.
- **cache\_hint** (str) – The cache hint.
- **coords** (*List*[tvm.relax.Expr]) – specifies the starting coordinates in the tensor data in the global memory

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.ptx_cp_async_bulk_tensor_tile_gather4_global_to_cluster"></a>
### `tvm.backend.cuda.op.ptx_cp_async_bulk_tensor_tile_gather4_global_to_cluster`

```python
tvm.backend.cuda.op.ptx_cp_async_bulk_tensor_tile_gather4_global_to_cluster(dim, dst_ptr, bar, tensormap_addr, cta_mask, cta_group, cache_hint, *coords, cache_policy=None)
```

TVM intrinsic to call
cp.async.bulk.tensor.dim.shared::cluster.global.tile::gather4.mbarrier::complete\_tx::bytes

**Parameters:**

- **dim** (int) – The dimension of the source tensor.
- **dst\_ptr** (tvm.relax.Expr) – The destination pointer to the shared memory.
- **bar** (tvm.relax.Expr) – The pointer to mbarrier variable.
- **tensormap\_addr** (tvm.relax.Expr) – The generic address of the tensor map object.
- **cta\_mask** (int) – The mask of the cta for multicast.
- **cta\_group** (int) – Must be either 1 or 2.
- **cache\_hint** (str) – The cache hint.
- **coords** (*List*[tvm.relax.Expr]) – The TMA coordinates followed by the 4 gather row indices.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.ptx_cp_async_bulk_tensor_shared_to_global"></a>
### `tvm.backend.cuda.op.ptx_cp_async_bulk_tensor_shared_to_global`

```python
tvm.backend.cuda.op.ptx_cp_async_bulk_tensor_shared_to_global(dim, src_ptr, tensormap_addr, cache_hint, *coords, cache_policy=None)
```

TVM intrinsic to call cp.async.bulk.tensor.dim.global.shared::cta.tile.bulk\_group

**Parameters:**

- **dim** (int) – The dimension of the copy tensor.
- **src\_ptr** (tvm.relax.Expr) – The source pointer to the shared memory.
- **tensormap\_addr** (tvm.relax.Expr) – The generic address of the tensor map object.
- **cache\_hint** (str) – The cache hint.
- **coords** (*List*[tvm.relax.Expr]) – specifies the starting coordinates in the tensor data in the global memory

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.ptx_cp_async_bulk_tensor_global_to_cluster_prefetch"></a>
### `tvm.backend.cuda.op.ptx_cp_async_bulk_tensor_global_to_cluster_prefetch`

```python
tvm.backend.cuda.op.ptx_cp_async_bulk_tensor_global_to_cluster_prefetch(dim, tensormap_addr, cache_hint, *coords, cache_policy=None)
```

TVM intrinsic to call cp.async.bulk.prefetch.tensor.dim.L2.global.tile

**Parameters:**

- **dim** (int) – The dimension of the source tensor.
- **tensormap\_addr** (tvm.relax.Expr) – The generic address of the tensor map object.
- **cache\_hint** (str) – The cache hint.
- **coords** (*List*[tvm.relax.Expr]) – specifies the starting coordinates in the tensor data in the global memory

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.ptx_cp_async_bulk_tensor_shared_to_global_reduce"></a>
### `tvm.backend.cuda.op.ptx_cp_async_bulk_tensor_shared_to_global_reduce`

```python
tvm.backend.cuda.op.ptx_cp_async_bulk_tensor_shared_to_global_reduce(dim, src_ptr, tensormap_addr, cache_hint, red_op, *coords, cache_policy=None)
```

TVM intrinsic to call cp.reduce.async.bulk.tensor.dim.dst.src.redOp

**Parameters:**

- **dim** (int) – The dimension of the copy tensor.
- **src\_ptr** (tvm.relax.Expr) – The source pointer to the shared memory.
- **tensormap\_addr** (tvm.relax.Expr) – The generic address of the tensor map object.
- **cache\_hint** (str) – The cache hint.
- **red\_op** (str) – The reduction operator.
- **coords** (*List*[tvm.relax.Expr]) – The coordinates of the tensor.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.ptx_cp_async_bulk_commit_group"></a>
### `tvm.backend.cuda.op.ptx_cp_async_bulk_commit_group`

```python
tvm.backend.cuda.op.ptx_cp_async_bulk_commit_group()
```

TVM intrinsic to call cp.async.bulk.tensor.commit\_group

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.ptx_cp_async_bulk_wait_group"></a>
### `tvm.backend.cuda.op.ptx_cp_async_bulk_wait_group`

```python
tvm.backend.cuda.op.ptx_cp_async_bulk_wait_group(n=0, read=True)
```

TVM intrinsic to call cp.async.bulk.tensor.wait\_group

**Parameters:**

- **n** (int) – The number of the most recent uncommitted pending cp.async groups to wait.
- **read** (bool) – Whether the wait is for read.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.ptx_barrier_cluster_arrive"></a>
### `tvm.backend.cuda.op.ptx_barrier_cluster_arrive`

```python
tvm.backend.cuda.op.ptx_barrier_cluster_arrive(sem='', aligned=True)
```

TVM intrinsic to call barrier.cluster.arrive{.sem}{.aligned}

**Parameters:**

- **sem** (str) – Either release or relaxed or empty string.
- **aligned** (bool) – Whether all threads in the warp must execute the same instruction.

<a id="tvm.backend.cuda.op.ptx_barrier_cluster_wait"></a>
### `tvm.backend.cuda.op.ptx_barrier_cluster_wait`

```python
tvm.backend.cuda.op.ptx_barrier_cluster_wait(acquire=False, aligned=True)
```

TVM intrinsic to call barrier.cluster.wait{.acquire}{.aligned}

**Parameters:**

- **acquire** (bool) – The memory synchronization
- **aligned** (bool) – Whether all threads in the warp must execute the same instruction.

<a id="tvm.backend.cuda.op.ptx_clc_try_cancel"></a>
### `tvm.backend.cuda.op.ptx_clc_try_cancel`

```python
tvm.backend.cuda.op.ptx_clc_try_cancel(handle, mbar)
```

TVM intrinsic to call clusterlaunchcontrol.try\_cancel.

Async-requests cancelling the next cluster’s launch (work-stealing): writes the
16B response handle to smem and signals `mbar` (complete\_tx, multicast to both
cluster CTAs).

**Parameters:**

- **handle** (tvm.relax.Expr) – Pointer to the 16B (uint4) smem response handle.
- **mbar** (tvm.relax.Expr) – Pointer to the mbarrier signalled when the handle lands.

<a id="tvm.backend.cuda.op.ptx_clc_query_cancel"></a>
### `tvm.backend.cuda.op.ptx_clc_query_cancel`

```python
tvm.backend.cuda.op.ptx_clc_query_cancel(handle)
```

TVM intrinsic to call clusterlaunchcontrol.query\_cancel.

Decodes the response handle written by [`ptx_clc_try_cancel()`](#tvm.backend.cuda.op.ptx_clc_try_cancel). Returns the
cancelled cluster’s first `ctaid.x`, or `0xFFFFFFFF` when no work was stolen.

**Parameters:**

**handle** (tvm.relax.Expr) – Pointer to the 16B (uint4) smem response handle.

<a id="tvm.backend.cuda.op.ptx_elect_sync"></a>
### `tvm.backend.cuda.op.ptx_elect_sync`

```python
tvm.backend.cuda.op.ptx_elect_sync()
```

TVM intrinsic to call elect.sync

<a id="tvm.backend.cuda.op.ptx_fence_mbarrier_init"></a>
### `tvm.backend.cuda.op.ptx_fence_mbarrier_init`

```python
tvm.backend.cuda.op.ptx_fence_mbarrier_init()
```

TVM intrinsic for PTX fence.mbarrier\_init.release.cluster instruction.

Generates: fence.mbarrier\_init.release.cluster;

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.ptx_fetch_register"></a>
### `tvm.backend.cuda.op.ptx_fetch_register`

```python
tvm.backend.cuda.op.ptx_fetch_register(bits, reg_name)
```

TVM intrinsic to tvm instrinsics to fetch PTX pre-defined registers

**Parameters:**

- **bits** (int) – The number of bits of the register.
- **reg\_name** (str) – The name of the register.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.ptx_mma"></a>
### `tvm.backend.cuda.op.ptx_mma`

```python
tvm.backend.cuda.op.ptx_mma(shape, a_layout, b_layout, d_type, a_type, b_type, c_type, d_ptrs, a_ptrs, b_ptrs, c_ptrs=None, saturate=False, bit_op=None)
```

TVM intrinsic for ptx tensor core mma instructions.
<https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-for-mma>

Each per-thread register of every operand is addressed by its OWN pointer
(one `void*` per b32/f32 register), so the register fragments need not be
contiguous in the register file. `d_ptrs` / `a_ptrs` / `b_ptrs` /
`c_ptrs` are lists of one pointer per 32-bit register (b32 for
fp16/bf16/tf32/int8 multiplicands, f32/f64 for the accumulator), enumerated
in the fixed PTX register order (see the gemm dispatch /
`tests/python/tirx-base/test_tir_ptx_mma.py`).

Within one b32 register the packed elements (e.g. 2 fp16 along k\_pack)
must stay contiguous (stride 1); only the b32 registers themselves may be
scattered.

**Parameters:**

- **shape** (str) – The shape of mma fragment.
- **a\_layout** (*Literal*[*"row"*, *"col"*]) – The layout of multiplicand fragment A.
- **b\_layout** (*Literal*[*"row"*, *"col"*]) – The layout of multiplicand fragment B.
- **d\_type** (str) – The data type of result fragment D.
- **a\_type** (str) – The data type of multiplicand fragment A.
- **b\_type** (str) – The data type of multiplicand fragment B.
- **c\_type** (str) – The data type of accumulator fragment C.
- **d\_ptrs** (*List*[tvm.relax.Expr]) – One pointer per result-fragment D register, in PTX order.
- **a\_ptrs** (*List*[tvm.relax.Expr]) – One pointer per multiplicand-A register, in PTX order.
- **b\_ptrs** (*List*[tvm.relax.Expr]) – One pointer per multiplicand-B register, in PTX order.
- **c\_ptrs** (*Optional*[*List*[tvm.relax.Expr]]) – One pointer per accumulator-C register, in PTX order. `None` (the
  default) means the accumulator is not used (beta == 0): codegen feeds
  a literal 0 for each C slot.
- **saturate** (bool) – The optional saturation at the output.
- **bit\_op** (*Optional*[*Literal*[*"xor"*, *"and"*]]) – The 1-bit operator (for the b1 subbyte form). `None` means unused.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.ptx_mma_legacy"></a>
### `tvm.backend.cuda.op.ptx_mma_legacy`

```python
tvm.backend.cuda.op.ptx_mma_legacy(*all_args, operator=None)
```

Legacy `ptx_mma` API.

Signature: `(shape, A_layout, B_layout, A_dtype, B_dtype, C_dtype,
multiplicand_a, a_index, multiplicand_b, b_index, accumulator,
c_index, saturate, operator=None)`. The accumulator is reused as
both input and output (no separate `d`/`c` slot), unlike
fork-native [`ptx_mma()`](#tvm.backend.cuda.op.ptx_mma) which distinguishes them. Translation:

- `a_dtype, b_dtype, c_dtype` → fork `a_type, b_type, c_type`
  (and reuse `c_dtype` as fork `d_type` since the accumulator
  dtype is the output dtype here).
- `(a_ptr, a_offset)` and `(b_ptr, b_offset)` → folded via
  [`tvm_access_ptr()`](#tvm.backend.cuda.op.tvm_access_ptr).
- `(accumulator, c_index)` → folded; passed for both `d_ptr` and
  `c_ptr` since the accumulator is reused as the output.

`T.ptx.mma.legacy` runs through `_dtype_forward` which prepends a
`dtype=` kwarg as a leading positional, so this function accepts
either 13 or 14 positional args.

<a id="tvm.backend.cuda.op.ptx_mma_sp_legacy"></a>
### `tvm.backend.cuda.op.ptx_mma_sp_legacy`

```python
tvm.backend.cuda.op.ptx_mma_sp_legacy(*all_args)
```

Legacy `ptx_mma_sp` API.

Signature: `(shape, A_layout, B_layout, A_dtype, B_dtype, C_dtype,
multiplicand_a, a_index, multiplicand_b, b_index, accumulator,
c_index, metadata, meta_index, sparse_selector, saturate)`.

`T.ptx.mma_sp.legacy` runs through `_dtype_forward` which prepends
a `dtype=` kwarg as a leading positional, so this function accepts
either 16 or 17 positional args.

<a id="tvm.backend.cuda.op.mma_store"></a>
### `tvm.backend.cuda.op.mma_store`

```python
tvm.backend.cuda.op.mma_store(dtype, m, n, dst_ptr, src_ptr, src_offset, dst_stride)
```

Store the result of PTX MMA into a destination pointer.

<a id="tvm.backend.cuda.op.mma_store_legacy"></a>
### `tvm.backend.cuda.op.mma_store_legacy`

```python
tvm.backend.cuda.op.mma_store_legacy(dtype, m, n, dst_ptr, src_ptr, src_offset, dst_stride)
```

mma\_store with apache-style pointer/offset semantics.

<a id="tvm.backend.cuda.op.mma_fill"></a>
### `tvm.backend.cuda.op.mma_fill`

```python
tvm.backend.cuda.op.mma_fill(dtype, local_size, local_ptr, offset)
```

Zero-initialize an MMA accumulation register.

<a id="tvm.backend.cuda.op.mma_fill_legacy"></a>
### `tvm.backend.cuda.op.mma_fill_legacy`

```python
tvm.backend.cuda.op.mma_fill_legacy(dtype, local_size, local_ptr, offset)
```

mma\_fill with apache-style pointer/offset semantics.

<a id="tvm.backend.cuda.op.ptx_ldmatrix"></a>
### `tvm.backend.cuda.op.ptx_ldmatrix`

```python
tvm.backend.cuda.op.ptx_ldmatrix(trans, num, dtype, smem_ptr, *dst_handles)
```

TVM intrinsic for ldmatrix.sync.aligned.m8n8.x{num}{.trans}.shared.{dtype}.

Mirrors the PTX ISA destination form: each output register is a separate
operand. Pass `T.address_of(buf[idx])` (or `buf.ptr_to([idx])`) for
each destination — the slots may be non-contiguous.

**Parameters:**

- **trans** (bool) – Apply the `.trans` modifier.
- **num** (int) – One of 1, 2, 4 — number of m8n8 fragments.
- **dtype** (str) – `"b16"` (4 bytes per fragment register) or `"b8"` (2 bytes per).
- **smem\_ptr** (tvm.relax.Expr) – Generic pointer to source shared memory.
- **\*dst\_handles** (tvm.relax.Expr) – N pointer-to-uint32 destinations, where
  `N = num if dtype == "b16" else num // 2`.
- **https** (*//docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-ldmatrix*)

<a id="tvm.backend.cuda.op.ptx_ldmatrix_legacy"></a>
### `tvm.backend.cuda.op.ptx_ldmatrix_legacy`

```python
tvm.backend.cuda.op.ptx_ldmatrix_legacy(*all_args)
```

Legacy `ptx_ldmatrix` API taking explicit offsets.

Signature: `(trans, num, dtype, local_ptr, local_offset, smem_ptr,
smem_offset)`. Offsets are folded into the pointers via
`tvm_access_ptr` and dispatched to the fork-native
[`ptx_ldmatrix()`](#tvm.backend.cuda.op.ptx_ldmatrix).

`T.ptx.ldmatrix_legacy` runs through `_dtype_forward` which
prepends a `dtype=` kwarg as a leading positional naming the buffer
element type — offsets are in elements of that dtype, not bytes, so
we forward it to `tvm_access_ptr` for correct scaling.

<a id="tvm.backend.cuda.op.ptx_stmatrix"></a>
### `tvm.backend.cuda.op.ptx_stmatrix`

```python
tvm.backend.cuda.op.ptx_stmatrix(trans, num, dtype, smem_ptr, *src_handles, shape='m8n8', space='shared')
```

TVM intrinsic for `stmatrix.sync.aligned.shape.x{num}{.trans}.space.{dtype}`.

Mirrors [`ptx_ldmatrix()`](#tvm.backend.cuda.op.ptx_ldmatrix): each source register is a separate operand.
Pass `T.address_of(buf[idx])` (or `buf.ptr_to([idx])`) for each
source — the slots may be non-contiguous.

**Parameters:**

- **trans** (bool) – Apply the `.trans` modifier (required for `shape == "m16n8"`).
- **num** (int) – One of 1, 2, 4 — number of m8n8 fragments per warp.
- **dtype** (str) – `".b16"` (4 bytes per fragment register) or `".b8"` (2 bytes per).
- **smem\_ptr** (tvm.relax.Expr) – Destination pointer in shared memory.
- **\*src\_handles** (tvm.relax.Expr) – `num` pointer-to-uint32 sources.
- **shape** (str, *keyword-only*, *default "m8n8"*) – `"m8n8"` or `"m16n8"`.
- **space** (str, *keyword-only*, *default "shared"*) – `"shared"` or `"shared::cta"`.
- **https** (*//docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-stmatrix*)

<a id="tvm.backend.cuda.op.ptx_wgmma_encode_matrix_descriptor"></a>
### `tvm.backend.cuda.op.ptx_wgmma_encode_matrix_descriptor`

```python
tvm.backend.cuda.op.ptx_wgmma_encode_matrix_descriptor(desc, addr, ldo, sdo, swizzle)
```

TVM intrinsic to create memory descriptor for wgmma instructions

**Parameters:**

- **desc** (tvm.relax.Expr) – The pointer to the shared memory descriptor.
- **addr** (tvm.relax.Expr) – The address of the matrix.
- **ldo** (tvm.relax.Expr) – The leading dimension offset.
- **sdo** (tvm.relax.Expr) – The stride dimension offset.
- **swizzle** (int) – The swizzle value (CUtensorMapSwizzle\_enum).

<a id="tvm.backend.cuda.op.ptx_wgmma_noop_barrier"></a>
### `tvm.backend.cuda.op.ptx_wgmma_noop_barrier`

```python
tvm.backend.cuda.op.ptx_wgmma_noop_barrier(reg)
```

TVM intrinsic to call “” : “+{format}”(reg)::”memory”

**Parameters:**

**reg** (tvm.relax.Expr) – The register to fence.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.ptx_wgmma_mma_async_ss"></a>
### `tvm.backend.cuda.op.ptx_wgmma_mma_async_ss`

```python
tvm.backend.cuda.op.ptx_wgmma_mma_async_ss(descA, descB, *accums, M, N, K, in_dtype, out_dtype, transA, transB, scaleA, scaleB, scaleD)
```

TVM intrinsic to call wgmma.mma\_async.sync.aligned.shape.dtype.atype.btype over 2 smem operators

**Parameters:**

- **M** (int) – The number of rows in matrix A and D.
- **N** (int) – The number of columns in matrix B and D.
- **K** (int) – The number of columns in matrix A and rows in matrix B.
- **in\_dtype** (str) – The data type of the input matrices.
- **out\_type** (str) – The data type of the output matrices.
- **transA** (bool) – True for M/N major, False for K major.
- **transB** (bool) – True for M/N major, False for K major.
- **scaleA** (float) – The scaling factor for matrix A.
- **scaleB** (float) – The scaling factor for matrix B.
- **scaleD** (tvm.relax.Expr) – True: D = A \* B + D, False: D = A \* B.
- **descA** (tvm.relax.Expr) – The SMEM descriptor of matrix A
- **descB** (tvm.relax.Expr) – The SMEM descriptor of matrix B
- **accums** (list) – The accumulators registers.

<a id="tvm.backend.cuda.op.ptx_wgmma_mma_async_rs"></a>
### `tvm.backend.cuda.op.ptx_wgmma_mma_async_rs`

```python
tvm.backend.cuda.op.ptx_wgmma_mma_async_rs(descB, *reg_list, M, N, K, in_dtype, out_dtype, transA, transB, scaleA, scaleB, scaleD)
```

**TVM intrinsic to call wgmma.mma\_async.sync.aligned.shape.dtype.atype.btype**

When A is in register and B is in shared memory

**Parameters:**

- **M** (int) – The number of rows in matrix A and D.
- **N** (int) – The number of columns in matrix B and D.
- **K** (int) – The number of columns in matrix A and rows in matrix B.
- **in\_dtype** (str) – The data type of the input matrices.
- **out\_type** (str) – The data type of the output matrices.
- **transA** (bool) – True for M/N major, False for K major.
- **transB** (bool) – True for M/N major, False for K major.
- **scaleA** (float) – The scaling factor for matrix A.
- **scaleB** (float) – The scaling factor for matrix B.
- **scaleD** (tvm.relax.Expr) – True: D = A \* B + D, False: D = A \* B.
- **descB** (tvm.relax.Expr) – The SMEM descriptor of matrix B
- **reg\_list** (list) – The A registers and accumulators registers.

<a id="tvm.backend.cuda.op.ptx_wgmma_fence"></a>
### `tvm.backend.cuda.op.ptx_wgmma_fence`

```python
tvm.backend.cuda.op.ptx_wgmma_fence()
```

TVM intrinsic to call wgmma.fence.sync.aligned

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.ptx_wgmma_commit_group"></a>
### `tvm.backend.cuda.op.ptx_wgmma_commit_group`

```python
tvm.backend.cuda.op.ptx_wgmma_commit_group()
```

TVM intrinsic to call wgmma.commit\_group.sync.aligned

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.ptx_wgmma_wait_group"></a>
### `tvm.backend.cuda.op.ptx_wgmma_wait_group`

```python
tvm.backend.cuda.op.ptx_wgmma_wait_group(n)
```

TVM intrinsic to call wgmma.wait\_group.sync.aligned

**Parameters:**

**n** (int) – The number of the most recent uncommitted pending wgmma groups to wait.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.ptx_setmaxnreg"></a>
### `tvm.backend.cuda.op.ptx_setmaxnreg`

```python
tvm.backend.cuda.op.ptx_setmaxnreg(inc: bool, reg_count)
```

TVM intrinsic to call setmaxnreg.action.sync.aligned.u32 imm-reg-count

**Parameters:**

- **inc** (bool) – True to increase the register count, False to decrease.
- **reg\_count** (int) – The register count.

<a id="tvm.backend.cuda.op.ptx_tcgen05_alloc"></a>
### `tvm.backend.cuda.op.ptx_tcgen05_alloc`

```python
tvm.backend.cuda.op.ptx_tcgen05_alloc(dst_ptr, n_cols, cta_group=1)
```

**TVM intrinsic to call tcgen05.alloc.cta\_group.sync.aligned**

Dynamically allocates the number of cols in tensor memory, and write
the address of allocated memory to shared memory.

**Parameters:**

- **dst\_ptr** (tvm.ir.Var) – The pointer to the destination shared memory.
- **n\_cols** (int) – The number of columns to allocate in tensor memory.
  Must be a multiple of 32 and a power of 2, and within the range [32, 512].
- **cta\_group** (int) – The number of CTA groups involved in the allocation.
  If cta\_group=1, one warp from CTA performs the allocation. Else, if cta\_group=2,
  one warp from each of the peer CTAs perform the allocation.

<a id="tvm.backend.cuda.op.ptx_tcgen05_dealloc"></a>
### `tvm.backend.cuda.op.ptx_tcgen05_dealloc`

```python
tvm.backend.cuda.op.ptx_tcgen05_dealloc(taddr, n_cols, cta_group=1)
```

**TVM intrinsic to call tcgen05.dealloc.cta\_group.sync.aligned**

Deallocates the tensor memory specified by the tensor memory address taddr.

**Parameters:**

- **taddr** (tvm.relax.Expr) – The address of previously allocated tensor memory, should be uint32\_t.
- **n\_cols** (int) – The number of columns to deallocate in tensor memory.
  Must be a multiple of 32 and a power of 2, and within the range [32, 512].
- **cta\_group** (int) – The number of CTA groups involved in the deallocation.
  If cta\_group=1, one warp from CTA performs the deallocation. Else, if cta\_group=2,
  one warp from each of the peer CTAs perform the deallocation.

<a id="tvm.backend.cuda.op.ptx_tcgen05_relinquish_alloc_permit"></a>
### `tvm.backend.cuda.op.ptx_tcgen05_relinquish_alloc_permit`

```python
tvm.backend.cuda.op.ptx_tcgen05_relinquish_alloc_permit(cta_group=1)
```

**TVM intrinsic to call tcgen05.relinquish\_alloc\_permit.cta\_group.sync.aligned**

The CTA of the executing thread is relinquishing the right to allocate
Tensor Memory after calling this op.

**Parameters:**

**cta\_group** (int) – The number of CTA groups involved in relinquishing.
If cta\_group=1, one warp from CTA performs the relinquishing. Else, if cta\_group=2,
one warp from each of the peer CTAs perform the relinquishing.

<a id="tvm.backend.cuda.op.ptx_tcgen05_encode_matrix_descriptor"></a>
### `tvm.backend.cuda.op.ptx_tcgen05_encode_matrix_descriptor`

```python
tvm.backend.cuda.op.ptx_tcgen05_encode_matrix_descriptor(desc, addr, ldo, sdo, swizzle)
```

TVM intrinsic to create memory descriptor for tcgen05 instructions

**Parameters:**

- **desc** (tvm.relax.Expr) – The pointer to the shared memory descriptor.
- **addr** (tvm.relax.Expr) – The address of the matrix.
- **ldo** (tvm.relax.Expr) – The leading dimension offset.
- **sdo** (tvm.relax.Expr) – The stride dimension offset.
- **swizzle** (int) – The swizzle value (CUtensorMapSwizzle\_enum).

<a id="tvm.backend.cuda.op.ptx_tcgen05_encode_instr_descriptor"></a>
### `tvm.backend.cuda.op.ptx_tcgen05_encode_instr_descriptor`

```python
tvm.backend.cuda.op.ptx_tcgen05_encode_instr_descriptor(desc, *, d_dtype, a_dtype, b_dtype, M, N, K, trans_a, trans_b, n_cta_groups=1, neg_a=False, neg_b=False, sat_d=False, is_sparse=False)
```

TVM intrinsic to create instruction descriptor for tcgen05 MMA without block scaling

**Parameters:**

- **desc** (tvm.relax.Expr) – The pointer to the instruction descriptor.
- **d\_dtype** (str) – The datatype of resultant matrix D.
- **a\_dtype** (str) – The datatype of multiplicand matrix A.
- **b\_dtype** (str) – The datatype of multiplicand matrix B.
- **M** (int) – The size of non-reduction dimension of Matrix A.
- **N** (int) – The size of non-reduction dimension of Matrix B.
- **K** (int) – The size of reduction dimension of Matrix A/B.
- **trans\_a** (bool) – Whether the multiplicand matrix A is transposed.
  True for M/N major, False for K major.
- **trans\_b** (bool) – Whether the multiplicand matrix B is transposed.
  True for M/N major, False for K major.
- **n\_cta\_groups** (int) – The number of CTA groups involved in the MMA operation.
- **neg\_a** (bool) – Whether to negate the multiplicand matrix A.
- **neg\_b** (bool) – Whether to negate the multiplicand matrix B.
- **sat\_d** (bool) – Whether to saturate the resultant matrix D.
- **is\_sparse** (bool) – Whether the MMA operation is sparse.

<a id="tvm.backend.cuda.op.ptx_tcgen05_encode_instr_descriptor_block_scaled"></a>
### `tvm.backend.cuda.op.ptx_tcgen05_encode_instr_descriptor_block_scaled`

```python
tvm.backend.cuda.op.ptx_tcgen05_encode_instr_descriptor_block_scaled(desc, *, d_dtype, a_dtype, b_dtype, sfa_dtype, sfb_dtype, sfa_tmem_addr, sfb_tmem_addr, M, N, K, trans_a, trans_b, n_cta_groups=1, neg_a=False, neg_b=False, is_sparse=False)
```

TVM intrinsic to create instruction descriptor for tcgen05 MMA with block scaling

**Parameters:**

- **desc** (tvm.relax.Expr) – The pointer to the instruction descriptor.
- **d\_dtype** (str) – The datatype of resultant matrix D.
- **a\_dtype** (str) – The datatype of multiplicand matrix A.
- **b\_dtype** (str) – The datatype of multiplicand matrix B.
- **sfa\_dtype** (str) – The datatype of scale factor matrix A.
- **sfb\_dtype** (str) – The datatype of scale factor matrix B.
- **sfa\_tmem\_addr** (tvm.relax.Expr) – The address of the scale factor matrix A in tensor memory, should be uint32\_t.
- **sfb\_tmem\_addr** (tvm.relax.Expr) – The address of the scale factor matrix B in tensor memory, should be uint32\_t.
- **M** (int) – The size of non-reduction dimension of Matrix A.
- **N** (int) – The size of non-reduction dimension of Matrix B.
- **K** (int) – The size of reduction dimension of Matrix A/B.
- **trans\_a** (bool) – Whether the multiplicand matrix A is transposed.
  True for M/N major, False for K major.
- **trans\_b** (bool) – Whether the multiplicand matrix B is transposed.
  True for M/N major, False for K major.
- **n\_cta\_groups** (int) – The number of CTA groups involved in the MMA operation.
- **neg\_a** (bool) – Whether to negate the multiplicand matrix A.
- **neg\_b** (bool) – Whether to negate the multiplicand matrix B.
- **is\_sparse** (bool) – Whether the MMA operation is sparse.

<a id="tvm.backend.cuda.op.ptx_tcgen05_mma"></a>
### `tvm.backend.cuda.op.ptx_tcgen05_mma`

```python
tvm.backend.cuda.op.ptx_tcgen05_mma(d_tmem_addr, a_operand, b_desc, i_desc, *disable_output_lane, d_dtype, a_dtype, b_dtype, use_a_tmem, cta_group, enable_input_d=1, scale_input_d=0, pred=None)
```

TVM intrinsic to call tcgen05.mma.cta\_group.kind without block scaling.

**Parameters:**

- **d\_dtype** (str) – The datatype of resultant matrix D.
- **a\_dtype** (str) – The datatype of multiplicand matrix A.
- **b\_dtype** (str) – The datatype of multiplicand matrix B.
- **d\_tmem\_addr** (tvm.relax.Expr) – The address of the resultant matrix D in tensor memory, should be uint32\_t.
- **a\_operand** (tvm.relax.Expr) – Either the matrix descriptor of multiplicand matrix A in shared memory,
  or the address of the multiplicand matrix A in tensor memory (uint32\_t).
- **b\_desc** (tvm.relax.Expr) – The matrix descriptor of multiplicand matrix B in shared memory.
- **i\_desc** (tvm.relax.Expr) – The instruction descriptor of the MMA operation.
- **use\_a\_tmem** (bool) – Whether the multiplicand matrix A is in tensor memory.
- **cta\_group** (int) – The number of CTA groups involved in the MMA operation.
- **enable\_input\_d** (tvm.relax.Expr) – Scale operand for the input accumulator C/D. The inline asm tests
  enable\_input\_d != 0: zero means D = A\*B, non-zero means D = A\*B + D.
- **scale\_input\_d** (int) – The optional scaling factor to scale input matrix D.
  D = A\*B+D \* (2 ^ - scale-input-d)
- **disable\_output\_lane** (list) – The lanes that should not be updated in the resultant matrix D.
- **pred** (*Optional*[tvm.relax.Expr]) – Runtime `uint32` instruction-level predicate. When given, emit
  `@p_issue tcgen05.mma...` with `p_issue = (pred != 0)`. Preserves
  PTX-level predicate semantics (single predicated SASS instruction).

<a id="tvm.backend.cuda.op.ptx_tcgen05_mma_block_scale"></a>
### `tvm.backend.cuda.op.ptx_tcgen05_mma_block_scale`

```python
tvm.backend.cuda.op.ptx_tcgen05_mma_block_scale(d_tmem_addr, a_operand, b_desc, sfa_tmem_addr, sfb_tmem_addr, i_desc, *, d_dtype, a_dtype, b_dtype, sfa_dtype, sfb_dtype, use_a_tmem, cta_group, enable_input_d=1)
```

**TVM intrinsic to call tcgen05.mma.cta\_group.kind.block\_scale**

Performs matrix multiplication with block scaling:
(A \* scale\_A) \* (B \* scale\_B) + D

**Parameters:**

- **d\_dtype** (str) – The datatype of resultant matrix D.
- **a\_dtype** (str) – The datatype of multiplicand matrix A.
- **b\_dtype** (str) – The datatype of multiplicand matrix B.
- **sfa\_dtype** (str) – The datatype of scale factor matrix A.
- **sfb\_dtype** (str) – The datatype of scale factor matrix B.
- **d\_tmem\_addr** (tvm.relax.Expr) – The address of the resultant matrix D in tensor memory, should be uint32\_t.
- **a\_operand** (tvm.relax.Expr) – Either the matrix descriptor of multiplicand matrix A in shared memory,
  or the address of the multiplicand matrix A in tensor memory (uint32\_t).
- **b\_desc** (tvm.relax.Expr) – The matrix descriptor of multiplicand matrix B in shared memory.
- **sfa\_tmem\_addr** (tvm.relax.Expr) – The address of the scale factor matrix A in tensor memory, should be uint32\_t.
- **sfb\_tmem\_addr** (tvm.relax.Expr) – The address of the scale factor matrix B in tensor memory, should be uint32\_t.
- **i\_desc** (tvm.relax.Expr) – The instruction descriptor of the MMA operation.
- **use\_a\_tmem** (bool) – Whether the multiplicand matrix A is in tensor memory.
- **cta\_group** (int) – The number of CTA groups involved in the MMA operation.
- **enable\_input\_d** (tvm.relax.Expr) – Scale operand for the input accumulator C/D. Zero means D = A\*B,
  non-zero means D = A\*B + D.

<a id="tvm.backend.cuda.op.ptx_tcgen05_mma_sp"></a>
### `tvm.backend.cuda.op.ptx_tcgen05_mma_sp`

```python
tvm.backend.cuda.op.ptx_tcgen05_mma_sp(d_tmem_addr, a_operand, b_desc, sp_tmem_addr, i_desc, *disable_output_lane, d_dtype, a_dtype, b_dtype, use_a_tmem, cta_group, enable_input_d=1, scale_input_d=0)
```

TVM intrinsic to call tcgen05.mma.sp.cta\_group.kind without block scaling.

**Parameters:**

- **d\_dtype** (str) – The datatype of resultant matrix D.
- **a\_dtype** (str) – The datatype of multiplicand matrix A.
- **b\_dtype** (str) – The datatype of multiplicand matrix B.
- **d\_tmem\_addr** (tvm.relax.Expr) – The address of the resultant matrix D in tensor memory, should be uint32\_t.
- **a\_operand** (tvm.relax.Expr) – Either the matrix descriptor of multiplicand matrix A in shared memory,
  or the address of the multiplicand matrix A in tensor memory (uint32\_t).
- **b\_desc** (tvm.relax.Expr) – The matrix descriptor of multiplicand matrix B in shared memory.
- **sp\_tmem\_addr** (tvm.relax.Expr) – The address of the metadata of sparse matrix in tensor memory, should be uint32\_t.
- **i\_desc** (tvm.relax.Expr) – The instruction descriptor of the MMA operation.
- **use\_a\_tmem** (bool) – Whether the multiplicand matrix A is in tensor memory.
- **cta\_group** (int) – The number of CTA groups involved in the MMA operation.
- **enable\_input\_d** (tvm.relax.Expr) – Scale operand for the input accumulator C/D. The inline asm tests
  enable\_input\_d != 0: zero means D = A\*B, non-zero means D = A\*B + D.
- **scale\_input\_d** (int) – The optional scaling factor to scale input matrix D.
  D = A\*B+D \* (2 ^ - scale-input-d)
- **disable\_output\_lane** (list) – The lanes that should not be updated in the resultant matrix D.

<a id="tvm.backend.cuda.op.ptx_tcgen05_mma_sp_block_scale"></a>
### `tvm.backend.cuda.op.ptx_tcgen05_mma_sp_block_scale`

```python
tvm.backend.cuda.op.ptx_tcgen05_mma_sp_block_scale(d_tmem_addr, a_operand, b_desc, sfa_tmem_addr, sfb_tmem_addr, sp_tmem_addr, i_desc, *, d_dtype, a_dtype, b_dtype, sfa_dtype, sfb_dtype, use_a_tmem, cta_group, enable_input_d=1)
```

**TVM intrinsic to call tcgen05.mma.sp.cta\_group.kind.block\_scale**

Performs sparse matrix multiplication with block scaling:
(A \* scale\_A) \* (B \* scale\_B) + D

**Parameters:**

- **d\_dtype** (str) – The datatype of resultant matrix D.
- **a\_dtype** (str) – The datatype of multiplicand matrix A.
- **b\_dtype** (str) – The datatype of multiplicand matrix B.
- **sfa\_dtype** (str) – The datatype of scale factor matrix A.
- **sfb\_dtype** (str) – The datatype of scale factor matrix B.
- **d\_tmem\_addr** (tvm.relax.Expr) – The address of the resultant matrix D in tensor memory, should be uint32\_t.
- **a\_operand** (tvm.relax.Expr) – Either the matrix descriptor of multiplicand matrix A in shared memory,
  or the address of the multiplicand matrix A in tensor memory (uint32\_t).
- **b\_desc** (tvm.relax.Expr) – The matrix descriptor of multiplicand matrix B in shared memory.
- **sfa\_tmem\_addr** (tvm.relax.Expr) – The address of the scale factor matrix A in tensor memory, should be uint32\_t.
- **sfb\_tmem\_addr** (tvm.relax.Expr) – The address of the scale factor matrix B in tensor memory, should be uint32\_t.
- **sp\_tmem\_addr** (tvm.relax.Expr) – The address of the metadata of sparse matrix in tensor memory, should be uint32\_t.
- **i\_desc** (tvm.relax.Expr) – The instruction descriptor of the MMA operation.
- **use\_a\_tmem** (bool) – Whether the multiplicand matrix A is in tensor memory.
- **cta\_group** (int) – The number of CTA groups involved in the MMA operation.
- **enable\_input\_d** (tvm.relax.Expr) – Scale operand for the input accumulator C/D. Zero means D = A\*B,
  non-zero means D = A\*B + D.

<a id="tvm.backend.cuda.op.ptx_tcgen05_fence_before_thread_sync"></a>
### `tvm.backend.cuda.op.ptx_tcgen05_fence_before_thread_sync`

```python
tvm.backend.cuda.op.ptx_tcgen05_fence_before_thread_sync()
```

TVM intrinsic to call tcgen05.fence::before\_thread\_sync
Orders all prior asynchronous tcgen05 operations relative to subsequent operations.

<a id="tvm.backend.cuda.op.ptx_tcgen05_fence_after_thread_sync"></a>
### `tvm.backend.cuda.op.ptx_tcgen05_fence_after_thread_sync`

```python
tvm.backend.cuda.op.ptx_tcgen05_fence_after_thread_sync()
```

TVM intrinsic to call tcgen05.fence::after\_thread\_sync
Orders all subsequent asynchronous tcgen05 operations relative to previous operations.

<a id="tvm.backend.cuda.op.ptx_tcgen05_cp"></a>
### `tvm.backend.cuda.op.ptx_tcgen05_cp`

```python
tvm.backend.cuda.op.ptx_tcgen05_cp(taddr, src_desc, *, shape, cta_group=1, multicast='', decompress='', row=0, col=0)
```

TVM intrinsic for the Blackwell tcgen05.cp PTX instruction.

The emitted PTX is:

```
tcgen05.cp.cta_group::{cta_group}.{shape}[.{multicast}][.{decompress}] [taddr], src_desc;
```

Each keyword argument maps 1:1 to a PTX token: read the call and you
know what instruction is emitted.

**Parameters:**

- **taddr** (tvm.relax.Expr) – Destination tensor-memory address (uint32). Callers typically pass
  `tmem_base + column_offset_in_uint32s` directly. Use the optional
  `row` / `col` keyword arguments only when the address needs
  runtime row/col composition via `get_tmem_addr` (high 16 bits row,
  low 16 bits col).
- **src\_desc** (tvm.relax.Expr) – The 64-bit shared-memory matrix descriptor.
- **shape** (str) – One of `"32x128b"`, `"4x256b"`, `"128x128b"`, `"128x256b"`,
  `"64x128b"`.
- **cta\_group** (int) – 1 or 2.
- **multicast** (str) – One of `""`, `"warpx4"`, `"warpx2::02_13"`, `"warpx2::01_23"`.
  `"32x128b"` requires `"warpx4"`; `"64x128b"` requires one of the
  `warpx2::*` values; other shapes require `""`.
- **decompress** (str) – Trailing PTX suffix for fp4/fp6 → fp8 on-the-fly decompression.
  One of `""`, `"b8x16.b4x16_p64"`, `"b8x16.b6x16_p32"`.
- **row** (tvm.relax.Expr) – Optional row/col offsets added to `taddr` at runtime. Default 0.
- **col** (tvm.relax.Expr) – Optional row/col offsets added to `taddr` at runtime. Default 0.

<a id="tvm.backend.cuda.op.ptx_tcgen05_shift"></a>
### `tvm.backend.cuda.op.ptx_tcgen05_shift`

```python
tvm.backend.cuda.op.ptx_tcgen05_shift(taddr, cta_group=1)
```

**TVM intrinsic to call tcgen05.shift.cta\_group.down**

Asynchronously shift down the rows of the matrix in Tensor Memory for a warp.

**Parameters:**

- **taddr** (tvm.relax.Expr) – The address of matrix in tensor memory, should be uint32\_t.
- **cta\_group** (int) – The number of CTA groups involved in the shift.
  If cta\_group=1, shift operation is performed in the Tensor Memory of current CTA.
  Else, shift operation is performed in the Tensor Memory of both the current CTA and
  the peer CTA.

<a id="tvm.backend.cuda.op.ptx_tcgen05_ld"></a>
### `tvm.backend.cuda.op.ptx_tcgen05_ld`

```python
tvm.backend.cuda.op.ptx_tcgen05_ld(src_addr, *regs, shape, num, row=0, col=0, pack=False)
```

TVM intrinsic for tcgen05.ld.sync.aligned — async collective load from TMEM.

Emits `tcgen05.ld.sync.aligned.{shape}.x{num}[.pack::16b].b32 {regs}, [addr];`

**Parameters:**

- **src\_addr** (tvm.relax.Expr) – Tensor-memory source address (uint32).
- **regs** (list[tvm.relax.Expr]) – Destination registers. Count depends on shape x num.
- **shape** (str) – One of `"16x32bx2"`, `"16x64b"`, `"16x128b"`, `"16x256b"`, `"32x32b"`.
- **num** (int) – Repeat factor along the columns. Power-of-two in [1, 128].
- **row** (tvm.relax.Expr) – Optional TMEM row/col offsets added to `src_addr` at runtime (row must be
  a multiple of 32). Default 0.
- **col** (tvm.relax.Expr) – Optional TMEM row/col offsets added to `src_addr` at runtime (row must be
  a multiple of 32). Default 0.
- **pack** (bool) – Pack two 16-bit chunks into a single 32-bit register.

<a id="tvm.backend.cuda.op.ptx_tcgen05_st"></a>
### `tvm.backend.cuda.op.ptx_tcgen05_st`

```python
tvm.backend.cuda.op.ptx_tcgen05_st(dst_addr, *regs, shape, num, row=0, col=0, unpack=False)
```

TVM intrinsic for tcgen05.st.sync.aligned — async collective store to TMEM.

Emits `tcgen05.st.sync.aligned.{shape}.x{num}[.unpack::16b].b32 [addr], {regs};`

**Parameters:**

- **dst\_addr** (tvm.relax.Expr) – Tensor-memory destination address (uint32).
- **regs** (list[tvm.relax.Expr]) – Source registers. Count depends on shape x num.
- **shape** (str) – One of `"16x32bx2"`, `"16x64b"`, `"16x128b"`, `"16x256b"`, `"32x32b"`.
- **num** (int) – Repeat factor along the columns. Power-of-two in [1, 128].
- **row** (tvm.relax.Expr) – Optional TMEM row/col offsets added to `dst_addr` at runtime (row must be
  a multiple of 32). Default 0.
- **col** (tvm.relax.Expr) – Optional TMEM row/col offsets added to `dst_addr` at runtime (row must be
  a multiple of 32). Default 0.
- **unpack** (bool) – Unpack a 32-bit register into two 16-bit chunks.

<a id="tvm.backend.cuda.op.ptx_tcgen05_wait_ld"></a>
### `tvm.backend.cuda.op.ptx_tcgen05_wait_ld`

```python
tvm.backend.cuda.op.ptx_tcgen05_wait_ld()
```

TVM intrinsic to call tcgen05.wait::ld.sync.aligned
Wait for the completion of all prior async tcgen05.ld operations.

<a id="tvm.backend.cuda.op.ptx_tcgen05_wait_st"></a>
### `tvm.backend.cuda.op.ptx_tcgen05_wait_st`

```python
tvm.backend.cuda.op.ptx_tcgen05_wait_st()
```

TVM intrinsic to call tcgen05.wait::st.sync.aligned
Wait for the completion of all prior async tcgen05.st operations.

<a id="tvm.backend.cuda.op.ptx_tcgen05_commit"></a>
### `tvm.backend.cuda.op.ptx_tcgen05_commit`

```python
tvm.backend.cuda.op.ptx_tcgen05_commit(bar, cta_group=1, cta_mask=0, *, pred=None)
```

TVM intrinsic to call tcgen05.commit.cta\_group

**Parameters:**

- **bar** (tvm.relax.Expr) – The pointer to mbarrier variable.
- **cta\_group** (int) – The number of CTA groups involved in previous tcgen05 operations.
- **cta\_mask** (int) – The mask of the CTAs in the cluster, used for multicast.
- **pred** (*Optional*[tvm.relax.Expr]) – Runtime `uint32` predicate. When given, emit
  `@p tcgen05.commit...` with `p = (pred != 0)`. This preserves
  PTX-level instruction predicate semantics (single predicated
  instruction in SASS), distinct from a C-level `if` branch.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.timer_init_cuda"></a>
### `tvm.backend.cuda.op.timer_init_cuda`

```python
tvm.backend.cuda.op.timer_init_cuda(profiler_buffer, profiler_tag, profiler_write_offset, num_groups, group_id)
```

TVM intrinsic for initializing the CUDA profiler, and store profiling result in a buffer.

**Parameters:**

- **profiler\_buffer** (tvm.ir.Var) – The buffer to store the profiling result.
- **profiler\_tag** (tvm.ir.Var) – Buffer of length 1 storing the base tag of the current thread.
- **profiler\_write\_offset** (tvm.ir.Var) – Buffer of length 1 storing the offset in buffer to write the next
  profiling result for the current thread.
- **num\_groups** (int) – The number of groups in the profiler.
- **group\_id** (tvm.relax.Expr) – The group id of the current thread.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.timer_start_cuda"></a>
### `tvm.backend.cuda.op.timer_start_cuda`

```python
tvm.backend.cuda.op.timer_start_cuda(event_type, profiler_buffer, profiler_tag, profiler_write_offset, profiler_write_stride, leader_cond)
```

TVM intrinsic for starting the timer for profiling a specific event, and storing profiling result in a buffer.

**Parameters:**

- **event\_type** (*Enum*) – The event to profile.
- **profiler\_buffer** (tvm.ir.Var) – The buffer to store the profiling result.
- **profiler\_tag** (tvm.ir.Var) – Buffer of length 1 storing the base tag of the current thread.
- **profiler\_write\_offset** (tvm.ir.Var) – Buffer of length 1 storing the offset in buffer to write the next
  profiling result for the current thread.
- **profiler\_write\_stride** (int) – The stride to advance in buffer in the next write.
- **leader\_cond** (tvm.relax.Expr) – The condition to check if the current thread is the leader.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.timer_end_cuda"></a>
### `tvm.backend.cuda.op.timer_end_cuda`

```python
tvm.backend.cuda.op.timer_end_cuda(event_type, profiler_buffer, profiler_tag, profiler_write_offset, profiler_write_stride, leader_cond)
```

TVM intrinsic for ending the timer for profiling a specific event, and storing profiling result in a buffer.

**Parameters:**

- **event\_type** (*Enum*) – The event to profile.
- **profiler\_buffer** (tvm.ir.Var) – The buffer to store the profiling result.
- **profiler\_tag** (tvm.ir.Var) – Buffer of length 1 storing the base tag of the current thread.
- **profiler\_write\_offset** (tvm.ir.Var) – Buffer of length 1 storing the offset in buffer to write the next
  profiling result for the current thread.
- **profiler\_write\_stride** (int) – The stride to advance in buffer in the next write.
- **leader\_cond** (tvm.relax.Expr) – The condition to check if the current thread is the leader.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.timer_finalize_cuda"></a>
### `tvm.backend.cuda.op.timer_finalize_cuda`

```python
tvm.backend.cuda.op.timer_finalize_cuda(profiler_buffer, profiler_tag, profiler_write_offset, profiler_write_stride, leader_cond)
```

TVM intrinsic for finalizing the CUDA profiler, and store profiling result in a buffer.

**Parameters:**

- **profiler\_buffer** (tvm.ir.Var) – The buffer to store the profiling result.
- **profiler\_tag** (tvm.ir.Var) – Buffer of length 1 storing the base tag of the current thread.
- **profiler\_write\_offset** (tvm.ir.Var) – Buffer of length 1 storing the offset in buffer to write the next
  profiling result for the current thread.
- **profiler\_write\_stride** (int) – The stride to advance in buffer in the next write.
- **leader\_cond** (tvm.relax.Expr) – The condition to check if the current thread is the leader.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.cuda_atomic_add"></a>
### `tvm.backend.cuda.op.cuda_atomic_add`

```python
tvm.backend.cuda.op.cuda_atomic_add(res_addr, value)
```

TVM intrinsic to call cuda atomic add instruction

**Parameters:**

- **res\_addr** (tvm.relax.Expr) – The result address.
- **value** (tvm.relax.Expr) – The value to add.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.cuda_thread_fence"></a>
### `tvm.backend.cuda.op.cuda_thread_fence`

```python
tvm.backend.cuda.op.cuda_thread_fence()
```

TVM intrinsic to call cuda thread fence instruction

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.cuda_warpgroup_sync"></a>
### `tvm.backend.cuda.op.cuda_warpgroup_sync`

```python
tvm.backend.cuda.op.cuda_warpgroup_sync(bar_no)
```

TVM intrinsic to synchronize a CUDA warpgroup via a named barrier.

**Parameters:**

**bar\_no** (tvm.relax.Expr) – The named barrier id to use for the warpgroup.

Notes

Synchronizes 128 threads in a warpgroup using bar.sync bar\_no, 128.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.cuda_syncthreads_and"></a>
### `tvm.backend.cuda.op.cuda_syncthreads_and`

```python
tvm.backend.cuda.op.cuda_syncthreads_and(cond)
```

TVM intrinsic to call cuda syncthreads\_and instruction

**Parameters:**

**cond** (tvm.relax.Expr) – The condition.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.cuda_syncthreads_or"></a>
### `tvm.backend.cuda.op.cuda_syncthreads_or`

```python
tvm.backend.cuda.op.cuda_syncthreads_or(cond)
```

TVM intrinsic to call cuda syncthreads\_or instruction

**Parameters:**

**cond** (tvm.relax.Expr) – The condition.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.cuda_nano_sleep"></a>
### `tvm.backend.cuda.op.cuda_nano_sleep`

```python
tvm.backend.cuda.op.cuda_nano_sleep(time)
```

TVM intrinsic to call cuda nano sleep instruction

**Parameters:**

**time** (tvm.relax.Expr) – The time to sleep.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.cuda_printf"></a>
### `tvm.backend.cuda.op.cuda_printf`

```python
tvm.backend.cuda.op.cuda_printf(fmt, *args)
```

TVM intrinsic to call cuda printf instruction

**Parameters:**

- **fmt** (str) – The format string.
- **\*args** (list) – The arguments to the format string.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.cuda_ldg"></a>
### `tvm.backend.cuda.op.cuda_ldg`

```python
tvm.backend.cuda.op.cuda_ldg(addr, dtype)
```

TVM intrinsic to call CUDA C++ \_\_ldg() function

**Parameters:**

- **addr** (tvm.relax.Expr) – The memory address to load.
- **dtype** (str) – The data type of the loaded value.
- **Returns**

<a id="tvm.backend.cuda.op.cuda_get_tmem_addr"></a>
### `tvm.backend.cuda.op.cuda_get_tmem_addr`

```python
tvm.backend.cuda.op.cuda_get_tmem_addr(addr, row_offset, col_offset)
```

TVM intrinsic to call cuda tmem address calculation

**Parameters:**

- **addr** (tvm.relax.Expr) – The memory address to calculate.
- **row\_offset** (tvm.relax.Expr) – The row offset to calculate.
- **col\_offset** (tvm.relax.Expr) – The column offset to calculate.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.cuda_cvta_generic_to_shared"></a>
### `tvm.backend.cuda.op.cuda_cvta_generic_to_shared`

```python
tvm.backend.cuda.op.cuda_cvta_generic_to_shared(ptr)
```

Convert a generic pointer to a shared-memory address (uint32).

Wraps `__cvta_generic_to_shared(ptr)`. Used by op-wrappers that
precompute the shared-memory address at the wrapper layer instead of
inside the asm helper body.

<a id="tvm.backend.cuda.op.cuda_smem_addr_from_uint64"></a>
### `tvm.backend.cuda.op.cuda_smem_addr_from_uint64`

```python
tvm.backend.cuda.op.cuda_smem_addr_from_uint64(cluster_addr)
```

Narrow a 64-bit cluster-mapped SMEM address to a 32-bit SMEM address.

Wraps `static_cast<unsigned int>(cluster_addr)`. Used by
cp.async.bulk.shared::cluster.\* op-wrappers.

<a id="tvm.backend.cuda.op.cuda_sm100_tma_2sm_mbarrier_addr"></a>
### `tvm.backend.cuda.op.cuda_sm100_tma_2sm_mbarrier_addr`

```python
tvm.backend.cuda.op.cuda_sm100_tma_2sm_mbarrier_addr(bar)
```

Compute the SM100 2SM TMA mbarrier shared-address operand.

<a id="tvm.backend.cuda.op.ptx_exp2"></a>
### `tvm.backend.cuda.op.ptx_exp2`

```python
tvm.backend.cuda.op.ptx_exp2(x)
```

TVM intrinsic for PTX fast exp2 approximation (ex2.approx.ftz.f32)

**Parameters:**

**x** (tvm.relax.Expr) – The float32 input value.

**Returns:**

**call** – The call expression returning 2^x (approximate).

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.ptx_rcp"></a>
### `tvm.backend.cuda.op.ptx_rcp`

```python
tvm.backend.cuda.op.ptx_rcp(x)
```

TVM intrinsic for PTX fast reciprocal approximation (rcp.approx.ftz.f32)

**Parameters:**

**x** (tvm.relax.Expr) – The float32 input value.

**Returns:**

**call** – The call expression returning 1/x (approximate).

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.ptx_any_sync"></a>
### `tvm.backend.cuda.op.ptx_any_sync`

```python
tvm.backend.cuda.op.ptx_any_sync(mask, pred)
```

TVM intrinsic for PTX warp-wide any predicate (\_\_any\_sync)

**Parameters:**

- **mask** (tvm.relax.Expr) – The thread mask (uint32).
- **pred** (tvm.relax.Expr) – The predicate value (int32).

**Returns:**

**call** – The call expression returning 1 if any thread in mask has pred != 0.

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.ptx_reduce3_max_f32"></a>
### `tvm.backend.cuda.op.ptx_reduce3_max_f32`

```python
tvm.backend.cuda.op.ptx_reduce3_max_f32(a, b, c)
```

TVM intrinsic to call 3-input max.f32 PTX instruction (sm\_100a+)

**Parameters:**

- **a** (tvm.relax.Expr) – The three float32 values to compare.
- **b** (tvm.relax.Expr) – The three float32 values to compare.
- **c** (tvm.relax.Expr) – The three float32 values to compare.

**Returns:**

**call** – The call expression returning max(a, b, c).

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.ptx_reduce3_min_f32"></a>
### `tvm.backend.cuda.op.ptx_reduce3_min_f32`

```python
tvm.backend.cuda.op.ptx_reduce3_min_f32(a, b, c)
```

TVM intrinsic to call 3-input min.f32 PTX instruction (sm\_100a+)

**Parameters:**

- **a** (tvm.relax.Expr) – The three float32 values to compare.
- **b** (tvm.relax.Expr) – The three float32 values to compare.
- **c** (tvm.relax.Expr) – The three float32 values to compare.

**Returns:**

**call** – The call expression returning min(a, b, c).

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.ptx_add_f32"></a>
### `tvm.backend.cuda.op.ptx_add_f32`

```python
tvm.backend.cuda.op.ptx_add_f32(d_addr, a, b, *, rounding='rn', ftz=False, sat=False)
```

PTX `add{.rnd}{.ftz}{.sat}.f32 [d_addr], a, b` — DPS form.

<a id="tvm.backend.cuda.op.ptx_add_f32x2"></a>
### `tvm.backend.cuda.op.ptx_add_f32x2`

```python
tvm.backend.cuda.op.ptx_add_f32x2(d_addr, a, b, *, rounding='rn', ftz=False)
```

PTX `add{.rnd}{.ftz}.f32x2 [d_addr], a, b` — DPS form.

a, b are packed-as-uint64 register operands (2 fp32 each).

<a id="tvm.backend.cuda.op.ptx_add_f64"></a>
### `tvm.backend.cuda.op.ptx_add_f64`

```python
tvm.backend.cuda.op.ptx_add_f64(d_addr, a, b, *, rounding='rn')
```

PTX `add{.rnd}.f64 [d_addr], a, b` — DPS form (no .ftz / .sat).

<a id="tvm.backend.cuda.op.ptx_sub_f32"></a>
### `tvm.backend.cuda.op.ptx_sub_f32`

```python
tvm.backend.cuda.op.ptx_sub_f32(d_addr, a, b, *, rounding='rn', ftz=False, sat=False)
```

PTX `sub{.rnd}{.ftz}{.sat}.f32 [d_addr], a, b` — DPS form.

<a id="tvm.backend.cuda.op.ptx_sub_f32x2"></a>
### `tvm.backend.cuda.op.ptx_sub_f32x2`

```python
tvm.backend.cuda.op.ptx_sub_f32x2(d_addr, a, b, *, rounding='rn', ftz=False)
```

PTX `sub{.rnd}{.ftz}.f32x2 [d_addr], a, b` — DPS form.

<a id="tvm.backend.cuda.op.ptx_sub_f64"></a>
### `tvm.backend.cuda.op.ptx_sub_f64`

```python
tvm.backend.cuda.op.ptx_sub_f64(d_addr, a, b, *, rounding='rn')
```

PTX `sub{.rnd}.f64 [d_addr], a, b` — DPS form.

<a id="tvm.backend.cuda.op.ptx_mul_f32"></a>
### `tvm.backend.cuda.op.ptx_mul_f32`

```python
tvm.backend.cuda.op.ptx_mul_f32(d_addr, a, b, *, rounding='rn', ftz=False, sat=False)
```

PTX `mul{.rnd}{.ftz}{.sat}.f32 [d_addr], a, b` — DPS form.

<a id="tvm.backend.cuda.op.ptx_mul_f32x2"></a>
### `tvm.backend.cuda.op.ptx_mul_f32x2`

```python
tvm.backend.cuda.op.ptx_mul_f32x2(d_addr, a, b, *, rounding='rn', ftz=False)
```

PTX `mul{.rnd}{.ftz}.f32x2 [d_addr], a, b` — DPS form.

<a id="tvm.backend.cuda.op.ptx_mul_f64"></a>
### `tvm.backend.cuda.op.ptx_mul_f64`

```python
tvm.backend.cuda.op.ptx_mul_f64(d_addr, a, b, *, rounding='rn')
```

PTX `mul{.rnd}.f64 [d_addr], a, b` — DPS form.

<a id="tvm.backend.cuda.op.ptx_fma_f32"></a>
### `tvm.backend.cuda.op.ptx_fma_f32`

```python
tvm.backend.cuda.op.ptx_fma_f32(d_addr, a, b, c, *, rounding='rn', ftz=False, sat=False)
```

PTX `fma{.rnd}{.ftz}{.sat}.f32 [d_addr], a, b, c` — DPS form.

<a id="tvm.backend.cuda.op.ptx_fma_f32x2"></a>
### `tvm.backend.cuda.op.ptx_fma_f32x2`

```python
tvm.backend.cuda.op.ptx_fma_f32x2(d_addr, a, b, c, *, rounding='rn', ftz=False)
```

PTX `fma{.rnd}{.ftz}.f32x2 [d_addr], a, b, c` — DPS form.

a, b, c are packed-as-uint64 register operands.

<a id="tvm.backend.cuda.op.ptx_fma_f64"></a>
### `tvm.backend.cuda.op.ptx_fma_f64`

```python
tvm.backend.cuda.op.ptx_fma_f64(d_addr, a, b, c, *, rounding='rn')
```

PTX `fma{.rnd}.f64 [d_addr], a, b, c` — DPS form.

<a id="tvm.backend.cuda.op.ptx_max_f32"></a>
### `tvm.backend.cuda.op.ptx_max_f32`

```python
tvm.backend.cuda.op.ptx_max_f32(a, b, *, ftz=False, nan=False)
```

TVM intrinsic for PTX `max{.ftz}{.NaN}.f32 d, a, b`.

2-operand form (distinct from [`ptx_reduce3_max_f32()`](#tvm.backend.cuda.op.ptx_reduce3_max_f32) which is the
3-operand SM\_100+ form). `.NaN` qualifier propagates NaN inputs to
the output; without it, NaN inputs are silently ignored.

**Parameters:**

- **a** (tvm.relax.Expr) – Float32 inputs.
- **b** (tvm.relax.Expr) – Float32 inputs.
- **ftz** (bool) – If True, flush subnormals to zero (`.ftz`).
- **nan** (bool) – If True, propagate NaN inputs (`.NaN`).

<a id="tvm.backend.cuda.op.ptx_griddepcontrol_wait"></a>
### `tvm.backend.cuda.op.ptx_griddepcontrol_wait`

```python
tvm.backend.cuda.op.ptx_griddepcontrol_wait()
```

TVM intrinsic for PTX `griddepcontrol.wait` (sm\_90+).

Blocks the current grid until prerequisite grids signalled via
[`ptx_griddepcontrol_launch_dependents()`](#tvm.backend.cuda.op.ptx_griddepcontrol_launch_dependents) have finished. Acts as a
full memory barrier.

<a id="tvm.backend.cuda.op.ptx_griddepcontrol_launch_dependents"></a>
### `tvm.backend.cuda.op.ptx_griddepcontrol_launch_dependents`

```python
tvm.backend.cuda.op.ptx_griddepcontrol_launch_dependents()
```

TVM intrinsic for PTX `griddepcontrol.launch_dependents` (sm\_90+).

Signals that the current grid has reached a point where dependent
grids may begin execution.

<a id="tvm.backend.cuda.op.ptx_ld_acquire"></a>
### `tvm.backend.cuda.op.ptx_ld_acquire`

```python
tvm.backend.cuda.op.ptx_ld_acquire(addr, return_type, ptx_type, *, scope='gpu', space='global', vec='', dst=None, cache_hint='', cache_policy=None, l1_evict='', l2_evict='', prefetch_size='')
```

TVM intrinsic for PTX `ld.acquire.scope{.ss}...` loads.

`scope`, state `space`, PTX `type` and TVM `return_type` are
explicit so callers can request either raw-bit or typed loads. The
optional `vec`/`dst` and cache arguments cover vector and cache-policy
forms of the same PTX instruction.

**Parameters:**

- **addr** (tvm.relax.Expr) – The memory address to load.
- **return\_type** (str) – TVM dtype returned by the load.
- **ptx\_type** (str) – PTX type suffix such as `"b32"`, `"u64"`, or `"s32"`.
- **scope** (str) – PTX memory scope: `"cta"`, `"cluster"`, `"gpu"`, or `"sys"`.
- **space** (str) – PTX state space suffix.

**Returns:**

**call** – The loaded value.

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.ptx_ld"></a>
### `tvm.backend.cuda.op.ptx_ld`

```python
tvm.backend.cuda.op.ptx_ld(addr, return_type, ptx_type, *, dst=None, weak=False, space='global', cop='', vec='', cache_hint='', cache_policy=None, l1_evict='', l2_evict='', prefetch_size='')
```

TVM intrinsic for PTX `ld{.weak}{.ss}{.cop}...` loads.

<a id="tvm.backend.cuda.op.ptx_ld_volatile"></a>
### `tvm.backend.cuda.op.ptx_ld_volatile`

```python
tvm.backend.cuda.op.ptx_ld_volatile(addr, return_type, ptx_type, *, space='global', vec='', dst=None, prefetch_size='')
```

TVM intrinsic for PTX `ld.volatile{.ss}...` loads.

<a id="tvm.backend.cuda.op.ptx_ld_global_acquire"></a>
### `tvm.backend.cuda.op.ptx_ld_global_acquire`

```python
tvm.backend.cuda.op.ptx_ld_global_acquire(res, addr)
```

TVM intrinsic to call the legacy ptx ld.global.acquire helper.

**Parameters:**

- **res** (tvm.relax.Expr) – The result of the load.
- **addr** (tvm.relax.Expr) – The memory address to load.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.ptx_map_shared_rank"></a>
### `tvm.backend.cuda.op.ptx_map_shared_rank`

```python
tvm.backend.cuda.op.ptx_map_shared_rank(ptr, rank)
```

TVM intrinsic to call ptx map\_shared\_rank instruction

**Parameters:**

- **ptr** (tvm.relax.Expr) – The generic pointer to the local shared memory, handle type
- **rank** (int) – The rank of the distributed shared memory.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.ptx_mapa"></a>
### `tvm.backend.cuda.op.ptx_mapa`

```python
tvm.backend.cuda.op.ptx_mapa(ptr, rank, *, space='', ptx_type='u64', return_type='uint64')
```

TVM intrinsic for PTX `mapa{.space}.type d, a, b`.

<a id="tvm.backend.cuda.op.cuda_atomic_cas"></a>
### `tvm.backend.cuda.op.cuda_atomic_cas`

```python
tvm.backend.cuda.op.cuda_atomic_cas(ptr, old_val, new_val)
```

TVM intrinsic to call cuda atomic cas instruction

**Parameters:**

- **ptr** (tvm.relax.Expr) – The pointer to the memory location.
- **old\_val** (tvm.relax.Expr) – The old value.
- **new\_val** (tvm.relax.Expr) – The new value.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.nvshmem_my_pe"></a>
### `tvm.backend.cuda.op.nvshmem_my_pe`

```python
tvm.backend.cuda.op.nvshmem_my_pe()
```

TVM intrinsic to call nvshmem\_my\_pe()

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.nvshmem_n_pes"></a>
### `tvm.backend.cuda.op.nvshmem_n_pes`

```python
tvm.backend.cuda.op.nvshmem_n_pes()
```

TVM intrinsic to call nvshmem\_n\_pes()

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.nvshmem_getmem_nbi"></a>
### `tvm.backend.cuda.op.nvshmem_getmem_nbi`

```python
tvm.backend.cuda.op.nvshmem_getmem_nbi(dst, src, nelems, pe)
```

TVM intrinsic to call nvshmem\_getmem\_nbi()

**Parameters:**

- **dst** (tvm.relax.Expr) – The pointer to the symmetric address or host/device address of the data object to be updated.
- **src** (tvm.relax.Expr) – The pointer to the symmetric address of the source data object.
- **nelems** (int) – The number of bytes to get per thread.
- **pe** (int) – The PE number of the remote PE.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.nvshmem_putmem_nbi"></a>
### `tvm.backend.cuda.op.nvshmem_putmem_nbi`

```python
tvm.backend.cuda.op.nvshmem_putmem_nbi(dst, src, nelems, pe)
```

TVM intrinsic to call nvshmem\_putmem\_nbi()

**Parameters:**

- **dst** (tvm.relax.Expr) – The pointer to the symmetric address of the destination data object.
- **src** (tvm.relax.Expr) – The pointer to the symmetric address or host/device address of the data object to be copied.
- **nelems** (int) – The number of bytes to put per thread.
- **pe** (int) – The PE number of the remote PE.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.nvshmem_getmem_nbi_warp"></a>
### `tvm.backend.cuda.op.nvshmem_getmem_nbi_warp`

```python
tvm.backend.cuda.op.nvshmem_getmem_nbi_warp(dst, src, nelems, pe)
```

TVM intrinsic to call nvshmem\_getmem\_nbi\_warp()

**Parameters:**

- **dst** (tvm.relax.Expr) – The pointer to the symmetric address or host/device address of the data object to be updated.
- **src** (tvm.relax.Expr) – The pointer to the symmetric address of the source data object.
- **nelems** (int) – The number of bytes to get per warp.
- **pe** (int) – The PE number of the remote PE.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.nvshmem_putmem_nbi_warp"></a>
### `tvm.backend.cuda.op.nvshmem_putmem_nbi_warp`

```python
tvm.backend.cuda.op.nvshmem_putmem_nbi_warp(dst, src, nelems, pe)
```

TVM intrinsic to call nvshmem\_putmem\_nbi\_warp()

**Parameters:**

- **dst** (tvm.relax.Expr) – The pointer to the symmetric address of the destination data object.
- **src** (tvm.relax.Expr) – The pointer to the symmetric address or host/device address of the data object to be copied.
- **nelems** (int) – The number of bytes to put per warp.
- **pe** (int) – The PE number of the remote PE.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.nvshmem_getmem_nbi_block"></a>
### `tvm.backend.cuda.op.nvshmem_getmem_nbi_block`

```python
tvm.backend.cuda.op.nvshmem_getmem_nbi_block(dst, src, nelems, pe)
```

TVM intrinsic to call nvshmem\_getmem\_nbi\_block()

**Parameters:**

- **dst** (tvm.relax.Expr) – The pointer to the symmetric address or host/device address of the data object to be updated.
- **src** (tvm.relax.Expr) – The pointer to the symmetric address of the source data object.
- **nelems** (int) – The number of bytes to get per block.
- **pe** (int) – The PE number of the remote PE.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.nvshmem_putmem_nbi_block"></a>
### `tvm.backend.cuda.op.nvshmem_putmem_nbi_block`

```python
tvm.backend.cuda.op.nvshmem_putmem_nbi_block(dst, src, nelems, pe)
```

TVM intrinsic to call nvshmem\_putmem\_nbi\_block()

**Parameters:**

- **dst** (tvm.relax.Expr) – The pointer to the symmetric address of the destination data object.
- **src** (tvm.relax.Expr) – The pointer to the symmetric address or host/device address of the data object to be copied.
- **nelems** (int) – The number of bytes to put per block.
- **pe** (int) – The PE number of the remote PE.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.nvshmem_signal_op"></a>
### `tvm.backend.cuda.op.nvshmem_signal_op`

```python
tvm.backend.cuda.op.nvshmem_signal_op(sig_addr, signal, sig_op, pe)
```

TVM intrinsic to call nvshmem\_signal\_op()

**Parameters:**

- **sig\_addr** (tvm.relax.Expr) – The pointer to the symmetric address of the signal word to be updated, must be uint64\_t\*.
- **signal** (*uint64\_t*) – The value used to update sig\_addr.
- **sig\_op** (str) – Operation used to update sig\_addr with signal, typical sig\_op values are “set” and “add”.
- **pe** (int) – The PE number of the remote PE.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.nvshmem_wait_until"></a>
### `tvm.backend.cuda.op.nvshmem_wait_until`

```python
tvm.backend.cuda.op.nvshmem_wait_until(ivar, cmp, cmp_value, type='uint64_t')
```

TVM intrinsic to call nvshmem\_wait\_until()

**Parameters:**

- **ivar** (tvm.relax.Expr) – The pointer to the symmetric address of a remotely accessible data object, must be TYPE\*.
- **cmp** (str) – The compare operator that compares ivar with cmp\_value.
- **cmp\_value** (*TYPE*) – The value to be compared with ivar.
- **type** (str) – The TYPE of ivar and cmp\_value.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.nvshmem_quiet"></a>
### `tvm.backend.cuda.op.nvshmem_quiet`

```python
tvm.backend.cuda.op.nvshmem_quiet()
```

TVM intrinsic to call nvshmem\_quiet()

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.nvshmem_putmem_signal_nbi"></a>
### `tvm.backend.cuda.op.nvshmem_putmem_signal_nbi`

```python
tvm.backend.cuda.op.nvshmem_putmem_signal_nbi(dst, src, nelems, sig_addr, signal, sig_op, pe)
```

TVM intrinsic to call nvshmem\_putmem\_signal\_nbi()

**Parameters:**

- **dst** (tvm.relax.Expr) – The pointer to the symmetric address of the data object to be updated on the remote PE.
- **src** (tvm.relax.Expr) – The pointer to the symmetric address or host/device address of data object containing the data to be copied.
- **nelems** (int) – The number of bytes to put per thread.
- **sig\_addr** (tvm.relax.Expr) – The pointer to the symmetric address of the signal data object to be updated on the remote PE as a signal, must be uint64\_t\*.
- **signal** (*uint64\_t*) – The unsigned 64-bit value that is used for updating the remote sig\_addr signal data object.
- **sig\_op** (str) – Signal operator that represents the type of update to be performed on the remote sig\_addr signal data object.
- **pe** (int) – The PE number of the remote PE.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.nvshmem_putmem_signal_nbi_warp"></a>
### `tvm.backend.cuda.op.nvshmem_putmem_signal_nbi_warp`

```python
tvm.backend.cuda.op.nvshmem_putmem_signal_nbi_warp(dst, src, nelems, sig_addr, signal, sig_op, pe)
```

TVM intrinsic to call nvshmem\_putmem\_signal\_nbi\_warp()

**Parameters:**

- **dst** (tvm.relax.Expr) – The pointer to the symmetric address of the data object to be updated on the remote PE.
- **src** (tvm.relax.Expr) – The pointer to the symmetric address or host/device address of data object containing the data to be copied.
- **nelems** (int) – The number of bytes to put per warp.
- **sig\_addr** (tvm.relax.Expr) – The pointer to the symmetric address of the signal data object to be updated on the remote PE as a signal, must be uint64\_t\*.
- **signal** (*uint64\_t*) – The unsigned 64-bit value that is used for updating the remote sig\_addr signal data object.
- **sig\_op** (str) – Signal operator that represents the type of update to be performed on the remote sig\_addr signal data object.
- **pe** (int) – The PE number of the remote PE.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.nvshmem_putmem_signal_nbi_block"></a>
### `tvm.backend.cuda.op.nvshmem_putmem_signal_nbi_block`

```python
tvm.backend.cuda.op.nvshmem_putmem_signal_nbi_block(dst, src, nelems, sig_addr, signal, sig_op, pe)
```

TVM intrinsic to call nvshmem\_putmem\_signal\_nbi\_block()

**Parameters:**

- **dst** (tvm.relax.Expr) – The pointer to the symmetric address of the data object to be updated on the remote PE.
- **src** (tvm.relax.Expr) – The pointer to the symmetric address or host/device address of data object containing the data to be copied.
- **nelems** (int) – The number of bytes to put per block.
- **sig\_addr** (tvm.relax.Expr) – The pointer to the symmetric address of the signal data object to be updated on the remote PE as a signal, must be uint64\_t\*.
- **signal** (*uint64\_t*) – The unsigned 64-bit value that is used for updating the remote sig\_addr signal data object.
- **sig\_op** (str) – Signal operator that represents the type of update to be performed on the remote sig\_addr signal data object.
- **pe** (int) – The PE number of the remote PE.

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.nvshmem_fence"></a>
### `tvm.backend.cuda.op.nvshmem_fence`

```python
tvm.backend.cuda.op.nvshmem_fence()
```

TVM intrinsic to call nvshmem\_fence()

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="tvm.backend.cuda.op.nvshmem_barrier_all"></a>
### `tvm.backend.cuda.op.nvshmem_barrier_all`

```python
tvm.backend.cuda.op.nvshmem_barrier_all()
```

TVM intrinsic to call nvshmem\_barrier\_all()

**Returns:**

**call** – The call expression.

**Return type:**

tvm.relax.Expr

<a id="module-tvm.backend.cuda.script"></a>
## tvm.backend.cuda.script

CUDA TVMScript namespaces.

<a id="tvm.backend.cuda.script.CUDANamespace"></a>
### `tvm.backend.cuda.script.CUDANamespace`

```python
class tvm.backend.cuda.script.CUDANamespace
```

The CUDA intrinsics submodule.

<a id="tvm.backend.cuda.script.NVSHMEMNamespace"></a>
### `tvm.backend.cuda.script.NVSHMEMNamespace`

```python
class tvm.backend.cuda.script.NVSHMEMNamespace
```

The NVSHMEM intrinsics submodule.

<a id="tvm.backend.cuda.script.PTXNamespace"></a>
### `tvm.backend.cuda.script.PTXNamespace`

```python
class tvm.backend.cuda.script.PTXNamespace
```

The PTX instruction submodule.

<a id="module-tvm.backend.cuda.operator"></a>
## tvm.backend.cuda.operator

CUDA backend operator registrations and helpers.

<a id="module-tvm.backend.cuda.target_tags"></a>
## tvm.backend.cuda.target_tags

NVIDIA CUDA target tags.

<a id="tvm.backend.cuda.target_tags.register_tag"></a>
### `tvm.backend.cuda.target_tags.register_tag`

```python
tvm.backend.cuda.target_tags.register_tag(name: str, config: dict[str, Any], override: bool = False) → Target | None
```

Add a user-defined tag into the target tag registry.

**Parameters:**

- **name** (str) – Name of the target, e.g. “nvidia/gtx1080ti”
- **config** (*Dict*[str, *Any*]) – The config dict used to create the target
- **override** (bool) – A boolean flag indicating if overriding existing tags are allowed.
  If False and the tag has been registered already, an exception will be thrown.

**Returns:**

**target** – The target corresponding to the tag
None if TVM is built in runtime-only mode.

**Return type:**

Optional[tvm.target.Target]

Examples

```python
register_tag("nvidia/gtx1080ti", config={
    "kind": "cuda",
    "arch": "sm_61",
})
```
