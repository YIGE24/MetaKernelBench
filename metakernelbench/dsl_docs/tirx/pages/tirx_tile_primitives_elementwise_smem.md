<!-- source: https://tvm.apache.org/docs/tirx/tile_primitives/elementwise/smem.html -->

<a id="elementwise-smem"></a>
# elementwise → smem

The `smem` variant lowers an elementwise op (`sqrt`, `exp`, `add`,
`fma`, …) when **all operands are in shared memory**. Like the copy
[copy → gmem\_smem](tirx_tile_primitives_copy_gmem_smem.md) variant it *synthesizes* a `[outer, threads, vec]`
partition from the execution scope, then applies the op to each (vectorized)
element. Source:
`python/tvm/backend/cuda/operator/tile_primitive/elementwise/smem.py`.

<a id="what-it-accepts"></a>
## What it accepts

`is_smem_ewise(spec)` builds the predicate:

```python
def check(op_call, sctx):
    if not sctx.is_target("cuda"): return False, "non-cuda target"
    if sctx.scope_kind not in ("thread", "warp", "warpgroup", "cta"): ...
    ok, reason = _all_threads_active(sctx)              # full scope
    plan, msg = spec.parse(op_call)                     # parse the op's operands
    for br in buffer_regions(plan):
        if not br.buffer.scope().startswith("shared"):  # every operand shared*
            return False, f"operand scope {br.buffer.scope()} != shared*"
        if br.buffer.layout is None: ...
    # + spec.check_extras (dtype rules) and anchor-layout validation
```

| Property | Requirement |
| --- | --- |
| target / scope / priority | `cuda`; `thread` / `warp` / `warpgroup` / `cta` (all active); priority `10` |
| operands | **every** operand (inputs and output) in `shared*` |
| op | any op in the registry (unary `sqrt`/`exp`/`zero`…, binary `add`/`mul`…, `fma`); `spec.check_extras` validates the dtype combo |
| layout | operands have layouts; the layout sets the **vector width** (the partition itself is synthesized from the scope’s thread count, not the layout) |

<a id="demonstration-program"></a>
## Demonstration program

A CTA takes the elementwise `sqrt` of a `32×32` `float32` shared tile
(adapted from `test_unary.py` — here a 256-thread CTA, so the partition is one
round):

```python
s_layout = TileLayout(S[(32, 32)]); full = (slice(0, 32), slice(0, 32))

@T.prim_func
def unary_op(A_ptr: T.handle):
    A = T.match_buffer(A_ptr, (32, 32), "float32", layout=s_layout)
    T.device_entry(); T.cta_id([1]); T.warp_id([8]); T.lane_id([32]); T.thread_id([256])
    A_smem = T.alloc_buffer((32, 32), "float32", scope="shared", layout=s_layout)
    Tx.cta.copy(A_smem[full], A[full])
    Tx.cta.sqrt(A_smem[full], A_smem[full])   # elementwise smem dispatch
    Tx.cta.copy(A[full], A_smem[full])
```

<a id="algorithm"></a>
## Algorithm

**1. Parse the op and check operands.** `spec.parse` turns the call into a plan
(inputs, output, the op); the predicate confirms every operand is shared.

**2. Synthesize the partition** from the scope’s **thread count** (as
[copy → gmem\_smem](tirx_tile_primitives_copy_gmem_smem.md) does): split the region into `[outer, threads, vec]`,
with the vector width taken from the layout’s innermost contiguous run. For
`32×32 = 1024` `float32` over 256 threads, `vec = 4` ⇒ `outer = 1`.

**3. Apply the op per element.** Instead of a copy, each (thread, round) reads its
`vec` elements, applies the op, and writes back — vectorized:

<a id="generated-tirx-ir"></a>
## Generated TIRx IR
```python
for f in range(1):                                # outer = 1
    A_smem[tid * 4 + vec] = T.sqrt(A_smem[tid * 4 + vec])
```

<a id="generated-cuda"></a>
## Generated CUDA

The `vec = 4` element bundle becomes a `float4` and the op is applied per
component:

```c
float4 v_ = *(float4*)(&A_smem_ptr[tid * 4]);
__1.x = sqrtf(v_.x);  __1.y = sqrtf(v_.y);
__1.z = sqrtf(v_.z);  __1.w = sqrtf(v_.w);
```

(Verified on `sm_100a` — the tile equals `sqrt(A)`.)

<a id="how-inputs-change-the-algorithm"></a>
## How inputs change the algorithm

| input | effect |
| --- | --- |
| op | unary → `sqrtf` / `expf` / … per component; binary → the two inputs combined (`a + b`); `fma` → `a * b + c` |
| dtype | sets the vector width (`vec = widest aligned` ⇒ the round count), as in [copy → gmem\_smem](tirx_tile_primitives_copy_gmem_smem.md) |
| scope | sets the thread axis and count, hence the synthesized partition |
