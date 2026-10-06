<!-- source: https://tvm.apache.org/docs/tirx/native_basics/cuda/compiling.html -->

<a id="compiling-and-inspecting"></a>
# Compiling and inspecting

Wrap the `PrimFunc` in an `IRModule` and compile with
`tvm.compile(mod, target=..., tir_pipeline="tirx")`; it runs the TIRx lowering
pipeline and returns an `Executable` you call directly. The arch (e.g.
`sm_100a`) is auto-detected from the device, so the target `"cuda"` is enough.

```python
target = tvm.target.Target("cuda")
exe = tvm.compile(tvm.IRModule({"main": scale}), target=target, tir_pipeline="tirx")
```

`tir_pipeline="tirx"` selects the TIRx lowering pipeline (`LowerTIRx` →
tile-primitive dispatch → host/device split → finalize). Compiling inside a
`with target:` block also works and lets the kernel pick up the target context.

<a id="inspecting-the-result"></a>
## Inspecting the result

Read the IR with `.show()` / `.script()`, and read the generated CUDA from the
compiled module.

```python
scale.show()                          # pretty-print the TIRx (TVMScript)
print(scale.script())                 # ... the same, as a string

# the generated CUDA C source, from the compiled Executable:
print(exe.mod.imports[0].inspect_source())
```

Debug aids: `T.print_buffer(C.data, "float32", False, False, 1, (M,))` emits a
runtime `printf` of a buffer into the kernel; `T.hint("message")` (statement
or `with` block) attaches structured hints that survive a script round-trip.

<a id="from-simple-to-complex"></a>
## From simple to complex

A natural native progression, each rung adding one capability:

1. **Elementwise** — `device_entry` + `thread_id` + a guarded store (the first
   kernel).
2. **Shared-memory reduction** — stage into `T.alloc_shared`, then a
   `cta_sync`-separated tree (shown in full below). Adds shared memory and a
   block barrier.
3. **Warp / block reduction** — `T.tvm_warp_shuffle_xor` or `T.cuda.cta_sum`
   to combine partial results across lanes/warps (the warp all-reduce in
   [CUDA C++/PTX intrinsics](tirx_native_basics_cuda_threads_sync.md)).
4. **Async pipeline** — `T.ptx.cp_async` (or TMA `cp_async.bulk.tensor`) with
   `T.ptx.mbarrier.*` to overlap loads with compute.

Rung 2 in full — a 256-element block sum via a shared-memory tree reduction
(shared buffer, `cta_sync`, a `while` loop, and a thread predicate):

```python
@T.prim_func
def block_sum(A_ptr: T.handle, out_ptr: T.handle):
    A = T.match_buffer(A_ptr, (256,), "float32")
    out = T.match_buffer(out_ptr, (1,), "float32")

    T.device_entry()
    bx = T.cta_id([1])
    tx = T.thread_id([256])

    sm = T.alloc_shared((256,), "float32")
    sm[tx] = A[tx]
    T.cuda.cta_sync()

    s = T.alloc_local((1,), "int32")
    s[0] = 128
    while s[0] >= 1:
        if tx < s[0]:
            sm[tx] += sm[tx + s[0]]
        T.cuda.cta_sync()
        s[0] = s[0] // 2

    if tx == 0:
        out[0] = sm[0]

exe = tvm.compile(tvm.IRModule({"main": block_sum}),
                  target=tvm.target.Target("cuda"), tir_pipeline="tirx")
a = torch.arange(256, device="cuda", dtype=torch.float32)
out = torch.zeros(1, device="cuda")
exe(a, out)                          # out[0] == 32640.0
```

The full tile-level GEMM/attention ladder (sync → TMA → warp specialization →
2-CTA cluster) is built on top of these and the dispatchable tile primitives in
[Tile Primitives](tirx_tile_primitives.md).

<a id="next-steps"></a>
## Next steps

- [Tensor Layout](tirx_layout.md) — how buffers map to physical resources (`TileLayout`).
- [Tile Primitives](tirx_tile_primitives.md) — the dispatchable ops these native idioms lower to.
