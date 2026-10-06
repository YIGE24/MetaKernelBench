<!-- source: https://tvm.apache.org/docs/tirx/native_basics/cuda/first_kernel.html -->

<a id="your-first-kernel"></a>
# Your first kernel

A complete write → compile → run → inspect loop. This kernel scales a vector by 2
with one block of 256 threads.

```python
import numpy as np
import tvm
from tvm.script import tirx as T

@T.prim_func
def scale(A_ptr: T.handle, B_ptr: T.handle):
    A = T.match_buffer(A_ptr, (256,), "float32")
    B = T.match_buffer(B_ptr, (256,), "float32")

    T.device_entry()                 # everything below runs on the device
    bx = T.cta_id([1])               # 1 block  (blockIdx)
    tx = T.thread_id([256])          # 256 threads per block (threadIdx)

    B[tx] = A[tx] * T.float32(2.0)

# compile for CUDA through the TIRx pipeline -> an Executable
exe = tvm.compile(tvm.IRModule({"main": scale}),
                  target=tvm.target.Target("cuda"), tir_pipeline="tirx")

dev = tvm.cuda(0)
a = tvm.runtime.tensor(np.random.rand(256).astype("float32"), device=dev)
b = tvm.runtime.tensor(np.zeros(256, "float32"), device=dev)
exe(a, b)                            # run
print(exe.mod.imports[0].inspect_source())   # the generated CUDA C
```

What the surrounding calls do:

- `tvm.IRModule({"main": scale})` wraps the `PrimFunc` into an *IRModule* — a
  named collection of functions; `"main"` is the entry point the compiler builds
  and the symbol you call.
- `tvm.compile(mod, target=..., tir_pipeline="tirx")` returns an **Executable**:
  the compiled host launcher plus device code, bundled together. You call it like
  a function (`exe(a, b)`); arguments are positional and match the kernel’s
  parameter order.
- `tvm.cuda(0)` is a handle to CUDA device 0; `tvm.runtime.tensor(arr,
  device=dev)` places data on that device as a TVM tensor (an `NDArray`).
- `exe.mod` is the underlying runtime `Module`; `exe.mod.imports[0]` is the
  imported device (CUDA) module, and its `.inspect_source()` returns the
  generated CUDA C.

For this kernel, that last `print` produces (boilerplate elided):

```c
extern "C" __global__ void __launch_bounds__(256)
scale_kernel(float* __restrict__ A_ptr, float* __restrict__ B_ptr) {
  int tx = ((int)threadIdx.x);
  B_ptr[tx] = A_ptr[tx] * 2.0f;
}
```

Every thread writes one element — a direct map from `B[tx] = A[tx] * 2.0` to the
generated indexing.

> **Note**
>
> The compiled `Executable` also accepts CUDA `torch` tensors **directly**
> (zero-copy, via DLPack) — no conversion step needed:
>
> @@BLOCK2@@

The following chapters expand each piece: [Defining a function](tirx_native_basics_cuda_functions.md), [Buffers and memory](tirx_native_basics_cuda_buffers.md),
[Control flow](tirx_native_basics_cuda_control_flow.md), [CUDA C++/PTX intrinsics](tirx_native_basics_cuda_threads_sync.md), and [Compiling and inspecting](tirx_native_basics_cuda_compiling.md).
