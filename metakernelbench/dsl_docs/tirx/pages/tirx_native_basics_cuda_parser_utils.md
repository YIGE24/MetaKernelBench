<!-- source: https://tvm.apache.org/docs/tirx/native_basics/cuda/parser_utils.html -->

<a id="parser-utilities"></a>
# Parser utilities

A few helpers act at **parse time** (when TVMScript is turned into TIRx), letting
you inline Python-computed values, factor out reusable fragments, and bundle
parser-side state.

<a id="t-meta-var-inline-a-python-value"></a>
## T.meta_var — inline a Python value

`T.meta_var(x)` tells the parser to treat `x` — a value computed in **Python** —
as a compile-time *meta* value and inline it directly into the IR, rather than
parse it as a script variable. It avoids a throwaway local, and it drives
metaprogramming: a plain Python `for` over a meta value unrolls in the parser.

```python
n = T.meta_var(4)              # n is a Python int, inlined
for j in range(n):            # unrolled at parse time
    acc[0] = acc[0] + A[tx, j]
```

<a id="t-inline-inline-functions"></a>
## @T.inline — inline functions

`@T.inline` defines a function whose body is **inlined at each call site** during
parsing — no call appears in the generated code. It follows Python’s lexical (LEGB)
scoping with late binding, so a parameter shadows an enclosing variable:

```python
@T.inline
def add_into(acc, x):
    acc[0] = acc[0] + x

add_into(s.acc, A[tx, j])     # inlined -> s.acc[0] = s.acc[0] + A[tx, j]
```

<a id="t-meta-class-parser-side-state-objects"></a>
## @T.meta_class — parser-side state objects

`@T.meta_class` marks a plain Python class whose **instances are parser meta
values**: their fields can hold buffers and scalars, so you can bundle related
allocations and state into one object and use it in the kernel body.

```python
@T.meta_class
class State:
    def __init__(self, smem):
        self.acc = T.alloc_local([1], "float32")
        self.buf = T.decl_buffer([64], "float16", smem, scope="shared.dyn")

s = State(smem.data)
s.acc[0] = T.float32(0.0)     # use its fields like ordinary buffers
# ... s.buf[i] ...
```

This is handy for grouping a kernel’s pipeline state (barriers, accumulators,
scratch views) instead of threading many separate locals through the body.

<a id="t-constexpr"></a>
## T.constexpr

`T.constexpr` marks a compile-time kernel parameter, baked in by `@T.jit`’s
`.specialize(...)`. See [Defining a function](tirx_native_basics_cuda_functions.md) for the details.
