# {name}

{description}

`/tests/task.py` defines the target computation through `reference(...)`, including its semantics and output structure. Implement that computation for an NVIDIA {gpu}, satisfy the correctness contract below, and make it run as quickly as possible. You may combine eager PyTorch, ordinary Python, and **{dsl_display}**, but every call to `solve(...)` must execute at least one {dsl_display} kernel that performs a required step of the computation—computing or moving data—on which at least one returned output tensor depends. The {dsl_display} contribution may be arbitrarily small; its value is reflected in the measured performance.

Grading records two outcomes:

- **Passed (P)**: all correctness checks on fresh inputs pass and timing completes successfully. This outcome is reported under `passed`.
- **Speed**: conditional on P, the speedup over eager PyTorch computed from median execution times. The uncapped speedup is recorded; 1.0x is not a ceiling, and higher is better.

P is the precondition for speed; without P, no performance is scored.

## Deliverables

### kernel.py

- `/app/kernel.py` must define `solve(...)`. The grader calls it positionally with exactly the arguments produced by `make_inputs()`. It must return an output that matches the structure of `reference(...)` and satisfies the correctness contract: every output tensor must have the same shape, dtype, and device, and every output container, including nested tuples and lists, must have the same type, length, and nesting.
- The submission consists only of `/app/kernel.py`, which must be at most {kernel_byte_limit} bytes. It may use preinstalled packages and the original files under `/tests` only as permitted below, but it must not depend on any other file you create; those files will be absent during grading.
- `solve(...)` is called repeatedly on fresh, independently sampled inputs. A {dsl_display} kernel may be compiled on first use, and the compiled artifact may be cached and reused. Every call must nevertheless execute the required {dsl_display} kernel and recompute the outputs from that call's arguments.
- You may specialize on facts that are invariant across every legal draw (one input sample) from `make_inputs()`, including fixed shapes, dtypes, configuration constants, and index structures. A fact qualifies as invariant only if the source of `/tests/task.py` proves it. Any value that may vary across draws must be obtained at run time from the current call's arguments.
- You may cache compiled artifacts, static metadata, and output buffers. You must not cache, precompute, or reuse any numerical result that depends on a particular input.

### SKILL.md

Create and maintain `/app/SKILL.md` alongside the kernel. It records **kernel knowledge** only: evidence-backed conclusions about how this computation maps efficiently to the target hardware, written so that they can transfer to another kernel DSL. It is not a problem summary, a step-by-step account of the final implementation, or a benchmark log. At the end of the attempt, the file is collected with `kernel.py` and may be provided to a future attempt on the same problem using a different kernel DSL.

Record:

- conclusions supported by substantive experiments: what changed, how the observation compared with the expectation, and what causal explanation the evidence supports;
- which parts of the computation are or are not worth moving out of eager PyTorch, and the evidence supporting that boundary;
- why unsuccessful kernel approaches failed, the conditions under which existing conclusions hold, and the questions most worth testing next.

Every conclusion must rest on concrete evidence: a measured change in correctness, a time or speedup reported by the evaluator, or a specific observation about generated code or hardware behavior. Include DSL-specific details only when they are needed to explain the evidence; state the final conclusions without relying on the current DSL's syntax or APIs.

Do not include legality rules, code, commands, API instructions, compilation or debugging narratives, submission mechanics, or unsupported generic optimization advice. Update the file as you experiment rather than reconstructing it from memory at the end. Use concise Markdown paragraphs or lists. Do not add a title or heading; one is added externally when the file is used.

## Implementation boundary

A valid submission may combine eager PyTorch, ordinary Python, and {dsl_display} kernels. The {dsl_display} portion may be arbitrarily small and may be a pure data-movement step, but it must implement part of the target computation. Merely routing an already valid eager PyTorch result through {dsl_display} does not count: bypassing the {dsl_display} step and returning the eager result must not produce the same valid outputs. Subject to the restrictions below, all remaining computation may be implemented with eager PyTorch or ordinary Python. A submission without a working {dsl_display} contribution is invalid and receives a score of 0, even if it is otherwise correct and fast.

### Required

1. On every legal call to `solve(...)`, at least one returned output tensor must depend on a tensor value computed or moved by a {dsl_display} kernel launched during that call.
2. Compiling a kernel without launching it, running it only for some legal inputs, discarding its result, independently recomputing and overwriting its contribution, or routing an already valid eager result through a semantics-preserving copy, cast, or layout change does not satisfy the requirement. A later eager PyTorch operation may consume or transform the {dsl_display} result as long as the returned output still depends on it.

### Allowed

1. Eager PyTorch and ordinary Python may implement any remaining part of the computation, allocate storage, move and convert tensor values, and dispatch on metadata such as shapes and dtypes.
2. Tensor values may flow freely between eager PyTorch operations and {dsl_display} kernels.
3. You may use any number of {dsl_display} kernels, cache compiled artifacts and static scheduling information, and use {dsl_display}'s own operations, hardware intrinsics, tensor cores, asynchronous execution mechanisms, and mixed-precision capabilities.
4. The internal algorithms and precision may differ from those of `reference(...)`, provided that the final outputs satisfy the grading contract.

### Forbidden

1. `solve(...)` must not call `reference`, `make_inputs`, or any computational helper in `/tests/task.py`, and it must not reuse their numerical results either directly or indirectly. You may inspect those functions, reproduce their logic in `kernel.py`, and import genuinely static configuration constants.
2. Tensor computation may use only ordinary Python, ordinary eager PyTorch operations, and {dsl_display}. Do not invoke another kernel language or compiler, including `torch.compile`, `torch.jit`, Triton, or raw CUDA C/C++. None of these may be accessed through inline-source facilities, runtime extensions, subprocesses, or ctypes. Libraries that eager PyTorch invokes internally are allowed, as are operations and hardware intrinsics officially provided by {dsl_display}. Every {dsl_display} kernel must be your own code in `kernel.py`; importing kernels from {dsl_display} kernel libraries, including those vendored inside PyTorch, is forbidden.
3. CUDA Graphs are forbidden because they remove launch overhead that the baseline still incurs.
4. `solve(...)` must compute each output solely from the current call's arguments and permitted invariant metadata. It must not inspect or otherwise exploit evaluation seeds, call order, the timing phase, evaluator state, or results from earlier calls, and it must not modify or circumvent the testing or grading process.

## Grading contract

### Correctness

- `solve(...)` must pass the correctness checks on `CORRECTNESS_DRAWS` independently sampled inputs. The type, length, and nesting of every output container, and the shape, dtype, and device of every output tensor, must match `reference(...)` exactly.
- Every integer or Boolean output must equal, element for element, the corresponding output from `reference(...)` evaluated at the original input dtypes.
- For floating-point outputs, grading computes both a reference output at the original input dtypes and a high-precision anchor. To construct the anchor, each floating-point input tensor whose dtype is in `ANCHOR_DTYPES` is converted to float64 before `reference(...)` is evaluated. The maximum absolute error of your output relative to the anchor must not exceed `ALLOWANCE_FACTOR * max(reference_error, allowance_floor)`, where `reference_error` is the original reference output's maximum absolute error relative to the anchor and `allowance_floor` is defined precisely by `matches` in `/tests/contract.py`. An output more accurate than the original reference output always passes; only an output that exceeds this error budget fails.
- You may choose the kernel's internal algorithm and precision, including the use of tensor cores, TF32, and mixed precision; grading checks only the final outputs. For outputs graded by exact equality, and for computations involving quantization, rounding, or dtype conversion, mathematically equivalent expressions may still produce different discrete results. Check the exact order of operations in `reference(...)`.

### Speed

- Each timing draw begins by synchronizing the device. The grader then records a start event, invokes `solve(...)` once, and synchronizes the entire device to wait for all GPU work to finish. This end-to-end measurement includes Python dispatch within `solve(...)`, all kernel launches, and all GPU work issued by eager PyTorch and {dsl_display}.
- After `WARMUP_DRAWS` warmup calls, grading takes the median over `TIMING_DRAWS` fresh inputs. The eager PyTorch baseline evaluates `reference(...)` using the same warmup seeds, timing seeds, and measurement procedure.
- The eager baseline uses the inputs' original dtypes and the `reference_numerics` settings from `/tests/contract.py`. The float64 anchor is used only for correctness and never as a performance baseline.
- Both correctness and timing inputs use fresh, unpredictable seeds. Grading also re-verifies the outputs on randomly selected timing inputs. The exact procedures are implemented by `median_ms` and `baseline_ms` in `/tests/contract.py`.
- `passed: yes` means that correctness passed and timing completed. It does not by itself mean that the implementation is faster than eager; consult the reported speedup for that comparison.

## Environment

- Your development sandbox provides {sandbox_cpus} CPU cores and {sandbox_memory} of memory, but no GPU or network access. Grading runs in the same software image on one NVIDIA {gpu} (sm_100a), accessible only through the submission script below. Any preinstalled package importable here is available at the same version during grading.
- `/tests/task.py` is the complete problem specification. `reference(*inputs)` defines the semantics and output structure, while `make_inputs()` defines the arguments' shapes, dtypes, fixed structure, and random distributions. The implementation must handle every legal draw that `make_inputs()` can produce.
- `/tests/contract.py` is the complete grading contract. Editing files under `/tests` does not affect grading, which uses its own copies.
- `make_inputs()` allocates on CUDA and therefore cannot run in the development sandbox. For local checks, write a reduced CPU version of `make_inputs()` that preserves the dtypes and any dimensions asserted by `reference(...)`, then use `contract.matches_reference(actual, reference, cpu_make_inputs, seed)`. This validates your understanding of the semantics and output structure rather than the GPU kernel itself.
- The {dsl_display} documentation is available under `/docs`. Inspect the directory first, then search it for specific APIs, architecture questions, or errors. Imports, tracing, and generated-code inspection that do not require a GPU can run locally; specify sm_100a whenever a target architecture is required. {dsl_display} reads kernel source from disk, so define kernels in `.py` files rather than in heredocs or `-c` strings. Executing a {dsl_display} GPU kernel requires the {gpu}.

## Submissions and budget

```bash
bash /tests/test.sh
```

- The script snapshots `/app/kernel.py` at submission time and evaluates that snapshot on the {gpu}. Every submission that actually enters evaluation receives a report containing its import status, correctness result, and `passed` value. A failed correctness check also reports the first mismatch it finds: the output, what differs, and for floating-point outputs how far the error exceeds the allowance. Once correctness passes and timing completes, the report also includes the kernel and eager median times and the speedup. Exceptions include their tracebacks; other failures, such as timeouts, include an error message. Output printed by `solve(...)` is not part of a report; a timeout report ends with the tail of the child's stdout, and a crash report with the tail of its stderr. You may continue editing the file while waiting, but those edits do not affect the submitted snapshot.
- Evaluation of a submitted kernel runs in a child process. Import, correctness checks, warmup, kernel timing, and the random correctness re-checks during timing must collectively finish within `EVALUATION_TIMEOUT_SEC` seconds; otherwise, the process is terminated and the submission fails.
- You have **{n_submissions} feedback submissions** in total. If a submission never enters evaluation because the file is oversized, unreadable, or fails to parse, or because an infrastructure failure occurs, it is returned with an explanation and does not consume a submission. After all submissions have been used, no further feedback submissions are accepted; the final evaluation described below still runs.
- The total wall-clock budget is **{budget_text}**, including time spent waiting for reports. The shell prompt shows the minutes used and left in this attempt, and every report shows the remaining submissions and time.
- Submissions provide feedback only. When the budget expires or you finish early, the version of `/app/kernel.py` then present on disk is evaluated again on a fresh {gpu}; that evaluation determines the final score. There is no bonus for finishing early, so remaining budget is best spent on further optimization. Keep a backup of the best-known working version, and ensure that `/app/kernel.py` contains it before you finish or the budget expires.

## Suggested workflow

1. Read `/tests/task.py` and `/tests/contract.py` first. Identify the inputs and outputs, the complete computational semantics, the invariant facts, the randomized components, and the correctness requirements for every output.
2. Decompose `reference(...)` into a dataflow and candidate kernels, and identify the likely bottleneck in light of the {gpu}'s hardware characteristics. Resolve uncertainties that would affect the design by consulting the documentation or by making one submission with an explicit hypothesis.
3. Make the first version valid from the outset: implement the computation with eager PyTorch and add the smallest genuine {dsl_display} kernel whose result contributes to a returned output. Establish importability, interface conformance, output structure, and numerical correctness first; then move more of the computation into {dsl_display} kernels and optimize.
4. Before submitting, run every check that does not require a GPU, including Python imports, DSL tracing, generated-code inspection, and any structural or plumbing checks that a reduced CPU input can support.
5. Submit a first candidate before a quarter of the budget has passed, even if it is a plain, unoptimized kernel: a correct slow kernel scores, an unfinished fast one does not. Then address failures in this order: import, structure, correctness, and performance. Change one key factor per experiment, and state the expected effect before submitting.
6. After each report, add only the newly acquired kernel knowledge and its evidence to `/app/SKILL.md`. Do not copy reports or narrate the submission process. Confirm that the `kernel.py` on disk remains the best known version.
7. During the last five minutes of the budget, stop high-risk experiments, perform the final checks, and retain the best version.
