<!-- source: https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/pipeline.html -->

<a id="module-cutlass.pipeline"></a>
# cutlass.pipeline

<a id="cutlass.pipeline.Agent"></a>
### `cutlass.pipeline.Agent`

```python
class cutlass.pipeline.Agent(value)
```

Bases: `Enum`

Agent indicates what is participating in the pipeline synchronization.

<a id="cutlass.pipeline.Agent.Thread"></a>
#### `cutlass.pipeline.Agent.Thread`

```python
Thread = 1
```

<a id="cutlass.pipeline.Agent.Warp"></a>
#### `cutlass.pipeline.Agent.Warp`

```python
Warp = 2
```

<a id="cutlass.pipeline.Agent.ThreadBlock"></a>
#### `cutlass.pipeline.Agent.ThreadBlock`

```python
ThreadBlock = 3
```

<a id="cutlass.pipeline.Agent.ThreadBlockCluster"></a>
#### `cutlass.pipeline.Agent.ThreadBlockCluster`

```python
ThreadBlockCluster = 4
```

<a id="cutlass.pipeline.CooperativeGroup"></a>
### `cutlass.pipeline.CooperativeGroup`

```python
class cutlass.pipeline.CooperativeGroup( agent: Agent, size: int | cutlass.cutlass_dsl.Int32 = 1, alignment: int | None = None, )
```

Bases: `object`

CooperativeGroup contains size restrictions for an Agent.

<a id="cutlass.pipeline.CooperativeGroup.__init__"></a>
#### `cutlass.pipeline.CooperativeGroup.__init__`

```python
__init__( agent: Agent, size: int | cutlass.cutlass_dsl.Int32 = 1, alignment: int | None = None, )
```

<a id="cutlass.pipeline.PipelineOp"></a>
### `cutlass.pipeline.PipelineOp`

```python
class cutlass.pipeline.PipelineOp(value)
```

Bases: `Enum`

PipelineOp assigns an operation to an agent corresponding to a specific hardware feature.

<a id="cutlass.pipeline.PipelineOp.AsyncThread"></a>
#### `cutlass.pipeline.PipelineOp.AsyncThread`

```python
AsyncThread = 1
```

<a id="cutlass.pipeline.PipelineOp.TCGen05Mma"></a>
#### `cutlass.pipeline.PipelineOp.TCGen05Mma`

```python
TCGen05Mma = 2
```

<a id="cutlass.pipeline.PipelineOp.TmaLoad"></a>
#### `cutlass.pipeline.PipelineOp.TmaLoad`

```python
TmaLoad = 3
```

<a id="cutlass.pipeline.PipelineOp.ClcLoad"></a>
#### `cutlass.pipeline.PipelineOp.ClcLoad`

```python
ClcLoad = 4
```

<a id="cutlass.pipeline.PipelineOp.TmaStore"></a>
#### `cutlass.pipeline.PipelineOp.TmaStore`

```python
TmaStore = 5
```

<a id="cutlass.pipeline.PipelineOp.Composite"></a>
#### `cutlass.pipeline.PipelineOp.Composite`

```python
Composite = 6
```

<a id="cutlass.pipeline.PipelineOp.AsyncLoad"></a>
#### `cutlass.pipeline.PipelineOp.AsyncLoad`

```python
AsyncLoad = 7
```

<a id="cutlass.pipeline.SyncObject"></a>
### `cutlass.pipeline.SyncObject`

```python
class cutlass.pipeline.SyncObject
```

Bases: `ABC`

Abstract base class for hardware synchronization primitives.

This class defines the interface for different types of hardware synchronization
mechanisms including shared memory barriers, named barriers, and fences.

<a id="cutlass.pipeline.SyncObject.arrive"></a>
#### `cutlass.pipeline.SyncObject.arrive`

```python
abstract arrive(*args: Any, **kwargs: Any) → None
```

<a id="cutlass.pipeline.SyncObject.wait"></a>
#### `cutlass.pipeline.SyncObject.wait`

```python
abstract wait(*args: Any, **kwargs: Any) → None
```

<a id="cutlass.pipeline.SyncObject.arrive_and_wait"></a>
#### `cutlass.pipeline.SyncObject.arrive_and_wait`

```python
abstract arrive_and_wait() → None
```

<a id="cutlass.pipeline.SyncObject.arrive_and_drop"></a>
#### `cutlass.pipeline.SyncObject.arrive_and_drop`

```python
abstract arrive_and_drop() → None
```

<a id="cutlass.pipeline.SyncObject.get_barrier"></a>
#### `cutlass.pipeline.SyncObject.get_barrier`

```python
abstract get_barrier() → cutlass.cute.typing.Pointer | int | None
```

<a id="cutlass.pipeline.SyncObject.max"></a>
#### `cutlass.pipeline.SyncObject.max`

```python
abstract max() → int | None
```

<a id="cutlass.pipeline.SyncObject._abc_impl"></a>
#### `cutlass.pipeline.SyncObject._abc_impl`

```python
_abc_impl = <_abc._abc_data object>
```

<a id="cutlass.pipeline.MbarrierLayout"></a>
### `cutlass.pipeline.MbarrierLayout`

```python
class cutlass.pipeline.MbarrierLayout(value)
```

Bases: `Enum`

Layout of mbarrier used for synchronization.

<a id="cutlass.pipeline.MbarrierLayout.V0"></a>
#### `cutlass.pipeline.MbarrierLayout.V0`

```python
V0 = 1
```

<a id="cutlass.pipeline.MbarrierArray"></a>
### `cutlass.pipeline.MbarrierArray`

```python
class cutlass.pipeline.MbarrierArray
```

Bases: [`SyncObject`](#cutlass.pipeline.SyncObject)

MbarrierArray implements an abstraction for an array of smem barriers.

<a id="cutlass.pipeline.MbarrierArray.__init__"></a>
#### `cutlass.pipeline.MbarrierArray.__init__`

```python
__init__( barrier_storage: cutlass.cute.typing.Pointer, num_stages: int, agent: tuple[PipelineOp, CooperativeGroup], tx_count: int = 0, mbarrier_layout: MbarrierLayout = MbarrierLayout.V0, name: str = '', *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

<a id="cutlass.pipeline.MbarrierArray.recast_to_new_op_type"></a>
#### `cutlass.pipeline.MbarrierArray.recast_to_new_op_type`

```python
recast_to_new_op_type( new_op_type: PipelineOp, ) → MbarrierArray
```

Creates a copy of MbarrierArray with a different op\_type without re-initializing barriers

<a id="cutlass.pipeline.MbarrierArray._mbar_scope"></a>
#### `cutlass.pipeline.MbarrierArray._mbar_scope`

```python
_mbar_scope(op: str) → Any
```

Return a Scope context manager for barrier identification.

Format: `name:op` (e.g. `smem_kv:wait`, `tmem_sp0:arrive`).
Profiling tools group by the `name` prefix and classify by the `op` suffix.

Usage:

```console
with self._mbar_scope("wait"):
    cute.arch.mbarrier_wait(...)
```

<a id="cutlass.pipeline.MbarrierArray.mbarrier_init"></a>
#### `cutlass.pipeline.MbarrierArray.mbarrier_init`

```python
mbarrier_init( *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

Initializes an array of mbarriers using warp 0.

<a id="cutlass.pipeline.MbarrierArray.arrive"></a>
#### `cutlass.pipeline.MbarrierArray.arrive`

```python
arrive( index: int, dst: int, cta_group: CtaGroup | None = None, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

Select the arrive corresponding to this MbarrierArray’s PipelineOp.

**Parameters:**

- **index** (*int*) – Index of the mbarrier in the array to arrive on
- **dst** (*int* | *None*) – Destination parameter for selective arrival, which can be either a mask or destination cta rank.
  When None, both `TCGen05Mma` and `AsyncThread` will arrive on their local mbarrier.
  - For `TCGen05Mma`, `dst` serves as a multicast mask (e.g., 0b1011 allows arrive signal to be multicast to CTAs
  in the cluster with rank = 0, 1, and 3).
  - For `AsyncThread`, `dst` serves as a destination cta rank (e.g., 3 means threads will arrive on
  the mbarrier with rank = 3 in the cluster).
- **cta\_group** (`cute.nvgpu.tcgen05.CtaGroup`, optional) – CTA group for `TCGen05Mma`, defaults to None for other op types

<a id="cutlass.pipeline.MbarrierArray.arrive_mbarrier"></a>
#### `cutlass.pipeline.MbarrierArray.arrive_mbarrier`

```python
arrive_mbarrier( index: int, dst_rank: int | None = None, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

<a id="cutlass.pipeline.MbarrierArray.arrive_cp_async_mbarrier"></a>
#### `cutlass.pipeline.MbarrierArray.arrive_cp_async_mbarrier`

```python
arrive_cp_async_mbarrier( index: int, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

<a id="cutlass.pipeline.MbarrierArray.arrive_tcgen05mma"></a>
#### `cutlass.pipeline.MbarrierArray.arrive_tcgen05mma`

```python
arrive_tcgen05mma( index: int, mask: int | None, cta_group: CtaGroup, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

<a id="cutlass.pipeline.MbarrierArray.arrive_and_expect_tx"></a>
#### `cutlass.pipeline.MbarrierArray.arrive_and_expect_tx`

```python
arrive_and_expect_tx( index: int, tx_count: int, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

<a id="cutlass.pipeline.MbarrierArray.arrive_and_expect_tx_with_dst"></a>
#### `cutlass.pipeline.MbarrierArray.arrive_and_expect_tx_with_dst`

```python
arrive_and_expect_tx_with_dst( index: int, tx_count: int, dst: int | None = None, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

<a id="cutlass.pipeline.MbarrierArray.try_wait"></a>
#### `cutlass.pipeline.MbarrierArray.try_wait`

```python
try_wait( index: int, phase: int, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cutlass_dsl.Boolean
```

<a id="cutlass.pipeline.MbarrierArray.test_wait"></a>
#### `cutlass.pipeline.MbarrierArray.test_wait`

```python
test_wait( index: int, phase: int, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cutlass_dsl.Boolean
```

<a id="cutlass.pipeline.MbarrierArray.wait"></a>
#### `cutlass.pipeline.MbarrierArray.wait`

```python
wait( index: int, phase: int, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

Wait on mbarrier.

**Parameters:**

- **index** – Index of the mbarrier in the array
- **phase** – Phase/parity to wait for (0 or 1)

<a id="cutlass.pipeline.MbarrierArray.arrive_and_wait"></a>
#### `cutlass.pipeline.MbarrierArray.arrive_and_wait`

```python
arrive_and_wait( index: int, phase: int, dst: int, cta_group: CtaGroup | None = None, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

<a id="cutlass.pipeline.MbarrierArray.arrive_and_drop"></a>
#### `cutlass.pipeline.MbarrierArray.arrive_and_drop`

```python
arrive_and_drop( *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

<a id="cutlass.pipeline.MbarrierArray.get_barrier"></a>
#### `cutlass.pipeline.MbarrierArray.get_barrier`

```python
get_barrier( index: int, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Pointer
```

<a id="cutlass.pipeline.MbarrierArray.max"></a>
#### `cutlass.pipeline.MbarrierArray.max`

```python
max() → int
```

<a id="cutlass.pipeline.MbarrierArray._abc_impl"></a>
#### `cutlass.pipeline.MbarrierArray._abc_impl`

```python
_abc_impl = <_abc._abc_data object>
```

<a id="cutlass.pipeline.NamedBarrier"></a>
### `cutlass.pipeline.NamedBarrier`

```python
class cutlass.pipeline.NamedBarrier( barrier_id: int | cutlass.cutlass_dsl.Int32, num_threads: int | cutlass.cutlass_dsl.Int32, )
```

Bases: [`SyncObject`](#cutlass.pipeline.SyncObject)

NamedBarrier is an abstraction for named barriers managed by hardware.
There are 16 named barriers available, with barrier\_ids 0-15.

See the PTX documentation.

<a id="cutlass.pipeline.NamedBarrier.barrier_id"></a>
#### `cutlass.pipeline.NamedBarrier.barrier_id`

```python
barrier_id: int | cutlass.cutlass_dsl.Int32
```

<a id="cutlass.pipeline.NamedBarrier.num_threads"></a>
#### `cutlass.pipeline.NamedBarrier.num_threads`

```python
num_threads: int | cutlass.cutlass_dsl.Int32
```

<a id="cutlass.pipeline.NamedBarrier.arrive"></a>
#### `cutlass.pipeline.NamedBarrier.arrive`

```python
arrive( *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

The aligned flavor of arrive is used when all threads in the CTA will execute the
same instruction. See PTX documentation.

<a id="cutlass.pipeline.NamedBarrier.arrive_unaligned"></a>
#### `cutlass.pipeline.NamedBarrier.arrive_unaligned`

```python
arrive_unaligned( *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

The unaligned flavor of arrive can be used with an arbitrary number of threads in the CTA.

<a id="cutlass.pipeline.NamedBarrier.wait"></a>
#### `cutlass.pipeline.NamedBarrier.wait`

```python
wait( *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

NamedBarriers do not have a standalone wait like mbarriers, only an arrive\_and\_wait.
If synchronizing two warps in a producer/consumer pairing, the arrive count would be
32 using mbarriers but 64 using NamedBarriers. Only threads from either the producer
or consumer are counted for mbarriers, while all threads participating in the sync
are counted for NamedBarriers.

<a id="cutlass.pipeline.NamedBarrier.wait_unaligned"></a>
#### `cutlass.pipeline.NamedBarrier.wait_unaligned`

```python
wait_unaligned( *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

<a id="cutlass.pipeline.NamedBarrier.arrive_and_wait"></a>
#### `cutlass.pipeline.NamedBarrier.arrive_and_wait`

```python
arrive_and_wait( *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

<a id="cutlass.pipeline.NamedBarrier.arrive_and_drop"></a>
#### `cutlass.pipeline.NamedBarrier.arrive_and_drop`

```python
arrive_and_drop( *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

<a id="cutlass.pipeline.NamedBarrier.sync"></a>
#### `cutlass.pipeline.NamedBarrier.sync`

```python
sync( *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

<a id="cutlass.pipeline.NamedBarrier.get_barrier"></a>
#### `cutlass.pipeline.NamedBarrier.get_barrier`

```python
get_barrier( *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → int | cutlass.cutlass_dsl.Int32
```

<a id="cutlass.pipeline.NamedBarrier.max"></a>
#### `cutlass.pipeline.NamedBarrier.max`

```python
max() → int
```

<a id="cutlass.pipeline.NamedBarrier.__init__"></a>
#### `cutlass.pipeline.NamedBarrier.__init__`

```python
__init__( barrier_id: int | cutlass.cutlass_dsl.Int32, num_threads: int | cutlass.cutlass_dsl.Int32, ) → None
```

<a id="cutlass.pipeline.NamedBarrier._abc_impl"></a>
#### `cutlass.pipeline.NamedBarrier._abc_impl`

```python
_abc_impl = <_abc._abc_data object>
```

<a id="cutlass.pipeline.PipelineOrder"></a>
### `cutlass.pipeline.PipelineOrder`

```python
class cutlass.pipeline.PipelineOrder( sync_object_full: SyncObject, depth: int, length: int, group_id: int, state: PipelineState, )
```

Bases: `object`

PipelineOrder is used for managing ordered pipeline execution with multiple groups.

This class implements a pipeline ordering mechanism where work is divided into groups
and stages, allowing for controlled progression through pipeline stages with proper
synchronization between different groups.

The pipeline ordering works as follows:
- The pipeline is divided into ‘length’ number of groups
- Each group has ‘depth’ number of stages
- Groups execute in a specific order with synchronization barriers
- Each group waits for the previous group to complete before proceeding

**Example:**

```python
# Create pipeline order with 3 groups, each with 2 stages
pipeline_order = PipelineOrder.create(
    barrier_storage=smem_ptr,      # shared memory pointer for barriers
    depth=2,                       # 2 stages per group
    length=3,                      # 3 groups total
    group_id=0,                    # current group ID (0, 1, or 2)
    producer_group=producer_warp   # cooperative group for producers
)

# In the pipeline loop
for stage in range(num_stages):
    pipeline_order.wait()          # Wait for previous group to complete
    # Process current stage
    pipeline_order.arrive()        # Signal completion to next group
```

<a id="cutlass.pipeline.PipelineOrder.sync_object_full"></a>
#### `cutlass.pipeline.PipelineOrder.sync_object_full`

```python
sync_object_full: SyncObject
```

<a id="cutlass.pipeline.PipelineOrder.depth"></a>
#### `cutlass.pipeline.PipelineOrder.depth`

```python
depth: int
```

<a id="cutlass.pipeline.PipelineOrder.length"></a>
#### `cutlass.pipeline.PipelineOrder.length`

```python
length: int
```

<a id="cutlass.pipeline.PipelineOrder.group_id"></a>
#### `cutlass.pipeline.PipelineOrder.group_id`

```python
group_id: int
```

<a id="cutlass.pipeline.PipelineOrder.state"></a>
#### `cutlass.pipeline.PipelineOrder.state`

```python
state: PipelineState
```

<a id="cutlass.pipeline.PipelineOrder.create"></a>
#### `cutlass.pipeline.PipelineOrder.create`

```python
static create( *, depth: int, length: int, group_id: int, producer_group: CooperativeGroup, barrier_storage: cutlass.cute.typing.Pointer | None = None, defer_sync: bool = False, name: str = '', ) → PipelineOrder
```

<a id="cutlass.pipeline.PipelineOrder.get_barrier_for_current_stage_idx"></a>
#### `cutlass.pipeline.PipelineOrder.get_barrier_for_current_stage_idx`

```python
get_barrier_for_current_stage_idx( group_id: int, state: PipelineState | None = None, ) → cutlass.cutlass_dsl.Int32
```

<a id="cutlass.pipeline.PipelineOrder.arrive"></a>
#### `cutlass.pipeline.PipelineOrder.arrive`

```python
arrive( state: PipelineState | None = None, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → PipelineState | None
```

<a id="cutlass.pipeline.PipelineOrder.wait"></a>
#### `cutlass.pipeline.PipelineOrder.wait`

```python
wait( state: PipelineState | None = None, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

<a id="cutlass.pipeline.PipelineOrder.__init__"></a>
#### `cutlass.pipeline.PipelineOrder.__init__`

```python
__init__( sync_object_full: SyncObject, depth: int, length: int, group_id: int, state: PipelineState, ) → None
```

<a id="cutlass.pipeline.TmaStoreFence"></a>
### `cutlass.pipeline.TmaStoreFence`

```python
class cutlass.pipeline.TmaStoreFence(num_stages: int = 0)
```

Bases: [`SyncObject`](#cutlass.pipeline.SyncObject)

TmaStoreFence is used for a multi-stage epilogue buffer.

<a id="cutlass.pipeline.TmaStoreFence.__init__"></a>
#### `cutlass.pipeline.TmaStoreFence.__init__`

```python
__init__(num_stages: int = 0) → None
```

<a id="cutlass.pipeline.TmaStoreFence.arrive"></a>
#### `cutlass.pipeline.TmaStoreFence.arrive`

```python
arrive( *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

<a id="cutlass.pipeline.TmaStoreFence.wait"></a>
#### `cutlass.pipeline.TmaStoreFence.wait`

```python
wait( *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

<a id="cutlass.pipeline.TmaStoreFence.arrive_and_wait"></a>
#### `cutlass.pipeline.TmaStoreFence.arrive_and_wait`

```python
arrive_and_wait( *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

<a id="cutlass.pipeline.TmaStoreFence.arrive_and_drop"></a>
#### `cutlass.pipeline.TmaStoreFence.arrive_and_drop`

```python
arrive_and_drop( *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

<a id="cutlass.pipeline.TmaStoreFence.get_barrier"></a>
#### `cutlass.pipeline.TmaStoreFence.get_barrier`

```python
get_barrier( *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

<a id="cutlass.pipeline.TmaStoreFence.max"></a>
#### `cutlass.pipeline.TmaStoreFence.max`

```python
max() → None
```

<a id="cutlass.pipeline.TmaStoreFence.tail"></a>
#### `cutlass.pipeline.TmaStoreFence.tail`

```python
tail( *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

<a id="cutlass.pipeline.TmaStoreFence._abc_impl"></a>
#### `cutlass.pipeline.TmaStoreFence._abc_impl`

```python
_abc_impl = <_abc._abc_data object>
```

<a id="cutlass.pipeline.PipelineUserType"></a>
### `cutlass.pipeline.PipelineUserType`

```python
class cutlass.pipeline.PipelineUserType(value)
```

Bases: `Enum`

An enumeration.

<a id="cutlass.pipeline.PipelineUserType.Producer"></a>
#### `cutlass.pipeline.PipelineUserType.Producer`

```python
Producer = 1
```

<a id="cutlass.pipeline.PipelineUserType.Consumer"></a>
#### `cutlass.pipeline.PipelineUserType.Consumer`

```python
Consumer = 2
```

<a id="cutlass.pipeline.PipelineUserType.ProducerConsumer"></a>
#### `cutlass.pipeline.PipelineUserType.ProducerConsumer`

```python
ProducerConsumer = 3
```

<a id="cutlass.pipeline.PipelineState"></a>
### `cutlass.pipeline.PipelineState`

```python
class cutlass.pipeline.PipelineState( stages: int, count: cutlass.cutlass_dsl.Int32, index: cutlass.cutlass_dsl.Int32, phase: cutlass.cutlass_dsl.Int32, )
```

Bases: `object`

Pipeline state contains an index and phase bit corresponding to the current position in the circular buffer.

<a id="cutlass.pipeline.PipelineState.__init__"></a>
#### `cutlass.pipeline.PipelineState.__init__`

```python
__init__( stages: int, count: cutlass.cutlass_dsl.Int32, index: cutlass.cutlass_dsl.Int32, phase: cutlass.cutlass_dsl.Int32, )
```

<a id="cutlass.pipeline.PipelineState.index"></a>
#### `cutlass.pipeline.PipelineState.index`

```python
property index
```

<a id="cutlass.pipeline.PipelineState.count"></a>
#### `cutlass.pipeline.PipelineState.count`

```python
property count
```

<a id="cutlass.pipeline.PipelineState.stages"></a>
#### `cutlass.pipeline.PipelineState.stages`

```python
property stages: int
```

<a id="cutlass.pipeline.PipelineState.phase"></a>
#### `cutlass.pipeline.PipelineState.phase`

```python
property phase
```

<a id="cutlass.pipeline.PipelineAsync"></a>
### `cutlass.pipeline.PipelineAsync`

```python
class cutlass.pipeline.PipelineAsync( sync_object_full: SyncObject, sync_object_empty: SyncObject, num_stages: int, producer_mask: cutlass.cutlass_dsl.Int32 | None, consumer_mask: cutlass.cutlass_dsl.Int32 | None, )
```

Bases: `object`

PipelineAsync is a generic pipeline class where both the producer and consumer are
AsyncThreads. It also serves as a base class for specialized pipeline classes.

This class implements a producer-consumer pipeline pattern where both sides operate
asynchronously. The pipeline maintains synchronization state using barrier objects
to coordinate between producer and consumer threads.

The pipeline state transitions of one pipeline entry(mbarrier) can be represented as:

Table 5 Pipeline State Transitions

| Barrier | State | p.acquire | p.commit | c.wait | c.release |
| --- | --- | --- | --- | --- | --- |
| empty\_bar | empty | <Return> | n/a | n/a |  |
| empty\_bar | wait | <Block> | n/a | n/a | -> empty |
| full\_bar | wait | n/a | -> full | <Block > | n/a |
| full\_bar | full | n/a |  | <Return> | n/a |

Where:

- p: producer
- c: consumer
- <Block>: This action is blocked until transition to a state allow it to proceed by other side
  - e.g. `p.acquire()` is blocked until `empty_bar` transition to `empty` state by `c.release()`

```text
Array of mbarriers as circular buffer:

     Advance Direction
   <-------------------

    Producer   Consumer
        |         ^
        V         |
   +-----------------+
 --|X|X|W|D|D|D|D|R|X|<-.
/  +-----------------+   \
|                        |
`------------------------'
```

Where:

- X: Empty buffer (initial state)
- W: Producer writing (producer is waiting for buffer to be empty)
- D: Data ready (producer has written data to buffer)
- R: Consumer reading (consumer is consuming data from buffer)

**Example:**

```python
# Create pipeline with 5 stages
pipeline = PipelineAsync.create(
    num_stages=5,                   # number of pipeline stages
    producer_group=producer_warp,
    consumer_group=consumer_warp
    barrier_storage=smem_ptr,       # smem pointer for array of mbarriers in shared memory
)

producer, consumer = pipeline.make_participants()
# Producer side
for i in range(num_iterations):
    handle = producer.acquire_and_advance()  # Wait for buffer to be empty & Move index to next stage
    # Write data to pipeline buffer
    handle.commit()   # Signal buffer is full

# Consumer side
for i in range(num_iterations):
    handle = consumer.wait_and_advance()     # Wait for buffer to be full & Move index to next stage
    # Read data from pipeline buffer
    handle.release()  # Signal buffer is empty
```

<a id="cutlass.pipeline.PipelineAsync.sync_object_full"></a>
#### `cutlass.pipeline.PipelineAsync.sync_object_full`

```python
sync_object_full: SyncObject
```

<a id="cutlass.pipeline.PipelineAsync.sync_object_empty"></a>
#### `cutlass.pipeline.PipelineAsync.sync_object_empty`

```python
sync_object_empty: SyncObject
```

<a id="cutlass.pipeline.PipelineAsync.num_stages"></a>
#### `cutlass.pipeline.PipelineAsync.num_stages`

```python
num_stages: int
```

<a id="cutlass.pipeline.PipelineAsync.producer_mask"></a>
#### `cutlass.pipeline.PipelineAsync.producer_mask`

```python
producer_mask: cutlass.cutlass_dsl.Int32 | None
```

<a id="cutlass.pipeline.PipelineAsync.consumer_mask"></a>
#### `cutlass.pipeline.PipelineAsync.consumer_mask`

```python
consumer_mask: cutlass.cutlass_dsl.Int32 | None
```

<a id="cutlass.pipeline.PipelineAsync._make_sync_object"></a>
#### `cutlass.pipeline.PipelineAsync._make_sync_object`

```python
static _make_sync_object( barrier_storage: cutlass.cute.typing.Pointer, num_stages: int, agent: tuple[PipelineOp, CooperativeGroup], tx_count: int = 0, name: str = '', phase: Literal['', 'full', 'empty'] = '', ) → SyncObject
```

Returns a SyncObject corresponding to an agent’s PipelineOp.

<a id="cutlass.pipeline.PipelineAsync.create"></a>
#### `cutlass.pipeline.PipelineAsync.create`

```python
static create( *, num_stages: int, producer_group: CooperativeGroup, consumer_group: CooperativeGroup, barrier_storage: cutlass.cute.typing.Pointer | None = None, producer_mask: cutlass.cutlass_dsl.Int32 | None = None, consumer_mask: cutlass.cutlass_dsl.Int32 | None = None, defer_sync: bool = False, name: str = '', ) → PipelineAsync
```

Creates and initializes a new PipelineAsync instance.

This helper function computes necessary attributes and returns an instance of PipelineAsync
with the specified configuration for producer and consumer synchronization.

**Parameters:**

- **barrier\_storage** (*cute.Pointer*) – Pointer to the shared memory address for this pipeline’s mbarriers
- **num\_stages** (*int*) – Number of buffer stages for this pipeline
- **producer\_group** ([*CooperativeGroup*](#cutlass.pipeline.CooperativeGroup)) – `CooperativeGroup` for the producer agent
- **consumer\_group** ([*CooperativeGroup*](#cutlass.pipeline.CooperativeGroup)) – `CooperativeGroup` for the consumer agent
- **producer\_mask** (*Int32*, *optional*) – Mask for signaling arrives for the producer agent
- **consumer\_mask** (*Int32*, *optional*) – Mask for signaling arrives for the consumer agent

**Raises:**

**ValueError** – If barrier\_storage is not a cute.Pointer instance

**Returns:**

A new `PipelineAsync` instance

**Return type:**

[PipelineAsync](#cutlass.pipeline.PipelineAsync)

<a id="cutlass.pipeline.PipelineAsync.producer_acquire"></a>
#### `cutlass.pipeline.PipelineAsync.producer_acquire`

```python
producer_acquire( state: PipelineState, try_acquire_token: cutlass.cutlass_dsl.Boolean | None = None, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

<a id="cutlass.pipeline.PipelineAsync.producer_try_acquire"></a>
#### `cutlass.pipeline.PipelineAsync.producer_try_acquire`

```python
producer_try_acquire( state: PipelineState, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cutlass_dsl.Boolean
```

<a id="cutlass.pipeline.PipelineAsync.producer_commit"></a>
#### `cutlass.pipeline.PipelineAsync.producer_commit`

```python
producer_commit( state: PipelineState, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

<a id="cutlass.pipeline.PipelineAsync.consumer_wait"></a>
#### `cutlass.pipeline.PipelineAsync.consumer_wait`

```python
consumer_wait( state: PipelineState, try_wait_token: cutlass.cutlass_dsl.Boolean | None = None, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

<a id="cutlass.pipeline.PipelineAsync.consumer_try_wait"></a>
#### `cutlass.pipeline.PipelineAsync.consumer_try_wait`

```python
consumer_try_wait( state: PipelineState, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cutlass_dsl.Boolean
```

<a id="cutlass.pipeline.PipelineAsync.consumer_release"></a>
#### `cutlass.pipeline.PipelineAsync.consumer_release`

```python
consumer_release( state: PipelineState, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

<a id="cutlass.pipeline.PipelineAsync.consumer_get_barrier"></a>
#### `cutlass.pipeline.PipelineAsync.consumer_get_barrier`

```python
consumer_get_barrier( state: PipelineState, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Pointer
```

<a id="cutlass.pipeline.PipelineAsync.producer_get_barrier"></a>
#### `cutlass.pipeline.PipelineAsync.producer_get_barrier`

```python
producer_get_barrier( state: PipelineState, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Pointer
```

<a id="cutlass.pipeline.PipelineAsync.producer_tail"></a>
#### `cutlass.pipeline.PipelineAsync.producer_tail`

```python
producer_tail( state: PipelineState, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

Make sure the last used buffer empty signal is visible to producer.
Producer tail is usually executed by producer before exit, to avoid dangling
mbarrier arrive signals after kernel exit.

**Parameters:**

**state** ([*PipelineState*](#cutlass.pipeline.PipelineState)) – The pipeline state that points to next useful buffer

<a id="cutlass.pipeline.PipelineAsync.make_producer"></a>
#### `cutlass.pipeline.PipelineAsync.make_producer`

```python
make_producer( *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → PipelineProducer
```

<a id="cutlass.pipeline.PipelineAsync.make_consumer"></a>
#### `cutlass.pipeline.PipelineAsync.make_consumer`

```python
make_consumer( *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → PipelineConsumer
```

<a id="cutlass.pipeline.PipelineAsync.make_participants"></a>
#### `cutlass.pipeline.PipelineAsync.make_participants`

```python
make_participants( *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → tuple[PipelineProducer, PipelineConsumer]
```

<a id="cutlass.pipeline.PipelineAsync.__init__"></a>
#### `cutlass.pipeline.PipelineAsync.__init__`

```python
__init__( sync_object_full: SyncObject, sync_object_empty: SyncObject, num_stages: int, producer_mask: cutlass.cutlass_dsl.Int32 | None, consumer_mask: cutlass.cutlass_dsl.Int32 | None, ) → None
```

<a id="cutlass.pipeline.PipelineCpAsync"></a>
### `cutlass.pipeline.PipelineCpAsync`

```python
class cutlass.pipeline.PipelineCpAsync( sync_object_full: SyncObject, sync_object_empty: SyncObject, num_stages: int, producer_mask: cutlass.cutlass_dsl.Int32 | None, consumer_mask: cutlass.cutlass_dsl.Int32 | None, )
```

Bases: [`PipelineAsync`](#cutlass.pipeline.PipelineAsync)

PipelineCpAsync is used for CpAsync producers and AsyncThread consumers (e.g. Hopper load mainloops).

<a id="cutlass.pipeline.PipelineCpAsync.create"></a>
#### `cutlass.pipeline.PipelineCpAsync.create`

```python
static create( *, barrier_storage: cutlass.cute.typing.Pointer, num_stages: cutlass.cutlass_dsl.Int32, producer_group: CooperativeGroup, consumer_group: CooperativeGroup, producer_mask: cutlass.cutlass_dsl.Int32 | None = None, consumer_mask: cutlass.cutlass_dsl.Int32 | None = None, defer_sync: bool = False, name: str = '', ) → PipelineCpAsync
```

Helper function that computes necessary attributes and returns a `PipelineCpAsync` instance.

**Parameters:**

- **barrier\_storage** (*cute.Pointer*) – Pointer to the shared memory address for this pipeline’s mbarriers
- **num\_stages** (*Int32*) – Number of buffer stages for this pipeline
- **producer\_group** ([*CooperativeGroup*](#cutlass.pipeline.CooperativeGroup)) – `CooperativeGroup` for the producer agent
- **consumer\_group** ([*CooperativeGroup*](#cutlass.pipeline.CooperativeGroup)) – `CooperativeGroup` for the consumer agent
- **producer\_mask** (*Int32*, *optional*) – Mask for signaling arrives for the producer agent, defaults to None
- **consumer\_mask** (*Int32*, *optional*) – Mask for signaling arrives for the consumer agent, defaults to None

**Returns:**

A new `PipelineCpAsync` instance configured with the provided parameters

**Return type:**

[PipelineCpAsync](#cutlass.pipeline.PipelineCpAsync)

<a id="cutlass.pipeline.PipelineCpAsync.__init__"></a>
#### `cutlass.pipeline.PipelineCpAsync.__init__`

```python
__init__( sync_object_full: SyncObject, sync_object_empty: SyncObject, num_stages: int, producer_mask: cutlass.cutlass_dsl.Int32 | None, consumer_mask: cutlass.cutlass_dsl.Int32 | None, ) → None
```

<a id="cutlass.pipeline.PipelineTmaAsync"></a>
### `cutlass.pipeline.PipelineTmaAsync`

```python
class cutlass.pipeline.PipelineTmaAsync( sync_object_full: SyncObject, sync_object_empty: SyncObject, num_stages: int, producer_mask: cutlass.cutlass_dsl.Int32 | None, consumer_mask: cutlass.cutlass_dsl.Int32 | None, is_signaling_thread: cutlass.cutlass_dsl.Boolean, )
```

Bases: [`PipelineAsync`](#cutlass.pipeline.PipelineAsync)

PipelineTmaAsync is used for TMA producers and AsyncThread consumers (e.g. Hopper mainloops).

<a id="cutlass.pipeline.PipelineTmaAsync.is_signaling_thread"></a>
#### `cutlass.pipeline.PipelineTmaAsync.is_signaling_thread`

```python
is_signaling_thread: cutlass.cutlass_dsl.Boolean
```

<a id="cutlass.pipeline.PipelineTmaAsync.init_empty_barrier_arrive_signal"></a>
#### `cutlass.pipeline.PipelineTmaAsync.init_empty_barrier_arrive_signal`

```python
static init_empty_barrier_arrive_signal( cta_layout_vmnk: cutlass.cute.typing.Layout, tidx: cutlass.cutlass_dsl.Int32, mcast_mode_mn: tuple[int, int] = (1, 1), ) → tuple[cutlass.cutlass_dsl.Int32, cutlass.cutlass_dsl.Boolean]
```

Initialize the empty barrier arrive signal.

This function determines which threads should signal empty barrier arrives based on the cluster layout
and multicast modes. It returns the destination CTA rank and whether the current thread should signal.

**Parameters:**

- **cta\_layout\_vmnk** (*cute.Layout*) – Layout describing the cluster shape and CTA arrangement
- **tidx** (*Int32*) – Thread index within the warp
- **mcast\_mode\_mn** (*tuple*[*int*, *int*]) – Tuple specifying multicast modes for m and n dimensions (each 0 or 1), defaults to (1,1)

**Raises:**

**AssertionError** – If both multicast modes are disabled (0,0)

**Returns:**

Tuple containing destination CTA rank and boolean indicating if current thread signals

**Return type:**

tuple[Int32, Boolean]

<a id="cutlass.pipeline.PipelineTmaAsync.create"></a>
#### `cutlass.pipeline.PipelineTmaAsync.create`

```python
static create( *, num_stages: int, producer_group: CooperativeGroup, consumer_group: CooperativeGroup, tx_count: int, barrier_storage: cutlass.cute.typing.Pointer | None = None, cta_layout_vmnk: cutlass.cute.typing.Layout | None = None, tidx: cutlass.cutlass_dsl.Int32 | None = None, mcast_mode_mn: tuple[int, int] = (1, 1), enable_multicast_signaling: bool = False, defer_sync: bool = False, name: str = '', ) → PipelineTmaAsync
```

Create a new `PipelineTmaAsync` instance.

**Parameters:**

- **num\_stages** (*int*) – Number of buffer stages for this pipeline
- **producer\_group** ([*CooperativeGroup*](#cutlass.pipeline.CooperativeGroup)) – `CooperativeGroup` for the producer agent
- **consumer\_group** ([*CooperativeGroup*](#cutlass.pipeline.CooperativeGroup)) – `CooperativeGroup` for the consumer agent
- **tx\_count** (*int*) – Number of bytes expected to be written to the transaction barrier for one stage
- **barrier\_storage** (*cute.Pointer*, *optional*) – Pointer to the shared memory address for this pipeline’s mbarriers, defaults to None
- **cta\_layout\_vmnk** (*cute.Layout*, *optional*) – Layout of the cluster shape, defaults to None
- **tidx** (*Int32*, *optional*) – Thread index to consumer async threads, defaults to None
- **mcast\_mode\_mn** (*tuple*[*int*, *int*], *optional*) – Tuple specifying multicast modes for m and n dimensions (each 0 or 1), defaults to (1,1)
- **enable\_multicast\_signaling** (*bool*, *optional*) – When `True`, the CooperativeGroup is expected
  to represent the number of threads in a CTA calling
  consumer\_wait/consumer\_release, and the actual arrive count is recomputed
  internally. Multicast is handled automatically based on cta\_layout\_vmnk and
  mcast\_mode\_mn. Defaults to `False`, which skips this logic and uses the
  consumer arrive count specified by the user.
- **defer\_sync** (*bool*, *optional*) – Bool specifying whether or not to skip the built-in mbarrier fence and sync for performance, defaults to False

**Raises:**

**ValueError** – If barrier\_storage is not a cute.Pointer instance

**Returns:**

New `PipelineTmaAsync` instance

**Return type:**

[PipelineTmaAsync](#cutlass.pipeline.PipelineTmaAsync)

<a id="cutlass.pipeline.PipelineTmaAsync.producer_acquire"></a>
#### `cutlass.pipeline.PipelineTmaAsync.producer_acquire`

```python
producer_acquire( state: PipelineState, try_acquire_token: cutlass.cutlass_dsl.Boolean | None = None, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

TMA producer commit conditionally waits on buffer empty and sets the transaction barrier.

<a id="cutlass.pipeline.PipelineTmaAsync.producer_commit"></a>
#### `cutlass.pipeline.PipelineTmaAsync.producer_commit`

```python
producer_commit( state: PipelineState, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

TMA producer commit is a noop since TMA instruction itself updates the transaction count.

<a id="cutlass.pipeline.PipelineTmaAsync.consumer_release"></a>
#### `cutlass.pipeline.PipelineTmaAsync.consumer_release`

```python
consumer_release( state: PipelineState, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

TMA consumer release conditionally signals the empty buffer to the producer.

<a id="cutlass.pipeline.PipelineTmaAsync.__init__"></a>
#### `cutlass.pipeline.PipelineTmaAsync.__init__`

```python
__init__( sync_object_full: SyncObject, sync_object_empty: SyncObject, num_stages: int, producer_mask: cutlass.cutlass_dsl.Int32 | None, consumer_mask: cutlass.cutlass_dsl.Int32 | None, is_signaling_thread: cutlass.cutlass_dsl.Boolean, ) → None
```

<a id="cutlass.pipeline.PipelineTmaUmma"></a>
### `cutlass.pipeline.PipelineTmaUmma`

```python
class cutlass.pipeline.PipelineTmaUmma( sync_object_full: SyncObject, sync_object_empty: SyncObject, num_stages: int, producer_mask: cutlass.cutlass_dsl.Int32 | None, consumer_mask: cutlass.cutlass_dsl.Int32 | None, is_leader_cta: bool, cta_group: CtaGroup, )
```

Bases: [`PipelineAsync`](#cutlass.pipeline.PipelineAsync)

PipelineTmaUmma is used for TMA producers and UMMA consumers (e.g. Blackwell mainloops).

<a id="cutlass.pipeline.PipelineTmaUmma.is_leader_cta"></a>
#### `cutlass.pipeline.PipelineTmaUmma.is_leader_cta`

```python
is_leader_cta: bool
```

<a id="cutlass.pipeline.PipelineTmaUmma.cta_group"></a>
#### `cutlass.pipeline.PipelineTmaUmma.cta_group`

```python
cta_group: CtaGroup
```

<a id="cutlass.pipeline.PipelineTmaUmma._make_sync_object"></a>
#### `cutlass.pipeline.PipelineTmaUmma._make_sync_object`

```python
_make_sync_object( barrier_storage: cutlass.cute.typing.Pointer, num_stages: int, agent: tuple[PipelineOp, CooperativeGroup], tx_count: int = 0, mbarrier_layout: MbarrierLayout = MbarrierLayout.V0, name: str = '', phase: Literal['', 'full', 'empty'] = '', *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → SyncObject
```

Returns a SyncObject corresponding to an agent’s PipelineOp.

<a id="cutlass.pipeline.PipelineTmaUmma._compute_mcast_arrival_mask"></a>
#### `cutlass.pipeline.PipelineTmaUmma._compute_mcast_arrival_mask`

```python
_compute_mcast_arrival_mask( cta_layout_vmnk: cutlass.cute.typing.Layout, mcast_mode_mn: tuple[int, int], *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cutlass_dsl.Int32
```

Computes a mask for signaling arrivals to multicasting threadblocks.

<a id="cutlass.pipeline.PipelineTmaUmma._compute_is_leader_cta"></a>
#### `cutlass.pipeline.PipelineTmaUmma._compute_is_leader_cta`

```python
_compute_is_leader_cta( cta_layout_vmnk: cutlass.cute.typing.Layout, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cutlass_dsl.Boolean
```

Computes leader threadblocks for 2CTA kernels. For 1CTA, all threadblocks are leaders.

<a id="cutlass.pipeline.PipelineTmaUmma.create"></a>
#### `cutlass.pipeline.PipelineTmaUmma.create`

```python
create( *, num_stages: int, producer_group: CooperativeGroup, consumer_group: CooperativeGroup, tx_count: int, barrier_storage: cutlass.cute.typing.Pointer | None = None, cta_layout_vmnk: cutlass.cute.typing.Layout | None = None, mcast_mode_mn: tuple[int, int] = (1, 1), enable_multicast_signaling: bool = False, defer_sync: bool = False, name: str = '', loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → PipelineTmaUmma
```

Creates and initializes a new PipelineTmaUmma instance.

**Parameters:**

- **num\_stages** (*int*) – Number of buffer stages for this pipeline
- **producer\_group** ([*CooperativeGroup*](#cutlass.pipeline.CooperativeGroup)) – CooperativeGroup for the producer agent
- **consumer\_group** ([*CooperativeGroup*](#cutlass.pipeline.CooperativeGroup)) – CooperativeGroup for the consumer agent
- **tx\_count** (*int*) – Number of bytes expected to be written to the transaction barrier for one stage
- **barrier\_storage** (*cute.Pointer*, *optional*) – Pointer to the shared memory address for this pipeline’s mbarriers
- **cta\_layout\_vmnk** (*cute.Layout*, *optional*) – Layout of the cluster shape
- **mcast\_mode\_mn** (*tuple*[*int*, *int*], *optional*) – Tuple specifying multicast modes for m and n dimensions (each 0 or 1)
- **enable\_multicast\_signaling** (*bool*, *optional*) – See docstring in PipelineTmaAsync.create() for details
- **defer\_sync** (*bool*, *optional*) – Bool specifying whether or not to skip the built-in mbarrier fence and sync for performance, defaults to False

**Raises:**

**ValueError** – If barrier\_storage is not a cute.Pointer instance

**Returns:**

A new PipelineTmaUmma instance configured with the provided parameters

**Return type:**

[PipelineTmaUmma](#cutlass.pipeline.PipelineTmaUmma)

<a id="cutlass.pipeline.PipelineTmaUmma.consumer_release"></a>
#### `cutlass.pipeline.PipelineTmaUmma.consumer_release`

```python
consumer_release( state: PipelineState, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

UMMA consumer release buffer empty, cta\_group needs to be provided.

<a id="cutlass.pipeline.PipelineTmaUmma.producer_acquire"></a>
#### `cutlass.pipeline.PipelineTmaUmma.producer_acquire`

```python
producer_acquire( state: PipelineState, try_acquire_token: cutlass.cutlass_dsl.Boolean | None = None, *, expected_tx: cutlass.cutlass_dsl.Int32 | None = None, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

TMA producer conditionally waits on buffer empty and sets the transaction barrier for leader threadblocks.

**Parameters:**

**expected\_tx** – Override the expected transaction byte count for this
acquire. When `None` (default), uses the `tx_count` from barrier init.
Pass a dynamic value for workloads where the byte count varies per
iteration (e.g. sparse GEMM with conditional metadata loading).

<a id="cutlass.pipeline.PipelineTmaUmma.producer_commit"></a>
#### `cutlass.pipeline.PipelineTmaUmma.producer_commit`

```python
producer_commit( state: PipelineState, ) → None
```

TMA producer commit is a noop since TMA instruction itself updates the transaction count.

<a id="cutlass.pipeline.PipelineTmaUmma.__init__"></a>
#### `cutlass.pipeline.PipelineTmaUmma.__init__`

```python
__init__( sync_object_full: SyncObject, sync_object_empty: SyncObject, num_stages: int, producer_mask: cutlass.cutlass_dsl.Int32 | None, consumer_mask: cutlass.cutlass_dsl.Int32 | None, is_leader_cta: bool, cta_group: CtaGroup, ) → None
```

<a id="cutlass.pipeline.PipelineAsyncUmma"></a>
### `cutlass.pipeline.PipelineAsyncUmma`

```python
class cutlass.pipeline.PipelineAsyncUmma( sync_object_full: SyncObject, sync_object_empty: SyncObject, num_stages: int, producer_mask: cutlass.cutlass_dsl.Int32 | None, consumer_mask: cutlass.cutlass_dsl.Int32 | None, cta_group: CtaGroup, )
```

Bases: [`PipelineAsync`](#cutlass.pipeline.PipelineAsync)

PipelineAsyncUmma is used for AsyncThread producers and UMMA consumers (e.g. Blackwell input fusion pipelines).

<a id="cutlass.pipeline.PipelineAsyncUmma.cta_group"></a>
#### `cutlass.pipeline.PipelineAsyncUmma.cta_group`

```python
cta_group: CtaGroup
```

<a id="cutlass.pipeline.PipelineAsyncUmma._compute_leading_cta_rank"></a>
#### `cutlass.pipeline.PipelineAsyncUmma._compute_leading_cta_rank`

```python
_compute_leading_cta_rank( cta_v_size: int, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cutlass_dsl.Int32
```

Computes the leading CTA rank.

<a id="cutlass.pipeline.PipelineAsyncUmma._compute_is_leader_cta"></a>
#### `cutlass.pipeline.PipelineAsyncUmma._compute_is_leader_cta`

```python
_compute_is_leader_cta( cta_layout_vmnk: cutlass.cute.typing.Layout, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cutlass_dsl.Boolean
```

Computes leader threadblocks for 2CTA kernels. For 1CTA, all threadblocks are leaders.

<a id="cutlass.pipeline.PipelineAsyncUmma._compute_peer_cta_mask"></a>
#### `cutlass.pipeline.PipelineAsyncUmma._compute_peer_cta_mask`

```python
_compute_peer_cta_mask( cta_layout_vmnk: cutlass.cute.typing.Layout, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cutlass_dsl.Int32
```

Computes a mask for signaling arrivals to multicasting threadblocks.

<a id="cutlass.pipeline.PipelineAsyncUmma.create"></a>
#### `cutlass.pipeline.PipelineAsyncUmma.create`

```python
create( *, num_stages: int, producer_group: CooperativeGroup, consumer_group: CooperativeGroup, barrier_storage: cutlass.cute.typing.Pointer | None = None, cta_layout_vmnk: cutlass.cute.typing.Layout | None = None, defer_sync: bool = False, name: str = '', loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → PipelineAsyncUmma
```

Creates and initializes a new PipelineAsyncUmma instance.

**Parameters:**

- **num\_stages** (*int*) – Number of buffer stages for this pipeline
- **producer\_group** ([*CooperativeGroup*](#cutlass.pipeline.CooperativeGroup)) – CooperativeGroup for the producer agent
- **consumer\_group** ([*CooperativeGroup*](#cutlass.pipeline.CooperativeGroup)) – CooperativeGroup for the consumer agent
- **barrier\_storage** (*cute.Pointer*, *optional*) – Pointer to the shared memory address for this pipeline’s mbarriers
- **cta\_layout\_vmnk** (*cute.Layout*, *optional*) – Layout of the cluster shape

**Raises:**

**ValueError** – If barrier\_storage is not a cute.Pointer instance

**Returns:**

A new PipelineAsyncUmma instance configured with the provided parameters

**Return type:**

[PipelineAsyncUmma](#cutlass.pipeline.PipelineAsyncUmma)

<a id="cutlass.pipeline.PipelineAsyncUmma.consumer_release"></a>
#### `cutlass.pipeline.PipelineAsyncUmma.consumer_release`

```python
consumer_release( state: PipelineState, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

UMMA consumer release buffer empty, cta\_group needs to be provided.

<a id="cutlass.pipeline.PipelineAsyncUmma.__init__"></a>
#### `cutlass.pipeline.PipelineAsyncUmma.__init__`

```python
__init__( sync_object_full: SyncObject, sync_object_empty: SyncObject, num_stages: int, producer_mask: cutlass.cutlass_dsl.Int32 | None, consumer_mask: cutlass.cutlass_dsl.Int32 | None, cta_group: CtaGroup, ) → None
```

<a id="cutlass.pipeline.PipelineUmmaAsync"></a>
### `cutlass.pipeline.PipelineUmmaAsync`

```python
class cutlass.pipeline.PipelineUmmaAsync( sync_object_full: SyncObject, sync_object_empty: SyncObject, num_stages: int, producer_mask: cutlass.cutlass_dsl.Int32 | None, consumer_mask: cutlass.cutlass_dsl.Int32 | None, cta_group: CtaGroup, )
```

Bases: [`PipelineAsync`](#cutlass.pipeline.PipelineAsync)

PipelineUmmaAsync is used for UMMA producers and AsyncThread consumers (e.g. Blackwell accumulator pipelines).

<a id="cutlass.pipeline.PipelineUmmaAsync.cta_group"></a>
#### `cutlass.pipeline.PipelineUmmaAsync.cta_group`

```python
cta_group: CtaGroup
```

<a id="cutlass.pipeline.PipelineUmmaAsync._compute_tmem_sync_mask"></a>
#### `cutlass.pipeline.PipelineUmmaAsync._compute_tmem_sync_mask`

```python
_compute_tmem_sync_mask( cta_layout_vmnk: cutlass.cute.typing.Layout, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cutlass_dsl.Int32
```

Computes a mask to signal completion of tmem buffers for 2CTA kernels.

<a id="cutlass.pipeline.PipelineUmmaAsync._compute_peer_cta_rank"></a>
#### `cutlass.pipeline.PipelineUmmaAsync._compute_peer_cta_rank`

```python
_compute_peer_cta_rank( *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cutlass_dsl.Int32
```

Computes a mask to signal release of tmem buffers for 2CTA kernels.

<a id="cutlass.pipeline.PipelineUmmaAsync.create"></a>
#### `cutlass.pipeline.PipelineUmmaAsync.create`

```python
create( *, num_stages: int, producer_group: CooperativeGroup, consumer_group: CooperativeGroup, barrier_storage: cutlass.cute.typing.Pointer | None = None, cta_layout_vmnk: cutlass.cute.typing.Layout | None = None, defer_sync: bool = False, name: str = '', loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → PipelineUmmaAsync
```

Creates an instance of PipelineUmmaAsync with computed attributes.

**Parameters:**

- **num\_stages** (*int*) – Number of buffer stages for this pipeline
- **producer\_group** ([*CooperativeGroup*](#cutlass.pipeline.CooperativeGroup)) – `CooperativeGroup` for the producer agent
- **consumer\_group** ([*CooperativeGroup*](#cutlass.pipeline.CooperativeGroup)) – `CooperativeGroup` for the consumer agent
- **barrier\_storage** (*cute.Pointer*, *optional*) – Pointer to the shared memory address for this pipeline’s mbarriers
- **cta\_layout\_vmnk** (*cute.Layout*, *optional*) – Layout of the cluster shape

**Raises:**

**ValueError** – If barrier\_storage is not a cute.Pointer instance

**Returns:**

New instance of `PipelineUmmaAsync`

**Return type:**

[PipelineUmmaAsync](#cutlass.pipeline.PipelineUmmaAsync)

<a id="cutlass.pipeline.PipelineUmmaAsync.producer_commit"></a>
#### `cutlass.pipeline.PipelineUmmaAsync.producer_commit`

```python
producer_commit( state: PipelineState, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

UMMA producer commit buffer full, cta\_group needs to be provided.

<a id="cutlass.pipeline.PipelineUmmaAsync.__init__"></a>
#### `cutlass.pipeline.PipelineUmmaAsync.__init__`

```python
__init__( sync_object_full: SyncObject, sync_object_empty: SyncObject, num_stages: int, producer_mask: cutlass.cutlass_dsl.Int32 | None, consumer_mask: cutlass.cutlass_dsl.Int32 | None, cta_group: CtaGroup, ) → None
```

<a id="cutlass.pipeline.PipelineClcFetchAsync"></a>
### `cutlass.pipeline.PipelineClcFetchAsync`

```python
class cutlass.pipeline.PipelineClcFetchAsync( sync_object_full: SyncObject, sync_object_empty: SyncObject, num_stages: int, producer_mask: cutlass.cutlass_dsl.Int32 | None, consumer_mask: cutlass.cutlass_dsl.Int32 | None, is_signaling_thread: cutlass.cutlass_dsl.Boolean, )
```

Bases: [`PipelineAsync`](#cutlass.pipeline.PipelineAsync)

PipelineClcFetchAsync implements a producer-consumer pipeline for Cluster Launch
Control based dynamic scheduling. Both producer and consumer operate asynchronously
using barrier synchronization to coordinate across pipeline stages and cluster CTAs.

- Producer: waits for empty buffer, signals full barrier with transection bytes
  across all CTAs in cluster, hardware autosignals each CTA’s mbarrier when
  transaction bytes are written, then the satte advance to next buffer slot.
- Consumer: waits for full barrier, then load respinse from local SMEM, then
  sigals CTA 0’s empty barrier to allow buffer reuse.

<a id="cutlass.pipeline.PipelineClcFetchAsync.sync_object_full"></a>
#### `cutlass.pipeline.PipelineClcFetchAsync.sync_object_full`

```python
sync_object_full: SyncObject
```

<a id="cutlass.pipeline.PipelineClcFetchAsync.sync_object_empty"></a>
#### `cutlass.pipeline.PipelineClcFetchAsync.sync_object_empty`

```python
sync_object_empty: SyncObject
```

<a id="cutlass.pipeline.PipelineClcFetchAsync.num_stages"></a>
#### `cutlass.pipeline.PipelineClcFetchAsync.num_stages`

```python
num_stages: int
```

<a id="cutlass.pipeline.PipelineClcFetchAsync.producer_mask"></a>
#### `cutlass.pipeline.PipelineClcFetchAsync.producer_mask`

```python
producer_mask: cutlass.cutlass_dsl.Int32 | None
```

<a id="cutlass.pipeline.PipelineClcFetchAsync.consumer_mask"></a>
#### `cutlass.pipeline.PipelineClcFetchAsync.consumer_mask`

```python
consumer_mask: cutlass.cutlass_dsl.Int32 | None
```

<a id="cutlass.pipeline.PipelineClcFetchAsync.is_signaling_thread"></a>
#### `cutlass.pipeline.PipelineClcFetchAsync.is_signaling_thread`

```python
is_signaling_thread: cutlass.cutlass_dsl.Boolean
```

<a id="cutlass.pipeline.PipelineClcFetchAsync._init_full_barrier_arrive_signal"></a>
#### `cutlass.pipeline.PipelineClcFetchAsync._init_full_barrier_arrive_signal`

```python
static _init_full_barrier_arrive_signal( cta_layout_vmnk: cutlass.cute.typing.Layout, tidx: cutlass.cutlass_dsl.Int32, ) → tuple[cutlass.cutlass_dsl.Int32, cutlass.cutlass_dsl.Boolean]
```

Computes producer barrier signaling parameters, returns destination CTA rank
(0 to cluster\_size-1) based on thread ID, and a boolean flag indicating if
this thread participates in signaling.

**Parameters:**

- **cta\_layout\_vmnk** – Cluster layout defining CTA count
- **tidx** – Thread ID within the CTA

<a id="cutlass.pipeline.PipelineClcFetchAsync.create"></a>
#### `cutlass.pipeline.PipelineClcFetchAsync.create`

```python
static create( *, num_stages: int, producer_group: CooperativeGroup, consumer_group: CooperativeGroup, tx_count: int, barrier_storage: cutlass.cute.typing.Pointer | None = None, producer_mask: cutlass.cutlass_dsl.Int32 | None = None, consumer_mask: cutlass.cutlass_dsl.Int32 | None = None, cta_layout_vmnk: cutlass.cute.typing.Layout | None = None, defer_sync: bool = False, name: str = '', ) → PipelineClcFetchAsync
```

This helper function computes any necessary attributes and returns an instance of PipelineClcFetchAsync.
:param barrier\_storage: Pointer to the shared memory address for this pipeline’s mbarriers
:type barrier\_storage: cute.Pointer
:param num\_stages: Number of buffer stages for this pipeline
:type num\_stages: int
:param producer\_group: CooperativeGroup for the producer agent
:type producer\_group: CooperativeGroup
:param consumer\_group: CooperativeGroup for the consumer agent
:type consumer\_group: CooperativeGroup
:param tx\_count: Number of bytes expected to be written to the transaction barrier for one stage
:type tx\_count: int
:param producer\_mask: Mask for signaling arrives for the producer agent, defaults to `None`
:type producer\_mask: Int32, optional
:param consumer\_mask: Mask for signaling arrives for the consumer agent, defaults to `None`
:type consumer\_mask: Int32, optional

<a id="cutlass.pipeline.PipelineClcFetchAsync.producer_acquire"></a>
#### `cutlass.pipeline.PipelineClcFetchAsync.producer_acquire`

```python
producer_acquire( state: PipelineState, try_acquire_token: cutlass.cutlass_dsl.Boolean | None = None, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

Producer acquire waits for empty buffer and sets transaction expectation on full barrier.

**Parameters:**

- **state** – Pipeline state pointing to the current buffer stage
- **try\_acquire\_token** – Optional token to skip the empty barrier wait

<a id="cutlass.pipeline.PipelineClcFetchAsync.consumer_wait"></a>
#### `cutlass.pipeline.PipelineClcFetchAsync.consumer_wait`

```python
consumer_wait( state: PipelineState, try_wait_token: cutlass.cutlass_dsl.Boolean | None = None, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

Consumer waits for full barrier to be signaled by hardware multicast.

**Parameters:**

- **state** – Pipeline state pointing to the current buffer stage
- **try\_wait\_token** – Optional token to skip the full barrier wait

<a id="cutlass.pipeline.PipelineClcFetchAsync.consumer_release"></a>
#### `cutlass.pipeline.PipelineClcFetchAsync.consumer_release`

```python
consumer_release( state: PipelineState, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

<a id="cutlass.pipeline.PipelineClcFetchAsync.producer_get_barrier"></a>
#### `cutlass.pipeline.PipelineClcFetchAsync.producer_get_barrier`

```python
producer_get_barrier( state: PipelineState, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Pointer
```

<a id="cutlass.pipeline.PipelineClcFetchAsync.consumer_get_barrier"></a>
#### `cutlass.pipeline.PipelineClcFetchAsync.consumer_get_barrier`

```python
consumer_get_barrier( state: PipelineState, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cute.typing.Pointer
```

<a id="cutlass.pipeline.PipelineClcFetchAsync.__init__"></a>
#### `cutlass.pipeline.PipelineClcFetchAsync.__init__`

```python
__init__( sync_object_full: SyncObject, sync_object_empty: SyncObject, num_stages: int, producer_mask: cutlass.cutlass_dsl.Int32 | None, consumer_mask: cutlass.cutlass_dsl.Int32 | None, is_signaling_thread: cutlass.cutlass_dsl.Boolean, ) → None
```

<a id="cutlass.pipeline.PipelineTmaMultiConsumersAsync"></a>
### `cutlass.pipeline.PipelineTmaMultiConsumersAsync`

```python
class cutlass.pipeline.PipelineTmaMultiConsumersAsync( sync_object_full: SyncObject, sync_object_empty: SyncObject, num_stages: int, producer_mask: cutlass.cutlass_dsl.Int32 | None, consumer_mask: cutlass.cutlass_dsl.Int32 | None, is_leader_cta: bool, sync_object_empty_umma: SyncObject, sync_object_empty_async: SyncObject, cta_group: CtaGroup, consumer_dst_rank_async: cutlass.cutlass_dsl.Int32 | None = None, is_signaling_thread: cutlass.cutlass_dsl.Boolean = True, )
```

Bases: [`PipelineAsync`](#cutlass.pipeline.PipelineAsync)

PipelineTmaMultiConsumersAsync is used for TMA producers and UMMA+Async consumers.

<a id="cutlass.pipeline.PipelineTmaMultiConsumersAsync.is_leader_cta"></a>
#### `cutlass.pipeline.PipelineTmaMultiConsumersAsync.is_leader_cta`

```python
is_leader_cta: bool
```

<a id="cutlass.pipeline.PipelineTmaMultiConsumersAsync.sync_object_empty_umma"></a>
#### `cutlass.pipeline.PipelineTmaMultiConsumersAsync.sync_object_empty_umma`

```python
sync_object_empty_umma: SyncObject
```

<a id="cutlass.pipeline.PipelineTmaMultiConsumersAsync.sync_object_empty_async"></a>
#### `cutlass.pipeline.PipelineTmaMultiConsumersAsync.sync_object_empty_async`

```python
sync_object_empty_async: SyncObject
```

<a id="cutlass.pipeline.PipelineTmaMultiConsumersAsync.cta_group"></a>
#### `cutlass.pipeline.PipelineTmaMultiConsumersAsync.cta_group`

```python
cta_group: CtaGroup
```

<a id="cutlass.pipeline.PipelineTmaMultiConsumersAsync.consumer_dst_rank_async"></a>
#### `cutlass.pipeline.PipelineTmaMultiConsumersAsync.consumer_dst_rank_async`

```python
consumer_dst_rank_async: cutlass.cutlass_dsl.Int32 | None = None
```

<a id="cutlass.pipeline.PipelineTmaMultiConsumersAsync.is_signaling_thread"></a>
#### `cutlass.pipeline.PipelineTmaMultiConsumersAsync.is_signaling_thread`

```python
is_signaling_thread: cutlass.cutlass_dsl.Boolean = True
```

<a id="cutlass.pipeline.PipelineTmaMultiConsumersAsync.create"></a>
#### `cutlass.pipeline.PipelineTmaMultiConsumersAsync.create`

```python
static create( *, num_stages: int, producer_group: CooperativeGroup, consumer_group_umma: CooperativeGroup, consumer_group_async: CooperativeGroup, tx_count: int, barrier_storage: cutlass.cute.typing.Pointer | None = None, cta_layout_vmnk: cutlass.cute.typing.Layout | None = None, mcast_mode_mn: tuple[int, int] = (1, 1), tidx: cutlass.cutlass_dsl.Int32 | None = None, enable_multicast_signaling: bool = False, defer_sync: bool = False, force_deprecated_per_lane_signaling: bool | None = None, name: str = '', ) → PipelineTmaMultiConsumersAsync
```

This helper function computes any necessary attributes and returns an instance of PipelineTmaMultiConsumersAsync.
:param barrier\_storage: Pointer to the smem address for this pipeline’s mbarriers
:type barrier\_storage: cute.Pointer
:param num\_stages: Number of buffer stages for this pipeline
:type num\_stages: Int32
:param producer\_group: CooperativeGroup for the producer agent
:type producer\_group: CooperativeGroup
:param consumer\_group\_umma: CooperativeGroup for the UMMA consumer agent
:type consumer\_group\_umma: CooperativeGroup
:param consumer\_group\_async: CooperativeGroup for the AsyncThread consumer agent
:type consumer\_group\_async: CooperativeGroup
:param tx\_count: Number of bytes expected to be written to the transaction barrier for one stage
:type tx\_count: int
:param cta\_layout\_vmnk: Layout of the cluster shape
:type cta\_layout\_vmnk: cute.Layout | None
:param mcast\_mode\_mn: Tuple specifying multicast modes for m and n dimensions (each 0 or 1)
:type mcast\_mode\_mn: tuple[int, int]
:param tidx: Thread index for computing AsyncThread consumer signaling, defaults to thread\_idx()[0]
:type tidx: Int32 | None
:param enable\_multicast\_signaling: See docstring in PipelineTmaAsync.create() for details
:type enable\_multicast\_signaling: bool, optional
:param force\_deprecated\_per\_lane\_signaling: **Deprecated.** Set `False` if your arrive count is a multiple of `WARP_SIZE` and you do not want the legacy fallback. Leave unset otherwise.
:type force\_deprecated\_per\_lane\_signaling: bool | None

<a id="cutlass.pipeline.PipelineTmaMultiConsumersAsync._init_empty_barrier_arrive_signal_2sm"></a>
#### `cutlass.pipeline.PipelineTmaMultiConsumersAsync._init_empty_barrier_arrive_signal_2sm`

```python
static _init_empty_barrier_arrive_signal_2sm( cta_layout_vmnk: cutlass.cute.typing.Layout, tidx: cutlass.cutlass_dsl.Int32, mcast_mode_mn: tuple[int, int] = (1, 1), ) → tuple[cutlass.cutlass_dsl.Int32, cutlass.cutlass_dsl.Boolean]
```

Identical to sm90.py PipelineTmaAsync.init\_empty\_barrier\_arrive\_signal except
that CTAs in the multicast will also signal CTAs with a different V-coordinate (i.e. leader/follower CTA pairs).

<a id="cutlass.pipeline.PipelineTmaMultiConsumersAsync.producer_acquire"></a>
#### `cutlass.pipeline.PipelineTmaMultiConsumersAsync.producer_acquire`

```python
producer_acquire( state: PipelineState, try_acquire_token: cutlass.cutlass_dsl.Boolean | None = None, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

TMA producer acquire waits on buffer empty and sets the transaction barrier for leader threadblocks.

<a id="cutlass.pipeline.PipelineTmaMultiConsumersAsync.producer_commit"></a>
#### `cutlass.pipeline.PipelineTmaMultiConsumersAsync.producer_commit`

```python
producer_commit( state: PipelineState, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

TMA producer commit is a noop since TMA instruction itself updates the transaction count.

<a id="cutlass.pipeline.PipelineTmaMultiConsumersAsync.consumer_wait"></a>
#### `cutlass.pipeline.PipelineTmaMultiConsumersAsync.consumer_wait`

```python
consumer_wait( state: PipelineState, try_wait_token: cutlass.cutlass_dsl.Boolean | None = None, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

Consumer waits for full barrier to be signaled.

<a id="cutlass.pipeline.PipelineTmaMultiConsumersAsync.consumer_try_wait"></a>
#### `cutlass.pipeline.PipelineTmaMultiConsumersAsync.consumer_try_wait`

```python
consumer_try_wait( state: PipelineState, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cutlass_dsl.Boolean
```

Non-blocking check if data is ready.

<a id="cutlass.pipeline.PipelineTmaMultiConsumersAsync.consumer_release"></a>
#### `cutlass.pipeline.PipelineTmaMultiConsumersAsync.consumer_release`

```python
consumer_release( state: PipelineState, op_type: PipelineOp = PipelineOp.TCGen05Mma, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

<a id="cutlass.pipeline.PipelineTmaMultiConsumersAsync.make_participants"></a>
#### `cutlass.pipeline.PipelineTmaMultiConsumersAsync.make_participants`

```python
make_participants( *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → tuple[PipelineProducer, PipelineConsumer, None]
```

Returns (producer, umma\_consumer, None).

<a id="cutlass.pipeline.PipelineTmaMultiConsumersAsync.__init__"></a>
#### `cutlass.pipeline.PipelineTmaMultiConsumersAsync.__init__`

```python
__init__( sync_object_full: SyncObject, sync_object_empty: SyncObject, num_stages: int, producer_mask: cutlass.cutlass_dsl.Int32 | None, consumer_mask: cutlass.cutlass_dsl.Int32 | None, is_leader_cta: bool, sync_object_empty_umma: SyncObject, sync_object_empty_async: SyncObject, cta_group: CtaGroup, consumer_dst_rank_async: cutlass.cutlass_dsl.Int32 | None = None, is_signaling_thread: cutlass.cutlass_dsl.Boolean = True, ) → None
```

<a id="cutlass.pipeline.PipelineTmaStore"></a>
### `cutlass.pipeline.PipelineTmaStore`

```python
class cutlass.pipeline.PipelineTmaStore( sync_object_full: SyncObject, sync_object_empty: SyncObject, num_stages: int, producer_mask: cutlass.cutlass_dsl.Int32 | None, consumer_mask: cutlass.cutlass_dsl.Int32 | None, )
```

Bases: [`PipelineAsync`](#cutlass.pipeline.PipelineAsync)

PipelineTmaStore is used for synchronizing TMA stores in the epilogue. It does not use mbarriers.

<a id="cutlass.pipeline.PipelineTmaStore.create"></a>
#### `cutlass.pipeline.PipelineTmaStore.create`

```python
static create( *, num_stages: int, producer_group: CooperativeGroup, ) → PipelineTmaStore
```

This helper function computes any necessary attributes and returns an instance of `PipelineTmaStore`.

**Parameters:**

- **num\_stages** (*int*) – Number of buffer stages for this pipeline
- **producer\_group** ([*CooperativeGroup*](#cutlass.pipeline.CooperativeGroup)) – `CooperativeGroup` for the producer agent

**Returns:**

A new `PipelineTmaStore` instance

**Return type:**

[PipelineTmaStore](#cutlass.pipeline.PipelineTmaStore)

<a id="cutlass.pipeline.PipelineTmaStore.producer_acquire"></a>
#### `cutlass.pipeline.PipelineTmaStore.producer_acquire`

```python
producer_acquire( *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

<a id="cutlass.pipeline.PipelineTmaStore.producer_commit"></a>
#### `cutlass.pipeline.PipelineTmaStore.producer_commit`

```python
producer_commit( *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

<a id="cutlass.pipeline.PipelineTmaStore.consumer_wait"></a>
#### `cutlass.pipeline.PipelineTmaStore.consumer_wait`

```python
consumer_wait( *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

<a id="cutlass.pipeline.PipelineTmaStore.consumer_release"></a>
#### `cutlass.pipeline.PipelineTmaStore.consumer_release`

```python
consumer_release( *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

<a id="cutlass.pipeline.PipelineTmaStore.producer_tail"></a>
#### `cutlass.pipeline.PipelineTmaStore.producer_tail`

```python
producer_tail( *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

<a id="cutlass.pipeline.PipelineTmaStore.__init__"></a>
#### `cutlass.pipeline.PipelineTmaStore.__init__`

```python
__init__( sync_object_full: SyncObject, sync_object_empty: SyncObject, num_stages: int, producer_mask: cutlass.cutlass_dsl.Int32 | None, consumer_mask: cutlass.cutlass_dsl.Int32 | None, ) → None
```

<a id="cutlass.pipeline.PipelineProducer"></a>
### `cutlass.pipeline.PipelineProducer`

```python
class cutlass.pipeline.PipelineProducer( pipeline: PipelineAsync, state: PipelineState, group: CooperativeGroup, )
```

Bases: `object`

A class representing a producer in an asynchronous pipeline.

This class manages the producer side of an asynchronous pipeline, handling
synchronization and state management for producing data. It provides methods for
acquiring, committing, and advancing through pipeline stages.

**Variables:**

- **\_\_pipeline** – The asynchronous pipeline this producer belongs to
- **\_\_state** – The current state of the producer in the pipeline
- **\_\_group** – The cooperative group this producer operates in

**Examples:**

```python
pipeline = PipelineAsync.create(...)
producer, consumer = pipeline.make_participants()
for i in range(iterations):
    # Try to acquire the current buffer without blocking
    try_acquire_token = producer.try_acquire()

    # Do something else independently
    ...

    # Wait for current buffer to be empty & Move index to next stage
    # If try_acquire_token is True, return immediately
    # If try_acquire_token is False, block until buffer is empty
    handle = producer.acquire_and_advance(try_acquire_token)

    # Produce data
    handle.commit()
```

<a id="cutlass.pipeline.PipelineProducer.ImmutableResourceHandle"></a>
#### `cutlass.pipeline.PipelineProducer.ImmutableResourceHandle`

```python
class ImmutableResourceHandle( _ImmutableResourceHandle__origin: cutlass.pipeline.sm90.PipelineAsync, _ImmutableResourceHandle__immutable_state: cutlass.pipeline.helpers.PipelineState, )
```

Bases: `ImmutableResourceHandle`

<a id="cutlass.pipeline.PipelineProducer.ImmutableResourceHandle.barrier"></a>
##### `cutlass.pipeline.PipelineProducer.ImmutableResourceHandle.barrier`

```python
property barrier: cutlass.cute.typing.Pointer
```

Get the barrier pointer for the current pipeline stage.

**Returns:**

Pointer to the barrier for the current stage

**Return type:**

cute.Pointer

<a id="cutlass.pipeline.PipelineProducer.ImmutableResourceHandle.commit"></a>
##### `cutlass.pipeline.PipelineProducer.ImmutableResourceHandle.commit`

```python
commit( *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

Signal that data production is complete for the current stage.

This allows consumers to start processing the data.

<a id="cutlass.pipeline.PipelineProducer.ImmutableResourceHandle.__init__"></a>
##### `cutlass.pipeline.PipelineProducer.ImmutableResourceHandle.__init__`

```python
__init__( _ImmutableResourceHandle__origin: PipelineAsync, _ImmutableResourceHandle__immutable_state: PipelineState, ) → None
```

<a id="cutlass.pipeline.PipelineProducer.__init__"></a>
#### `cutlass.pipeline.PipelineProducer.__init__`

```python
__init__( pipeline: PipelineAsync, state: PipelineState, group: CooperativeGroup, ) → None
```

Initialize a new Producer instance.

**Parameters:**

- **pipeline** ([*PipelineAsync*](#cutlass.pipeline.PipelineAsync)) – The pipeline this producer belongs to
- **state** ([*PipelineState*](#cutlass.pipeline.PipelineState)) – Initial pipeline state
- **group** ([*CooperativeGroup*](#cutlass.pipeline.CooperativeGroup)) – The cooperative group for synchronization

<a id="cutlass.pipeline.PipelineProducer.__pipeline"></a>
#### `cutlass.pipeline.PipelineProducer.__pipeline`

```python
__pipeline: PipelineAsync
```

<a id="cutlass.pipeline.PipelineProducer.__state"></a>
#### `cutlass.pipeline.PipelineProducer.__state`

```python
__state: PipelineState
```

<a id="cutlass.pipeline.PipelineProducer.__group"></a>
#### `cutlass.pipeline.PipelineProducer.__group`

```python
__group: CooperativeGroup
```

<a id="cutlass.pipeline.PipelineProducer.clone"></a>
#### `cutlass.pipeline.PipelineProducer.clone`

```python
clone() → PipelineProducer
```

Create a new Producer instance with the same state.

<a id="cutlass.pipeline.PipelineProducer.reset"></a>
#### `cutlass.pipeline.PipelineProducer.reset`

```python
reset( *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

Reset the count of how many handles this producer has committed.

<a id="cutlass.pipeline.PipelineProducer.current_handle"></a>
#### `cutlass.pipeline.PipelineProducer.current_handle`

```python
current_handle() → ImmutableResourceHandle
```

Get the current handle for the producer.

<a id="cutlass.pipeline.PipelineProducer.acquire"></a>
#### `cutlass.pipeline.PipelineProducer.acquire`

```python
acquire( try_acquire_token: cutlass.cutlass_dsl.Boolean | None = None, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, **kwargs: Any, ) → ImmutableResourceHandle
```

Wait for the current buffer to be empty before producing data.
This is a blocking operation.

**Parameters:**

**try\_acquire\_token** (*Optional*[*Boolean*]) – Optional token to try to acquire the buffer

**Returns:**

A handle to the producer for committing the data

**Return type:**

[ImmutableResourceHandle](#cutlass.pipeline.PipelineProducer.ImmutableResourceHandle)

<a id="cutlass.pipeline.PipelineProducer.advance"></a>
#### `cutlass.pipeline.PipelineProducer.advance`

```python
advance( *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

Move to the next pipeline stage.

<a id="cutlass.pipeline.PipelineProducer.acquire_and_advance"></a>
#### `cutlass.pipeline.PipelineProducer.acquire_and_advance`

```python
acquire_and_advance( try_acquire_token: cutlass.cutlass_dsl.Boolean | None = None, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, **kwargs: Any, ) → ImmutableResourceHandle
```

Acquire the current buffer and advance to the next pipeline stage.

This method combines the acquire() and advance() operations into a single call.
It first waits for the current buffer to be empty before producing data,
then advances the pipeline to the next stage.

**Parameters:**

**try\_acquire\_token** (*Optional*[*Boolean*]) – Token indicating whether to try non-blocking acquire.
If True, returns immediately without waiting. If False or None, blocks
until buffer is empty.

**Returns:**

A handle to the producer that can be used to commit data to the
acquired buffer stage

**Return type:**

[ImmutableResourceHandle](#cutlass.pipeline.PipelineProducer.ImmutableResourceHandle)

<a id="cutlass.pipeline.PipelineProducer.try_acquire"></a>
#### `cutlass.pipeline.PipelineProducer.try_acquire`

```python
try_acquire( *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cutlass_dsl.Boolean
```

Attempt to acquire the current buffer without blocking.

This method tries to acquire the current buffer stage for producing data
without waiting. It can be used to check buffer availability before
committing to a blocking acquire operation.

**Returns:**

A boolean token indicating whether the buffer was successfully acquired

**Return type:**

Boolean

<a id="cutlass.pipeline.PipelineProducer.commit"></a>
#### `cutlass.pipeline.PipelineProducer.commit`

```python
commit( handle: ImmutableResourceHandle | None = None, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

Signal that data production is complete for the current stage.

This allows consumers to start processing the data.

**Parameters:**

**handle** (*Optional*[[*ImmutableResourceHandle*](#cutlass.pipeline.PipelineProducer.ImmutableResourceHandle)]) – Optional handle to commit, defaults to None

**Raises:**

**AssertionError** – If provided handle does not belong to this producer

<a id="cutlass.pipeline.PipelineProducer.tail"></a>
#### `cutlass.pipeline.PipelineProducer.tail`

```python
tail( *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

Ensure all used buffers are properly synchronized before producer exit.

This should be called before the producer finishes to avoid dangling signals.

<a id="cutlass.pipeline.PipelineConsumer"></a>
### `cutlass.pipeline.PipelineConsumer`

```python
class cutlass.pipeline.PipelineConsumer( pipeline: PipelineAsync, state: PipelineState, group: CooperativeGroup, )
```

Bases: `object`

A class representing a consumer in an asynchronous pipeline.

The Consumer class manages the consumer side of an asynchronous pipeline, handling
synchronization and state management for consuming data. It provides methods for
waiting, releasing, and advancing through pipeline stages.

**Variables:**

- **\_\_pipeline** – The asynchronous pipeline this consumer belongs to
- **\_\_state** – The current state of the consumer in the pipeline
- **\_\_group** – The cooperative group this consumer operates in

**Examples:**

```python
pipeline = PipelineAsync.create(...)
producer, consumer = pipeline.make_participants()
for i in range(iterations):
    # Try to wait for buffer to be full
    try_wait_token = consumer.try_wait()

    # Do something else independently
    ...

    # Wait for buffer to be full & Move index to next stage
    # If try_wait_token is True, return immediately
    # If try_wait_token is False, block until buffer is full
    handle = consumer.wait_and_advance(try_wait_token)

    # Consume data
    handle.release(  )  # Signal buffer is empty

    # Alternative way to do this is:
    # handle.release()  # Signal buffer is empty
```

<a id="cutlass.pipeline.PipelineConsumer.ImmutableResourceHandle"></a>
#### `cutlass.pipeline.PipelineConsumer.ImmutableResourceHandle`

```python
class ImmutableResourceHandle( _ImmutableResourceHandle__origin: cutlass.pipeline.sm90.PipelineAsync, _ImmutableResourceHandle__immutable_state: cutlass.pipeline.helpers.PipelineState, )
```

Bases: `ImmutableResourceHandle`

<a id="cutlass.pipeline.PipelineConsumer.ImmutableResourceHandle.barrier"></a>
##### `cutlass.pipeline.PipelineConsumer.ImmutableResourceHandle.barrier`

```python
property barrier: cutlass.cute.typing.Pointer
```

Get the barrier pointer for the current pipeline stage.

**Returns:**

Pointer to the barrier for the current stage

**Return type:**

cute.Pointer

<a id="cutlass.pipeline.PipelineConsumer.ImmutableResourceHandle.release"></a>
##### `cutlass.pipeline.PipelineConsumer.ImmutableResourceHandle.release`

```python
release( *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

Signal that data production is complete for the current stage.
This allows consumers to start processing the data.

<a id="cutlass.pipeline.PipelineConsumer.ImmutableResourceHandle.__init__"></a>
##### `cutlass.pipeline.PipelineConsumer.ImmutableResourceHandle.__init__`

```python
__init__( _ImmutableResourceHandle__origin: PipelineAsync, _ImmutableResourceHandle__immutable_state: PipelineState, ) → None
```

<a id="cutlass.pipeline.PipelineConsumer.__init__"></a>
#### `cutlass.pipeline.PipelineConsumer.__init__`

```python
__init__( pipeline: PipelineAsync, state: PipelineState, group: CooperativeGroup, ) → None
```

Initialize a new Consumer instance.

**Parameters:**

- **pipeline** ([*PipelineAsync*](#cutlass.pipeline.PipelineAsync)) – The pipeline this consumer belongs to
- **state** ([*PipelineState*](#cutlass.pipeline.PipelineState)) – Initial pipeline state
- **group** ([*CooperativeGroup*](#cutlass.pipeline.CooperativeGroup)) – The cooperative group for synchronization

<a id="cutlass.pipeline.PipelineConsumer.__pipeline"></a>
#### `cutlass.pipeline.PipelineConsumer.__pipeline`

```python
__pipeline: PipelineAsync
```

<a id="cutlass.pipeline.PipelineConsumer.__group"></a>
#### `cutlass.pipeline.PipelineConsumer.__group`

```python
__group: CooperativeGroup
```

<a id="cutlass.pipeline.PipelineConsumer.__state"></a>
#### `cutlass.pipeline.PipelineConsumer.__state`

```python
__state: PipelineState
```

<a id="cutlass.pipeline.PipelineConsumer.clone"></a>
#### `cutlass.pipeline.PipelineConsumer.clone`

```python
clone() → PipelineConsumer
```

Create a new Consumer instance with the same state.

<a id="cutlass.pipeline.PipelineConsumer.reset"></a>
#### `cutlass.pipeline.PipelineConsumer.reset`

```python
reset( *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

Reset the count of how many handles this consumer has consumed.

<a id="cutlass.pipeline.PipelineConsumer.current_handle"></a>
#### `cutlass.pipeline.PipelineConsumer.current_handle`

```python
current_handle() → ImmutableResourceHandle
```

Get the current handle for the consumer.

<a id="cutlass.pipeline.PipelineConsumer.wait"></a>
#### `cutlass.pipeline.PipelineConsumer.wait`

```python
wait( try_wait_token: cutlass.cutlass_dsl.Boolean | None = None, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → ImmutableResourceHandle
```

Wait for data to be ready in the current buffer. This is a blocking operation
that will not return until data is available.

**Parameters:**

**try\_wait\_token** (*Optional*[*Boolean*]) – Token used to attempt a non-blocking wait for the buffer.
If provided and True, returns immediately if buffer is not ready.

**Returns:**

An immutable handle to the consumer that can be used to release the buffer
once data consumption is complete

**Return type:**

[ImmutableResourceHandle](#cutlass.pipeline.PipelineConsumer.ImmutableResourceHandle)

<a id="cutlass.pipeline.PipelineConsumer.advance"></a>
#### `cutlass.pipeline.PipelineConsumer.advance`

```python
advance( *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

Advance the consumer to the next pipeline stage.

This updates the internal state to point to the next buffer in the pipeline.
Should be called after consuming data from the current buffer.

<a id="cutlass.pipeline.PipelineConsumer.wait_and_advance"></a>
#### `cutlass.pipeline.PipelineConsumer.wait_and_advance`

```python
wait_and_advance( try_wait_token: cutlass.cutlass_dsl.Boolean | None = None, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → ImmutableResourceHandle
```

Atomically wait for data and advance to next pipeline stage.

This is a convenience method that combines wait() and advance() into a single
atomic operation. It will block until data is available in the current buffer,
then automatically advance to the next stage.

**Parameters:**

**try\_wait\_token** (*Optional*[*Boolean*]) – Token used to attempt a non-blocking wait for the buffer.
If provided and True, returns immediately if buffer is not ready.

**Returns:**

An immutable handle to the consumer that can be used to release the buffer
once data consumption is complete

**Return type:**

[ImmutableResourceHandle](#cutlass.pipeline.PipelineConsumer.ImmutableResourceHandle)

<a id="cutlass.pipeline.PipelineConsumer.try_wait"></a>
#### `cutlass.pipeline.PipelineConsumer.try_wait`

```python
try_wait( *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → cutlass.cutlass_dsl.Boolean
```

Non-blocking check if data is ready in the current buffer.

This method provides a way to test if data is available without blocking.
Unlike wait(), this will return immediately regardless of buffer state.

**Returns:**

True if data is ready to be consumed, False if the buffer is not yet ready

**Return type:**

Boolean

<a id="cutlass.pipeline.PipelineConsumer.release"></a>
#### `cutlass.pipeline.PipelineConsumer.release`

```python
release( handle: ImmutableResourceHandle | None = None, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

Signal that data consumption is complete for the current stage.
This allows producers to start producing new data.

<a id="cutlass.pipeline.make_pipeline_state"></a>
### `cutlass.pipeline.make_pipeline_state`

```python
cutlass.pipeline.make_pipeline_state( type: PipelineUserType, stages: int, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → PipelineState
```

Creates a pipeline state. Producers are assumed to start with an empty buffer and have a flipped phase bit of 1.

<a id="cutlass.pipeline.pipeline_init_arrive"></a>
### `cutlass.pipeline.pipeline_init_arrive`

```python
cutlass.pipeline.pipeline_init_arrive( cluster_shape_mn: cutlass.cute.typing.Layout | None = None, is_relaxed: bool = False, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

Fences the mbarrier\_init and sends an arrive if using clusters.

<a id="cutlass.pipeline.pipeline_init_wait"></a>
### `cutlass.pipeline.pipeline_init_wait`

```python
cutlass.pipeline.pipeline_init_wait( cluster_shape_mn: cutlass.cute.typing.Layout | None = None, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

Syncs the threadblock or cluster

<a id="cutlass.pipeline.agent_sync"></a>
### `cutlass.pipeline.agent_sync`

```python
cutlass.pipeline.agent_sync( group: Agent, is_relaxed: bool = False, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

Syncs all threads within an agent.

<a id="cutlass.pipeline.arrive"></a>
### `cutlass.pipeline.arrive`

```python
cutlass.pipeline.arrive( barrier_id: int, num_threads: int, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

The aligned flavor of arrive is used when all threads in the CTA will execute the
same instruction. See PTX documentation.

<a id="cutlass.pipeline.arrive_unaligned"></a>
### `cutlass.pipeline.arrive_unaligned`

```python
cutlass.pipeline.arrive_unaligned( barrier_id: int, num_threads: int, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

The unaligned flavor of arrive can be used with an arbitrary number of threads in the CTA.

<a id="cutlass.pipeline.wait"></a>
### `cutlass.pipeline.wait`

```python
cutlass.pipeline.wait( *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

NamedBarriers do not have a standalone wait like mbarriers, only an arrive\_and\_wait.
If synchronizing two warps in a producer/consumer pairing, the arrive count would be
32 using mbarriers but 64 using NamedBarriers. Only threads from either the producer
or consumer are counted for mbarriers, while all threads participating in the sync
are counted for NamedBarriers.

<a id="cutlass.pipeline.wait_unaligned"></a>
### `cutlass.pipeline.wait_unaligned`

```python
cutlass.pipeline.wait_unaligned( barrier_id: int, num_threads: int, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

<a id="cutlass.pipeline.arrive_and_wait"></a>
### `cutlass.pipeline.arrive_and_wait`

```python
cutlass.pipeline.arrive_and_wait( barrier_id: int, num_threads: int, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```

<a id="cutlass.pipeline.sync"></a>
### `cutlass.pipeline.sync`

```python
cutlass.pipeline.sync( barrier_id: int = 0, *, loc: cutlass._mlir.ir.Location | None = None, ip: cutlass._mlir.ir.InsertionPoint | None = None, ) → None
```
