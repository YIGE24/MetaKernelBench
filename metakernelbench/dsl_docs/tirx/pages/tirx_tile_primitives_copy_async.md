<!-- source: https://tvm.apache.org/docs/tirx/tile_primitives/copy_async.html -->

<a id="copy-async"></a>
# copy_async

Asynchronous copy. Every variant emits only the *issue* instruction — the caller is
responsible for completion (`cp.async` commit/wait for `ldgsts`; mbarrier
arrive/wait for the bulk-tensor and dsmem paths; `tcgen05.commit` /
`tcgen05.wait` for the tensor-memory paths). Selection is by the source/dest
memory pair and scope.

| Variant | Pair | Prio | Issue instruction |
| --- | --- | --- | --- |
| [copy\_async → ldgsts](tirx_tile_primitives_copy_async_ldgsts.md) | global → shared | 20 | `cp.async` (LDGSTS), per-thread vectorized |
| [copy\_async → tma](tirx_tile_primitives_copy_async_tma.md) | global ↔ shared | 10 | `cp.async.bulk.tensor` (TMA, descriptor-driven, single-thread) |
| [copy\_async → dsmem](tirx_tile_primitives_copy_async_dsmem.md) | shared → shared (cross-CTA) | 10 | `cp.async.bulk` shared::cluster (`mapa` remote address) |
| [copy\_async → tcgen05\_cp](tirx_tile_primitives_copy_async_tcgen05_cp.md) | shared → tmem | 10 | `tcgen05.cp.32x128b.warpx4` (matrix-descriptor driven) |
| [copy\_async → tcgen05\_ldst](tirx_tile_primitives_copy_async_tcgen05_ldst.md) | tmem ↔ register | 10 | `tcgen05.ld` / `tcgen05.st` (warpgroup, atom-matched) |

- [copy\_async → ldgsts](tirx_tile_primitives_copy_async_ldgsts.md)
- [copy\_async → tma](tirx_tile_primitives_copy_async_tma.md)
- [copy\_async → dsmem](tirx_tile_primitives_copy_async_dsmem.md)
- [copy\_async → tcgen05\_cp](tirx_tile_primitives_copy_async_tcgen05_cp.md)
- [copy\_async → tcgen05\_ldst](tirx_tile_primitives_copy_async_tcgen05_ldst.md)
