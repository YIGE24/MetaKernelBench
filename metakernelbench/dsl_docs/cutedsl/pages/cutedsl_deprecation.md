<!-- source: https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/deprecation.html -->

<a id="deprecation-policy"></a>
# Deprecation Policy

<a id="purpose"></a>
## Purpose

The goal of this policy is to evolve the DSL and its APIs while keeping user
programs stable. Features or APIs are deprecated only when they are redundant,
unsafe, or block better designs.

<a id="deprecation-process"></a>
## Deprecation Process

**Step 1 — Soft Deprecation**

When a feature is considered for removal, it is first annotated with the
`@deprecated` decorator or `DeprecationWarning` and documented with a
suggested alternative. At this stage, the feature continues to work normally.

Users are encouraged to provide feedback and describe their use cases.
If there is strong justification, we may keep or redesign the feature.

**Step 2 — Removal (the subsequent release)**

If no valid use cases remain, the deprecated feature will be removed in the
following **minor** release.

> **Note**
>
> The release version follows the format `<major>.<minor>.<patch>`.

<a id="communication"></a>
## Communication

All deprecations are announced through:

- This page
- In-code warning messages

<a id="soft-deprecations"></a>
## Soft Deprecations

**Version 4.2.1**

- `cute.arch.warpgroup_reg_alloc` and `cute.arch.warpgroup_reg_dealloc`
  → Scheduled for deprecation. Use `cute.arch.setmaxregister_increase` and `cute.arch.setmaxregister_decrease` instead.
- `alignment` argument in `CooperativeGroup` constructor
  → Scheduled for deprecation. It was unused; no replacement is suggested.
- `cute.AddressSpace`
  → Supported today as a quiet compatibility alias. It is scheduled for
  deprecation in a later release, where accesses will start emitting a
  deprecation warning. Use `cutlass.AddressSpace` for new code. The
  lowercase address-space members such as `gmem`, `smem`, `rmem`,
  `tmem`, and `dsmem` keep the same spelling.

<a id="deprecated-features"></a>
## Deprecated Features

*(None currently.)*
