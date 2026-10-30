# Methodology

## Overview

The toolkit extracts process memory from an uncompressed LiME v1 dump by combining three kinds of evidence:

1. Physical-memory ranges recorded in the LiME file.
2. ARM64 page-table structures captured in physical memory.
3. Kernel symbols and BTF metadata from the kernel represented by the dump.

The profile and dump must correspond to the same target kernel. The toolkit deliberately avoids guessing kernel structure offsets, KASLR relocation, physical address widths, or missing physical-memory contents.

## Acquisition and profile consistency

`build_profile.py` acquires and normalizes kernel metadata into `profile.json` and associated source artifacts. The acquisition is performed on the target system and requires effective UID 0 for the runtime metadata used by the builder.

The profile records kernel identity and build metadata, symbols, BTF types, configuration, boot/KASLR information, VMCOREINFO values, memory-map information, and artifact provenance where available. Compare the profile's kernel release and build ID with the dump's provenance before analysis. A mismatch can invalidate symbol resolution, structure offsets, and page-table interpretation.

A profile marked invalid is rejected by default. The CLI's `--allow-invalid-profile` option explicitly bypasses that strict rejection; it does not repair or validate the profile.

## Symbol and KASLR resolution

Runtime symbol resolution follows an evidence-based order:

1. A nonzero address from captured `/proc/kallsyms`.
2. A nonzero VMCOREINFO symbol address.
3. A `System.map` link-time address plus a captured runtime relocation, when relocation is allowed and available.
4. An unrelocated `System.map` address only when KASLR is disabled or was not compiled into the kernel.

If KASLR may be active and no defensible runtime address or relocation is available, symbol resolution fails rather than assuming the link-time address is the runtime address.

For kernel-image virtual-to-physical conversion on ARM64, the preferred `kimage_voffset` source is VMCOREINFO (`NUMBER(kimage_voffset)` or its supported alias). If absent, the profile may derive the offset from the runtime `_text` symbol and the start of exactly one visible, unredacted `/proc/iomem` `Kernel code` range:

`kimage_voffset = runtime _text - physical Kernel code start`

The derived offset must be positive and consistent with the captured evidence. This conversion is intended for kernel-image symbols, not arbitrary vmalloc or linear-map pointers.

## Physical-memory parsing

`LimeReader` parses the uncompressed LiME v1 stream and records each captured physical range and its corresponding file offset. It validates headers, versions, range bounds, truncation, and overlapping physical ranges.

A physical read must be fully covered by captured ranges. If a request crosses a physical-memory hole, `PhysicalAddressNotCaptured` is raised. The reader does not silently pad holes with zeroes, because that would turn missing evidence into fabricated memory contents.

## ARM64 virtual-to-physical translation

`ARM64Memory` derives translation geometry from the target profile and, when necessary, an explicitly supplied PA width. It supports 4 KiB, 16 KiB, and 64 KiB translation granules, with VA and PA widths up to 48 bits.

The walker validates TTBR0/user and TTBR1/kernel virtual-address ranges, follows table descriptors, resolves block mappings and page mappings, and rejects invalid or unsupported descriptor encodings. A 52-bit VA/LVA or PA/LPA2 configuration is not supported.

The kernel page-table root is resolved from the physical address of `swapper_pg_dir`, using the profile's kernel-image VA-to-PA evidence when possible. The `--kernel-pgd-paddr` option is an explicit alternative for an independently verified root physical address. The `--pa-bits` option is intended only for a PA width established by independent evidence; it must not be guessed.

Top-byte tag stripping is opt-in in the memory API. It is not silently applied to every address.

## Task and process enumeration

`ProcessEnumerator` resolves the runtime `init_task` symbol and the `task_struct.tasks` member offset from BTF. It walks the circular task list, checks for null or repeated links, enforces a traversal limit, and obtains process metadata using target-kernel member offsets.

`list` reports process leaders by default (`pid == tgid`). The `--all-tasks` option includes individual threads. Kernel threads are identified by a null `task_struct.mm` pointer and do not have a user address space to read.

## VMA enumeration

The enumerator supports two VMA representations:

- The classic linked list, using `mm_struct.mmap` and `vm_area_struct.vm_next` when both are available in BTF.
- Maple Tree storage through `mm_struct.mm_mt` when the classic list representation is unavailable and the required Maple Tree metadata exists.

VMA start and end addresses are read from `vm_area_struct`; the end address is exclusive. `vm_flags`, page offset, and backing-file pointer are included when their corresponding BTF members are available.

### Maple Tree traversal

Maple Tree layout offsets, structure sizes, and node-type enum values are resolved from the target BTF rather than copied from a different kernel build. The traversal stack carries four values:

- `node_address`: address of the current Maple node.
- `node_type`: target-kernel Maple node type.
- `depth`: traversal depth, used as a safety bound.
- `node_max`: maximum index covered by the current node, inherited from its parent.

The root begins with the maximum 64-bit index. When descending through a range or allocation-range node, the child maximum is taken from the corresponding parent pivot, or inherited for the final slot. This maximum is required when determining the active slots in 64-bit leaf/range nodes.

Dense and 64-bit leaf nodes contain VMA pointers. Range and allocation-range nodes contain encoded child-node pointers. Allocation-range active-slot information is obtained from its metadata; leaf/range active-slot calculation uses the final pivot and inherited maximum, following the target Maple Tree representation.

Maple Tree encodings and metadata are kernel implementation details. Validate traversal against synthetic nodes and dumps from each supported target-kernel family before treating results as complete. A profile with insufficient BTF must fail explicitly rather than falling back to guessed offsets.

## Process-memory reads and pattern searches

The public `read` CLI command reads from a process virtual address. Its optional `--offset` is added to `--address`; it is not a VMA index or VMA-relative offset. The command does not enforce VMA boundaries.

The `find` command enumerates a process's VMAs and scans them in chunks. It uses the internal VMA-relative read helper to keep each chunk within the selected VMA. Unreadable chunks are skipped, and the search does not match across unreadable gaps.

## Forensic interpretation

A successful translation or read establishes that the requested bytes were available through the captured page tables and captured physical ranges. It does not, by itself, establish that the profile and dump are from the same boot, that every relevant page was captured, or that a pointer is semantically valid. Correlate kernel identity, source-artifact hashes, runtime symbols, and memory-map evidence before drawing conclusions.
