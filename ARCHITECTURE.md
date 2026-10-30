# Architecture

## Design

The toolkit separates physical-memory parsing, kernel-profile interpretation, ARM64 address translation, kernel data-structure traversal, and the command-line interface. Each layer has a narrow responsibility and reports missing or inconsistent evidence explicitly.

```text
                  build_profile.py
                         |
                         v
                  kernel-profile/
                    profile.json
                         |
                         v
lime_ps.py ---> KernelProfile <--- profile source artifacts
    |
    v
ProcessEnumerator
    |              \
    |               +---- BTF offsets, sizes, and enum values
    v
ARM64Memory <---- kernel page-table root / process PGD
    |
    v
LimeReader <------ uncompressed LiME v1 dump
```

## Modules

### `build_profile.py`

Runs on the target system to acquire kernel metadata and generate the normalized profile. The profile provides kernel identity, symbol data, BTF types, configuration, boot/KASLR information, VMCOREINFO, memory-map evidence, and source-artifact provenance as available.

### `kernel_profile.py`

Loads `profile.json`, validates its format/version and basic section types, and exposes metadata to the other modules. Its responsibilities include:

- Kernel release, build ID, architecture, page size, VA bits, and PA bits.
- Validation status and issues.
- Source-artifact lookup and SHA-256 metadata.
- Runtime and link-time symbol resolution.
- ARM64 kernel-image `kimage_voffset` derivation and kernel-image symbol VA-to-PA conversion.
- BTF type, structure-size, member-offset, bit-offset, and member-type lookup, including anonymous struct/union members.
- Kernel configuration and loaded-module metadata.

Frequently used symbol, BTF, member, and source lookups are cached for repeated memory-walker operations.

### `lime_reader.py`

Parses uncompressed LiME v1 dumps. It records captured physical ranges, maps physical addresses to file offsets, and supports physical reads across adjacent captured ranges. It rejects malformed headers, unsupported versions, truncation, overlapping physical ranges, and reads that cross uncaptured holes.

### `arm64_memory.py`

Implements ARM64 virtual-to-physical translation and virtual-memory reads. It:

- Derives page-table geometry from page size, VA width, and PA width.
- Supports 4 KiB, 16 KiB, and 64 KiB granules and up to 48-bit VA/PA widths.
- Validates TTBR0 and TTBR1 address ranges.
- Walks table descriptors and resolves page and block mappings.
- Reads descriptors and translated physical memory through `LimeReader`.
- Resolves the kernel page-table root from verified evidence or an explicitly supplied physical address.
- Supports optional top-byte tag handling through the memory API.

Unsupported translation geometries and invalid descriptors raise explicit errors.

### `process_enumerator.py`

Uses `KernelProfile` and `ARM64Memory` to inspect kernel task and process data. It:

- Walks the circular `init_task.tasks` list using BTF-derived offsets.
- Enumerates process leaders or all tasks/threads.
- Resolves a process's `mm_struct.pgd` to a physical page-table root.
- Enumerates VMAs using either the classic linked list or Maple Tree.
- Reads process virtual memory and bounded slices of enumerated VMAs.

#### Maple Tree implementation

The Maple Tree walker obtains node layouts and `maple_type` enum values from target BTF. It supports dense nodes, 64-bit leaf nodes, 64-bit range nodes, and allocation-range nodes when the required BTF records are present.

Its traversal stack is:

```python
# (node_address, node_type, depth, node_max)
stack: list[tuple[int, int, int, int]] = []
```

`node_max` is the maximum index covered by the current node. The root is initialized to the maximum 64-bit index. When traversing a child slot, the maximum is inherited from the parent pivot associated with that slot; the final slot inherits the parent's maximum.

The inherited maximum is used with the final pivot to determine the active slot count for 64-bit leaf/range nodes. Allocation-range nodes use their `metadata.end` value. Dense nodes use the dense slot array. Leaf slots contain VMA pointers; internal range/arange slots contain encoded child-node pointers.

The walker rejects unsupported node types, invalid child encodings, repeated nodes, duplicate VMA pointers, excessive depth, and traversal-limit violations. The logic should be covered by synthetic fixtures for partially populated and full nodes, final-pivot boundary cases, nested nodes, and malformed metadata.

### `lime_ps.py`

Provides the public CLI:

| Command | Function |
|---|---|
| `list` | List process leaders, or all tasks with `--all-tasks`. |
| `maps` | Enumerate VMAs for a PID. |
| `read` | Read process virtual memory from `--address`, with optional additive `--offset`. |
| `find` | Search readable, captured VMA chunks for a text pattern. |

The former `read-region` CLI command has been removed. VMA-relative bounded reads remain an internal `ProcessEnumerator` API used by `find`; they are not exposed as a separate CLI subcommand.

The `read` command writes a hexdump to standard output by default or raw bytes to `--output`. CLI limits are 64 MiB per read and 1 MiB for hexdump output.

## Data flow

### Process listing

1. `lime_ps.py` loads the profile and opens the LiME dump.
2. `ARM64Memory` validates the target ARM64 translation geometry.
3. `ProcessEnumerator` resolves `init_task` and BTF offsets.
4. Kernel virtual reads are translated through the kernel page-table root and serviced by `LimeReader`.
5. Task records are formatted by the CLI.

### Process memory read

1. The enumerator locates the requested task and obtains its `mm_struct`.
2. It reads `mm_struct.pgd` and translates that kernel pointer to a physical page-table root.
3. `ARM64Memory` walks the process page tables for the requested VA.
4. The translated physical span is read from captured LiME ranges.
5. The CLI prints a hexdump or writes the raw bytes.

### Pattern search

1. The enumerator lists the process VMAs.
2. The CLI scans each VMA in bounded chunks using the internal VMA-relative read helper.
3. Unreadable chunks are skipped; the match tail is reset so a result cannot span an unreadable gap.
4. Matching virtual addresses and their VMA-relative offsets are reported.

## Error boundaries

The modules use distinct exception families so callers can distinguish profile-format/lookup failures, LiME-format or missing-physical-range failures, ARM64 translation failures, and process/VMA traversal failures. Unsupported or inconsistent metadata should stop the relevant operation rather than trigger guessed offsets or fabricated memory contents.
