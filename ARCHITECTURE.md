# Architectural Implementation

This toolkit prioritizes strict hardware-level parsing over OS-level heuristics. It strictly delineates the boundaries between physical memory acquisition, hardware page-table translation, and kernel data structure traversal.

## ARM64 Page-Table Translation
The `arm64_memory.py` module performs autonomous page-table walking to translate virtual addresses (VAs) to physical addresses (PAs). 
- **Supported Geometries:** It dynamically adapts to 4 KiB, 16 KiB, or 64 KiB translation granules and supports up to 48-bit virtual and physical addresses. 
- **Translation Mechanics:** The walker traverses up to four translation levels, parsing valid table and block descriptors while verifying TTBR0 (user) and TTBR1 (kernel) canonical address ranges.
- **Tagged Pointers:** It supports an opt-in mechanism to strip the top byte (bits 63:56) for AArch64 tagged pointers.

## Strict Physical Memory Parsing
The `lime_reader.py` module enforces strict boundary validation on LiME v1 dumps. Rather than silently zero-padding missing data, it explicitly tracks captured physical ranges and raises a `PhysicalAddressNotCaptured` exception if a page-table walk or memory read attempts to cross an uncaptured physical hole.

## BTF-Driven Process Traversal
To eliminate the brittleness of hardcoded structure offsets, `process_enumerator.py` derives all metadata dynamically from the target's BTF (BPF Type Format).
- **Task Iteration:** It resolves the runtime address of `init_task` and traverses the circular linked list via `task_struct.tasks`, explicitly checking for cyclic loops or corrupt kernel links.
- **VMA Enumeration:** The enumerator supports both the classic `mm_struct.mmap` linked list and the modern Maple Tree (`mm_struct.mm_mt`). Maple Tree internal nodes (allocation-range nodes, dense leaves, and 64-bit leaf/range nodes) are traversed by extracting precise layout constants, sizes, and type enums directly from the target kernel's BTF data.