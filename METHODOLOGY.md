# Methodology and Limitations

This toolkit treats process memory extraction as a deterministic mapping problem driven by explicit kernel profiles. It does not attempt to guess or brute-force undocumented offsets.

## Symbol and Address Resolution
Runtime symbol addresses are not blindly trusted from link-time artifacts (`System.map`). 
*   The `build_profile.py` script attempts to determine KASLR relocation by comparing link-time addresses against `/proc/kallsyms` or VMCOREINFO.
*   If runtime symbols are obfuscated (e.g., zeroed out due to `kptr_restrict`), the toolkit falls back to calculating the `kimage_voffset` by correlating the runtime `_text` address with unredacted "Kernel code" physical bounds discovered in `/proc/iomem`.

## Known Limitations
*   **Hardware Extensions:** ARM64 52-bit physical addressing (LPA2) and 52-bit virtual addressing (LVA) are explicitly unsupported; attempts to walk these geometries will raise an `UnsupportedTranslationGeometry` exception.
*   **BTF Dependency:** Because `process_enumerator.py` relies exclusively on BTF for traversing anonymous structs, unions, and the Maple Tree, target kernels lacking `/sys/kernel/btf/vmlinux` cannot be profiled or analyzed.
*   **Page-Table Swapping:** The toolkit identifies invalid or unmapped page-table descriptors but does not currently reconstruct swapped-out memory pages from disk artifacts.