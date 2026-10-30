# ARM64 LiME Forensics Toolkit

The ARM64 LiME Forensics Toolkit extracts and inspects Linux process memory from uncompressed LiME v1 physical-memory dumps. It translates ARM64 virtual addresses through the captured page tables and resolves kernel structures using metadata from the target kernel's BTF (BPF Type Format).

The toolkit uses only the Python standard library; no third-party Python packages are required.

## Requirements

- Python 3.10 or newer.
- An uncompressed LiME v1 memory dump.
- A kernel profile generated from the same target kernel and boot context as the dump.
- Matching kernel BTF metadata for structure and Maple Tree inspection.

## Installation

Clone the repository and enter the toolkit directory:

```bash
git clone https://github.com/abdullahbagyapan/arm64-lime-forensics.git
cd arm64-lime-forensics/arm64-lime
```

No `pip install` step is required.

## Acquisition

Acquire the target kernel metadata before analyzing its memory dump. The `build_profile.py` script collects and normalizes kernel metadata, including runtime symbols, `System.map`, kernel configuration, BTF types, and supporting artifacts.

Run the acquisition on the target system:

```bash
sudo python3 build_profile.py kernel-profile --acquire
```

Acquisition requires effective UID 0 to access the runtime kernel metadata and artifacts.

Keep the generated profile and the corresponding memory dump together. The profile must describe the kernel represented in the dump; using metadata from a different kernel build or boot can produce incorrect translations and structure interpretation.

## Analysis

`lime_ps.py` is the command-line frontend for process enumeration, VMA inspection, virtual-memory reads, and byte-pattern searches.

Global options must appear before the subcommand:

| Option | Purpose |
|---|---|
| `--profile DIR` | Directory containing `profile.json` and its source artifacts. |
| `--dump FILE` | Uncompressed LiME v1 dump. |
| `--pa-bits BITS` | Explicit target physical-address width, only when independently evidenced. |
| `--kernel-pgd-paddr ADDRESS` | Independently verified physical address of `swapper_pg_dir`. |
| `--allow-invalid-profile` | Permit loading a profile whose validation status is invalid; use only when the implications are understood. |

### List processes and tasks

List process leaders:

```bash
python3 lime_ps.py --profile kernel-profile --dump memory.lime list
```

Include individual threads/tasks:

```bash
python3 lime_ps.py --profile kernel-profile --dump memory.lime list --all-tasks
```

### Enumerate VMAs

List the virtual memory areas for a process:

```bash
python3 lime_ps.py --profile kernel-profile --dump memory.lime maps --pid 1234
```

The output includes each VMA's start and end virtual addresses, size, raw `vm_flags` value when available, page offset, and `vm_file` pointer when available.

### Read process virtual memory

Read bytes starting at a process virtual address:

```bash
python3 lime_ps.py --profile kernel-profile --dump memory.lime \
    read --pid 1234 --address 0x7fff0000 --size 4096
```

Apply an additional byte offset to the supplied address:

```bash
python3 lime_ps.py --profile kernel-profile --dump memory.lime \
    read --pid 1234 --address 0x7fff0000 --offset 0x100 --size 128
```

The effective address is `--address + --offset`. The offset defaults to zero. `read` is the only public CLI memory-read subcommand; the former `read-region` command has been removed.

By default, bytes are printed as a hexadecimal dump with an ASCII column. Use `--output` to write raw bytes to a file:

```bash
python3 lime_ps.py --profile kernel-profile --dump memory.lime \
    read --pid 1234 --address 0x7fff0000 --size 4096 --output region.bin
```

The CLI limits an individual read to 64 MiB. Hexadecimal terminal output is limited to 1 MiB; use `--output` for larger reads. A direct `read` is address-based and does not require selecting a VMA or enforce a VMA boundary. Ensure the supplied virtual-address range is the range you intend to inspect.

### Find a byte pattern

Search a process's readable, captured VMA chunks for an ASCII or UTF-8 string:

```bash
python3 lime_ps.py --profile kernel-profile --dump memory.lime \
    find --pid 1234 --pattern "secret"
```

Limit the number of reported matches (default: 20):

```bash
python3 lime_ps.py --profile kernel-profile --dump memory.lime \
    find --pid 1234 --pattern "secret" --max-hits 5
```

Unreadable or uncaptured chunks are skipped, and matches are not joined across those gaps. The search uses VMA-relative reads internally; this is an implementation detail, not a separate CLI command.

## Methodology and limitations

The toolkit treats memory extraction as a mapping and metadata-resolution problem. It does not guess undocumented structure offsets or silently fill missing physical-memory ranges.

### Symbol and address resolution

Runtime symbol resolution prefers nonzero runtime addresses from captured `/proc/kallsyms`, then VMCOREINFO symbols. When those are unavailable, it can use `System.map` together with a defensible KASLR relocation. An unrelocated `System.map` address is used only when KASLR is disabled or was not compiled into the kernel.

For ARM64 kernel-image VA-to-PA conversion, `kimage_voffset` is taken from VMCOREINFO when available. Otherwise, the profile can derive it from the runtime `_text` address and the physical start of the captured `/proc/iomem` `Kernel code` range. The fallback requires unredacted, consistent evidence; it does not invent an offset when the evidence is missing.

### Known limitations

- **Address-size extensions:** 52-bit physical addressing (LPA2) and 52-bit virtual addressing (LVA) are unsupported. The current page-table walker supports up to 48-bit VA and PA widths.
- **Translation granules:** Supported ARM64 granules are 4 KiB, 16 KiB, and 64 KiB.
- **BTF:** Structure offsets and Maple Tree layouts are resolved from target-kernel BTF. The profile must contain the required types and members. Maple Tree inspection may require building the profile with `--all-btf`.
- **Swapped pages:** Invalid or unmapped page-table descriptors are reported, but swapped-out pages are not reconstructed from swap artifacts.
- **Physical-memory holes:** Reads that cross uncaptured physical ranges fail explicitly; missing bytes are not zero-filled.
- **Read boundaries:** The CLI `read` command reads from a supplied process virtual address and does not restrict the request to a VMA. Use `maps` to inspect VMA boundaries before selecting an address range.

See [methodology.md](methodology.md) for the analysis assumptions and [architecture.md](architecture.md) for module responsibilities and traversal details.
