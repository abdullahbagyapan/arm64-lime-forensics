# ARM64 LiME Forensics Toolkit

This toolkit extracts ARM64 Linux process memory by independently translating page tables and resolving operating system structures dynamically via BTF metadata. It operates exclusively on uncompressed LiME v1 physical memory dumps and requires no third-party Python modules.

## Installation & Setup

Because this toolkit relies entirely on the Python standard library, there is no need for `pip install`, virtual environments, or external dependencies. It requires **Python 3.10 or higher** (due to modern type hinting and dataclass features).

**1. Clone the repository:**
```bash
git clone github.com/abdullahbagyapan/arm64-lime-forensics.git
cd arm64-lime-forensics/arm64-lime
```

## Acquisition

Before analyzing a memory dump, you must acquire the target kernel's metadata. The `build_profile.py` script captures runtime structures, `System.map`, kernel configuration, and BTF types into a normalized JSON profile.

Execute the builder on the target system:
```bash
sudo python3 build_profile.py kernel-profile --acquire
```

> Note: Acquisition requires effective UID 0 to access runtime kernel metadata.


## Analysis

The `lime_ps.py` utility serves as the primary forensic frontend for analyzing the captured LiME dump alongside the generated profile.

### Core Commands

- **List Processes**: Enumerate process leaders (or all threads via `--all-tasks`) from the kernel's circular `init_task.tasks` list.

```bash
python3 lime_ps.py --profile kernel-profile --dump memory.lime list
```

- **Enumerate VMAs**: List the virtual memory areas for a specific PID, including start/end addresses, sizes, and raw `vm_flags`.

```bash
python3 lime_ps.py --profile kernel-profile --dump memory.lime maps --pid 1234
```

- **Read Virtual Memory**: Extract raw bytes from a specific virtual address or a bounded slice of a listed VMA.

```bash
python3 lime_ps.py --profile kernel-profile --dump memory.lime read-region --pid 1234 --index 0 --size 4096 --output region.bin
```

- **Find Pattern**: Search for specific ASCII or UTF-8 byte sequences within the readable, captured chunks of a process's virtual memory.

```bash
python3 lime_ps.py --profile kernel-profile --dump memory.lime find --pid 1234 --pattern "secret"
```

