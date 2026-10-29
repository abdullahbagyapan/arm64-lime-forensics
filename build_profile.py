#!/usr/bin/env python3
"""
ARM64 Linux kernel-profile acquisition and profile builder.

This script replaces both:
    get_metadata.sh
    build_profile.py

Modes
-----
Build an existing acquisition directory:
    python3 build_profile.py kernel-profile

Acquire metadata from the currently running Linux kernel and then build:
    sudo python3 build_profile.py kernel-profile --acquire

Acquire with an exact matching vmlinux ELF:
    sudo python3 build_profile.py kernel-profile --acquire \
        --vmlinux /path/to/vmlinux

The acquisition mode requires effective UID 0.

The resulting directory can contain, among other artifacts:
    System.map
    vmlinux.btf
    vmlinux
    config
    cmdline
    uname.txt
    os-release
    kallsyms
    iomem
    cpuinfo
    modules.txt
    meminfo
    vmstat
    zoneinfo
    buddyinfo
    pagetypeinfo
    slabinfo
    allocinfo
    kernel-notes.bin
    module-sections/
    module-parameters/
    module-info/
    btf-modules/
    memory-blocks/
    cpu-topology/
    numa/
    device-tree/
    device-tree.dtb
    acpi-tables/
    security/
    hardening/
    FILES.txt
    SHA256SUMS
    profile.json

No third-party Python modules are required.
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import gzip
import hashlib
import json
import os
import re
import shlex
import shutil
import stat
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

# ===========================================================================
# Constants
# ===========================================================================

CORE_FILES = (
    "System.map",
    "vmlinux.btf",
)

PROFILE_FORMAT = "lime-arm64-kernel-profile"
PROFILE_VERSION = 2

DEFAULT_SYMBOLS = {
    "init_task",
    "init_mm",
    "_text",
    "_stext",
    "_etext",
    "_end",
    "swapper_pg_dir",
    "idmap_pg_dir",
    "tramp_pg_dir",
    "empty_zero_page",
    "memstart_addr",
    "kimage_voffset",
    "physvirt_offset",
    "vmemmap",
    "high_memory",
    "modules",
    "module_kset",
}

DEFAULT_BTF_TYPES = {
    "task_struct",
    "list_head",
    "mm_struct",
    "vm_area_struct",
    "maple_tree",
    "maple_node",
    "maple_range_64",
    "maple_arange_64",
    "maple_metadata",
    "maple_type",
    "cred",
    "files_struct",
    "fdtable",
    "file",
    "path",
    "dentry",
    "vfsmount",
    "mount",
    "signal_struct",
    "sighand_struct",
    "nsproxy",
    "pid",
    "upid",
    "pid_namespace",
    "thread_struct",
    "thread_info",
    "page",
}

IMPORTANT_CONFIG_PREFIXES = (
    "CONFIG_ARM64",
    "CONFIG_PAGE_SIZE",
    "CONFIG_PGTABLE_LEVELS",
    "CONFIG_RANDOMIZE_BASE",
    "CONFIG_RELOCATABLE",
    "CONFIG_DEBUG_INFO",
    "CONFIG_DEBUG_INFO_BTF",
    "CONFIG_IKCONFIG",
    "CONFIG_IKCONFIG_PROC",
    "CONFIG_KALLSYMS",
    "CONFIG_KALLSYMS_ALL",
    "CONFIG_MODULES",
    "CONFIG_SMP",
    "CONFIG_NUMA",
    "CONFIG_VMAP_STACK",
    "CONFIG_THREAD_INFO_IN_TASK",
    "CONFIG_PID_NS",
    "CONFIG_USER_NS",
    "CONFIG_MEMCG",
)


# ===========================================================================
# Errors
# ===========================================================================

class ProfileError(RuntimeError):
    """Profile generation or acquisition failed."""

class AcquisitionError(ProfileError):
    """Running-kernel metadata acquisition failed."""


# ===========================================================================
# Generic helpers
# ===========================================================================

def log(message: str) -> None:
    print(f"[+] {message}")

def warn(message: str) -> None:
    print(f"[!] {message}", file=sys.stderr)

def _utc_now() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat().replace("+00:00", "Z")

def _utc_now_seconds() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def read_text(path: Path) -> str:
    """
    Read UTF-8 text defensively.
    An absent optional artifact returns an empty string.
    """
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except (FileNotFoundError, PermissionError, OSError):
        return ""

def read_text_strict(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="strict")

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def hexaddr(value: int) -> str:
    return f"0x{value:x}"

def parse_num(value: str) -> int:
    return int(value, 0)

def file_info(path: Path) -> dict[str, Any]:
    st = path.stat()
    return {
        "path": path.name,
        "size": st.st_size,
        "sha256": sha256_file(path),
    }

def _safe_relpath(root: Path, path: Path) -> Optional[str]:
    try:
        resolved_root = root.resolve()
        resolved = path.resolve()
        resolved.relative_to(resolved_root)
        return path.relative_to(root).as_posix()
    except (ValueError, OSError):
        return None

def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def _write_text(path: Path, value: str) -> None:
    _ensure_parent(path)
    path.write_text(value, encoding="utf-8")

def _write_text_line(path: Path, value: str) -> None:
    _write_text(path, value.rstrip("\n") + "\n")

def _read_proc_uptime() -> str:
    try:
        raw = Path("/proc/uptime").read_text(encoding="ascii", errors="strict")
        return raw.split()[0]
    except (OSError, IndexError, UnicodeError):
        return ""

def _require_root() -> None:
    """
    Require root for acquisition.
    os.geteuid() is intentionally used rather than checking $USER or sudo environment variables.
    """
    geteuid = getattr(os, "geteuid", None)
    if geteuid is None:
        raise AcquisitionError("--acquire is supported only on Unix-like systems exposing effective UID")
    if geteuid() != 0:
        raise AcquisitionError("--acquire requires root privileges. Run this command with sudo or as UID 0.")


# ===========================================================================
# Acquisition helpers
# ===========================================================================

def _source_readable_file(path: Path) -> bool:
    """
    Check whether a path represents something that can be opened as a file.
    procfs/sysfs pseudo-files do not always behave like ordinary disk files,
    so access/open behavior matters more than inode type alone.
    """
    try:
        return path.exists() and not path.is_dir() and os.access(path, os.R_OK)
    except OSError:
        return False

def _copy_optional(source: Path, destination: Path, *, binary: bool = True) -> bool:
    if not _source_readable_file(source):
        warn(f"not readable or absent: {source}")
        return False
    
    _ensure_parent(destination)
    try:
        if binary:
            with source.open("rb") as src:
                with destination.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
        else:
            with source.open("r", encoding="utf-8", errors="replace") as src:
                with destination.open("w", encoding="utf-8") as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
        log(f"captured {destination}")
        return True
    except OSError as exc:
        try:
            destination.unlink(missing_ok=True)
        except OSError:
            pass
        warn(f"failed to copy {source}: {exc}")
        return False

def _text_optional(source: Path, destination: Path) -> bool:
    if not _source_readable_file(source):
        warn(f"not readable or absent: {source}")
        return False
    
    _ensure_parent(destination)
    try:
        with source.open("rb") as src:
            raw = src.read()
        
        text = raw.decode("utf-8", errors="replace")
        destination.write_text(text, encoding="utf-8")
        log(f"captured {destination}")
        return True
    except OSError as exc:
        try:
            destination.unlink(missing_ok=True)
        except OSError:
            pass
        warn(f"failed to read {source}: {exc}")
        return False

def _copy_device_tree_text_property(source: Path, destination: Path) -> bool:
    """
    Convert NUL-separated device-tree string properties into readable text.
    """
    if not _source_readable_file(source):
        return False
    
    try:
        raw = source.read_bytes()
        text = raw.replace(b"\0", b"\n").decode("utf-8", errors="replace")
        _write_text(destination, text)
        return True
    except OSError as exc:
        warn(f"failed to capture device-tree property {source}: {exc}")
        return False

def _capture_kernel_config(output_dir: Path, kernel_release: str) -> None:
    boot_config = Path(f"/boot/config-{kernel_release}")
    destination = output_dir / "config"
    
    if _source_readable_file(boot_config):
        _copy_optional(boot_config, destination, binary=False)
        return

    proc_config = Path("/proc/config.gz")
    if _source_readable_file(proc_config):
        try:
            with gzip.open(proc_config, "rt", encoding="utf-8", errors="replace") as source:
                content = source.read()
            _write_text(destination, content)
            log("captured config from /proc/config.gz")
            return
        except (OSError, EOFError, gzip.BadGzipFile) as exc:
            try:
                destination.unlink(missing_ok=True)
            except OSError:
                pass
            warn(f"could not decompress /proc/config.gz: {exc}")
            return
            
    warn("kernel config unavailable; profile builder will mark derived values unknown")

def _capture_cpu_topology(output_dir: Path) -> None:
    base = Path("/sys/devices/system/cpu")
    for key in ("online", "offline", "possible", "present"):
        _text_optional(base / key, output_dir / "cpu-topology" / key)

def _capture_numa(output_dir: Path) -> None:
    base = Path("/sys/devices/system/node")
    for key in ("online", "possible"):
        _text_optional(base / key, output_dir / "numa" / key)
        
    if not base.is_dir():
        return
        
    for node_path in sorted(base.glob("node[0-9]*")):
        if not node_path.is_dir():
            continue
        node = node_path.name
        for key in ("meminfo", "cpulist", "distance"):
            _text_optional(node_path / key, output_dir / "numa" / node / key)

def _capture_memory_blocks(output_dir: Path) -> None:
    base = Path("/sys/devices/system/memory")
    if not base.is_dir():
        return
        
    for block_path in sorted(base.glob("memory[0-9]*")):
        if not block_path.is_dir():
            continue
        block = block_path.name
        for key in ("phys_index", "state", "removable", "valid_zones"):
            _text_optional(block_path / key, output_dir / "memory-blocks" / block / key)

def _capture_modules(output_dir: Path) -> None:
    sys_module = Path("/sys/module")
    if not sys_module.is_dir():
        return
        
    kernel_btf = Path("/sys/kernel/btf")
    for module_path in sorted(sys_module.iterdir()):
        if not module_path.is_dir():
            continue
            
        module = module_path.name
        _text_optional(module_path / "refcnt", output_dir / "module-info" / module / "refcnt")
        
        sections = module_path / "sections"
        if sections.is_dir():
            for section_path in sorted(sections.iterdir()):
                if section_path.is_dir():
                    continue
                _text_optional(section_path, output_dir / "module-sections" / module / section_path.name)
                
        parameters = module_path / "parameters"
        if parameters.is_dir():
            for parameter in sorted(parameters.iterdir()):
                if parameter.is_dir():
                    continue
                if not os.access(parameter, os.R_OK):
                    continue
                _text_optional(parameter, output_dir / "module-parameters" / module / f"{parameter.name}.txt")
                
        module_btf = kernel_btf / module
        if _source_readable_file(module_btf):
            _copy_optional(module_btf, output_dir / "btf-modules" / f"{module}.btf")

def _capture_firmware(output_dir: Path) -> None:
    _copy_optional(Path("/sys/firmware/fdt"), output_dir / "device-tree.dtb")
    
    dt_base = Path("/sys/firmware/devicetree/base")
    if dt_base.is_dir():
        for rel in ("model", "compatible", "name", "#address-cells", "#size-cells", "#interrupt-cells"):
            source = dt_base / rel
            if not _source_readable_file(source):
                continue
            destination_name = rel.replace("/", "_") + ".txt"
            _copy_device_tree_text_property(source, output_dir / "device-tree" / destination_name)
            
        chosen_bootargs = dt_base / "chosen" / "bootargs"
        if _source_readable_file(chosen_bootargs):
            _copy_device_tree_text_property(chosen_bootargs, output_dir / "device-tree" / "chosen-bootargs.txt")
            
        chosen_stdout = dt_base / "chosen" / "stdout-path"
        if _source_readable_file(chosen_stdout):
            _copy_device_tree_text_property(chosen_stdout, output_dir / "device-tree" / "chosen-stdout-path.txt")
            
    acpi_base = Path("/sys/firmware/acpi/tables")
    if acpi_base.is_dir():
        for table in sorted(acpi_base.iterdir()):
            if table.is_dir():
                continue
            if not os.access(table, os.R_OK):
                continue
            _copy_optional(table, output_dir / "acpi-tables" / table.name)

def _capture_security(output_dir: Path) -> None:
    _text_optional(Path("/sys/kernel/security/lockdown"), output_dir / "security" / "lockdown.txt")
    _text_optional(Path("/proc/sys/kernel/randomize_va_space"), output_dir / "hardening" / "randomize_va_space.txt")
    _text_optional(Path("/proc/sys/kernel/kptr_restrict"), output_dir / "hardening" / "kptr_restrict.txt")
    _text_optional(Path("/proc/sys/kernel/dmesg_restrict"), output_dir / "hardening" / "dmesg_restrict.txt")

def _inventory_paths(output_dir: Path) -> list[tuple[str, Path]]:
    """Return all raw acquisition artifacts eligible for FILES.txt/SHA256SUMS."""
    result: list[tuple[str, Path]] = []
    for path in output_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(output_dir).as_posix()
        if rel in {"SHA256SUMS", "FILES.txt", "profile.json"}:
            continue
        result.append((rel, path))
    result.sort(key=lambda item: item[0])
    return result

def _write_manifests(output_dir: Path) -> None:
    items = _inventory_paths(output_dir)
    files_txt = output_dir / "FILES.txt"
    sha_txt = output_dir / "SHA256SUMS"
    
    with files_txt.open("w", encoding="utf-8") as handle:
        for rel, _ in items:
            handle.write(f"./{rel}\n")
            
    with sha_txt.open("w", encoding="utf-8") as handle:
        for rel, path in items:
            digest = sha256_file(path)
            handle.write(f"{digest}  ./{rel}\n")

def acquire_metadata(output_dir: Path, *, vmlinux_path: Optional[Path] = None) -> None:
    """Acquire metadata from the currently running Linux kernel."""
    _require_root()
    if not sys.platform.startswith("linux"):
        raise AcquisitionError("--acquire is supported only on Linux")
        
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        os.chmod(output_dir, stat.S_IRWXU)
    except OSError:
        pass
        
    try:
        uname = os.uname()
    except AttributeError as exc:
        raise AcquisitionError("os.uname() is unavailable") from exc
        
    kernel_release = uname.release
    log(f"Kernel release: {kernel_release}")
    log(f"Output: {output_dir}")

    _write_text_line(output_dir / "capture-start-utc.txt", _utc_now_seconds())
    _write_text_line(output_dir / "capture-start-uptime.txt", _read_proc_uptime())

    _copy_optional(Path(f"/boot/System.map-{kernel_release}"), output_dir / "System.map", binary=False)
    _copy_optional(Path("/sys/kernel/btf/vmlinux"), output_dir / "vmlinux.btf")
    _capture_kernel_config(output_dir, kernel_release)
    
    if vmlinux_path is not None:
        source_vmlinux = vmlinux_path.expanduser().resolve()
        if not source_vmlinux.is_file():
            raise AcquisitionError(f"--vmlinux does not point to a readable file: {source_vmlinux}")
        _copy_optional(source_vmlinux, output_dir / "vmlinux")

    _write_text_line(output_dir / "kernel-release.txt", kernel_release)
    uname_text = " ".join((uname.sysname, uname.nodename, uname.release, uname.version, uname.machine))
    _write_text_line(output_dir / "uname.txt", uname_text)
    
    _text_optional(Path("/proc/version"), output_dir / "proc-version.txt")
    _text_optional(Path("/etc/os-release"), output_dir / "os-release")
    _text_optional(Path("/proc/cmdline"), output_dir / "cmdline")

    runtime_text_files = (
        ("/proc/kallsyms", "kallsyms"),
        ("/proc/iomem", "iomem"),
        ("/proc/cpuinfo", "cpuinfo"),
        ("/proc/modules", "modules.txt"),
        ("/proc/meminfo", "meminfo"),
        ("/proc/vmstat", "vmstat"),
        ("/proc/zoneinfo", "zoneinfo"),
        ("/proc/buddyinfo", "buddyinfo"),
        ("/proc/pagetypeinfo", "pagetypeinfo"),
        ("/proc/slabinfo", "slabinfo"),
        ("/proc/allocinfo", "allocinfo"),
    )
    for source, destination in runtime_text_files:
        _text_optional(Path(source), output_dir / destination)

    runtime_identity = (
        ("/sys/kernel/ostype", "runtime-ostype.txt"),
        ("/sys/kernel/osrelease", "runtime-osrelease.txt"),
        ("/sys/kernel/version", "runtime-version.txt"),
        ("/proc/sys/kernel/tainted", "tainted.txt"),
    )
    for source, destination in runtime_identity:
        _text_optional(Path(source), output_dir / destination)

    _copy_optional(Path("/sys/kernel/notes"), output_dir / "kernel-notes.bin")
    _text_optional(Path("/sys/kernel/vmcoreinfo"), output_dir / "vmcoreinfo-location.txt")
    
    _capture_cpu_topology(output_dir)
    _capture_numa(output_dir)
    _capture_memory_blocks(output_dir)
    _capture_modules(output_dir)
    _capture_firmware(output_dir)
    _capture_security(output_dir)

    _write_text_line(output_dir / "capture-end-utc.txt", _utc_now_seconds())
    _write_text_line(output_dir / "capture-end-uptime.txt", _read_proc_uptime())
    _write_manifests(output_dir)

    core_present = [name for name in CORE_FILES if (output_dir / name).is_file() and (output_dir / name).stat().st_size > 0]
    
    log("Metadata acquisition complete.")
    log("Core artifacts present: " + (" ".join(core_present) if core_present else "(none)"))
    
    missing_core = [name for name in CORE_FILES if name not in core_present]
    if missing_core:
        raise AcquisitionError(
            "Acquisition completed, but required core artifact(s) are missing: "
            + ", ".join(missing_core)
            + ". Ensure the matching System.map and /sys/kernel/btf/vmlinux are available."
        )


# ===========================================================================
# SHA256SUMS
# ===========================================================================

_SHA_LINE = re.compile(r"^([0-9a-fA-F]{64})\s+[* ]?(.+)$")

def parse_sha256sums(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in read_text(path).splitlines():
        line = raw.strip()
        if not line:
            continue
        match = _SHA_LINE.match(line)
        if not match:
            continue
        name = match.group(2).strip()
        while name.startswith("./"):
            name = name[2:]
        result[name] = match.group(1).lower()
    return result

def verify_sha256sums(profile_dir: Path, expected: dict[str, str]) -> dict[str, Any]:
    entries: dict[str, Any] = {}
    overall = True
    
    for name, wanted in sorted(expected.items()):
        path = profile_dir / name
        if name == "SHA256SUMS":
            entries[name] = {"status": "self-entry-ignored", "expected": wanted, "actual": None}
            continue
            
        if not path.is_file():
            entries[name] = {"status": "missing", "expected": wanted, "actual": None}
            overall = False
            continue
            
        actual = sha256_file(path)
        ok = (actual == wanted)
        entries[name] = {"status": "ok" if ok else "mismatch", "expected": wanted, "actual": actual}
        overall = (overall and ok)

    manifested = {name for name in expected if name not in {"SHA256SUMS", "FILES.txt", "profile.json"}}
    actual_files: set[str] = set()
    
    for path in profile_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.name in {"SHA256SUMS", "FILES.txt", "profile.json"}:
            continue
        rel = _safe_relpath(profile_dir, path)
        if rel is not None:
            actual_files.add(rel)
            
    unmanifested = sorted(actual_files - manifested)
    if unmanifested:
        entries["__unmanifested_files__"] = {"status": "unmanifested", "files": unmanifested}
        overall = False
        
    return {"ok": overall, "files": entries, "unmanifested_files": unmanifested}


# ===========================================================================
# OS identity / uname / command line
# ===========================================================================

def parse_os_release(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in read_text(path).splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        try:
            parsed = shlex.split(value, posix=True)
            result[key] = parsed[0] if len(parsed) == 1 else value.strip('"')
        except ValueError:
            result[key] = value.strip().strip('"')
    return result

def parse_uname(path: Path) -> dict[str, Any]:
    raw = read_text(path).strip()
    tokens = raw.split()
    known_arches = {"aarch64", "arm64", "x86_64", "riscv64", "ppc64", "ppc64le", "s390x"}
    arch = next((token for token in reversed(tokens) if token in known_arches), None)
    return {
        "raw": raw,
        "sysname": tokens[0] if tokens else None,
        "hostname": tokens[1] if len(tokens) > 1 else None,
        "release_hint": tokens[2] if len(tokens) > 2 else None,
        "architecture_hint": arch,
    }

def parse_cmdline(path: Path) -> dict[str, Any]:
    raw = read_text(path).strip()
    try:
        argv = shlex.split(raw)
    except ValueError:
        argv = raw.split()
        
    params: dict[str, list[Optional[str]]] = {}
    for token in argv:
        if "=" in token:
            key, value = token.split("=", 1)
        else:
            key = token
            value = None
        params.setdefault(key, []).append(value)
        
    return {"raw": raw, "argv": argv, "parameters": params}


# ===========================================================================
# Kernel config
# ===========================================================================

def parse_kernel_config(path: Path) -> dict[str, Any]:
    all_values: dict[str, Any] = {}
    for raw in read_text(path).splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("# CONFIG_") and line.endswith(" is not set"):
            name = line[len("# "):].split()[0]
            all_values[name] = False
            continue
        if "=" not in line or not line.startswith("CONFIG_"):
            continue
            
        name, value = line.split("=", 1)
        if value == "y":
            parsed: Any = True
        elif value == "n":
            parsed = False
        elif value.startswith('"') and value.endswith('"'):
            parsed = value[1:-1]
        else:
            try:
                parsed = int(value, 0)
            except ValueError:
                parsed = value
        all_values[name] = parsed
        
    relevant = {
        key: value for key, value in all_values.items()
        if any(key.startswith(prefix) for prefix in IMPORTANT_CONFIG_PREFIXES)
    }
    
    return {
        "all_count": len(all_values),
        "relevant": dict(sorted(relevant.items())),
        "_all": all_values,
    }

def derive_page_size(config: dict[str, Any]) -> Optional[int]:
    selected = [
        value for key, value in (
            ("CONFIG_ARM64_4K_PAGES", 4096),
            ("CONFIG_ARM64_16K_PAGES", 16384),
            ("CONFIG_ARM64_64K_PAGES", 65536),
        ) if config.get(key) is True
    ]
    if len(set(selected)) == 1:
        return selected[0]
    if len(set(selected)) > 1:
        return None
        
    value = config.get("CONFIG_PAGE_SIZE")
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None

def derive_va_bits(config: dict[str, Any]) -> Optional[int]:
    direct = config.get("CONFIG_ARM64_VA_BITS")
    if isinstance(direct, int) and not isinstance(direct, bool):
        return direct
        
    for bits in (36, 39, 42, 47, 48, 52):
        if config.get(f"CONFIG_ARM64_VA_BITS_{bits}") is True:
            return bits
    return None

def derive_pa_bits(config: dict[str, Any]) -> Optional[int]:
    direct = config.get("CONFIG_ARM64_PA_BITS")
    if isinstance(direct, int) and not isinstance(direct, bool):
        return direct
    if config.get("CONFIG_ARM64_PA_BITS_52") is True:
        return 52
    if config.get("CONFIG_ARM64") is True and config.get("CONFIG_ARM64_PA_BITS_52") is False:
        return 48
    return None


# ===========================================================================
# Symbols
# ===========================================================================

@dataclass
class Symbol:
    address: int
    sym_type: str
    name: str
    module: Optional[str] = None

def parse_symbol_file(path: Path) -> list[Symbol]:
    symbols: list[Symbol] = []
    for raw in read_text(path).splitlines():
        fields = raw.split()
        if len(fields) < 3:
            continue
        addr_s, sym_type, name = fields[:3]
        module = None
        if len(fields) >= 4 and fields[3].startswith("[") and fields[3].endswith("]"):
            module = fields[3][1:-1]
        try:
            address = int(addr_s, 16)
        except ValueError:
            continue
        symbols.append(Symbol(address, sym_type, name, module))
    return symbols

def symbol_multimap(symbols: Iterable[Symbol]) -> dict[str, list[Symbol]]:
    result: dict[str, list[Symbol]] = {}
    for sym in symbols:
        result.setdefault(sym.name, []).append(sym)
    return result

def serialize_symbol(sym: Symbol) -> dict[str, Any]:
    result: dict[str, Any] = {
        "address": hexaddr(sym.address),
        "type": sym.sym_type,
    }
    if sym.module:
        result["module"] = sym.module
    return result

def selected_symbols(system_map: list[Symbol], kallsyms: list[Symbol], names: set[str], all_symbols: bool) -> dict[str, Any]:
    sm = symbol_multimap(system_map)
    ks = symbol_multimap(kallsyms)
    requested = sorted(set(sm) | set(ks)) if all_symbols else sorted(names)
    
    output: dict[str, Any] = {}
    for name in requested:
        entry: dict[str, Any] = {}
        if name in sm:
            entry["system_map"] = [serialize_symbol(sym) for sym in sm[name]]
        if name in ks:
            entry["kallsyms"] = [serialize_symbol(sym) for sym in ks[name]]
        if entry:
            output[name] = entry
    return output

def derive_kaslr(system_map: list[Symbol], kallsyms: list[Symbol]) -> dict[str, Any]:
    sm = symbol_multimap(system_map)
    ks = symbol_multimap(kallsyms)
    candidates = ("_text", "_stext", "_etext", "init_task", "init_mm")
    
    observations: list[dict[str, Any]] = []
    deltas: list[int] = []
    
    for name in candidates:
        if name not in sm or name not in ks:
            continue
            
        a = next((sym for sym in sm[name] if sym.module is None), sm[name][0])
        b = next((sym for sym in ks[name] if sym.module is None), ks[name][0])
        
        if b.address == 0:
            observations.append({"symbol": name, "status": "kallsyms-hidden"})
            continue
            
        delta = b.address - a.address
        deltas.append(delta)
        observations.append({
            "symbol": name,
            "system_map": hexaddr(a.address),
            "runtime": hexaddr(b.address),
            "delta": hexaddr(delta) if delta >= 0 else f"-0x{-delta:x}",
        })
        
    if not deltas:
        return {"status": "unresolved", "runtime_offset": None, "consistent": None, "observations": observations}
        
    unique = sorted(set(deltas))
    consistent = (len(unique) == 1)
    runtime_offset = None
    if consistent:
        runtime_offset = hexaddr(unique[0]) if unique[0] >= 0 else f"-0x{-unique[0]:x}"
        
    return {
        "status": "resolved" if consistent else "inconsistent",
        "runtime_offset": runtime_offset,
        "consistent": consistent,
        "observations": observations,
    }

def kallsyms_visibility(symbols: list[Symbol]) -> dict[str, Any]:
    if not symbols:
        return {"status": "missing", "zero_ratio": None}
        
    zeros = sum(1 for sym in symbols if sym.address == 0)
    ratio = zeros / len(symbols)
    
    if ratio == 1.0:
        status = "fully-hidden"
    elif ratio > 0.5:
        status = "mostly-hidden"
    elif ratio > 0:
        status = "partially-hidden"
    else:
        status = "visible"
        
    return {
        "status": status,
        "symbol_count": len(symbols),
        "zero_address_count": zeros,
        "zero_ratio": ratio,
    }


# ===========================================================================
# /proc/iomem
# ===========================================================================

_IOMEM_RE = re.compile(r"^(\s*)([0-9a-fA-F]+)-([0-9a-fA-F]+)\s*:\s*(.+?)\s*$")

def parse_iomem(path: Path) -> dict[str, Any]:
    ranges: list[dict[str, Any]] = []
    ram: list[dict[str, Any]] = []
    
    for raw in read_text(path).splitlines():
        match = _IOMEM_RE.match(raw)
        if not match:
            continue
            
        indent, start_s, end_s, name = match.groups()
        start = int(start_s, 16)
        end = int(end_s, 16)
        
        item = {
            "start": hexaddr(start),
            "end": hexaddr(end),
            "size": end - start + 1,
            "name": name,
            "indent": len(indent),
        }
        ranges.append(item)
        if name == "System RAM":
            ram.append(item.copy())
            
    redacted = bool(ranges) and all(int(item["start"], 0) == 0 and int(item["end"], 0) == 0 for item in ranges)
    return {"range_count": len(ranges), "system_ram": ram, "redacted": redacted, "ranges": ranges}


# ===========================================================================
# /proc/cpuinfo
# ===========================================================================

def parse_cpuinfo(path: Path) -> dict[str, Any]:
    records: list[dict[str, str]] = []
    current: dict[str, str] = {}
    
    for raw in read_text(path).splitlines():
        if not raw.strip():
            if current:
                records.append(current)
                current = {}
            continue
        if ":" not in raw:
            continue
        key, value = raw.split(":", 1)
        current[key.strip()] = value.strip()
        
    if current:
        records.append(current)
        
    processors = [record for record in records if "processor" in record]
    feature_sets: list[set[str]] = []
    for record in processors:
        features = record.get("Features")
        if features is not None:
            feature_sets.append(set(features.split()))
            
    common_features = sorted(set.intersection(*feature_sets)) if feature_sets else []
    implementers = sorted({record.get("CPU implementer") for record in processors if record.get("CPU implementer")})
    parts = sorted({record.get("CPU part") for record in processors if record.get("CPU part")})
    
    return {
        "processor_count": len(processors),
        "common_features": common_features,
        "implementers": implementers,
        "parts": parts,
        "records": records,
    }


# ===========================================================================
# /proc/modules
# ===========================================================================

def parse_modules(path: Path) -> dict[str, Any]:
    modules: list[dict[str, Any]] = []
    for raw in read_text(path).splitlines():
        fields = raw.split()
        if len(fields) < 6:
            continue
        name, size_s, ref_s, deps_s, state, address_s = fields[:6]
        
        try:
            size = int(size_s)
            refcount = int(ref_s)
        except ValueError:
            continue
            
        try:
            address = int(address_s, 16)
            address_value: Optional[str] = hexaddr(address)
        except ValueError:
            address_value = None
            
        modules.append({
            "name": name,
            "size": size,
            "refcount": refcount,
            "dependencies": [] if deps_s == "-" else deps_s.rstrip(",").split(","),
            "state": state,
            "address": address_value,
        })
        
    return {"count": len(modules), "modules": modules}


# ===========================================================================
# Native BTF parser
# ===========================================================================

BTF_MAGIC = 0xEB9F

BTF_KIND_NAMES = {
    0: "UNKN", 1: "INT", 2: "PTR", 3: "ARRAY", 4: "STRUCT", 5: "UNION", 6: "ENUM",
    7: "FWD", 8: "TYPEDEF", 9: "VOLATILE", 10: "CONST", 11: "RESTRICT", 12: "FUNC",
    13: "FUNC_PROTO", 14: "VAR", 15: "DATASEC", 16: "FLOAT", 17: "DECL_TAG",
    18: "TYPE_TAG", 19: "ENUM64",
}

@dataclass
class BTFType:
    type_id: int
    name: str
    kind: int
    kind_name: str
    vlen: int
    kflag: int
    size_or_type: int
    extra: Any

class BTF:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.endian = "<"
        self.header: dict[str, Any] = {}
        self.types: list[Optional[BTFType]] = [None]
        self.strings = b""
        self._named_index: dict[str, list[BTFType]] = {}
        self._parse()
        self._build_name_index()

    def _u32(self, blob: bytes, off: int) -> int:
        return struct.unpack_from(self.endian + "I", blob, off)[0]

    def _s32(self, blob: bytes, off: int) -> int:
        return struct.unpack_from(self.endian + "i", blob, off)[0]

    def _build_name_index(self) -> None:
        index: dict[str, list[BTFType]] = {}
        for item in self.types[1:]:
            if item is None or not item.name:
                continue
            index.setdefault(item.name, []).append(item)
        self._named_index = index

    def _parse(self) -> None:
        if len(self.data) < 24:
            raise ProfileError("BTF file is too small")
            
        magic_le = struct.unpack_from("<H", self.data, 0)[0]
        magic_be = struct.unpack_from(">H", self.data, 0)[0]
        
        if magic_le == BTF_MAGIC:
            self.endian = "<"
        elif magic_be == BTF_MAGIC:
            self.endian = ">"
        else:
            raise ProfileError(f"Invalid BTF magic: expected 0x{BTF_MAGIC:04x}")
            
        header = struct.unpack_from(self.endian + "HBBIIIII", self.data, 0)
        magic, version, flags, hdr_len, type_off, type_len, str_off, str_len = header
        
        if hdr_len < 24 or hdr_len > len(self.data):
            raise ProfileError(f"Invalid BTF hdr_len={hdr_len}")
            
        base = hdr_len
        type_start = base + type_off
        type_end = type_start + type_len
        str_start = base + str_off
        str_end = str_start + str_len
        
        if type_end > len(self.data) or str_end > len(self.data):
            raise ProfileError("BTF sections exceed file length")
            
        self.header = {
            "magic": f"0x{magic:04x}",
            "version": version,
            "flags": flags,
            "header_length": hdr_len,
            "type_length": type_len,
            "string_length": str_len,
            "endianness": "little" if self.endian == "<" else "big",
        }
        self.strings = self.data[str_start:str_end]
        
        blob = self.data[type_start:type_end]
        off = 0
        type_id = 1
        
        while off < len(blob):
            if off + 12 > len(blob):
                raise ProfileError("Truncated BTF type record")
                
            name_off, info, size_or_type = struct.unpack_from(self.endian + "III", blob, off)
            off += 12
            
            vlen = info & 0xFFFF
            kind = (info >> 24) & 0x1F
            kflag = (info >> 31) & 1
            name = self.string(name_off)
            
            extra, consumed = self._parse_extra(blob, off, kind, vlen, kflag)
            off += consumed
            
            self.types.append(BTFType(
                type_id=type_id,
                name=name,
                kind=kind,
                kind_name=BTF_KIND_NAMES.get(kind, f"KIND_{kind}"),
                vlen=vlen,
                kflag=kflag,
                size_or_type=size_or_type,
                extra=extra,
            ))
            type_id += 1
            
        if off != len(blob):
            raise ProfileError("BTF type section ended at unexpected offset")

    def string(self, offset: int) -> str:
        if offset == 0:
            return ""
        if offset < 0 or offset >= len(self.strings):
            return f"<bad-string-offset:{offset}>"
        end = self.strings.find(b"\0", offset)
        if end < 0:
            end = len(self.strings)
        return self.strings[offset:end].decode("utf-8", errors="replace")

    def _parse_extra(self, blob: bytes, off: int, kind: int, vlen: int, kflag: int) -> tuple[Any, int]:
        if kind == 1:
            if off + 4 > len(blob):
                raise ProfileError("Truncated BTF INT")
            value = self._u32(blob, off)
            return {"encoding": (value >> 24) & 0x0F, "offset": (value >> 16) & 0xFF, "bits": value & 0xFF}, 4
            
        if kind == 3:
            if off + 12 > len(blob):
                raise ProfileError("Truncated BTF ARRAY")
            elem_type, index_type, nelems = struct.unpack_from(self.endian + "III", blob, off)
            return {"element_type": elem_type, "index_type": index_type, "count": nelems}, 12
            
        if kind in (4, 5):
            need = vlen * 12
            if off + need > len(blob):
                raise ProfileError("Truncated BTF STRUCT/UNION")
            members = []
            pos = off
            for _ in range(vlen):
                name_off, type_id, offset_value = struct.unpack_from(self.endian + "III", blob, pos)
                pos += 12
                if kflag:
                    bitfield_size = (offset_value >> 24) & 0xFF
                    bit_offset = offset_value & 0x00FFFFFF
                else:
                    bitfield_size = 0
                    bit_offset = offset_value
                members.append({
                    "name": self.string(name_off),
                    "type_id": type_id,
                    "bit_offset": bit_offset,
                    "byte_offset": bit_offset // 8,
                    "bitfield_size": bitfield_size,
                })
            return members, need
            
        if kind == 6:
            need = vlen * 8
            if off + need > len(blob):
                raise ProfileError("Truncated BTF ENUM")
            values = []
            pos = off
            for _ in range(vlen):
                name_off = self._u32(blob, pos)
                value = self._s32(blob, pos + 4)
                pos += 8
                values.append({"name": self.string(name_off), "value": value})
            return values, need
            
        if kind == 13:
            need = vlen * 8
            if off + need > len(blob):
                raise ProfileError("Truncated BTF FUNC_PROTO")
            params = []
            pos = off
            for _ in range(vlen):
                name_off, type_id = struct.unpack_from(self.endian + "II", blob, pos)
                pos += 8
                params.append({"name": self.string(name_off), "type_id": type_id})
            return params, need
            
        if kind == 14:
            if off + 4 > len(blob):
                raise ProfileError("Truncated BTF VAR")
            linkage = self._u32(blob, off)
            return {"linkage": linkage}, 4
            
        if kind == 15:
            need = vlen * 12
            if off + need > len(blob):
                raise ProfileError("Truncated BTF DATASEC")
            vars_ = []
            pos = off
            for _ in range(vlen):
                type_id, sec_offset, size = struct.unpack_from(self.endian + "III", blob, pos)
                pos += 12
                vars_.append({"type_id": type_id, "offset": sec_offset, "size": size})
            return vars_, need
            
        if kind == 17:
            if off + 4 > len(blob):
                raise ProfileError("Truncated BTF DECL_TAG")
            component_idx = self._s32(blob, off)
            return {"component_idx": component_idx}, 4
            
        if kind == 19:
            need = vlen * 12
            if off + need > len(blob):
                raise ProfileError("Truncated BTF ENUM64")
            values = []
            pos = off
            for _ in range(vlen):
                name_off, lo, hi = struct.unpack_from(self.endian + "III", blob, pos)
                pos += 12
                value = (hi << 32) | lo
                if kflag and (hi & 0x80000000):
                    value -= (1 << 64)
                values.append({"name": self.string(name_off), "value": value})
            return values, need
            
        if kind in (0, 2, 7, 8, 9, 10, 11, 12, 16, 18):
            return None, 0
            
        raise ProfileError(f"Unsupported BTF kind {kind}")

    def type_by_id(self, type_id: int) -> Optional[BTFType]:
        if type_id <= 0 or type_id >= len(self.types):
            return None
        return self.types[type_id]

    def named_types(self, name: str) -> list[BTFType]:
        return list(self._named_index.get(name, []))

    def resolve_modifiers(self, type_id: int) -> tuple[int, list[str]]:
        modifiers: list[str] = []
        seen: set[int] = set()
        while type_id not in seen:
            seen.add(type_id)
            item = self.type_by_id(type_id)
            if item is None:
                break
            if item.kind in (8, 9, 10, 11, 18):
                if item.kind_name == "TYPEDEF" and item.name:
                    modifiers.append(f"typedef:{item.name}")
                else:
                    modifiers.append(item.kind_name.lower())
                type_id = item.size_or_type
                continue
            break
        return type_id, modifiers

    def describe_type_ref(self, type_id: int) -> dict[str, Any]:
        resolved_id, modifiers = self.resolve_modifiers(type_id)
        item = self.type_by_id(resolved_id)
        if item is None:
            return {
                "type_id": type_id,
                "resolved_type_id": resolved_id,
                "kind": "VOID" if resolved_id == 0 else "UNKNOWN",
                "name": None,
                "modifiers": modifiers,
            }
        return {
            "type_id": type_id,
            "resolved_type_id": resolved_id,
            "kind": item.kind_name,
            "name": item.name or None,
            "modifiers": modifiers,
        }

    def serialize_type(self, item: BTFType, include_members: bool = True) -> dict[str, Any]:
        output: dict[str, Any] = {
            "type_id": item.type_id,
            "name": item.name or None,
            "kind": item.kind_name,
            "vlen": item.vlen,
        }
        if item.kind in (1, 4, 5, 6, 15, 16, 19):
            output["size"] = item.size_or_type
        else:
            output["type"] = item.size_or_type
            
        if item.kind in (4, 5) and include_members:
            members: dict[str, dict[str, Any]] = {}
            for index, member in enumerate(item.extra):
                member_name = member["name"]
                key = member_name if member_name else f"__anon_member_{index}"
                member_value = {
                    "member_name": member_name,
                    "offset": member["byte_offset"],
                    "bit_offset": member["bit_offset"],
                    "bitfield_size": member["bitfield_size"],
                    **self.describe_type_ref(member["type_id"]),
                }
                if key in members:
                    key = f"{key}__duplicate_{index}"
                members[key] = member_value
            output["members"] = members
        elif item.extra is not None:
            output["extra"] = item.extra
        return output

    def export_named(self, names: set[str], all_btf: bool) -> dict[str, Any]:
        if all_btf:
            chosen = [item for item in self.types[1:] if item is not None]
        else:
            chosen_by_id: dict[int, BTFType] = {}
            for name in names:
                for item in self.named_types(name):
                    chosen_by_id[item.type_id] = item
                    
            pending = list(chosen_by_id.values())
            while pending:
                parent = pending.pop()
                if parent.kind not in (4, 5) or not isinstance(parent.extra, list):
                    continue
                for member in parent.extra:
                    if member.get("name"):
                        continue
                    resolved_id, _ = self.resolve_modifiers(member.get("type_id", 0))
                    child = self.type_by_id(resolved_id)
                    if child is None or child.kind not in (4, 5) or child.name:
                        continue
                    if child.type_id not in chosen_by_id:
                        chosen_by_id[child.type_id] = child
                        pending.append(child)
            chosen = sorted(chosen_by_id.values(), key=lambda item: item.type_id)
            
        exported: dict[str, list[dict[str, Any]]] = {}
        for item in chosen:
            key = item.name or f"__anon_type_{item.type_id}"
            exported.setdefault(key, []).append(self.serialize_type(item))
            
        return {
            "header": self.header,
            "type_count": len(self.types) - 1,
            "selected_type_count": len(chosen),
            "types": exported,
        }

def parse_btf_file(path: Path, names: set[str], all_btf: bool) -> dict[str, Any]:
    data = path.read_bytes()
    btf = BTF(data)
    result = btf.export_named(names, all_btf)
    result["file_size"] = len(data)
    result["sha256"] = hashlib.sha256(data).hexdigest()
    return result


# ===========================================================================
# Metadata / provenance helpers
# ===========================================================================

def build_sources(profile_dir: Path) -> dict[str, Any]:
    """Index all raw files, including optional and nested acquisition artifacts."""
    sources: dict[str, Any] = {}
    for path in sorted(profile_dir.rglob("*")):
        if not path.is_file() or path.name == "profile.json":
            continue
        rel = _safe_relpath(profile_dir, path)
        if rel is None:
            continue
        info = file_info(path)
        info["path"] = rel
        sources[rel] = info
    return sources

def ensure_required_files(profile_dir: Path) -> None:
    missing = [name for name in CORE_FILES if not (profile_dir / name).is_file()]
    if missing:
        raise ProfileError("Missing core profile files: " + ", ".join(missing))

def architecture_from_inputs(config: dict[str, Any], uname: dict[str, Any], cpuinfo: dict[str, Any]) -> dict[str, Any]:
    config_arm64 = config.get("CONFIG_ARM64") is True
    uname_arch = uname.get("architecture_hint")
    cpuinfo_arm_hint = bool(cpuinfo.get("implementers") or cpuinfo.get("parts"))
    normalized = "arm64" if (config_arm64 or uname_arch in ("aarch64", "arm64")) else None
    
    return {
        "normalized": normalized,
        "config_arm64": config_arm64,
        "uname_hint": uname_arch,
        "cpuinfo_arm_hint": cpuinfo_arm_hint,
    }

def parse_key_value_file(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for raw in read_text(path).splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        key = parts[0].rstrip(":")
        value_text = " ".join(parts[1:])
        try:
            value: Any = int(value_text, 0)
        except ValueError:
            try:
                value = int(value_text)
            except ValueError:
                value = value_text
        result[key] = value
    return result

def parse_meminfo(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for raw in read_text(path).splitlines():
        match = re.match(r"^([^:]+):\s*(\d+)(?:\s+(\S+))?", raw)
        if not match:
            continue
        key, number, unit = match.groups()
        result[key.strip()] = {"value": int(number), "unit": unit}
    return result

def parse_physical_blocks(root: Path) -> list[dict[str, Any]]:
    base = root / "memory-blocks"
    blocks: list[dict[str, Any]] = []
    if not base.is_dir():
        return blocks
        
    for block in sorted(base.iterdir()):
        if not block.is_dir():
            continue
        item: dict[str, Any] = {"name": block.name}
        for key in ("phys_index", "state", "removable", "valid_zones"):
            value = read_text(block / key).strip()
            if not value:
                continue
            if key == "phys_index":
                try:
                    item[key] = int(value, 0)
                except ValueError:
                    item[key] = value
            else:
                item[key] = value
        blocks.append(item)
    return blocks

def parse_cpu_topology(root: Path) -> dict[str, Optional[str]]:
    base = root / "cpu-topology"
    keys = ("online", "offline", "possible", "present")
    return {key: (read_text(base / key).strip() or None) for key in keys}

def parse_numa(root: Path) -> dict[str, Any]:
    base = root / "numa"
    output: dict[str, Any] = {}
    
    for name in ("online", "possible"):
        value = read_text(base / name).strip()
        if value:
            output[name] = value
            
    nodes: dict[str, Any] = {}
    if base.is_dir():
        for node_dir in sorted(base.glob("node[0-9]*")):
            if not node_dir.is_dir():
                continue
            node: dict[str, Any] = {}
            for name in ("meminfo", "cpulist", "distance"):
                value = read_text(node_dir / name).strip()
                if value:
                    node[name] = value.splitlines()
            if node:
                nodes[node_dir.name] = node
    output["nodes"] = nodes
    return output

def parse_module_sections(root: Path) -> dict[str, dict[str, str]]:
    base = root / "module-sections"
    result: dict[str, dict[str, str]] = {}
    if not base.is_dir():
        return result
        
    for module_dir in sorted(base.iterdir()):
        if not module_dir.is_dir():
            continue
        sections: dict[str, str] = {}
        for path in sorted(module_dir.iterdir()):
            if not path.is_file():
                continue
            value = read_text(path).strip()
            if value:
                sections[path.name] = value
        if sections:
            result[module_dir.name] = sections
    return result

def parse_module_runtime_info(root: Path) -> dict[str, dict[str, Any]]:
    base = root / "module-info"
    result: dict[str, dict[str, Any]] = {}
    if not base.is_dir():
        return result
        
    for module_dir in sorted(base.iterdir()):
        if not module_dir.is_dir():
            continue
        item: dict[str, Any] = {}
        refcnt = read_text(module_dir / "refcnt").strip()
        if refcnt:
            try:
                item["refcount"] = int(refcnt, 0)
            except ValueError:
                item["refcount_raw"] = refcnt
        if item:
            result[module_dir.name] = item
    return result

def parse_module_parameters(root: Path) -> dict[str, dict[str, str]]:
    base = root / "module-parameters"
    result: dict[str, dict[str, str]] = {}
    if not base.is_dir():
        return result
        
    for module_dir in sorted(base.iterdir()):
        if not module_dir.is_dir():
            continue
        params: dict[str, str] = {}
        for path in sorted(module_dir.iterdir()):
            if not path.is_file():
                continue
            value = read_text(path).strip()
            if value:
                params[path.name.removesuffix(".txt")] = value
        if params:
            result[module_dir.name] = params
    return result

def parse_module_btf(root: Path) -> dict[str, dict[str, Any]]:
    base = root / "btf-modules"
    result: dict[str, dict[str, Any]] = {}
    if not base.is_dir():
        return result
        
    for path in sorted(base.glob("*.btf")):
        if path.is_file():
            result[path.stem] = {
                "path": path.relative_to(root).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    return result

def parse_vmcoreinfo_notes(path: Path) -> dict[str, Any]:
    """Extract VMCOREINFO from a Linux ELF notes blob without guessing values."""
    if not path.is_file():
        return {"available": False, "values": {}, "symbols": {}, "errors": []}
        
    blob = path.read_bytes()
    note_records: list[dict[str, Any]] = []
    errors: list[str] = []
    selected = None
    
    for endian in ("<", ">"):
        records: list[dict[str, Any]] = []
        offset = 0
        try:
            while offset + 12 <= len(blob):
                namesz, descsz, ntype = struct.unpack_from(endian + "III", blob, offset)
                offset += 12
                if namesz > len(blob) or descsz > len(blob):
                    raise ValueError("impossible note length")
                    
                name_end = offset + namesz
                if name_end > len(blob):
                    raise ValueError("truncated note name")
                    
                name = blob[offset:name_end].rstrip(b"\0").decode("ascii", errors="replace")
                offset = (name_end + 3) & ~3
                
                desc_end = offset + descsz
                if desc_end > len(blob):
                    raise ValueError("truncated note descriptor")
                    
                desc = blob[offset:desc_end]
                offset = (desc_end + 3) & ~3
                
                records.append({"name": name, "type": ntype, "desc": desc})
            if records:
                selected = records
                break
        except (ValueError, struct.error):
            continue
            
    if selected is None:
        return {"available": False, "values": {}, "symbols": {}, "errors": ["No valid ELF note sequence parsed"]}
        
    gnu_build_id = next((record["desc"].hex() for record in selected if record["name"] == "GNU" and record["type"] == 3), None)
    vm_descs = [record for record in selected if record["name"].startswith("VMCOREINFO")]
    
    for record in vm_descs:
        desc_text = record["desc"].decode("utf-8", errors="replace")
        note_records.append({
            "name": record["name"],
            "type": record["type"],
            "descriptor_length": len(record["desc"]),
        })
        
        values: dict[str, str] = {}
        symbols: dict[str, str] = {}
        for line in desc_text.splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            
            if key.startswith("SYMBOL(") and key.endswith(")"):
                symbols[key[7:-1]] = value
            else:
                values[key] = value
                
        return {
            "available": True,
            "values": values,
            "symbols": symbols,
            "notes": note_records,
            "build_id": gnu_build_id,
            "errors": errors,
        }
        
    return {"available": False, "values": {}, "symbols": {}, "notes": note_records, "build_id": gnu_build_id, "errors": errors}

def parse_vmlinux_elf(path: Path) -> dict[str, Any]:
    """Read ELF identity and GNU build ID from an optional matching vmlinux."""
    if not path.is_file():
        return {"available": False}
        
    try:
        with path.open("rb") as handle:
            ident = handle.read(16)
            if len(ident) < 16 or ident[:4] != b"\x7fELF":
                return {"available": False, "error": "not an ELF file"}
                
            elf_class = ident[4]
            data_encoding = ident[5]
            if elf_class not in (1, 2) or data_encoding not in (1, 2):
                return {"available": False, "error": "unsupported ELF class or byte order"}
                
            endian = "<" if data_encoding == 1 else ">"
            handle.seek(0)
            
            if elf_class == 2:
                header = handle.read(64)
                if len(header) < 64:
                    raise ValueError("truncated ELF64 header")
                e_type, e_machine = struct.unpack_from(endian + "HH", header, 16)
                e_phoff = struct.unpack_from(endian + "Q", header, 32)[0]
                e_phentsize, e_phnum = struct.unpack_from(endian + "HH", header, 54)
                ph_fmt = endian + "IIQQQQQQ"
                min_ph = struct.calcsize(ph_fmt)
            else:
                header = handle.read(52)
                if len(header) < 52:
                    raise ValueError("truncated ELF32 header")
                e_type, e_machine = struct.unpack_from(endian + "HH", header, 16)
                e_phoff = struct.unpack_from(endian + "I", header, 28)[0]
                e_phentsize, e_phnum = struct.unpack_from(endian + "HH", header, 42)
                ph_fmt = endian + "IIIIIIII"
                min_ph = struct.calcsize(ph_fmt)
                
            if e_phnum > 8192 or e_phentsize < min_ph:
                raise ValueError("invalid ELF program-header table")
                
            notes: list[tuple[int, int]] = []
            for index in range(e_phnum):
                handle.seek(e_phoff + index * e_phentsize)
                entry = handle.read(e_phentsize)
                if len(entry) < min_ph:
                    raise ValueError("truncated ELF program header")
                fields = struct.unpack_from(ph_fmt, entry, 0)
                p_type = fields[0]
                if elf_class == 2:
                    p_offset = fields[2]
                    p_filesz = fields[5]
                else:
                    p_offset = fields[1]
                    p_filesz = fields[4]
                if p_type == 4 and p_filesz:
                    notes.append((p_offset, p_filesz))
                    
            build_id = None
            for offset, size in notes:
                if size > 64 * 1024 * 1024:
                    continue
                handle.seek(offset)
                payload = handle.read(size)
                pos = 0
                while pos + 12 <= len(payload):
                    namesz, descsz, ntype = struct.unpack_from(endian + "III", payload, pos)
                    pos += 12
                    name_end = pos + namesz
                    if name_end > len(payload):
                        break
                    name = payload[pos:name_end].rstrip(b"\0")
                    pos = (name_end + 3) & ~3
                    desc_end = pos + descsz
                    if desc_end > len(payload):
                        break
                    desc = payload[pos:desc_end]
                    pos = (desc_end + 3) & ~3
                    
                    if name == b"GNU" and ntype == 3:
                        build_id = desc.hex()
                        break
                if build_id:
                    break
                    
        machine_names = {183: "AArch64", 62: "x86_64", 243: "RISC-V"}
        return {
            "available": True,
            "path": path.name,
            "elf_class": 64 if elf_class == 2 else 32,
            "endianness": "little" if data_encoding == 1 else "big",
            "type": e_type,
            "machine": e_machine,
            "machine_name": machine_names.get(e_machine, f"EM_{e_machine}"),
            "build_id": build_id,
            "size": path.stat().st_size,
        }
    except (OSError, ValueError, struct.error) as exc:
        return {"available": False, "error": str(exc)}

def parse_device_tree(root: Path) -> dict[str, Any]:
    base = root / "device-tree"
    properties: dict[str, Any] = {}
    
    if base.is_dir():
        for path in sorted(base.rglob("*.txt")):
            rel = path.relative_to(base).as_posix()
            value = read_text(path).strip()
            if value:
                properties[rel] = value
                
    dtb = root / "device-tree.dtb"
    return {
        "dtb_present": dtb.is_file(),
        "dtb_size": dtb.stat().st_size if dtb.is_file() else None,
        "properties": properties,
    }

def parse_firmware(root: Path) -> dict[str, Any]:
    acpi_dir = root / "acpi-tables"
    acpi = []
    
    if acpi_dir.is_dir():
        for path in sorted(acpi_dir.rglob("*")):
            if path.is_file():
                acpi.append({
                    "path": path.relative_to(root).as_posix(),
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                })
                
    return {
        "device_tree": parse_device_tree(root),
        "acpi_tables": acpi,
    }

def parse_capture_metadata(root: Path) -> dict[str, Any]:
    names = ("capture-start-utc.txt", "capture-start-uptime.txt", "capture-end-utc.txt", "capture-end-uptime.txt")
    return {name: (read_text(root / name).strip() or None) for name in names}

def parse_vmcoreinfo_location(path: Path) -> dict[str, Any]:
    raw = read_text(path).strip()
    result: dict[str, Any] = {"raw": raw or None}
    
    if not raw:
        result["available"] = False
        return result
        
    tokens = raw.split()
    parsed = []
    for token in tokens:
        try:
            parsed.append(hexaddr(int(token, 0)))
        except ValueError:
            parsed.append(token)
            
    result["tokens"] = parsed
    result["available"] = True
    return result

def _vm_num(values: dict[str, str], *names: str) -> Optional[int]:
    for name in names:
        value = values.get(name)
        if value is None:
            continue
        try:
            return int(value, 0)
        except ValueError:
            try:
                return int(value, 10)
            except ValueError:
                continue
    return None

def _enrich_selected_symbols(selected: dict[str, Any], vm_symbols: dict[str, str]) -> None:
    for name, value in vm_symbols.items():
        try:
            address = int(value, 0)
        except (TypeError, ValueError):
            continue
        selected.setdefault(name, {}).setdefault("vmcoreinfo", []).append({
            "address": hexaddr(address),
            "type": "VMCOREINFO",
        })


# ===========================================================================
# Validation
# ===========================================================================

def validate_profile(profile: dict[str, Any]) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    integrity = profile.get("integrity", {}).get("sha256sums", {})
    
    if integrity.get("ok") is False:
        issues.append({
            "severity": "error",
            "code": "HASH_MISMATCH",
            "message": "One or more files fail SHA256SUMS verification.",
        })
    elif integrity.get("status") == "not-provided":
        issues.append({
            "severity": "warning",
            "code": "HASH_MANIFEST_MISSING",
            "message": "SHA256SUMS was not available; raw-file integrity was not independently checked.",
        })
        
    kernel = profile.get("kernel", {})
    elf = kernel.get("elf", {})
    
    if elf.get("available") and elf.get("machine_name") != "AArch64":
        issues.append({
            "severity": "error",
            "code": "VMLINUX_NOT_AARCH64",
            "message": "Optional vmlinux ELF is not identified as AArch64.",
        })
        
    if kernel.get("build_id_match") is False:
        issues.append({
            "severity": "error",
            "code": "VMLINUX_BUILD_ID_MISMATCH",
            "message": "The optional vmlinux ELF build ID differs from the captured running-kernel notes.",
        })
        
    architecture = kernel.get("architecture", {}).get("normalized")
    if architecture != "arm64":
        issues.append({
            "severity": "error",
            "code": "NOT_ARM64",
            "message": "Profile does not positively identify an ARM64 kernel.",
        })
        
    if kernel.get("page_size") is None:
        issues.append({
            "severity": "warning",
            "code": "PAGE_SIZE_UNKNOWN",
            "message": "ARM64 page size could not be derived from captured configuration or VMCOREINFO.",
        })
        
    if kernel.get("va_bits") is None:
        issues.append({
            "severity": "warning",
            "code": "VA_BITS_UNKNOWN",
            "message": "ARM64 VA bits could not be derived from captured configuration or VMCOREINFO.",
        })
        
    kall = profile.get("symbols", {}).get("kallsyms_visibility", {})
    if kall.get("status") in ("fully-hidden", "mostly-hidden"):
        issues.append({
            "severity": "warning",
            "code": "KALLSYMS_RESTRICTED",
            "message": "Runtime symbol addresses in /proc/kallsyms are hidden or mostly hidden.",
        })
        
    kaslr = profile.get("boot", {}).get("kaslr", {})
    if kaslr.get("status") == "inconsistent":
        issues.append({
            "severity": "error",
            "code": "KASLR_DELTA_INCONSISTENT",
            "message": "Runtime/link-time symbol deltas are inconsistent.",
        })
    elif (
        kaslr.get("status") == "unresolved"
        and profile.get("boot", {}).get("kaslr_compiled")
        and not profile.get("boot", {}).get("nokaslr")
    ):
        issues.append({
            "severity": "warning",
            "code": "KASLR_DELTA_UNRESOLVED",
            "message": "KASLR is compiled in but runtime relocation could not be derived.",
        })
        
    if profile.get("memory", {}).get("iomem", {}).get("redacted"):
        issues.append({
            "severity": "warning",
            "code": "IOMEM_REDACTED",
            "message": "/proc/iomem appears to contain redacted address ranges.",
        })
        
    if not profile.get("btf", {}).get("types"):
        issues.append({
            "severity": "warning",
            "code": "BTF_TYPES_EMPTY",
            "message": "BTF parsed, but no named types were included in the cache.",
        })
        
    errors = sum(issue["severity"] == "error" for issue in issues)
    warnings = sum(issue["severity"] == "warning" for issue in issues)
    
    return {
        "status": "invalid" if errors else ("warning" if warnings else "valid"),
        "error_count": errors,
        "warning_count": warnings,
        "issues": issues,
    }


# ===========================================================================
# Profile construction
# ===========================================================================

def build_profile(
    profile_dir: Path,
    *,
    all_symbols: bool = False,
    all_btf: bool = False,
    include_full_config: bool = False,
) -> dict[str, Any]:
    profile_dir = profile_dir.resolve()
    ensure_required_files(profile_dir)
    sources = build_sources(profile_dir)
    sums_path = profile_dir / "SHA256SUMS"
    
    expected_hashes = parse_sha256sums(sums_path) if sums_path.is_file() else {}
    if expected_hashes:
        hash_verification = verify_sha256sums(profile_dir, expected_hashes)
        hash_verification["status"] = "verified" if hash_verification["ok"] else "failed"
    else:
        hash_verification = {"ok": None, "status": "not-provided", "files": {}}
        
    uname = parse_uname(profile_dir / "uname.txt")
    release_file = read_text(profile_dir / "kernel-release.txt").strip()
    kernel_release = release_file or uname.get("release_hint")
    
    os_release = parse_os_release(profile_dir / "os-release")
    cmdline = parse_cmdline(profile_dir / "cmdline")
    
    parsed_config = parse_kernel_config(profile_dir / "config")
    all_config = parsed_config.pop("_all")
    
    cpuinfo = parse_cpuinfo(profile_dir / "cpuinfo")
    modules = parse_modules(profile_dir / "modules.txt")
    iomem = parse_iomem(profile_dir / "iomem")
    
    system_map_syms = parse_symbol_file(profile_dir / "System.map")
    kallsyms_syms = parse_symbol_file(profile_dir / "kallsyms")
    kaslr = derive_kaslr(system_map_syms, kallsyms_syms)
    
    btf = parse_btf_file(profile_dir / "vmlinux.btf", DEFAULT_BTF_TYPES, all_btf)
    
    vmcoreinfo = parse_vmcoreinfo_notes(profile_dir / "kernel-notes.bin")
    vm_values = vmcoreinfo.get("values", {})
    vm_symbols = vmcoreinfo.get("symbols", {})
    
    page_size = _vm_num(vm_values, "NUMBER(PAGE_SIZE)", "PAGESIZE", "PAGE_SIZE") or derive_page_size(all_config)
    va_bits = _vm_num(vm_values, "NUMBER(VA_BITS)", "VA_BITS") or derive_va_bits(all_config)
    pa_bits = _vm_num(vm_values, "NUMBER(PA_BITS)", "PA_BITS") or derive_pa_bits(all_config)
    
    config_serialized: dict[str, Any] = {
        "entry_count": parsed_config["all_count"],
        "relevant": parsed_config["relevant"],
    }
    if include_full_config:
        config_serialized["all"] = dict(sorted(all_config.items()))
        
    selected = selected_symbols(system_map_syms, kallsyms_syms, DEFAULT_SYMBOLS, all_symbols)
    _enrich_selected_symbols(selected, vm_symbols)
    
    vm_values_selected = {
        key: value for key, value in vm_values.items()
        if (
            key in {
                "OSRELEASE", "PAGESIZE", "NUMBER(PAGE_SIZE)", "NUMBER(VA_BITS)", 
                "NUMBER(PA_BITS)", "NUMBER(PHYS_OFFSET)", "NUMBER(KERNELOFFSET)",
                "SYMBOL(swapper_pg_dir)", "SYMBOL(init_task)", "SYMBOL(init_mm)",
            }
            or key.startswith("OFFSET(")
            or key.startswith("SIZE(")
        )
    }
    
    elf = parse_vmlinux_elf(profile_dir / "vmlinux")
    
    profile: dict[str, Any] = {
        "format": PROFILE_FORMAT,
        "format_version": PROFILE_VERSION,
        "created_utc": _utc_now(),
        "kernel": {
            "release": kernel_release,
            "uname": uname,
            "architecture": architecture_from_inputs(all_config, uname, cpuinfo),
            "page_size": page_size,
            "va_bits": va_bits,
            "pa_bits": pa_bits,
            "build_id": None,
            "runtime_build_id": vmcoreinfo.get("build_id"),
            "build_id_match": None,
            "elf": elf,
        },
        "os": os_release,
        "boot": {
            "cmdline": cmdline,
            "kaslr_compiled": bool(all_config.get("CONFIG_RANDOMIZE_BASE")),
            "nokaslr": "nokaslr" in cmdline.get("parameters", {}),
            "kaslr": kaslr,
        },
        "config": config_serialized,
        "symbols": {
            "system_map_count": len(system_map_syms),
            "kallsyms_count": len(kallsyms_syms),
            "kallsyms_visibility": kallsyms_visibility(kallsyms_syms),
            "selected": selected,
        },
        "memory": {
            "iomem": iomem,
            "physical_blocks": parse_physical_blocks(profile_dir),
            "meminfo": parse_meminfo(profile_dir / "meminfo"),
            "vmstat": parse_key_value_file(profile_dir / "vmstat"),
            "zoneinfo_lines": read_text(profile_dir / "zoneinfo").splitlines(),
            "buddyinfo_lines": read_text(profile_dir / "buddyinfo").splitlines(),
            "pagetypeinfo_lines": read_text(profile_dir / "pagetypeinfo").splitlines(),
            "slabinfo_lines": read_text(profile_dir / "slabinfo").splitlines(),
            "allocinfo_lines": read_text(profile_dir / "allocinfo").splitlines(),
            "numa": parse_numa(profile_dir),
        },
        "cpu": {
            **cpuinfo,
            "topology": parse_cpu_topology(profile_dir),
        },
        "modules": {
            **modules,
            "sections": parse_module_sections(profile_dir),
            "runtime_info": parse_module_runtime_info(profile_dir),
            "parameters": parse_module_parameters(profile_dir),
            "btf_modules": parse_module_btf(profile_dir),
        },
        "btf": btf,
        "vmcoreinfo": {
            "notes": vmcoreinfo,
            "selected_values": vm_values_selected,
            "location": parse_vmcoreinfo_location(profile_dir / "vmcoreinfo-location.txt"),
        },
        "runtime": {
            "proc_version": read_text(profile_dir / "proc-version.txt").strip() or None,
            "tainted": read_text(profile_dir / "tainted.txt").strip() or None,
            "ostype": read_text(profile_dir / "runtime-ostype.txt").strip() or None,
            "osrelease": read_text(profile_dir / "runtime-osrelease.txt").strip() or None,
            "version": read_text(profile_dir / "runtime-version.txt").strip() or None,
            "security": {
                "lockdown": read_text(profile_dir / "security" / "lockdown.txt").strip() or None,
                "randomize_va_space": read_text(profile_dir / "hardening" / "randomize_va_space.txt").strip() or None,
                "kptr_restrict": read_text(profile_dir / "hardening" / "kptr_restrict.txt").strip() or None,
                "dmesg_restrict": read_text(profile_dir / "hardening" / "dmesg_restrict.txt").strip() or None,
            },
        },
        "firmware": parse_firmware(profile_dir),
        "capture": parse_capture_metadata(profile_dir),
        "sources": sources,
        "integrity": {
            "sha256sums": hash_verification,
        },
    }

    elf_info = profile["kernel"]["elf"]
    if "vmlinux" in sources:
        elf_info["sha256"] = sources["vmlinux"].get("sha256")
        
    if elf_info.get("available") and elf_info.get("machine_name") == "AArch64":
        profile["kernel"]["build_id"] = elf_info.get("build_id")
        
    runtime_build_id = profile["kernel"].get("runtime_build_id")
    elf_build_id = profile["kernel"].get("build_id")
    
    if runtime_build_id and elf_build_id:
        profile["kernel"]["build_id_match"] = (runtime_build_id.lower() == elf_build_id.lower())
        
    profile["validation"] = validate_profile(profile)
    return profile


# ===========================================================================
# Output
# ===========================================================================

def print_summary(profile: dict[str, Any]) -> None:
    kernel = profile["kernel"]
    boot = profile["boot"]
    validation = profile["validation"]
    btf = profile["btf"]
    
    integrity_ok = profile["integrity"]["sha256sums"].get("ok")
    integrity_label = "OK" if integrity_ok is True else ("FAILED" if integrity_ok is False else "NOT PROVIDED")
    
    print(f"Format              : {profile['format']} v{profile['format_version']}")
    print(f"Kernel release      : {kernel['release']}")
    print(f"Architecture        : {kernel['architecture']['normalized']}")
    print(f"Page size           : {kernel['page_size']}")
    print(f"VA bits             : {kernel['va_bits']}")
    print(f"PA bits             : {kernel['pa_bits']}")
    print(f"Build ID            : {kernel.get('build_id')}")
    print(f"KASLR compiled      : {boot['kaslr_compiled']}")
    print(f"nokaslr             : {boot['nokaslr']}")
    print(f"Runtime relocation  : {boot['kaslr']['runtime_offset']}")
    print(f"System.map symbols  : {profile['symbols']['system_map_count']}")
    print(f"kallsyms symbols    : {profile['symbols']['kallsyms_count']}")
    print(f"BTF types           : {btf['type_count']}")
    print(f"BTF cached          : {btf['selected_type_count']}")
    print(f"System RAM ranges   : {len(profile['memory']['iomem']['system_ram'])}")
    print(f"Loaded modules      : {profile['modules']['count']}")
    print(f"Source artifacts    : {len(profile['sources'])}")
    print(f"Hash verification   : {integrity_label}")
    print(f"Validation          : {validation['status'].upper()}")
    
    for issue in validation["issues"]:
        print(f"  [{issue['severity'].upper()}] {issue['code']}: {issue['message']}")

def write_profile_atomic(output: Path, profile: dict[str, Any], *, compact: bool = False) -> None:
    """
    Atomically replace profile.json.
    This prevents an interrupted write from leaving a partially-written
    profile file that could later be mistaken for a valid acquisition.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.tmp-{os.getpid()}"
    
    if compact:
        encoded = json.dumps(profile, sort_keys=True, separators=(",", ":")) + "\n"
    else:
        encoded = json.dumps(profile, indent=2, sort_keys=True) + "\n"
        
    try:
        temporary.write_text(encoded, encoding="utf-8")
        os.replace(temporary, output)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


# ===========================================================================
# CLI
# ===========================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Acquire running ARM64 Linux kernel metadata and/or build a normalized LiME kernel profile."
    )
    parser.add_argument(
        "profile_dir",
        type=Path,
        help="Profile directory. With --acquire, metadata is captured here before profile.json is generated.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output JSON path. Defaults to PROFILE_DIR/profile.json.",
    )
    parser.add_argument(
        "--acquire",
        action="store_true",
        help="Acquire metadata from the currently running Linux kernel before building the profile. Requires effective UID 0.",
    )
    parser.add_argument(
        "--vmlinux",
        type=Path,
        help="Optional exact matching uncompressed vmlinux ELF to copy into PROFILE_DIR during --acquire.",
    )
    parser.add_argument(
        "--all-symbols",
        action="store_true",
        help="Embed all System.map and kallsyms symbols instead of the default required subset.",
    )
    parser.add_argument(
        "--all-btf",
        action="store_true",
        help="Embed all BTF types. This can make profile.json very large.",
    )
    parser.add_argument(
        "--full-config",
        action="store_true",
        help="Embed all kernel config values in profile.json.",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Write compact JSON instead of pretty-printed JSON.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return exit status 1 when final profile validation status is invalid.",
    )
    return parser

def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    profile_dir = args.profile_dir.expanduser().resolve()
    
    # --vmlinux only has acquisition semantics.
    if args.vmlinux is not None and not args.acquire:
        parser.error("--vmlinux requires --acquire")
        
    try:
        if args.acquire:
            acquire_metadata(profile_dir, vmlinux_path=args.vmlinux)
        elif not profile_dir.is_dir():
            print(f"error: not a directory: {profile_dir}", file=sys.stderr)
            return 2
            
        output = args.output.expanduser().resolve() if args.output is not None else (profile_dir / "profile.json")
        
        profile = build_profile(
            profile_dir,
            all_symbols=args.all_symbols,
            all_btf=args.all_btf,
            include_full_config=args.full_config,
        )
        
        write_profile_atomic(output, profile, compact=args.compact)
    except (OSError, ProfileError, ValueError, struct.error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
        
    print(f"[+] Wrote {output}")
    print_summary(profile)
    
    if args.strict and profile["validation"]["status"] == "invalid":
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())