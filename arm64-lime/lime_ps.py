#!/usr/bin/env python3
"""List processes, show VMAs, or read process virtual memory from a LiME dump."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from arm64_memory import ARM64Memory, ARM64MemoryError
from kernel_profile import KernelProfile, KernelProfileError
from lime_reader import LimeReader, LimeReaderError
from process_enumerator import (
    MemoryRegion,
    ProcessEnumerationError,
    ProcessEnumerator,
)


def _number(value: str) -> int:
    try:
        return int(value, 0)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid integer: {value!r}") from exc


def _print_processes(enumerator: ProcessEnumerator, all_tasks: bool) -> None:
    entries = enumerator.list_tasks() if all_tasks else enumerator.list_processes()
    print(f"{'PID':>7} {'TGID':>7} {'PPID':>7} {'MM':>18}  COMM")
    for item in entries:
        ppid = "-" if item.parent_pid is None else str(item.parent_pid)
        mm = "-" if item.mm_address == 0 else f"0x{item.mm_address:x}"
        print(f"{item.pid:7d} {item.tgid:7d} {ppid:>7} {mm:>18}  {item.comm}")
    print(f"\n{len(entries)} {'task(s)' if all_tasks else 'process(es)'} found in captured task list")


def _print_regions(regions: list[MemoryRegion]) -> None:
    print(f"{'IDX':>4} {'START':>18} {'END':>18} {'SIZE':>12} {'FLAGS':>12} {'Pgoff':>12} {'vm_file':>18}")
    for index, region in enumerate(regions):
        flags = "-" if region.flags is None else f"0x{region.flags:x}"
        pgoff = "-" if region.page_offset is None else f"0x{region.page_offset:x}"
        file_pointer = "-" if region.file_pointer is None else f"0x{region.file_pointer:x}"
        print(
            f"{index:4d} 0x{region.start:016x} 0x{region.end:016x} {region.size:12d} "
            f"{flags:>12} {pgoff:>12} {file_pointer:>18}"
        )
    print(f"\n{len(regions)} region(s)")

def _find_pattern_in_region(
    processes: ProcessEnumerator,
    pid: int,
    region: MemoryRegion,
    pattern: bytes,
    chunk_size: int = 4096,
) -> list[int]:
    """Find a byte pattern in the readable parts of a process VMA."""
    if not pattern:
        raise ValueError("pattern must not be empty")

    chunk_size = max(chunk_size, len(pattern))
    matches: list[int] = []
    tail = b""
    offset = 0

    while offset < region.size:
        size = min(chunk_size, region.size - offset)

        try:
            data = processes.read_memory_region(
                pid,
                region,
                offset=offset,
                size=size,
            )
        except ProcessEnumerationError:
            # Some VMA pages may be unmapped or absent from the dump.
            # Skip this chunk and don't match across an unreadable gap.
            tail = b""
            offset += size
            continue

        chunk_address = region.start + offset
        combined = tail + data
        combined_address = chunk_address - len(tail)
        position = 0

        while True:
            position = combined.find(pattern, position)
            if position < 0:
                break

            address = combined_address + position

            # Include matches spanning chunk boundaries, without
            # reporting a match that was already wholly scanned.
            if (
                address >= region.start
                and address + len(pattern) <= region.end
                and address + len(pattern) > chunk_address
            ):
                matches.append(address)

            position += 1

        # Preserve enough bytes to detect matches across chunks.
        tail = (
            combined[-(len(pattern) - 1):]
            if len(pattern) > 1
            else b""
        )

        offset += size

    return matches

def _print_hexdump(
    data: bytes,
    base_address: int,
    *,
    width: int = 16,
) -> None:
    """Print memory as address, hexadecimal bytes, and ASCII."""
    hex_width = width * 3 - 1

    for offset in range(0, len(data), width):
        chunk = data[offset:offset + width]

        hex_part = " ".join(
            f"{byte:02x}" for byte in chunk
        ).ljust(hex_width)

        ascii_part = "".join(
            chr(byte) if 0x20 <= byte <= 0x7e else "."
            for byte in chunk
        )

        print(
            f"{base_address + offset:016x}  "
            f"{hex_part}  |{ascii_part}|"
        )

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, help="Directory containing profile.json and raw artifacts")
    parser.add_argument("--dump", required=True, help="Uncompressed LiME v1 dump")
    parser.add_argument("--pa-bits", type=_number, help="Target PA width, only when independently evidenced")
    parser.add_argument("--kernel-pgd-paddr", type=_number, help="Verified physical address of swapper_pg_dir")
    parser.add_argument("--allow-invalid-profile", action="store_true", help="Permit use of a profile marked invalid")

    commands = parser.add_subparsers(dest="command", required=True)
    list_cmd = commands.add_parser("list", help="List process leaders (or all tasks/threads)")
    list_cmd.add_argument("--all-tasks", action="store_true", help="Include individual threads, not just process leaders")

    find_cmd = commands.add_parser(
        "find",
        help="Find a byte pattern in a process's virtual memory",
    )
    find_cmd.add_argument("--pid", required=True, type=_number)
    find_cmd.add_argument(
        "--pattern",
        required=True,
        help="ASCII or UTF-8 text to search for",
    )
    find_cmd.add_argument(
        "--max-hits",
        type=_number,
        default=20,
        help="Maximum number of matches to display (default: 20)",
    )

    maps_cmd = commands.add_parser("maps", help="List a PID's virtual memory areas")
    maps_cmd.add_argument("--pid", required=True, type=_number)

    read_cmd = commands.add_parser(
        "read",
        help="Read bytes at a process virtual address",
    )
    read_cmd.add_argument("--pid", required=True, type=_number)
    read_cmd.add_argument("--address", required=True, type=_number)
    read_cmd.add_argument(
        "--offset",
        type=_number,
        default=0,
        help="Additional byte offset from --address (default: 0)",
    )
    read_cmd.add_argument("--size", required=True, type=_number)
    read_cmd.add_argument(
        "--output",
        help="Write raw bytes to this file; otherwise print hex",
    )

    region_cmd = commands.add_parser("read-region", help="Read a bounded slice of a listed VMA")
    region_cmd.add_argument("--pid", required=True, type=_number)
    region_cmd.add_argument("--index", required=True, type=_number, help="Zero-based row index from the PID's maps output")
    region_cmd.add_argument("--offset", type=_number, default=0, help="Offset from the region start")
    region_cmd.add_argument("--size", type=_number, required=True, help="Number of bytes to read")
    region_cmd.add_argument("--output", help="Write raw bytes to this file; otherwise print hex")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command in {"read", "read-region"}:
        if args.offset < 0:
            parser.error("--offset must be non-negative")
        if args.size < 0:
            parser.error("--size must be non-negative")
        if args.size > 64 * 1024 * 1024:
            parser.error("CLI reads are limited to 64 MiB per request")
        if not args.output and args.size > 1024 * 1024:
            parser.error("Hex output is limited to 1 MiB; use --output for larger reads")

    try:
        profile = KernelProfile(
            args.profile,
            require_valid=not args.allow_invalid_profile,
        )
        with LimeReader(args.dump) as lime:
            memory = ARM64Memory(
                profile,
                lime,
                kernel_pgd_paddr=args.kernel_pgd_paddr,
                pa_bits=args.pa_bits,
            )
            processes = ProcessEnumerator(profile, memory)

            if args.command == "list":
                _print_processes(processes, args.all_tasks)
                return 0
            if args.command == "maps":
                process = processes.find_process(args.pid)
                print(f"PID {process.pid} ({process.comm}); task=0x{process.task_address:x}; mm=0x{process.mm_address:x}")
                _print_regions(processes.list_memory_regions(args.pid))
                return 0

            if args.command == "find":
                if args.max_hits <= 0:
                    parser.error("--max-hits must be positive")

                pattern = args.pattern.encode("utf-8")

                process = processes.find_process(args.pid)
                regions = processes.list_memory_regions(args.pid)
                chunk_size = profile.page_size or 4096
                total = 0

                for index, region in enumerate(regions):
                    matches = _find_pattern_in_region(
                        processes,
                        args.pid,
                        region,
                        pattern,
                        chunk_size=chunk_size,
                    )

                    for address in matches:
                        offset = address - region.start

                        print(
                            f"PID {process.pid} ({process.comm})"
                        )
                        print(f"  Region index : {index}")
                        print(
                            f"  Region       : "
                            f"0x{region.start:016x}-0x{region.end:016x}"
                        )
                        print(f"  Offset       : 0x{offset:x} ({offset})")
                        print(f"  Match VA     : 0x{address:016x}")
                        print(f"  Sequence VA  : 0x{address + 0x10:016x}")
                        print(f"  Value VA     : 0x{address + 0x18:016x}")
                        print()

                        total += 1
                        if total >= args.max_hits:
                            break

                    if total >= args.max_hits:
                        break

                if total == 0:
                    print(
                        "Pattern not found in readable, captured VMA chunks.",
                        file=sys.stderr,
                    )
                    return 1

                print(f"{total} match(es) found")
                return 0
            
            if args.command == "read":
                read_address = args.address + args.offset

                data = processes.read_process_memory(
                    args.pid,
                    read_address,
                    args.size,
                )

                display_address = read_address

            else:
                regions = processes.list_memory_regions(args.pid)

                if args.index < 0 or args.index >= len(regions):
                    raise ProcessEnumerationError(
                        f"Region index {args.index} out of range; "
                        f"PID {args.pid} has {len(regions)} regions"
                    )

                region = regions[args.index]

                data = processes.read_memory_region(
                    args.pid,
                    region,
                    offset=args.offset,
                    size=args.size,
                )

                display_address = region.start + args.offset

            if args.output:
                output = Path(args.output)
                output.write_bytes(data)
                print(f"Wrote {len(data)} bytes to {output}")
            else:
                _print_hexdump(data, display_address)

            return 0
    except (KernelProfileError, LimeReaderError, ARM64MemoryError, ProcessEnumerationError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
