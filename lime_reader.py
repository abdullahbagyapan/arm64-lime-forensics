#!/usr/bin/env python3
"""Strict, efficient reader for uncompressed LiME v1 physical-memory dumps."""
from __future__ import annotations

import bisect
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

LIME_MAGIC = 0x4C694D45
LIME_VERSION = 1
HEADER_SIZE = 32
_HEADER_FORMAT = "IIQQ8s"


class LimeReaderError(RuntimeError):
    pass


class LimeFormatError(LimeReaderError):
    pass


class PhysicalAddressNotCaptured(LimeReaderError):
    pass


@dataclass(frozen=True, slots=True)
class LimeRange:
    start: int
    end: int
    file_offset: int
    header_offset: int

    @property
    def size(self) -> int:
        return self.end - self.start + 1

    def contains(self, paddr: int) -> bool:
        return self.start <= paddr <= self.end

    def physical_to_file(self, paddr: int) -> int:
        if not self.contains(paddr):
            raise ValueError(f"0x{paddr:x} is not in this LiME range")
        return self.file_offset + (paddr - self.start)


class LimeReader:
    """Read physical memory without silently filling uncaptured holes."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise LimeReaderError(f"LiME dump does not exist: {self.path}")
        self._fh = self.path.open("rb")
        self._fd = self._fh.fileno()
        self._closed = False
        try:
            self.file_size = os.fstat(self._fd).st_size
            self.endianness, self.ranges = self._parse_ranges()
            self._range_starts = tuple(r.start for r in self.ranges)
        except Exception:
            self._fh.close()
            self._closed = True
            raise

    def _pread_exact(self, size: int, offset: int) -> bytes:
        data = os.pread(self._fd, size, offset)
        if len(data) != size:
            raise LimeFormatError(
                f"Short file read at 0x{offset:x}: expected {size}, got {len(data)}"
            )
        return data

    def _parse_ranges(self) -> tuple[str, list[LimeRange]]:
        if self.file_size < HEADER_SIZE:
            raise LimeFormatError("File is too small to contain a LiME v1 header")
        first = self._pread_exact(HEADER_SIZE, 0)
        if struct.unpack_from("<I", first, 0)[0] == LIME_MAGIC:
            endian = "<"
        elif struct.unpack_from(">I", first, 0)[0] == LIME_MAGIC:
            endian = ">"
        else:
            raise LimeFormatError("LiME magic not found at offset 0")

        ranges: list[LimeRange] = []
        offset = 0
        while offset < self.file_size:
            remaining = self.file_size - offset
            if remaining < HEADER_SIZE:
                raise LimeFormatError(
                    f"Trailing {remaining} byte(s) at offset 0x{offset:x}; incomplete LiME header"
                )
            raw = self._pread_exact(HEADER_SIZE, offset)
            magic, version, start, end, _ = struct.unpack(endian + _HEADER_FORMAT, raw)
            if magic != LIME_MAGIC:
                raise LimeFormatError(
                    f"Invalid LiME magic at file offset 0x{offset:x}: 0x{magic:08x}"
                )
            if version != LIME_VERSION:
                raise LimeFormatError(
                    f"Unsupported LiME version {version} at file offset 0x{offset:x}"
                )
            if end < start:
                raise LimeFormatError(
                    f"Invalid physical range 0x{start:x}-0x{end:x} at 0x{offset:x}"
                )
            data_offset = offset + HEADER_SIZE
            size = end - start + 1
            next_offset = data_offset + size
            if next_offset > self.file_size:
                available = max(0, self.file_size - data_offset)
                raise LimeFormatError(
                    f"Truncated LiME range 0x{start:x}-0x{end:x}: expected {size} data bytes, found {available}"
                )
            ranges.append(LimeRange(start, end, data_offset, offset))
            offset = next_offset

        ranges.sort(key=lambda r: (r.start, r.end))
        for prev, cur in zip(ranges, ranges[1:]):
            if cur.start <= prev.end:
                raise LimeFormatError(
                    f"Overlapping physical ranges: 0x{prev.start:x}-0x{prev.end:x} and 0x{cur.start:x}-0x{cur.end:x}"
                )
        if not ranges:
            raise LimeFormatError("No physical-memory ranges found")
        return endian, ranges

    def __enter__(self) -> "LimeReader":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed:
            self._fh.close()
            self._closed = True

    def _ensure_open(self) -> None:
        if self._closed or self._fh.closed:
            raise LimeReaderError("LiME reader is closed")

    def iter_ranges(self) -> Iterator[LimeRange]:
        return iter(self.ranges)

    @property
    def summary(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "file_size": self.file_size,
            "endianness": "little" if self.endianness == "<" else "big",
            "range_count": len(self.ranges),
            "physical_start": self.ranges[0].start,
            "physical_end": self.ranges[-1].end,
            "captured_bytes": sum(r.size for r in self.ranges),
            "ranges": [
                {"start": f"0x{r.start:x}", "end": f"0x{r.end:x}", "size": r.size, "file_offset": r.file_offset}
                for r in self.ranges
            ],
        }

    def _find_range_index(self, paddr: int) -> int:
        idx = bisect.bisect_right(self._range_starts, paddr) - 1
        if idx >= 0 and self.ranges[idx].contains(paddr):
            return idx
        return -1

    def _find_range(self, paddr: int) -> LimeRange | None:
        idx = self._find_range_index(paddr)
        return self.ranges[idx] if idx >= 0 else None

    def contains_phys(self, paddr: int, size: int = 1) -> bool:
        if paddr < 0 or size < 0:
            return False
        if size == 0:
            return True
        try:
            self._validate_captured_span(paddr, size)
            return True
        except PhysicalAddressNotCaptured:
            return False

    def _validate_captured_span(self, paddr: int, size: int) -> None:
        idx = self._find_range_index(paddr)
        if idx < 0:
            raise PhysicalAddressNotCaptured(
                f"Physical address 0x{paddr:x} is not captured in {self.path}"
            )
        cursor, left = paddr, size
        while left:
            item = self.ranges[idx]
            if not item.contains(cursor):
                raise PhysicalAddressNotCaptured(
                    f"Physical read 0x{paddr:x}+0x{size:x} crosses an uncaptured physical hole at 0x{cursor:x}"
                )
            take = min(left, item.end - cursor + 1)
            cursor += take
            left -= take
            if left:
                idx += 1
                if idx >= len(self.ranges) or self.ranges[idx].start != cursor:
                    raise PhysicalAddressNotCaptured(
                        f"Physical read 0x{paddr:x}+0x{size:x} crosses an uncaptured physical hole at 0x{cursor:x}"
                    )

    def read_phys(self, paddr: int, size: int) -> bytes:
        self._ensure_open()
        if not isinstance(paddr, int) or isinstance(paddr, bool) or paddr < 0:
            raise ValueError("paddr must be a non-negative integer")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValueError("size must be a non-negative integer")
        if size == 0:
            return b""

        self._validate_captured_span(paddr, size)
        idx = self._find_range_index(paddr)
        cursor, left = paddr, size
        out = bytearray(size)
        out_pos = 0
        while left:
            item = self.ranges[idx]
            take = min(left, item.end - cursor + 1)
            chunk = os.pread(self._fd, take, item.physical_to_file(cursor))
            if len(chunk) != take:
                raise LimeFormatError(
                    f"Short read for physical address 0x{cursor:x}: expected {take}, got {len(chunk)}"
                )
            out[out_pos:out_pos + take] = chunk
            out_pos += take
            cursor += take
            left -= take
            idx += 1
        return bytes(out)

    def _byteorder(self) -> str:
        return "little" if self.endianness == "<" else "big"

    def read_u8(self, paddr: int) -> int:
        return self.read_phys(paddr, 1)[0]

    def read_u16(self, paddr: int) -> int:
        return int.from_bytes(self.read_phys(paddr, 2), self._byteorder())

    def read_u32(self, paddr: int) -> int:
        return int.from_bytes(self.read_phys(paddr, 4), self._byteorder())

    def read_u64(self, paddr: int) -> int:
        return int.from_bytes(self.read_phys(paddr, 8), self._byteorder())
