#!/usr/bin/env python3
"""ARM64 page-table walker over a LiME physical-memory dump.

Supported today: little- or big-endian AArch64 descriptors using 4 KiB,
16 KiB, or 64 KiB granules, up to 48-bit virtual and physical addresses.
Unsupported extensions (including 52-bit PA/LPA2 descriptors) fail explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from kernel_profile import KernelProfile, KernelProfileError
from lime_reader import (
    LimeReader,
    LimeReaderError,
    PhysicalAddressNotCaptured,
)


class ARM64MemoryError(RuntimeError):
    """Base error for ARM64 virtual memory operations."""


class UnsupportedTranslationGeometry(ARM64MemoryError):
    """The target's page-table geometry cannot be identified or is unsupported."""


class NonCanonicalVirtualAddress(ARM64MemoryError):
    """A virtual address does not fit the target's configured VA width."""


class UnmappedVirtualAddress(ARM64MemoryError):
    """The page tables contain an invalid/unmapped descriptor for the VA."""


class InvalidPageTableDescriptor(ARM64MemoryError):
    """A descriptor has a reserved or unsupported type/encoding."""


class KernelPageTableRootUnavailable(ARM64MemoryError):
    """The physical address of swapper_pg_dir could not be established."""


@dataclass(frozen=True, slots=True)
class PageTableGeometry:
    page_size: int
    va_bits: int
    pa_bits: int
    page_shift: int
    index_bits: int
    levels: int
    start_level: int
    first_level_bits: int

    @classmethod
    def create(
        cls,
        *,
        page_size: int | None,
        va_bits: int | None,
        pa_bits: int | None,
    ) -> "PageTableGeometry":
        if page_size not in (4096, 16384, 65536):
            raise UnsupportedTranslationGeometry(
                f"Unsupported or unknown ARM64 page size: {page_size!r}; "
                "expected 4096, 16384, or 65536"
            )

        if (
            va_bits is None
            or isinstance(va_bits, bool)
            or not isinstance(va_bits, int)
        ):
            raise UnsupportedTranslationGeometry("Target VA_BITS is unknown")

        if (
            pa_bits is None
            or isinstance(pa_bits, bool)
            or not isinstance(pa_bits, int)
        ):
            raise UnsupportedTranslationGeometry(
                "Target PA_BITS is unknown. Supply pa_bits explicitly only "
                "when you have evidence for the target's physical address size."
            )

        page_shift = page_size.bit_length() - 1
        index_bits = page_shift - 3

        if not (page_shift + 1 <= va_bits <= 48):
            raise UnsupportedTranslationGeometry(
                f"VA_BITS={va_bits} is outside the currently supported range "
                f"{page_shift + 1}..48 "
                "(52-bit/LVA layouts are not implemented)"
            )

        if not (page_shift + 1 <= pa_bits <= 48):
            raise UnsupportedTranslationGeometry(
                f"PA_BITS={pa_bits} is outside the supported range "
                f"{page_shift + 1}..48; 52-bit PA/LPA2 is not implemented"
            )

        va_index_width = va_bits - page_shift
        levels = (va_index_width + index_bits - 1) // index_bits
        max_levels = 3 if page_size == 65536 else 4

        if levels < 1 or levels > max_levels:
            raise UnsupportedTranslationGeometry(
                f"Unsupported number of translation levels ({levels}) for "
                f"page_size={page_size}, va_bits={va_bits}"
            )

        start_level = 4 - levels
        first_level_bits = va_index_width - index_bits * (levels - 1)

        if not (1 <= first_level_bits <= index_bits):
            raise UnsupportedTranslationGeometry(
                "Inconsistent page-table geometry: invalid first-level index width"
            )

        if start_level < 0 or start_level > 3:
            raise UnsupportedTranslationGeometry(
                "Unsupported ARM64 translation start level"
            )

        return cls(
            page_size=page_size,
            va_bits=va_bits,
            pa_bits=pa_bits,
            page_shift=page_shift,
            index_bits=index_bits,
            levels=levels,
            start_level=start_level,
            first_level_bits=first_level_bits,
        )

    @property
    def address_mask(self) -> int:
        return (
            ((1 << self.pa_bits) - 1)
            & ~((1 << self.page_shift) - 1)
        )

    def level_shift(self, depth: int) -> int:
        return self.page_shift + self.index_bits * (
            self.levels - depth - 1
        )

    def index_for(self, va_index: int, depth: int) -> int:
        bits = self.first_level_bits if depth == 0 else self.index_bits
        shift = self.level_shift(depth)
        return (va_index >> shift) & ((1 << bits) - 1)


@dataclass(frozen=True, slots=True)
class TranslationResult:
    virtual_address: int
    physical_address: int
    mapping_size: int
    level: int
    descriptor: int
    table_paddr: int
    descriptor_paddr: int
    is_block: bool


class ARM64Memory:
    """Translate ARM64 virtual addresses and read memory from a LiME dump.

    Args:
        profile: Loaded KernelProfile describing the dump's target kernel.
        lime: Open LimeReader for the raw physical-memory dump.
        kernel_pgd_paddr: Optional, independently established physical address
            of swapper_pg_dir. If omitted, derive it only when captured evidence
            establishes the runtime symbol and kernel-image VA-to-PA offset.
        pa_bits: Explicit PA width override. Required if the profile does not
            establish PA_BITS. Do not guess this value on evidentiary data.
        strip_top_byte_tag: If true, strips bits 63:56 for addresses supplied
            to public translation/read calls. This is opt-in for tagged user VAs.
    """

    def __init__(
        self,
        profile: KernelProfile,
        lime: LimeReader,
        *,
        kernel_pgd_paddr: int | None = None,
        pa_bits: int | None = None,
        strip_top_byte_tag: bool = False,
    ) -> None:
        self.profile = profile
        self.lime = lime
        self.strip_top_byte_tag = bool(strip_top_byte_tag)

        if profile.architecture != "arm64":
            raise UnsupportedTranslationGeometry(
                f"Kernel profile architecture is {profile.architecture!r}, "
                "not 'arm64'"
            )

        profile_pa_bits = profile.pa_bits

        if (
            pa_bits is not None
            and profile_pa_bits is not None
            and pa_bits != profile_pa_bits
        ):
            raise UnsupportedTranslationGeometry(
                f"Explicit pa_bits={pa_bits} conflicts with "
                f"profile PA_BITS={profile_pa_bits}"
            )

        effective_pa_bits = (
            pa_bits if pa_bits is not None else profile_pa_bits
        )

        self.geometry = PageTableGeometry.create(
            page_size=profile.page_size,
            va_bits=profile.va_bits,
            pa_bits=effective_pa_bits,
        )

        self._explicit_kernel_pgd_paddr = kernel_pgd_paddr
        self._resolved_kernel_pgd_paddr: int | None = None

        if kernel_pgd_paddr is not None:
            self._validate_pgd_paddr(
                kernel_pgd_paddr,
                name="kernel_pgd_paddr",
            )

    @staticmethod
    def _require_uint64(value: int, label: str) -> None:
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or not 0 <= value < (1 << 64)
        ):
            raise ValueError(
                f"{label} must be an unsigned 64-bit integer"
            )

    def _validate_pgd_paddr(
        self,
        value: int,
        *,
        name: str = "pgd_paddr",
    ) -> None:
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
        ):
            raise ValueError(
                f"{name} must be a non-negative integer"
            )

        if value & (self.geometry.page_size - 1):
            raise ValueError(
                f"{name} 0x{value:x} is not aligned to "
                f"{self.geometry.page_size}-byte pages"
            )

        if value >= (1 << self.geometry.pa_bits):
            raise ValueError(
                f"{name} exceeds the configured PA_BITS width"
            )

    def _normalize_virtual_address(
        self,
        vaddr: int,
        strip_tag: bool | None,
    ) -> tuple[int, int]:
        """Validate an AArch64 VA and return its translation index.

        For ARM64, TTBR0 addresses occupy the lower VA_BITS range,
        while TTBR1 addresses occupy the upper VA_BITS range.

        This deliberately does not use the x86-style sign-extension
        test on bit VA_BITS - 1.
        """
        self._require_uint64(vaddr, "vaddr")

        strip = (
            self.strip_top_byte_tag
            if strip_tag is None
            else strip_tag
        )

        if strip:
            # Opt-in for tagged pointers. Sign-extend bit 55 to safely support 
            # both user (TTBR0) and kernel (TTBR1) tagged pointers.
            vaddr = (vaddr | (0xFF << 56)) if (vaddr & (1 << 55)) else (vaddr & ~(0xFF << 56))

        va_bits = self.geometry.va_bits

        # Bits above the configured VA range distinguish TTBR0 and TTBR1.
        upper = vaddr >> va_bits
        upper_mask = (1 << (64 - va_bits)) - 1

        if vaddr & (1 << 63):
            # Upper/kernel address: all bits above VA_BITS must be ones.
            expected_upper = upper_mask
        else:
            # Lower/user address: all bits above VA_BITS must be zeros.
            expected_upper = 0

        if upper != expected_upper:
            raise NonCanonicalVirtualAddress(
                f"Virtual address 0x{vaddr:x} is outside the ARM64 "
                f"TTBR0/TTBR1 ranges for VA_BITS={va_bits}"
            )

        # Page-table indices use the low VA_BITS of the address.
        va_index = vaddr & ((1 << va_bits) - 1)

        return vaddr, va_index
    @property
    def kernel_pgd_paddr(self) -> int:
        """Physical address of the kernel's swapper page-table root.

        The physical address is derived only from the profile's captured
        runtime symbol and kernel-image VA-to-PA evidence. Otherwise, callers
        must supply an independently established root PA.
        """
        if self._explicit_kernel_pgd_paddr is not None:
            return self._explicit_kernel_pgd_paddr

        if self._resolved_kernel_pgd_paddr is not None:
            return self._resolved_kernel_pgd_paddr

        try:
            paddr = (
                self.profile.kernel_image_symbol_physical_address(
                    "swapper_pg_dir"
                )
            )
        except KernelProfileError as exc:
            raise KernelPageTableRootUnavailable(
                "Cannot derive the physical kernel page-table root. Supply "
                "kernel_pgd_paddr only if independently verified, or provide "
                "runtime swapper_pg_dir plus VMCOREINFO kimage_voffset or "
                "unredacted /proc/iomem Kernel code evidence."
            ) from exc

        self._validate_pgd_paddr(
            paddr,
            name="derived_kernel_pgd_paddr",
        )

        if not self.lime.contains_phys(
            paddr,
            self.geometry.page_size,
        ):
            raise KernelPageTableRootUnavailable(
                f"Derived swapper_pg_dir PA 0x{paddr:x} "
                "is not fully present in the LiME dump"
            )

        self._resolved_kernel_pgd_paddr = paddr
        return paddr

    @lru_cache(maxsize=131072)
    def _read_descriptor(self, paddr: int) -> int:
        try:
            raw = self.lime.read_phys(paddr, 8)
        except (
            LimeReaderError,
            PhysicalAddressNotCaptured,
        ) as exc:
            raise ARM64MemoryError(
                f"Cannot read page-table descriptor at physical address "
                f"0x{paddr:x}: {exc}"
            ) from exc

        byteorder = (
            "little"
            if self.lime.endianness == "<"
            else "big"
        )

        return int.from_bytes(raw, byteorder)

    def translate(
        self,
        vaddr: int,
        *,
        pgd_paddr: int | None = None,
        strip_tag: bool | None = None,
    ) -> TranslationResult:
        """Walk a supplied page-table root and return mapping details.

        pgd_paddr is a physical root-table address. When omitted, the kernel's
        swapper_pg_dir is used. For user processes, translate mm_struct->pgd
        into a physical address and pass it as pgd_paddr.
        """
        original, va_index = self._normalize_virtual_address(
            vaddr,
            strip_tag,
        )

        root = (
            self.kernel_pgd_paddr
            if pgd_paddr is None
            else pgd_paddr
        )

        self._validate_pgd_paddr(root)

        table_paddr = root
        geometry = self.geometry

        for depth in range(geometry.levels):
            level = geometry.start_level + depth
            index = geometry.index_for(va_index, depth)
            entry_paddr = table_paddr + index * 8

            descriptor = self._read_descriptor(entry_paddr)
            valid = descriptor & 1
            desc_type = descriptor & 0b11

            level_shift = geometry.level_shift(depth)
            mapping_size = 1 << level_shift

            if not valid:
                raise UnmappedVirtualAddress(
                    f"VA 0x{original:x} is unmapped at level {level} "
                    f"(descriptor PA 0x{entry_paddr:x}, "
                    f"descriptor 0x{descriptor:x})"
                )

            is_last = depth == geometry.levels - 1

            if is_last:
                if desc_type != 0b11:
                    raise InvalidPageTableDescriptor(
                        f"Invalid leaf descriptor 0x{descriptor:x} "
                        f"at level {level} for VA 0x{original:x}"
                    )

                page_base = descriptor & geometry.address_mask
                offset = va_index & (geometry.page_size - 1)

                return TranslationResult(
                    virtual_address=original,
                    physical_address=page_base + offset,
                    mapping_size=geometry.page_size,
                    level=level,
                    descriptor=descriptor,
                    table_paddr=table_paddr,
                    descriptor_paddr=entry_paddr,
                    is_block=False,
                )

            if desc_type == 0b11:
                # Table descriptor. The output address is masked separately
                # from attributes such as PXNTable and UXNTable.
                next_table = descriptor & geometry.address_mask
                table_paddr = next_table
                continue

            if desc_type == 0b01:
                if level == 0:
                    raise InvalidPageTableDescriptor(
                        f"Block descriptors are not valid at level 0 "
                        f"(VA 0x{original:x})"
                    )

                lower_reserved = (
                    (mapping_size - 1)
                    & ~((1 << geometry.page_shift) - 1)
                )

                if descriptor & lower_reserved:
                    raise InvalidPageTableDescriptor(
                        f"Misaligned block descriptor 0x{descriptor:x} "
                        f"at level {level}"
                    )

                block_base = (
                    descriptor
                    & geometry.address_mask
                    & ~(mapping_size - 1)
                )

                offset = va_index & (mapping_size - 1)

                return TranslationResult(
                    virtual_address=original,
                    physical_address=block_base + offset,
                    mapping_size=mapping_size,
                    level=level,
                    descriptor=descriptor,
                    table_paddr=table_paddr,
                    descriptor_paddr=entry_paddr,
                    is_block=True,
                )

            raise InvalidPageTableDescriptor(
                f"Reserved descriptor type 0b{desc_type:02b} "
                f"at level {level} for VA 0x{original:x}: "
                f"0x{descriptor:x}"
            )

        raise InvalidPageTableDescriptor(
            "Page-table walk ended without a leaf or block"
        )

    def virt_to_phys(
        self,
        vaddr: int,
        *,
        pgd_paddr: int | None = None,
        strip_tag: bool | None = None,
    ) -> int:
        return self.translate(
            vaddr,
            pgd_paddr=pgd_paddr,
            strip_tag=strip_tag,
        ).physical_address

    def read_virtual(
        self,
        vaddr: int,
        size: int,
        *,
        pgd_paddr: int | None = None,
        strip_tag: bool | None = None,
    ) -> bytes:
        """Read a virtual byte span, re-walking at each mapping boundary."""
        if (
            not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
        ):
            raise ValueError(
                "size must be a non-negative integer"
            )

        self._require_uint64(vaddr, "vaddr")

        if size == 0:
            return b""

        # Resolve the root once so omitted root consistently means kernel
        # mappings throughout the operation.
        root = (
            self.kernel_pgd_paddr
            if pgd_paddr is None
            else pgd_paddr
        )

        output = bytearray()
        cursor = vaddr
        remaining = size

        while remaining:
            result = self.translate(
                cursor,
                pgd_paddr=root,
                strip_tag=strip_tag,
            )

            # TranslationResult.physical_address already includes the offset
            # within the mapped page/block. Mapping bases are aligned to their
            # mapping size, so derive the remaining bytes without normalizing
            # the same VA a second time.
            offset_in_mapping = (
                result.physical_address & (result.mapping_size - 1)
            )

            chunk_size = min(
                remaining,
                result.mapping_size - offset_in_mapping,
            )

            try:
                output.extend(
                    self.lime.read_phys(
                        result.physical_address,
                        chunk_size,
                    )
                )
            except (
                LimeReaderError,
                PhysicalAddressNotCaptured,
            ) as exc:
                raise ARM64MemoryError(
                    f"Virtual read at VA 0x{cursor:x} maps to uncaptured "
                    f"physical memory (PA 0x{result.physical_address:x}): "
                    f"{exc}"
                ) from exc

            cursor += chunk_size
            remaining -= chunk_size

            if cursor >= (1 << 64) and remaining:
                raise NonCanonicalVirtualAddress(
                    "Virtual read overflowed the 64-bit address space"
                )

        return bytes(output)

    def read_kernel_virtual(
        self,
        vaddr: int,
        size: int,
    ) -> bytes:
        return self.read_virtual(
            vaddr,
            size,
            pgd_paddr=self.kernel_pgd_paddr,
        )

    def read_process_virtual(
        self,
        pgd_paddr: int,
        vaddr: int,
        size: int,
        *,
        strip_tag: bool | None = None,
    ) -> bytes:
        """Read a process virtual address using that process's physical PGD."""
        return self.read_virtual(
            vaddr,
            size,
            pgd_paddr=pgd_paddr,
            strip_tag=strip_tag,
        )

    def kernel_pointer_to_phys(
        self,
        kernel_vaddr: int,
    ) -> int:
        """Translate a kernel virtual pointer using swapper_pg_dir."""
        return self.virt_to_phys(
            kernel_vaddr,
            pgd_paddr=self.kernel_pgd_paddr,
        )


__all__ = [
    "ARM64Memory",
    "ARM64MemoryError",
    "InvalidPageTableDescriptor",
    "KernelPageTableRootUnavailable",
    "NonCanonicalVirtualAddress",
    "PageTableGeometry",
    "TranslationResult",
    "UnmappedVirtualAddress",
    "UnsupportedTranslationGeometry",
]