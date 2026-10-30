#!/usr/bin/env python3
"""
KernelProfile consumer for lime-arm64-kernel-profile.

Consumes:
    kernel-profile/profile.json

Provides:
    - Kernel metadata and validation
    - Runtime symbol resolution
    - ARM64 kernel-image virtual-to-physical address conversion
    - BTF structure/member lookup, including anonymous structs and unions
    - Raw artifact provenance and hash metadata
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


class KernelProfileError(RuntimeError):
    """Base error for profile loading and lookup failures."""


class KernelProfileFormatError(KernelProfileError):
    """Raised when profile.json has an unsupported or malformed format."""


class KernelProfileLookupError(KernelProfileError):
    """Raised when a requested symbol, type, or member cannot be found."""


class KernelProfile:
    FORMAT = "lime-arm64-kernel-profile"
    SUPPORTED_VERSIONS = {1, 2}

    def __init__(
        self,
        profile_dir: str | Path,
        *,
        require_valid: bool = False,
    ) -> None:
        self.profile_dir = Path(profile_dir).expanduser().resolve()
        self.profile_path = self.profile_dir / "profile.json"

        if not self.profile_dir.is_dir():
            raise KernelProfileError(
                f"Profile directory does not exist: {self.profile_dir}"
            )

        if not self.profile_path.is_file():
            raise KernelProfileError(
                f"Missing profile.json: {self.profile_path}"
            )

        try:
            self.data: dict[str, Any] = json.loads(
                self.profile_path.read_text(
                    encoding="utf-8",
                    errors="strict",
                )
            )
        except json.JSONDecodeError as exc:
            raise KernelProfileFormatError(
                f"Invalid JSON in {self.profile_path}: {exc}"
            ) from exc

        if not isinstance(self.data, dict):
            raise KernelProfileFormatError(
                "The profile root must be a JSON object"
            )

        self._validate_format()

        # Lazily built index for resolving anonymous BTF aggregates by type ID.
        self._btf_type_id_index: (
            dict[int, dict[str, Any]] | None
        ) = None

        # The profile is immutable for the lifetime of this object. Cache hot
        # symbol/BTF/source lookups used repeatedly by memory walkers.
        self._symbol_entry_cache: dict[str, dict[str, Any]] = {}
        self._runtime_symbol_cache: dict[tuple[str, bool], int] = {}
        self._btf_variant_cache: dict[str, list[dict[str, Any]]] = {}
        self._member_cache: dict[tuple[str, str], dict[str, Any]] = {}
        self._source_info_cache: dict[str, dict[str, Any]] = {}

        if require_valid and self.validation_status != "valid":
            issues = "; ".join(
                f"{item.get('code')}: {item.get('message')}"
                for item in self.validation_issues
                if item.get("severity") == "error"
            )
            raise KernelProfileFormatError(
                "Kernel profile validation failed: "
                f"{issues or 'unknown error'}"
            )

    # ------------------------------------------------------------------
    # Profile validation and basic metadata
    # ------------------------------------------------------------------

    def _validate_format(self) -> None:
        fmt = self.data.get("format")
        version = self.data.get("format_version")

        if fmt != self.FORMAT:
            raise KernelProfileFormatError(
                f"Unsupported profile format: {fmt!r}; "
                f"expected {self.FORMAT!r}"
            )

        if version not in self.SUPPORTED_VERSIONS:
            raise KernelProfileFormatError(
                f"Unsupported profile version: {version!r}; "
                f"supported versions: {sorted(self.SUPPORTED_VERSIONS)}"
            )

        for key in ("kernel", "sources", "symbols", "btf"):
            if not isinstance(self.data.get(key, {}), dict):
                raise KernelProfileFormatError(
                    f"Profile {key!r} section is malformed"
                )

        symbols = self.data.get("symbols", {})
        btf = self.data.get("btf", {})

        if not isinstance(symbols.get("selected", {}), dict):
            raise KernelProfileFormatError(
                "Profile symbol cache is malformed"
            )

        if not isinstance(btf.get("types", {}), dict):
            raise KernelProfileFormatError(
                "Profile BTF type cache is malformed"
            )

    @property
    def format_version(self) -> int:
        return int(self.data["format_version"])

    @property
    def kernel_release(self) -> Optional[str]:
        return self.data.get("kernel", {}).get("release")

    @property
    def build_id(self) -> Optional[str]:
        value = self.data.get("kernel", {}).get("build_id")
        return str(value) if value else None

    @property
    def architecture(self) -> Optional[str]:
        return (
            self.data
            .get("kernel", {})
            .get("architecture", {})
            .get("normalized")
        )

    @property
    def page_size(self) -> Optional[int]:
        value = self.data.get("kernel", {}).get("page_size")
        return int(value) if value is not None else None

    @property
    def va_bits(self) -> Optional[int]:
        value = self.data.get("kernel", {}).get("va_bits")
        return int(value) if value is not None else None

    @property
    def pa_bits(self) -> Optional[int]:
        value = self.data.get("kernel", {}).get("pa_bits")
        return int(value) if value is not None else None

    @property
    def validation_status(self) -> str:
        return (
            self.data
            .get("validation", {})
            .get("status", "unknown")
        )

    @property
    def validation_issues(self) -> list[dict[str, Any]]:
        issues = self.data.get("validation", {}).get("issues", [])
        return list(issues) if isinstance(issues, list) else []

    @property
    def capture_metadata(self) -> dict[str, Any]:
        value = self.data.get("capture", {})
        return dict(value) if isinstance(value, dict) else {}

    @property
    def source_files(self) -> list[str]:
        sources = self.data.get("sources", {})
        return sorted(sources) if isinstance(sources, dict) else []

    @property
    def kaslr_compiled(self) -> bool:
        return bool(
            self.data
            .get("boot", {})
            .get("kaslr_compiled", False)
        )

    @property
    def nokaslr(self) -> bool:
        return bool(
            self.data
            .get("boot", {})
            .get("nokaslr", False)
        )

    @property
    def runtime_relocation(self) -> Optional[int]:
        value = (
            self.data
            .get("boot", {})
            .get("kaslr", {})
            .get("runtime_offset")
        )
        return self.parse_address(value) if value is not None else None

    # ------------------------------------------------------------------
    # VMCOREINFO and ARM64 kernel-image address conversion
    # ------------------------------------------------------------------

    @property
    def vmcoreinfo_values(self) -> dict[str, Any]:
        values = (
            self.data.get("vmcoreinfo", {})
            .get("notes", {})
            .get("values", {})
        )
        return dict(values) if isinstance(values, dict) else {}

    @property
    def vmcoreinfo_symbols(self) -> dict[str, Any]:
        symbols = (
            self.data.get("vmcoreinfo", {})
            .get("notes", {})
            .get("symbols", {})
        )
        return dict(symbols) if isinstance(symbols, dict) else {}

    def vmcoreinfo_number(self, *names: str) -> Optional[int]:
        """Return a numeric VMCOREINFO value using supplied key aliases."""
        values = self.vmcoreinfo_values

        for name in names:
            if name not in values:
                continue

            value = values[name]

            if isinstance(value, bool):
                continue

            if isinstance(value, int):
                return value

            if not isinstance(value, str):
                continue

            try:
                return int(value, 0)
            except ValueError:
                try:
                    return int(value, 10)
                except ValueError:
                    continue

        return None

    def kernel_image_voffset(self) -> int:
        """Obtain ARM64 kimage_voffset from captured evidence.

        VMCOREINFO takes precedence. When it does not provide the value, derive
        it from the runtime _text address and the physical start of the
        captured /proc/iomem "Kernel code" range.

        The fallback requires one unredacted Kernel code range and a usable
        runtime _text address. It does not invent an offset when evidence is
        missing or inconsistent.
        """
        if self.architecture != "arm64":
            raise KernelProfileLookupError(
                "Kernel-image VA-to-PA conversion supports ARM64 only"
            )

        vmcore_voffset = self.vmcoreinfo_number(
            "NUMBER(kimage_voffset)",
            "kimage_voffset",
        )

        if vmcore_voffset is not None:
            return vmcore_voffset

        iomem = self.data.get("memory", {}).get("iomem", {})

        if not isinstance(iomem, dict) or iomem.get("redacted"):
            raise KernelProfileLookupError(
                "VMCOREINFO kimage_voffset is unavailable and /proc/iomem "
                "does not provide unredacted kernel image addresses"
            )

        ranges = iomem.get("ranges", [])

        if not isinstance(ranges, list):
            raise KernelProfileFormatError(
                "Profile /proc/iomem ranges are malformed"
            )

        code_ranges: list[tuple[int, int]] = []
        image_ranges: list[tuple[int, int, str]] = []

        for item in ranges:
            if not isinstance(item, dict):
                continue

            name = str(item.get("name", "")).strip()

            try:
                start = self.parse_address(item.get("start"))
                end = self.parse_address(item.get("end"))
            except KernelProfileFormatError:
                continue

            if start <= 0 or end < start:
                continue

            if name == "Kernel code":
                code_ranges.append((start, end))

            if name in {"Kernel code", "Kernel data", "Kernel bss"}:
                image_ranges.append((start, end, name))

        if len(code_ranges) != 1:
            raise KernelProfileLookupError(
                "Cannot derive kimage_voffset: expected exactly one visible "
                "'Kernel code' range in captured /proc/iomem"
            )

        try:
            text_vaddr = self.symbol_runtime_address("_text")
        except KernelProfileError as exc:
            raise KernelProfileLookupError(
                "Cannot derive kimage_voffset: runtime _text address "
                "is unavailable"
            ) from exc

        code_start_paddr, _ = code_ranges[0]
        voffset = text_vaddr - code_start_paddr

        if voffset <= 0:
            raise KernelProfileFormatError(
                "Derived kimage_voffset is non-positive; kernel symbols and "
                "/proc/iomem may not describe the same boot"
            )

        if text_vaddr - voffset != code_start_paddr:
            raise KernelProfileFormatError(
                "Derived kimage_voffset does not reproduce "
                "the Kernel code start"
            )

        if not any(
            start <= code_start_paddr <= end and name == "Kernel code"
            for start, end, name in image_ranges
        ):
            raise KernelProfileFormatError(
                "Kernel code physical range is inconsistent in "
                "captured /proc/iomem"
            )

        return voffset

    def kernel_image_symbol_physical_address(self, name: str) -> int:
        """Translate a kernel-image symbol VA to its physical address.

        Intended for kernel-image symbols such as swapper_pg_dir. This is
        not a general conversion for arbitrary vmalloc or linear-map pointers.
        """
        if self.architecture != "arm64":
            raise KernelProfileLookupError(
                "Kernel-image VA-to-PA conversion supports ARM64 only"
            )

        vaddr = self.symbol_runtime_address(name)
        voffset = self.kernel_image_voffset()
        paddr = vaddr - voffset

        if paddr < 0:
            raise KernelProfileFormatError(
                f"{name} - kimage_voffset produced a negative physical address"
            )

        iomem = self.data.get("memory", {}).get("iomem", {})
        ranges = iomem.get("ranges", []) if isinstance(iomem, dict) else []

        image_ranges: list[tuple[int, int]] = []

        for item in ranges if isinstance(ranges, list) else []:
            if not isinstance(item, dict):
                continue

            if item.get("name") not in {
                "Kernel code",
                "Kernel data",
                "Kernel bss",
            }:
                continue

            try:
                start = self.parse_address(item.get("start"))
                end = self.parse_address(item.get("end"))
            except KernelProfileFormatError:
                continue

            if start > 0 and end >= start:
                image_ranges.append((start, end))

        if image_ranges and not any(
            start <= paddr <= end for start, end in image_ranges
        ):
            raise KernelProfileLookupError(
                f"Resolved physical address 0x{paddr:x} for {name} falls "
                "outside the captured kernel image ranges; check that "
                "metadata matches the dump"
            )

        return paddr

    # ------------------------------------------------------------------
    # Address helpers
    # ------------------------------------------------------------------

    @staticmethod
    def parse_address(value: Any) -> int:
        """Convert hexadecimal or decimal profile addresses to integers."""
        if isinstance(value, int) and not isinstance(value, bool):
            return value

        if not isinstance(value, str):
            raise KernelProfileFormatError(
                f"Invalid address value: {value!r}"
            )

        try:
            return int(value, 0)
        except ValueError as exc:
            raise KernelProfileFormatError(
                f"Invalid address string: {value!r}"
            ) from exc

    @staticmethod
    def format_address(value: int) -> str:
        if value < 0:
            return f"-0x{-value:x}"
        return f"0x{value:x}"

    # ------------------------------------------------------------------
    # Source artifacts and provenance
    # ------------------------------------------------------------------

    def source_info(self, filename: str) -> dict[str, Any]:
        """Look up a source by relative path or unique basename."""
        cache_key = str(filename)
        cached = self._source_info_cache.get(cache_key)
        if cached is not None:
            return cached

        sources = self.data.get("sources", {})

        if not isinstance(sources, dict):
            raise KernelProfileFormatError(
                "Profile sources map is malformed"
            )

        key = str(filename).replace("\\", "/")

        if key.startswith("/") or any(
            part == ".." for part in key.split("/")
        ):
            raise KernelProfileFormatError(
                f"Invalid source selector: {filename!r}"
            )

        while key.startswith("./"):
            key = key[2:]

        if key in sources:
            result = sources[key]
            if isinstance(result, dict):
                self._source_info_cache[cache_key] = result
            return result

        matches = [
            info for name, info in sources.items()
            if Path(str(name)).name == Path(key).name
        ]

        if len(matches) == 1:
            self._source_info_cache[cache_key] = matches[0]
            return matches[0]

        if len(matches) > 1:
            raise KernelProfileLookupError(
                f"Source basename {filename!r} is ambiguous; "
                "use a relative path"
            )

        raise KernelProfileLookupError(
            f"Unknown profile source: {filename}"
        )

    def source_path(self, filename: str) -> Path:
        """Resolve a source path and prevent paths escaping the profile dir."""
        info = self.source_info(filename)
        rel = info.get("path")

        if not isinstance(rel, str) or not rel:
            raise KernelProfileFormatError(
                f"Source {filename!r} has no valid relative path"
            )

        candidate = Path(rel)

        if candidate.is_absolute():
            raise KernelProfileFormatError(
                f"Absolute source paths are not permitted: {rel!r}"
            )

        path = (self.profile_dir / candidate).resolve()

        try:
            path.relative_to(self.profile_dir)
        except ValueError as exc:
            raise KernelProfileFormatError(
                f"Source path escapes profile directory: {rel!r}"
            ) from exc

        return path

    def source_sha256(self, filename: str) -> Optional[str]:
        value = self.source_info(filename).get("sha256")
        return str(value) if value else None

    def verify_source_exists(self, filename: str) -> bool:
        try:
            return self.source_path(filename).is_file()
        except KernelProfileError:
            return False

    # ------------------------------------------------------------------
    # Kernel symbols
    # ------------------------------------------------------------------

    def _symbol_entry(self, name: str) -> dict[str, Any]:
        cached = self._symbol_entry_cache.get(name)
        if cached is not None:
            return cached

        symbols = (
            self.data
            .get("symbols", {})
            .get("selected", {})
        )

        try:
            entry = symbols[name]
        except KeyError as exc:
            raise KernelProfileLookupError(
                f"Symbol not present in cached profile: {name}"
            ) from exc

        if not isinstance(entry, dict):
            raise KernelProfileFormatError(
                f"Symbol cache entry for {name!r} is malformed"
            )

        self._symbol_entry_cache[name] = entry
        return entry

    @staticmethod
    def _first_symbol_address(
        entries: list[dict[str, Any]] | None,
    ) -> Optional[int]:
        if not entries:
            return None

        for entry in entries:
            if not isinstance(entry, dict):
                continue

            value = entry.get("address")
            if value is None:
                continue

            address = KernelProfile.parse_address(value)

            # Restricted kallsyms can expose symbols with address zero.
            if address != 0:
                return address

        return None

    def symbol_link_address(self, name: str) -> int:
        """Return the symbol address from System.map."""
        entry = self._symbol_entry(name)

        address = self._first_symbol_address(
            entry.get("system_map")
        )

        if address is None:
            raise KernelProfileLookupError(
                f"No usable System.map address for symbol: {name}"
            )

        return address

    def symbol_runtime_address(
        self,
        name: str,
        *,
        allow_relocation: bool = True,
    ) -> int:
        """Resolve a runtime symbol address from captured evidence.

        Resolution order:
          1. Non-zero runtime /proc/kallsyms address
          2. Non-zero VMCOREINFO symbol address
          3. System.map plus a derived relocation
          4. Unrelocated System.map only if KASLR is disabled or not compiled

        The method does not silently assume that link-time and runtime
        addresses are identical when KASLR may be active.
        """
        cache_key = (name, allow_relocation)
        cached = self._runtime_symbol_cache.get(cache_key)
        if cached is not None:
            return cached

        entry = self._symbol_entry(name)

        runtime = self._first_symbol_address(
            entry.get("kallsyms")
        )
        if runtime is not None:
            self._runtime_symbol_cache[cache_key] = runtime
            return runtime

        vmcore = self._first_symbol_address(
            entry.get("vmcoreinfo")
        )
        if vmcore is not None:
            self._runtime_symbol_cache[cache_key] = vmcore
            return vmcore

        link = self._first_symbol_address(
            entry.get("system_map")
        )
        if link is None:
            raise KernelProfileLookupError(
                f"No usable address for symbol: {name}"
            )

        if allow_relocation:
            delta = self.runtime_relocation
            if delta is not None:
                result = link + delta
                self._runtime_symbol_cache[cache_key] = result
                return result

        if self.nokaslr or not self.kaslr_compiled:
            self._runtime_symbol_cache[cache_key] = link
            return link

        raise KernelProfileLookupError(
            f"Cannot determine runtime address for {name!r}: kallsyms is "
            "hidden, VMCOREINFO has no runtime address, and KASLR "
            "relocation is unresolved"
        )

    def has_symbol(self, name: str) -> bool:
        return name in (
            self.data
            .get("symbols", {})
            .get("selected", {})
        )

    # ------------------------------------------------------------------
    # BTF type helpers
    # ------------------------------------------------------------------

    def _btf_type_variants(
        self,
        type_name: str,
    ) -> list[dict[str, Any]]:
        cached = self._btf_variant_cache.get(type_name)
        if cached is not None:
            return cached

        types = (
            self.data
            .get("btf", {})
            .get("types", {})
        )

        variants = types.get(type_name)

        if not variants:
            raise KernelProfileLookupError(
                f"BTF type not cached in profile: {type_name}"
            )

        if not isinstance(variants, list):
            raise KernelProfileFormatError(
                f"BTF type entry for {type_name!r} is malformed"
            )

        self._btf_variant_cache[type_name] = variants
        return variants

    def btf_type(
        self,
        type_name: str,
        *,
        kind: Optional[str] = None,
    ) -> dict[str, Any]:
        variants = self._btf_type_variants(type_name)

        if kind is not None:
            kind = kind.upper()
            variants = [
                item for item in variants
                if str(item.get("kind", "")).upper() == kind
            ]

        if not variants:
            suffix = f" with kind {kind}" if kind else ""
            raise KernelProfileLookupError(
                f"BTF type {type_name!r}{suffix} not found"
            )

        if len(variants) > 1:
            if kind is None:
                struct_variants = [
                    item for item in variants
                    if item.get("kind") == "STRUCT"
                ]

                if len(struct_variants) == 1:
                    return struct_variants[0]

            raise KernelProfileLookupError(
                f"Ambiguous BTF type {type_name!r}: "
                f"{len(variants)} variants"
            )

        return variants[0]

    def has_type(self, type_name: str) -> bool:
        return type_name in (
            self.data
            .get("btf", {})
            .get("types", {})
        )

    def struct_size(self, struct_name: str) -> int:
        struct = self.btf_type(
            struct_name,
            kind="STRUCT",
        )

        size = struct.get("size")
        if size is None:
            raise KernelProfileFormatError(
                f"BTF STRUCT {struct_name!r} has no size"
            )

        return int(size)

    def _btf_type_by_id(
        self,
        type_id: Any,
    ) -> Optional[dict[str, Any]]:
        """Find a serialized BTF record by original BTF type ID."""
        if not isinstance(type_id, int) or isinstance(type_id, bool):
            return None

        if self._btf_type_id_index is None:
            index: dict[int, dict[str, Any]] = {}
            types = self.data.get("btf", {}).get("types", {})

            if isinstance(types, dict):
                for variants in types.values():
                    if not isinstance(variants, list):
                        continue

                    for item in variants:
                        if not isinstance(item, dict):
                            continue

                        ident = item.get("type_id")

                        if (
                            isinstance(ident, int)
                            and not isinstance(ident, bool)
                        ):
                            index.setdefault(ident, item)

            self._btf_type_id_index = index

        return self._btf_type_id_index.get(type_id)

    @staticmethod
    def _member_local_bit_offset(
        member: dict[str, Any],
    ) -> int:
        """Get a member's local offset in bits."""
        value = member.get("bit_offset")

        if value is not None:
            return int(value)

        return int(member.get("offset", 0)) * 8

    def _find_member_in_aggregate(
        self,
        type_info: dict[str, Any],
        member_name: str,
        *,
        base_bit_offset: int = 0,
        seen_type_ids: frozenset[int] = frozenset(),
    ) -> Optional[dict[str, Any]]:
        """Find a direct or anonymous struct/union member.

        Current profiles preserve unnamed BTF members under unique keys such
        as ``__anon_member_0`` and retain the original field name in
        ``member_name``. This method handles both that representation and
        older profiles using empty member-name keys.

        Parent and child bit offsets are accumulated so the returned offset
        is relative to the originally requested structure.
        """
        type_id = type_info.get("type_id")

        if isinstance(type_id, int):
            if type_id in seen_type_ids:
                return None
            seen_type_ids = seen_type_ids | {type_id}

        members = type_info.get("members")

        if not isinstance(members, dict):
            return None

        # Prefer a named direct member.
        direct = members.get(member_name)

        if isinstance(direct, dict):
            absolute_bits = (
                base_bit_offset
                + self._member_local_bit_offset(direct)
            )

            result = dict(direct)
            result["bit_offset"] = absolute_bits
            result["offset"] = absolute_bits // 8

            return result

        nested_results: list[dict[str, Any]] = []

        for child_name, child_member in members.items():
            if not isinstance(child_member, dict):
                continue

            original_name = child_member.get(
                "member_name",
                child_name,
            )

            is_anonymous = (
                original_name in (None, "", "(anon)", "__anon")
                or child_name in ("", "(anon)", "__anon")
                or child_name.startswith("__anon_member_")
            )

            if not is_anonymous:
                continue

            kind = str(child_member.get("kind", "")).upper()
            if kind not in ("STRUCT", "UNION"):
                continue

            child_type_id = child_member.get("resolved_type_id")
            if child_type_id is None:
                child_type_id = child_member.get("type_id")

            child_type = self._btf_type_by_id(child_type_id)

            if not isinstance(child_type, dict):
                continue

            child_kind = str(
                child_type.get("kind", "")
            ).upper()

            if child_kind not in ("STRUCT", "UNION"):
                continue

            child_base = (
                base_bit_offset
                + self._member_local_bit_offset(child_member)
            )

            found = self._find_member_in_aggregate(
                child_type,
                member_name,
                base_bit_offset=child_base,
                seen_type_ids=seen_type_ids,
            )

            if found is not None:
                nested_results.append(found)

        if not nested_results:
            return None

        offsets = {
            int(
                item.get(
                    "bit_offset",
                    int(item.get("offset", 0)) * 8,
                )
            )
            for item in nested_results
        }

        if len(offsets) > 1:
            raise KernelProfileLookupError(
                f"Ambiguous member {member_name!r} inside anonymous "
                f"aggregates of BTF type "
                f"{type_info.get('name') or type_info.get('type_id')}"
            )

        return nested_results[0]

    def member(
        self,
        struct_name: str,
        member_name: str,
    ) -> dict[str, Any]:
        """Return a member record, including anonymous aggregate members."""
        cache_key = (struct_name, member_name)
        cached = self._member_cache.get(cache_key)
        if cached is not None:
            return cached

        struct = self.btf_type(
            struct_name,
            kind="STRUCT",
        )

        found = self._find_member_in_aggregate(
            struct,
            member_name,
        )

        if found is None:
            raise KernelProfileLookupError(
                f"Member {struct_name}.{member_name} not found in direct "
                "or anonymous BTF struct/union members. If the profile "
                "contains only selected BTF types, rebuild with --all-btf."
            )

        self._member_cache[cache_key] = found
        return found

    def has_member(
        self,
        struct_name: str,
        member_name: str,
    ) -> bool:
        try:
            self.member(struct_name, member_name)
            return True
        except KernelProfileLookupError:
            return False

    def member_offset(
        self,
        struct_name: str,
        member_name: str,
    ) -> int:
        """Return the byte offset of a member."""
        member = self.member(struct_name, member_name)
        offset = member.get("offset")

        if offset is None:
            raise KernelProfileFormatError(
                f"Member {struct_name}.{member_name} has no byte offset"
            )

        return int(offset)

    def member_bit_offset(
        self,
        struct_name: str,
        member_name: str,
    ) -> int:
        member = self.member(struct_name, member_name)
        value = member.get("bit_offset")

        if value is None:
            return self.member_offset(
                struct_name,
                member_name,
            ) * 8

        return int(value)

    def member_bitfield_size(
        self,
        struct_name: str,
        member_name: str,
    ) -> int:
        member = self.member(struct_name, member_name)
        return int(member.get("bitfield_size", 0))

    def member_type(
        self,
        struct_name: str,
        member_name: str,
    ) -> dict[str, Any]:
        member = self.member(struct_name, member_name)

        return {
            "type_id": member.get("type_id"),
            "resolved_type_id": member.get("resolved_type_id"),
            "kind": member.get("kind"),
            "name": member.get("name"),
            "modifiers": list(member.get("modifiers", [])),
        }

    # ------------------------------------------------------------------
    # Physical memory metadata
    # ------------------------------------------------------------------

    @property
    def system_ram_ranges(self) -> list[tuple[int, int]]:
        """Return (start, end) pairs from the profile's System RAM map."""
        result: list[tuple[int, int]] = []

        ranges = (
            self.data
            .get("memory", {})
            .get("iomem", {})
            .get("system_ram", [])
        )

        for item in ranges:
            try:
                start = self.parse_address(item["start"])
                end = self.parse_address(item["end"])
            except (KeyError, KernelProfileFormatError, TypeError):
                continue

            if start <= end:
                result.append((start, end))

        return sorted(result)

    def physical_address_in_system_ram(
        self,
        paddr: int,
    ) -> bool:
        return any(
            start <= paddr <= end
            for start, end in self.system_ram_ranges
        )

    # ------------------------------------------------------------------
    # Kernel configuration and modules
    # ------------------------------------------------------------------

    def config(
        self,
        name: str,
        default: Any = None,
    ) -> Any:
        cfg = self.data.get("config", {})

        full = cfg.get("all")
        if isinstance(full, dict) and name in full:
            return full[name]

        relevant = cfg.get("relevant", {})
        if isinstance(relevant, dict):
            return relevant.get(name, default)

        return default

    @property
    def modules(self) -> list[dict[str, Any]]:
        modules = self.data.get("modules", {}).get("modules", [])
        return list(modules) if isinstance(modules, list) else []

    def module(self, name: str) -> dict[str, Any]:
        for module in self.modules:
            if module.get("name") == name:
                return module

        raise KernelProfileLookupError(
            f"Loaded module not found: {name}"
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        relocation = self.runtime_relocation

        return {
            "kernel_release": self.kernel_release,
            "build_id": self.build_id,
            "architecture": self.architecture,
            "page_size": self.page_size,
            "va_bits": self.va_bits,
            "pa_bits": self.pa_bits,
            "kaslr_compiled": self.kaslr_compiled,
            "nokaslr": self.nokaslr,
            "runtime_relocation": (
                self.format_address(relocation)
                if relocation is not None
                else None
            ),
            "validation_status": self.validation_status,
            "source_count": len(self.source_files),
            "system_ram_ranges": [
                {
                    "start": self.format_address(start),
                    "end": self.format_address(end),
                }
                for start, end in self.system_ram_ranges
            ],
        }

    def __repr__(self) -> str:
        return (
            f"KernelProfile("
            f"release={self.kernel_release!r}, "
            f"arch={self.architecture!r}, "
            f"page_size={self.page_size!r}, "
            f"va_bits={self.va_bits!r}, "
            f"validation={self.validation_status!r}"
            f")"
        )


if __name__ == "__main__":
    import argparse
    import pprint

    parser = argparse.ArgumentParser()
    parser.add_argument("profile_dir")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--symbol")
    parser.add_argument("--struct")
    parser.add_argument("--member")

    args = parser.parse_args()

    profile = KernelProfile(
        args.profile_dir,
        require_valid=args.strict,
    )

    pprint.pp(profile.summary())

    if args.symbol:
        address = profile.symbol_runtime_address(args.symbol)
        print(
            f"{args.symbol} = "
            f"{profile.format_address(address)}"
        )

    if args.struct:
        print(
            f"sizeof(struct {args.struct}) = "
            f"{profile.struct_size(args.struct)}"
        )

    if args.struct and args.member:
        offset = profile.member_offset(args.struct, args.member)
        print(
            f"offsetof(struct {args.struct}, {args.member}) = "
            f"{offset:#x}"
        )