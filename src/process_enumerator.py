#!/usr/bin/env python3
"""Enumerate Linux tasks and inspect process VMAs using target BTF metadata.

The implementation deliberately depends on BTF member offsets and runtime
kernel symbols. It does not hard-code task_struct/mm_struct layouts.

VMA traversal supports both the classic ``mm_struct.mmap`` linked list and
Maple Tree ``mm_struct.mm_mt`` storage. Maple Tree node layouts and member
offsets are taken from the target kernel's BTF; unsupported or inconsistent
encodings are rejected rather than guessed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

from arm64_memory import ARM64Memory, ARM64MemoryError, TranslationResult
from kernel_profile import KernelProfile, KernelProfileError


class ProcessEnumerationError(RuntimeError):
    """Base error for task-list and process metadata operations."""


class ProcessNotFound(ProcessEnumerationError):
    """No process leader with the requested PID was found in the task list."""


class ProcessHasNoUserAddressSpace(ProcessEnumerationError):
    """The requested task has no mm_struct (normally a kernel thread)."""


class UnsupportedVmaLayout(ProcessEnumerationError):
    """The target kernel's VMA representation is not supported by this walker."""


class CorruptKernelList(ProcessEnumerationError):
    """A kernel linked list is malformed, cyclic, or exceeds safety limits."""


@dataclass(frozen=True, slots=True)
class ProcessInfo:
    pid: int
    tgid: int
    comm: str
    task_address: int
    mm_address: int
    parent_pid: Optional[int] = None

    @property
    def is_kernel_thread(self) -> bool:
        return self.mm_address == 0


@dataclass(frozen=True, slots=True)
class MemoryRegion:
    """A Linux virtual memory area; ``end`` is exclusive, as in Linux."""

    start: int
    end: int
    flags: Optional[int]
    page_offset: Optional[int]
    file_pointer: Optional[int]
    vma_address: int

    @property
    def size(self) -> int:
        return self.end - self.start

    def contains(self, address: int) -> bool:
        return self.start <= address < self.end


class ProcessEnumerator:
    """Walk ``init_task.tasks`` and enumerate process VMAs.

    Args:
        profile: Profile built from the same target kernel as the dump.
        memory: Configured ARM64Memory instance for the same LiME dump.
        max_tasks: Safety limit for task-list traversal.
        max_vmas: Safety limit for VMA and Maple Tree node traversal.

    ``list_processes`` reports process leaders (``pid == tgid``), matching the
    usual process-oriented view of ``ps`` more closely than a list of every
    kernel task. ``list_tasks`` includes individual threads too.
    """

    TASK_COMM_LEN = 16
    POINTER_SIZE = 8
    INT_SIZE = 4

    # Maple Tree encoding constants from include/linux/maple_tree.h.
    _MAPLE_ROOT_NODE = 0x02
    _MAPLE_NODE_MASK = 0xFF
    _MAPLE_NODE_TYPE_SHIFT = 3
    _MAPLE_ENODE_NULL = 0x04

    _MAPLE_MAX_INDEX = (1 << 64) - 1

    _MAPLE_DENSE = 0
    _MAPLE_LEAF_64 = 1
    _MAPLE_RANGE_64 = 2
    _MAPLE_ARANGE_64 = 3

    def __init__(
        self,
        profile: KernelProfile,
        memory: ARM64Memory,
        *,
        max_tasks: int = 1_000_000,
        max_vmas: int = 1_000_000,
    ) -> None:
        if profile.architecture != "arm64":
            raise ProcessEnumerationError(
                f"Expected an arm64 profile, got {profile.architecture!r}"
            )

        if max_tasks <= 0 or max_vmas <= 0:
            raise ValueError("max_tasks and max_vmas must be positive")

        self.profile = profile
        self.memory = memory
        self.max_tasks = max_tasks
        self.max_vmas = max_vmas

        self.byteorder = (
            "little"
            if memory.lime.endianness == "<"
            else "big"
        )

        # The kernel profile and memory dump are immutable during analysis.
        # Cache metadata lookups that otherwise occur for every task/VMA.
        self._offset_cache: dict[
            tuple[str, tuple[str, ...], bool],
            Optional[int],
        ] = {}

        self._maple_layout_cache: Optional[dict[str, int]] = None

        # PID -> physical PGD address.
        self._process_pgd_cache: dict[int, int] = {}

    # ------------------------------------------------------------------
    # BTF / kernel read helpers
    # ------------------------------------------------------------------

    def _offset(
        self,
        struct_name: str,
        *member_names: str,
        required: bool = True,
    ) -> Optional[int]:
        """Resolve and cache a BTF member offset.

        Multiple member names may be supplied to support kernel-version
        aliases such as ``vm_flags`` versus ``__vm_flags``.
        """
        cache_key = (
            struct_name,
            member_names,
            required,
        )

        if cache_key in self._offset_cache:
            return self._offset_cache[cache_key]

        for member_name in member_names:
            if self.profile.has_member(
                struct_name,
                member_name,
            ):
                result = self.profile.member_offset(
                    struct_name,
                    member_name,
                )
                self._offset_cache[cache_key] = result
                return result

        if required:
            joined = " or ".join(
                f"{struct_name}.{name}"
                for name in member_names
            )
            raise ProcessEnumerationError(
                f"Required BTF member not found: {joined}. "
                "Rebuild the profile with the matching kernel's "
                "vmlinux.btf."
            )

        self._offset_cache[cache_key] = None
        return None

    def _read_kernel(
        self,
        address: int,
        size: int,
    ) -> bytes:
        try:
            data = self.memory.read_kernel_virtual(
                address,
                size,
            )
        except (
            ARM64MemoryError,
            KernelProfileError,
            ValueError,
        ) as exc:
            raise ProcessEnumerationError(
                f"Cannot read kernel VA 0x{address:x} "
                f"({size} bytes): {exc}"
            ) from exc

        if len(data) != size:
            raise ProcessEnumerationError(
                f"Short kernel read at VA 0x{address:x}: "
                f"wanted {size}, got {len(data)}"
            )

        return data

    def _read_uint(
        self,
        address: int,
        size: int,
    ) -> int:
        return int.from_bytes(
            self._read_kernel(address, size),
            self.byteorder,
            signed=False,
        )

    def _read_sint(
        self,
        address: int,
        size: int,
    ) -> int:
        return int.from_bytes(
            self._read_kernel(address, size),
            self.byteorder,
            signed=True,
        )

    def _read_pointer(
        self,
        address: int,
    ) -> int:
        return self._read_uint(
            address,
            self.POINTER_SIZE,
        )

    # ------------------------------------------------------------------
    # Task enumeration
    # ------------------------------------------------------------------

    def _read_task(
        self,
        task_address: int,
    ) -> ProcessInfo:
        pid_offset = self._offset(
            "task_struct",
            "pid",
        )
        tgid_offset = self._offset(
            "task_struct",
            "tgid",
        )
        comm_offset = self._offset(
            "task_struct",
            "comm",
        )

        mm_offset = self._offset(
            "task_struct",
            "mm",
            required=False,
        )

        parent_offset = self._offset(
            "task_struct",
            "real_parent",
            "parent",
            required=False,
        )

        pid = self._read_sint(
            task_address + int(pid_offset),
            self.INT_SIZE,
        )

        tgid = self._read_sint(
            task_address + int(tgid_offset),
            self.INT_SIZE,
        )

        comm_raw = self._read_kernel(
            task_address + int(comm_offset),
            self.TASK_COMM_LEN,
        )

        comm = comm_raw.split(
            b"\0",
            1,
        )[0].decode(
            "utf-8",
            errors="replace",
        )

        mm_address = (
            self._read_pointer(
                task_address + int(mm_offset)
            )
            if mm_offset is not None
            else 0
        )

        parent_pid: Optional[int] = None

        if parent_offset is not None:
            parent_address = self._read_pointer(
                task_address + int(parent_offset)
            )

            if parent_address:
                try:
                    tgid_field = self._offset(
                        "task_struct",
                        "tgid",
                    )

                    parent_pid = self._read_sint(
                        parent_address
                        + int(tgid_field),
                        self.INT_SIZE,
                    )

                except ProcessEnumerationError:
                    # Parent information is ancillary. A stale or
                    # uncaptured parent pointer must not discard an
                    # otherwise usable task record.
                    parent_pid = None

        return ProcessInfo(
            pid=pid,
            tgid=tgid,
            comm=comm,
            task_address=task_address,
            mm_address=mm_address,
            parent_pid=parent_pid,
        )

    def iter_tasks(
        self,
    ) -> Iterator[ProcessInfo]:
        """Yield every task_struct in the circular ``init_task.tasks`` list."""
        tasks_offset = self._offset(
            "task_struct",
            "tasks",
        )

        try:
            init_task = self.profile.symbol_runtime_address(
                "init_task"
            )
        except KernelProfileError as exc:
            raise ProcessEnumerationError(
                "Cannot resolve runtime init_task. "
                "The profile needs usable runtime symbols "
                "or a defensible KASLR relocation."
            ) from exc

        list_head = (
            init_task
            + int(tasks_offset)
        )

        next_link = self._read_pointer(
            list_head
        )

        if next_link == 0:
            raise CorruptKernelList(
                "init_task.tasks.next is NULL"
            )

        # init_task itself is part of the circular task list.
        count = 1
        seen_links: set[int] = {
            list_head,
        }

        yield self._read_task(
            init_task
        )

        while next_link != list_head:
            if next_link == 0:
                raise CorruptKernelList(
                    "Encountered a NULL task-list link"
                )

            if next_link in seen_links:
                raise CorruptKernelList(
                    f"Task list loops at link "
                    f"0x{next_link:x} before returning "
                    "to init_task"
                )

            if count >= self.max_tasks:
                raise CorruptKernelList(
                    f"Task list exceeded "
                    f"max_tasks={self.max_tasks}; "
                    "refusing to continue"
                )

            seen_links.add(
                next_link
            )

            task_address = (
                next_link
                - int(tasks_offset)
            )

            if task_address <= 0:
                raise CorruptKernelList(
                    "Invalid task_struct address "
                    "derived from list link "
                    f"0x{next_link:x}"
                )

            yield self._read_task(
                task_address
            )

            count += 1

            next_link = self._read_pointer(
                next_link
            )

    def list_tasks(
        self,
    ) -> list[ProcessInfo]:
        return list(
            self.iter_tasks()
        )

    def iter_processes(
        self,
    ) -> Iterator[ProcessInfo]:
        """Yield process leaders (``pid == tgid``), including kernel threads."""
        for task in self.iter_tasks():
            # PID 0 is the boot/idle task and is omitted from the
            # ordinary process-oriented view.
            if (
                task.pid > 0
                and task.pid == task.tgid
            ):
                yield task

    def list_processes(
        self,
    ) -> list[ProcessInfo]:
        """Return process leaders sorted by PID for readable CLI output."""
        return sorted(
            self.iter_processes(),
            key=lambda item: (
                item.pid,
                item.task_address,
            ),
        )

    def find_process(
        self,
        pid: int,
    ) -> ProcessInfo:
        if (
            not isinstance(pid, int)
            or isinstance(pid, bool)
            or pid < 0
        ):
            raise ValueError(
                "pid must be a non-negative integer"
            )

        for process in self.iter_tasks():
            if process.pid == pid:
                return process

        raise ProcessNotFound(
            f"Process PID {pid} not found "
            "in the captured task list"
        )

    # ------------------------------------------------------------------
    # Process address-space translation
    # ------------------------------------------------------------------

    def process_pgd_paddr(
        self,
        pid: int,
    ) -> int:
        """Resolve ``task_struct.mm->pgd`` to its physical table address."""
        cached = self._process_pgd_cache.get(
            pid
        )

        if cached is not None:
            return cached

        process = self.find_process(
            pid
        )

        if process.mm_address == 0:
            raise ProcessHasNoUserAddressSpace(
                f"PID {pid} ({process.comm}) "
                "has no mm_struct; it is likely "
                "a kernel thread"
            )

        pgd_offset = self._offset(
            "mm_struct",
            "pgd",
        )

        pgd_kernel_va = self._read_pointer(
            process.mm_address
            + int(pgd_offset)
        )

        if pgd_kernel_va == 0:
            raise ProcessEnumerationError(
                f"PID {pid}: mm_struct.pgd "
                "is NULL in captured memory"
            )

        try:
            result = self.memory.kernel_pointer_to_phys(
                pgd_kernel_va
            )
        except ARM64MemoryError as exc:
            raise ProcessEnumerationError(
                f"PID {pid}: cannot translate "
                f"mm_struct.pgd pointer "
                f"0x{pgd_kernel_va:x} "
                "to a physical address: "
                f"{exc}"
            ) from exc

        self._process_pgd_cache[
            pid
        ] = result

        return result

    def translate_process_address(
        self,
        pid: int,
        address: int,
    ) -> TranslationResult:
        """Translate a process virtual address using its own page tables."""
        pgd_paddr = self.process_pgd_paddr(pid)

        try:
            return self.memory.translate(
                address,
                pgd_paddr=pgd_paddr,
            )
        except (ARM64MemoryError, ValueError) as exc:
            raise ProcessEnumerationError(
                f"Cannot translate PID {pid} VA 0x{address:x}: {exc}"
            ) from exc

    def read_process_memory(
        self,
        pid: int,
        address: int,
        size: int,
    ) -> bytes:
        """Read bytes from a PID's virtual address."""
        if size < 0:
            raise ValueError(
                "size must be non-negative"
            )

        pgd_paddr = self.process_pgd_paddr(
            pid
        )

        try:
            return self.memory.read_process_virtual(
                pgd_paddr,
                address,
                size,
            )
        except ARM64MemoryError as exc:
            raise ProcessEnumerationError(
                f"Cannot read PID {pid} "
                f"VA 0x{address:x}+0x{size:x}: "
                f"{exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Maple Tree metadata
    # ------------------------------------------------------------------

    def _maple_layout(
        self,
    ) -> dict[str, int]:
        """Resolve Maple Tree structure offsets and sizes from target BTF."""
        if self._maple_layout_cache is not None:
            return self._maple_layout_cache

        required_types = (
            "maple_tree",
            "maple_node",
            "maple_range_64",
            "maple_arange_64",
            "maple_metadata",
            "maple_type",
        )

        missing_types = [
            name
            for name in required_types
            if not self.profile.has_type(name)
        ]

        required_members = (
            ("mm_struct", "mm_mt"),
            ("maple_tree", "ma_root"),
            ("maple_node", "mr64"),
            ("maple_node", "ma64"),
            ("maple_node", "slot"),
            ("maple_range_64", "slot"),
            ("maple_range_64", "pivot"),
            ("maple_range_64", "meta"),
            ("maple_arange_64", "pivot"),
            ("maple_arange_64", "slot"),
            ("maple_arange_64", "gap"),
            ("maple_arange_64", "meta"),
            ("maple_metadata", "end"),
        )

        missing_members = [
            f"{struct}.{member}"
            for struct, member
            in required_members
            if not self.profile.has_member(
                struct,
                member,
            )
        ]

        if missing_types or missing_members:
            missing = (
                missing_types
                + missing_members
            )

            raise UnsupportedVmaLayout(
                "The profile lacks BTF needed "
                "for Maple Tree traversal: "
                + ", ".join(missing)
                + ". Rebuild profile.json with "
                "--all-btf using the matching kernel."
            )

        try:
            layout = {
                "mm_mt":
                    self.profile.member_offset(
                        "mm_struct",
                        "mm_mt",
                    ),

                "ma_root":
                    self.profile.member_offset(
                        "maple_tree",
                        "ma_root",
                    ),

                "node_mr64":
                    self.profile.member_offset(
                        "maple_node",
                        "mr64",
                    ),

                "node_ma64":
                    self.profile.member_offset(
                        "maple_node",
                        "ma64",
                    ),

                "node_slot":
                    self.profile.member_offset(
                        "maple_node",
                        "slot",
                    ),

                "node_size":
                    self.profile.struct_size(
                        "maple_node"
                    ),

                "range_slot":
                    self.profile.member_offset(
                        "maple_range_64",
                        "slot",
                    ),

                "range_pivot":
                    self.profile.member_offset(
                        "maple_range_64",
                        "pivot",
                    ),

                "range_meta":
                    self.profile.member_offset(
                        "maple_range_64",
                        "meta",
                    ),

                "range_size":
                    self.profile.struct_size(
                        "maple_range_64"
                    ),

                "arange_pivot":
                    self.profile.member_offset(
                        "maple_arange_64",
                        "pivot",
                    ),

                "arange_slot":
                    self.profile.member_offset(
                        "maple_arange_64",
                        "slot",
                    ),

                "arange_gap":
                    self.profile.member_offset(
                        "maple_arange_64",
                        "gap",
                    ),

                "arange_meta":
                    self.profile.member_offset(
                        "maple_arange_64",
                        "meta",
                    ),

                "arange_size":
                    self.profile.struct_size(
                        "maple_arange_64"
                    ),

                "meta_end":
                    self.profile.member_offset(
                        "maple_metadata",
                        "end",
                    ),
            }

        except (
            KernelProfileError,
            ProcessEnumerationError,
            ValueError,
        ) as exc:
            raise UnsupportedVmaLayout(
                "Cannot resolve Maple Tree "
                f"layout from BTF: {exc}"
            ) from exc

        # Enum values come from the target kernel's BTF rather
        # than being inferred from another build.
        try:
            try:
                enum_info = self.profile.btf_type(
                    "maple_type",
                    kind="ENUM",
                )
            except KernelProfileError:
                enum_info = self.profile.btf_type(
                    "maple_type",
                    kind="ENUM64",
                )

            enum_items = enum_info.get(
                "extra",
                [],
            )

            enum_values = {
                str(item.get("name")):
                    int(item.get("value"))
                for item in enum_items
                if (
                    isinstance(item, dict)
                    and item.get("name") is not None
                    and item.get("value") is not None
                )
            }

            for key, enum_name in (
                (
                    "type_dense",
                    "maple_dense",
                ),
                (
                    "type_leaf_64",
                    "maple_leaf_64",
                ),
                (
                    "type_range_64",
                    "maple_range_64",
                ),
                (
                    "type_arange_64",
                    "maple_arange_64",
                ),
            ):
                if enum_name not in enum_values:
                    raise UnsupportedVmaLayout(
                        "Target BTF maple_type "
                        f"enum lacks {enum_name}"
                    )

                layout[key] = enum_values[
                    enum_name
                ]

        except KernelProfileError as exc:
            raise UnsupportedVmaLayout(
                "Cannot resolve Maple Tree "
                "node type enum from BTF: "
                f"{exc}"
            ) from exc

        # --------------------------------------------------------------
        # maple_range_64
        # --------------------------------------------------------------

        if (
            layout["range_slot"]
            <= layout["range_pivot"]
        ):
            raise UnsupportedVmaLayout(
                "Invalid maple_range_64 "
                "pivot/slot layout"
            )

        if (
            (
                layout["range_slot"]
                - layout["range_pivot"]
            )
            % 8
        ):
            raise UnsupportedVmaLayout(
                "Unexpected maple_range_64 "
                "pivot layout"
            )

        if (
            (
                layout["range_size"]
                - layout["range_slot"]
            )
            % 8
        ):
            raise UnsupportedVmaLayout(
                "Unexpected maple_range_64 "
                "slot layout"
            )

        layout["range_pivot_count"] = (
            layout["range_slot"]
            - layout["range_pivot"]
        ) // 8

        layout["range_slot_count"] = (
            layout["range_size"]
            - layout["range_slot"]
        ) // 8

        if (
            layout["range_pivot_count"] + 1
            != layout["range_slot_count"]
        ):
            raise UnsupportedVmaLayout(
                "Maple range64 pivot/slot count is inconsistent"
            )

        # --------------------------------------------------------------
        # maple_arange_64
        # --------------------------------------------------------------

        if not (
            layout["arange_pivot"]
            < layout["arange_slot"]
            < layout["arange_gap"]
            < layout["arange_meta"]
        ):
            raise UnsupportedVmaLayout(
                "Invalid maple_arange_64 "
                "pivot/slot/gap/metadata layout"
            )

        if (
            (
                layout["arange_slot"]
                - layout["arange_pivot"]
            )
            % 8
        ):
            raise UnsupportedVmaLayout(
                "Unexpected maple_arange_64 "
                "pivot layout"
            )

        if (
            (
                layout["arange_gap"]
                - layout["arange_slot"]
            )
            % 8
        ):
            raise UnsupportedVmaLayout(
                "Unexpected maple_arange_64 "
                "slot layout"
            )

        if (
            (
                layout["arange_meta"]
                - layout["arange_gap"]
            )
            % 8
        ):
            raise UnsupportedVmaLayout(
                "Unexpected maple_arange_64 "
                "gap layout"
            )

        layout["arange_pivot_count"] = (
            layout["arange_slot"]
            - layout["arange_pivot"]
        ) // 8

        layout["arange_slot_count"] = (
            layout["arange_gap"]
            - layout["arange_slot"]
        ) // 8

        layout["arange_gap_count"] = (
            layout["arange_meta"]
            - layout["arange_gap"]
        ) // 8

        if (
            layout["arange_pivot_count"] + 1
            != layout["arange_slot_count"]
        ):
            raise UnsupportedVmaLayout(
                "Maple arange64 pivot/slot count is inconsistent"
            )


        # Dense node slots occupy the remainder of struct maple_node
        # starting at maple_node.slot.
        if (
            layout["node_size"]
            < layout["node_slot"]
        ):
            raise UnsupportedVmaLayout(
                "Invalid maple_node slot layout"
            )

        if (
            (
                layout["node_size"]
                - layout["node_slot"]
            )
            % 8
        ):
            raise UnsupportedVmaLayout(
                "Unexpected maple_node dense "
                "slot layout"
            )

        layout["dense_slot_count"] = (
            layout["node_size"]
            - layout["node_slot"]
        ) // 8

        if not (
            1
            <= layout["range_pivot_count"]
            <= 64
        ):
            raise UnsupportedVmaLayout(
                "Unsupported maple_range_64 "
                "pivot count"
            )

        if not (
            2
            <= layout["range_slot_count"]
            <= 64
        ):
            raise UnsupportedVmaLayout(
                "Unsupported maple_range_64 "
                "slot count"
            )

        if not (
            2
            <= layout["arange_pivot_count"]
            <= 64
        ):
            raise UnsupportedVmaLayout(
                "Unsupported maple_arange_64 "
                "pivot count"
            )

        if not (
            2
            <= layout["arange_slot_count"]
            <= 64
        ):
            raise UnsupportedVmaLayout(
                "Unsupported maple_arange_64 "
                "slot count"
            )

        if (
            layout["arange_gap_count"]
            != layout["arange_slot_count"]
        ):
            raise UnsupportedVmaLayout(
                "maple_arange_64 gap array "
                "count does not match slot count"
            )

        if not (
            2
            <= layout["dense_slot_count"]
            <= 64
        ):
            raise UnsupportedVmaLayout(
                "Unsupported maple_node "
                "dense slot count"
            )

        self._maple_layout_cache = layout

        return layout

    @staticmethod
    def _maple_node_type(
        encoded: int,
    ) -> int:
        return (
            encoded
            >> ProcessEnumerator._MAPLE_NODE_TYPE_SHIFT
        ) & 0xF

    def _maple_active_range_slots(
        self,
        node_address: int,
        layout: dict[str, int],
        *,
        node_max: int,
    ) -> int:
        """Return active slots in a Maple leaf/range node.
        
        Follows the Linux ma_data_end() logic.
        """
        base = (
            node_address
            + layout["node_mr64"]
        )

        last_pivot_index = (
           layout["range_pivot_count"] - 1
        )


        last_pivot_address = (
            base
            + layout["range_pivot"]
            + last_pivot_index * 8
        )

        last_pivot = self._read_uint(
            last_pivot_address,
            8,
        )

        if last_pivot == 0:
            end_address = (
                base
                + layout["range_meta"]
                + layout["meta_end"]
            )

            end_index = self._read_uint(
                end_address,
                1,
            )

            if end_index >= layout["range_slot_count"] - 1:
                raise CorruptKernelList(
                    "Invalid Maple Tree "
                    f"metadata.end={end_index} "
                    f"at node 0x{node_address:x}"
                )

        elif last_pivot > node_max:
            raise CorruptKernelList(
                "Maple Tree final pivot exceeds "
                f"node maximum at 0x{node_address:x}: "
                f"pivot=0x{last_pivot:x}, "
                f"maximum=0x{node_max:x}"
            )

        elif last_pivot == node_max:
            end_index = last_pivot_index

        else:
            end_index = layout["range_slot_count"] - 1

        return end_index + 1

    def _maple_child_max(
        self,
        node_address: int,
        node_type: int,
        slot_index: int,
        node_max: int,
        layout: dict[str, int],
    ) -> int:
        """Return the maximum index covered by a child node."""
        if node_type == layout["type_range_64"]:
            base = node_address + layout["node_mr64"]
            pivot_offset = layout["range_pivot"]
            pivot_count = layout["range_pivot_count"]

        elif node_type == layout["type_arange_64"]:
            base = node_address + layout["node_ma64"]
            pivot_offset = layout["arange_pivot"]
            pivot_count = layout["arange_pivot_count"]

        else:
            raise CorruptKernelList(
                f"Cannot determine child range for "
                f"Maple node type {node_type}"
            )

        if slot_index >= pivot_count:
            return node_max

        child_max = self._read_uint(
            base + pivot_offset + slot_index * 8,
            8,
        )

        if child_max > node_max:
            raise CorruptKernelList(
                "Maple Tree child pivot exceeds parent maximum: "
                f"child=0x{child_max:x}, "
                f"parent=0x{node_max:x}"
            )

        return child_max


    def _maple_active_arange_slots(
        self,
        node_address: int,
        layout: dict[str, int],
    ) -> int:
        """Return active slots for a Maple allocation-range node."""
        end_address = (
            node_address
            + layout["node_ma64"]
            + layout["arange_meta"]
            + layout["meta_end"]
        )

        end_index = self._read_uint(
            end_address,
            1,
        )

        if (
            end_index
            >= layout["arange_slot_count"]
        ):
            raise CorruptKernelList(
                "Invalid maple_arange_64 "
                f"metadata.end={end_index} "
                f"at node 0x{node_address:x}; "
                "slot count is "
                f"{layout['arange_slot_count']}"
            )

        return end_index + 1

    # ------------------------------------------------------------------
    # Maple Tree VMA walker
    # ------------------------------------------------------------------

    def _maple_vma_addresses(
        self,
        process: ProcessInfo,
    ) -> list[int]:
        """Walk ``mm_struct.mm_mt`` and return VMA pointers.

        Supports dense leaves, 64-bit leaf/range nodes and allocation-range
        internal nodes. Layout details come exclusively from target BTF.
        """
        layout = self._maple_layout()

        tree_address = (
            process.mm_address
            + layout["mm_mt"]
        )

        root_value = self._read_pointer(
            tree_address
            + layout["ma_root"]
        )

        if root_value == 0:
            return []

        vma_addresses: list[int] = []
        seen_vmas: set[int] = set()
        seen_nodes: set[int] = set()

        # (node_address, node_type, depth, node_max)
        stack: list[
            tuple[int, int, int, int]
        ] = []

        # A root node carries MAPLE_ROOT_NODE in its low bits.
        # A single-entry tree instead stores a raw VMA pointer.
        if (
            root_value & 0x3
        ) == self._MAPLE_ROOT_NODE:
            root_type = self._maple_node_type(
                root_value
            )

            root_address = (
                root_value
                & ~self._MAPLE_NODE_MASK
            )

            if (
                root_address == 0
                or root_address < 0x1000
            ):
                raise CorruptKernelList(
                    "Invalid Maple Tree root "
                    f"node pointer 0x{root_value:x}"
                )

            stack.append(
                (
                    root_address,
                    root_type,
                    0,
                    self._MAPLE_MAX_INDEX,
                )
            )

        else:
            if root_value & 0x7:
                raise CorruptKernelList(
                    "Unexpected tagged Maple "
                    f"Tree root entry "
                    f"0x{root_value:x}"
                )

            if root_value < 0x1000:
                raise CorruptKernelList(
                    "Invalid Maple Tree root "
                    f"VMA pointer 0x{root_value:x}"
                )

            vma_addresses.append(
                root_value
            )
            seen_vmas.add(
                root_value
            )

        while stack:
            (
                node_address,
                node_type,
                depth,
                node_max,
            ) = stack.pop()

            if depth > 32:
                raise CorruptKernelList(
                    "Maple Tree depth exceeds "
                    "32 levels"
                )

            if node_address in seen_nodes:
                raise CorruptKernelList(
                    "Maple Tree node repeated "
                    "or cyclic at "
                    f"0x{node_address:x}"
                )

            if (
                len(seen_nodes)
                >= self.max_vmas
            ):
                raise CorruptKernelList(
                    "Maple Tree exceeded "
                    f"max_vmas={self.max_vmas} "
                    "nodes"
                )

            seen_nodes.add(
                node_address
            )

            supported_types = (
                layout["type_dense"],
                layout["type_leaf_64"],
                layout["type_range_64"],
                layout["type_arange_64"],
            )

            if node_type not in supported_types:
                raise CorruptKernelList(
                    "Unknown Maple Tree "
                    f"node type {node_type} "
                    f"at 0x{node_address:x}"
                )

            if node_type == layout["type_dense"]:
                slot_base = (
                    node_address
                    + layout["node_slot"]
                )

                slot_count = layout["dense_slot_count"]

            elif node_type == layout["type_arange_64"]:
                slot_base = (
                    node_address
                    + layout["node_ma64"]
                    + layout["arange_slot"]
                )

                slot_count = self._maple_active_arange_slots(
                    node_address,
                    layout,
                )

            else:
                slot_base = (
                    node_address
                    + layout["node_mr64"]
                    + layout["range_slot"]
                )

                slot_count = self._maple_active_range_slots(
                    node_address,
                    layout,
                    node_max=node_max,
                )

            # Dense and leaf_64 nodes contain VMA pointers.
            # range_64 and arange_64 contain encoded child nodes.
            is_leaf = node_type in (
                layout["type_dense"],
                layout["type_leaf_64"],
            )

            for slot_index in range(
                slot_count
            ):
                slot_address = (
                    slot_base
                    + slot_index * 8
                )

                entry = self._read_pointer(
                    slot_address
                )

                # Linux Maple Tree reserves low-valued entries.
                # Null, retry, dead-node and other reserved values
                # must not be interpreted as pointers.
                if entry < 0x1000:
                    continue

                is_encoded_child = (
                    (entry & 0x7)
                    == self._MAPLE_ENODE_NULL
                )

                if is_leaf:
                    if is_encoded_child:
                        raise CorruptKernelList(
                            "Encoded child node in "
                            "Maple leaf slot at "
                            f"0x{slot_address:x}"
                        )

                    if entry & 0x7:
                        raise CorruptKernelList(
                            "Unaligned VMA pointer "
                            f"0x{entry:x} "
                            "in Maple Tree"
                        )

                    if entry in seen_vmas:
                        raise CorruptKernelList(
                            "VMA pointer "
                            f"0x{entry:x} occurs "
                            "more than once in "
                            "Maple Tree"
                        )

                    seen_vmas.add(
                        entry
                    )

                    vma_addresses.append(
                        entry
                    )

                    if (
                        len(vma_addresses)
                        > self.max_vmas
                    ):
                        raise CorruptKernelList(
                            "Maple Tree exceeded "
                            f"max_vmas={self.max_vmas} "
                            "entries"
                        )

                    continue

                # Internal-node slots must contain encoded
                # Maple child pointers.
                if not is_encoded_child:
                    raise CorruptKernelList(
                        "Non-node pointer "
                        f"0x{entry:x} in internal "
                        "Maple Tree slot "
                        f"{slot_index} of node "
                        f"0x{node_address:x}"
                    )

                child_type = (
                    self._maple_node_type(
                        entry
                    )
                )

                child_address = (
                    entry
                    & ~self._MAPLE_NODE_MASK
                )

                if (
                    child_address == 0
                    or child_address < 0x1000
                ):
                    raise CorruptKernelList(
                        "Invalid Maple child "
                        f"pointer 0x{entry:x}"
                    )

                child_max = self._maple_child_max(
                    node_address,
                    node_type,
                    slot_index,
                    node_max,
                    layout,
                )


                stack.append(
                    (
                        child_address,
                        child_type,
                        depth + 1,
                        child_max,
                    )
                )

        return vma_addresses

    # ------------------------------------------------------------------
    # VMA enumeration
    # ------------------------------------------------------------------

    def list_memory_regions(
        self,
        pid: int,
    ) -> list[MemoryRegion]:
        """Enumerate VMAs from either a classic list or Maple Tree.

        A VMA's end address is exclusive.

        ``flags`` is the raw target ``vm_flags`` bitmask and is deliberately
        not decoded into permissions here.
        """
        process = self.find_process(
            pid
        )

        if process.mm_address == 0:
            raise ProcessHasNoUserAddressSpace(
                f"PID {pid} ({process.comm}) "
                "has no user mm_struct"
            )

        mmap_offset = self._offset(
            "mm_struct",
            "mmap",
            required=False,
        )

        next_offset = self._offset(
            "vm_area_struct",
            "vm_next",
            required=False,
        )

        vma_addresses: list[int] = []

        # --------------------------------------------------------------
        # Classic pre-Maple Tree linked-list layout
        # --------------------------------------------------------------

        if (
            mmap_offset is not None
            and next_offset is not None
        ):
            vma_address = self._read_pointer(
                process.mm_address
                + int(mmap_offset)
            )

            seen: set[int] = set()

            while vma_address:
                if vma_address in seen:
                    raise CorruptKernelList(
                        f"VMA list for PID {pid} "
                        "repeats vm_area_struct "
                        f"at 0x{vma_address:x}"
                    )

                if (
                    len(vma_addresses)
                    >= self.max_vmas
                ):
                    raise CorruptKernelList(
                        f"VMA list for PID {pid} "
                        "exceeded "
                        f"max_vmas={self.max_vmas}"
                    )

                seen.add(
                    vma_address
                )

                vma_addresses.append(
                    vma_address
                )

                vma_address = self._read_pointer(
                    vma_address
                    + int(next_offset)
                )

        # --------------------------------------------------------------
        # Modern Maple Tree layout
        # --------------------------------------------------------------

        elif self.profile.has_member(
            "mm_struct",
            "mm_mt",
        ):
            vma_addresses = (
                self._maple_vma_addresses(
                    process
                )
            )

        else:
            raise UnsupportedVmaLayout(
                "BTF has neither "
                "mm_struct.mmap nor "
                "mm_struct.mm_mt"
            )

        # --------------------------------------------------------------
        # vm_area_struct fields
        # --------------------------------------------------------------

        start_offset = self._offset(
            "vm_area_struct",
            "vm_start",
        )

        end_offset = self._offset(
            "vm_area_struct",
            "vm_end",
        )

        flags_offset = self._offset(
            "vm_area_struct",
            "vm_flags",
            "__vm_flags",
            required=False,
        )

        pgoff_offset = self._offset(
            "vm_area_struct",
            "vm_pgoff",
            "vm_offset",
            required=False,
        )

        file_offset = self._offset(
            "vm_area_struct",
            "vm_file",
            required=False,
        )

        regions: list[
            MemoryRegion
        ] = []

        for vma_address in vma_addresses:
            start = self._read_uint(
                vma_address
                + int(start_offset),
                self.POINTER_SIZE,
            )

            end = self._read_uint(
                vma_address
                + int(end_offset),
                self.POINTER_SIZE,
            )

            if end <= start:
                raise CorruptKernelList(
                    "Invalid VMA range "
                    f"0x{start:x}-0x{end:x} "
                    f"at 0x{vma_address:x}"
                )

            flags = (
                self._read_uint(
                    vma_address
                    + int(flags_offset),
                    self.POINTER_SIZE,
                )
                if flags_offset is not None
                else None
            )

            page_offset = (
                self._read_uint(
                    vma_address
                    + int(pgoff_offset),
                    self.POINTER_SIZE,
                )
                if pgoff_offset is not None
                else None
            )

            file_pointer = (
                self._read_pointer(
                    vma_address
                    + int(file_offset)
                )
                if file_offset is not None
                else None
            )

            regions.append(
                MemoryRegion(
                    start=start,
                    end=end,
                    flags=flags,
                    page_offset=page_offset,
                    file_pointer=file_pointer,
                    vma_address=vma_address,
                )
            )

        return sorted(
            regions,
            key=lambda region: (
                region.start,
                region.end,
            ),
        )

    # ------------------------------------------------------------------
    # VMA reads
    # ------------------------------------------------------------------

    def read_memory_region(
        self,
        pid: int,
        region: MemoryRegion,
        *,
        offset: int = 0,
        size: Optional[int] = None,
    ) -> bytes:
        """Read a bounded slice of a previously enumerated MemoryRegion."""
        if (
            offset < 0
            or offset > region.size
        ):
            raise ValueError(
                "offset must be within the region"
            )

        read_size = (
            region.size - offset
            if size is None
            else size
        )

        if (
            read_size < 0
            or offset + read_size
            > region.size
        ):
            raise ValueError(
                "requested slice exceeds "
                "the memory region"
            )

        return self.read_process_memory(
            pid,
            region.start + offset,
            read_size,
        )


__all__ = [
    "CorruptKernelList",
    "MemoryRegion",
    "ProcessEnumerationError",
    "ProcessHasNoUserAddressSpace",
    "ProcessInfo",
    "ProcessNotFound",
    "ProcessEnumerator",
    "UnsupportedVmaLayout",
]