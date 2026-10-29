import json
import struct
import tempfile
import unittest
from pathlib import Path

from arm64_memory import ARM64Memory
from kernel_profile import KernelProfile
from lime_reader import LIME_MAGIC, LIME_VERSION, LimeReader


class EndToEndSyntheticCaptureTests(unittest.TestCase):
    def test_lime_profile_page_walk_and_read(self):
        # A deterministic physical-memory image with a 4 KiB ARM64 mapping:
        # VA 0x400000 -> PA 0x9000, and VA 0x401000 -> PA 0xA000.
        phys = bytearray(0xC000)
        struct.pack_into("<Q", phys, 0x1000, 0x2000 | 0b11)
        struct.pack_into("<Q", phys, 0x2000 + 2 * 8, 0x3000 | 0b11)
        struct.pack_into("<Q", phys, 0x3000, 0x9000 | 0b11)
        struct.pack_into("<Q", phys, 0x3000 + 8, 0xA000 | 0b11)
        phys[0x9000:0xA000] = bytes(i & 0xFF for i in range(4096))
        phys[0xA000:0xB000] = b"Q" * 4096

        header = struct.pack(
            "<IIQQ8s",
            LIME_MAGIC,
            LIME_VERSION,
            0,
            len(phys) - 1,
            b"\0" * 8,
        )

        profile_data = {
            "format": KernelProfile.FORMAT,
            "format_version": 2,
            "kernel": {
                "architecture": {"normalized": "arm64"},
                "page_size": 4096,
                "va_bits": 39,
                "pa_bits": 48,
            },
            "sources": {},
            "symbols": {"selected": {}},
            "btf": {"types": {}},
            "boot": {"kaslr_compiled": False, "nokaslr": True,
                     "kaslr": {"runtime_offset": None}},
            "validation": {"status": "valid", "issues": []},
        }

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "memory.lime").write_bytes(header + phys)
            (root / "profile.json").write_text(
                json.dumps(profile_data), encoding="utf-8"
            )

            profile = KernelProfile(root)
            with LimeReader(root / "memory.lime") as lime:
                memory = ARM64Memory(
                    profile,
                    lime,
                    kernel_pgd_paddr=0x1000,
                )
                result = memory.translate(0x400123, pgd_paddr=0x1000)
                self.assertEqual(result.physical_address, 0x9123)
                self.assertEqual(
                    memory.read_virtual(0x400FF8, 16, pgd_paddr=0x1000),
                    bytes(range(0xF8, 0x100)) + b"Q" * 8,
                )


if __name__ == "__main__":
    unittest.main()
