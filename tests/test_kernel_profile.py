import json
import tempfile
import unittest
from pathlib import Path

from kernel_profile import (
    KernelProfile,
    KernelProfileFormatError,
    KernelProfileLookupError,
)


def sample_profile(version=2):
    return {
        "format": KernelProfile.FORMAT,
        "format_version": version,
        "kernel": {
            "release": "test-kernel",
            "architecture": {"normalized": "arm64"},
            "page_size": 4096,
            "va_bits": 39,
            "pa_bits": 48,
        },
        "sources": {},
        "symbols": {
            "selected": {
                "test_symbol": {
                    "kallsyms": [{"address": "0xffff000000001000", "type": "T"}]
                },
                "link_symbol": {
                    "system_map": [{"address": "0x1000", "type": "T"}]
                },
            },
        },
        "btf": {
            "types": {
                "outer": [{
                    "type_id": 1,
                    "name": "outer",
                    "kind": "STRUCT",
                    "size": 16,
                    "members": {
                        "__anon_member_0": {
                            "member_name": "",
                            "type_id": 2,
                            "resolved_type_id": 2,
                            "kind": "STRUCT",
                            "name": None,
                            "offset": 4,
                            "bit_offset": 32,
                            "bitfield_size": 0,
                            "modifiers": [],
                        }
                    },
                }],
                "__anon_type_2": [{
                    "type_id": 2,
                    "name": None,
                    "kind": "STRUCT",
                    "size": 8,
                    "members": {
                        "pid": {
                            "member_name": "pid",
                            "type_id": 3,
                            "resolved_type_id": 4,
                            "kind": "INT",
                            "name": "int",
                            "offset": 0,
                            "bit_offset": 0,
                            "bitfield_size": 0,
                            "modifiers": [],
                        }
                    },
                }],
            },
        },
        "boot": {
            "kaslr_compiled": False,
            "nokaslr": True,
            "kaslr": {"runtime_offset": None},
        },
        "validation": {"status": "valid", "issues": []},
    }


class KernelProfileTests(unittest.TestCase):
    def open_profile(self, data):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "profile.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return KernelProfile(tmp.name)

    def test_rejects_boolean_profile_version(self):
        with self.assertRaises(KernelProfileFormatError):
            self.open_profile(sample_profile(True))

    def test_symbol_runtime_resolution_uses_runtime_symbol(self):
        profile = self.open_profile(sample_profile())
        self.assertEqual(
            profile.symbol_runtime_address("test_symbol"),
            0xFFFF000000001000,
        )

    def test_system_map_address_allowed_when_kaslr_disabled(self):
        profile = self.open_profile(sample_profile())
        self.assertEqual(
            profile.symbol_runtime_address("link_symbol"),
            0x1000,
        )

    def test_anonymous_aggregate_member_offset(self):
        profile = self.open_profile(sample_profile())
        self.assertEqual(profile.member_offset("outer", "pid"), 4)
        self.assertEqual(profile.member_bit_offset("outer", "pid"), 32)

    def test_missing_member_fails_explicitly(self):
        profile = self.open_profile(sample_profile())
        with self.assertRaises(KernelProfileLookupError):
            profile.member_offset("outer", "does_not_exist")


if __name__ == "__main__":
    unittest.main()
