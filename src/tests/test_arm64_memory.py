import struct
import unittest

from arm64_memory import (
    ARM64Memory,
    NonCanonicalVirtualAddress,
    UnmappedVirtualAddress,
    UnsupportedTranslationGeometry,
    PageTableGeometry,
)


class FakeProfile:
    architecture = "arm64"
    page_size = 4096
    va_bits = 39
    pa_bits = 48


class FakeLime:
    endianness = "<"

    def __init__(self, size=0x20000):
        self.memory = bytearray(size)

    def write_u64(self, paddr, value):
        struct.pack_into("<Q", self.memory, paddr, value)

    def read_phys(self, paddr, size):
        if paddr < 0 or size < 0 or paddr + size > len(self.memory):
            raise ValueError("outside fake captured memory")
        return bytes(self.memory[paddr:paddr + size])

    def contains_phys(self, paddr, size=1):
        return paddr >= 0 and size >= 0 and paddr + size <= len(self.memory)


def setup_4k_page_tables(lime):
    # VA 0x400000 maps through three levels: root 0x1000, L2 0x2000,
    # L3 0x3000. At L2 index 2, two adjacent virtual pages map to PAs
    # 0x9000 and 0xA000.
    lime.write_u64(0x1000, 0x2000 | 0b11)
    lime.write_u64(0x2000 + 2 * 8, 0x3000 | 0b11)
    lime.write_u64(0x3000 + 0 * 8, 0x9000 | 0b11)
    lime.write_u64(0x3000 + 1 * 8, 0xA000 | 0b11)
    lime.memory[0x9000:0xA000] = bytes(i & 0xFF for i in range(4096))
    lime.memory[0xA000:0xB000] = b"Z" * 4096


class PageTableGeometryTests(unittest.TestCase):
    def test_supported_granules_create_geometry(self):
        cases = (
            (4096, 39, 48),
            (4096, 48, 48),
            (16384, 48, 48),
            (65536, 48, 48),
        )
        for page_size, va_bits, pa_bits in cases:
            with self.subTest(page_size=page_size, va_bits=va_bits):
                geom = PageTableGeometry.create(
                    page_size=page_size,
                    va_bits=va_bits,
                    pa_bits=pa_bits,
                )
                self.assertEqual(geom.page_size, page_size)
                self.assertGreaterEqual(geom.levels, 1)
                self.assertLessEqual(geom.levels, 4)

    def test_known_geometry_level_counts(self):
        expected = {
            (4096, 39): (3, 1),
            (4096, 48): (4, 0),
            (16384, 48): (4, 0),
            (65536, 48): (3, 1),
        }
        for (page_size, va_bits), (levels, start_level) in expected.items():
            with self.subTest(page_size=page_size, va_bits=va_bits):
                geom = PageTableGeometry.create(
                    page_size=page_size,
                    va_bits=va_bits,
                    pa_bits=48,
                )
                self.assertEqual(geom.levels, levels)
                self.assertEqual(geom.start_level, start_level)

    def test_unknown_or_unsupported_geometry_fails(self):
        for kwargs in (
            dict(page_size=None, va_bits=39, pa_bits=48),
            dict(page_size=4096, va_bits=None, pa_bits=48),
            dict(page_size=4096, va_bits=39, pa_bits=None),
            dict(page_size=4096, va_bits=52, pa_bits=48),
            dict(page_size=4096, va_bits=48, pa_bits=52),
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(UnsupportedTranslationGeometry):
                    PageTableGeometry.create(**kwargs)


class ARM64TranslationTests(unittest.TestCase):
    def setUp(self):
        self.lime = FakeLime()
        setup_4k_page_tables(self.lime)
        self.memory = ARM64Memory(
            FakeProfile(),
            self.lime,
            kernel_pgd_paddr=0x1000,
        )

    def test_physical_address_to_pfn(self):
        result = self.memory.translate(0x400123, pgd_paddr=0x1000)

        page_shift = self.memory.geometry.page_shift

        pfn = (result.physical_address >> page_shift)

        self.assertEqual(page_shift, 12)
        self.assertEqual(pfn, 0x9)

    def test_translate_user_page(self):
        result = self.memory.translate(0x400123, pgd_paddr=0x1000)

        self.assertEqual(result.physical_address, 0x9123)
        self.assertEqual(result.mapping_size, 4096)
        self.assertEqual(result.level, 3)
        self.assertFalse(result.is_block)

    def test_translation_result_metadata(self):
        result = self.memory.translate(0x400123, pgd_paddr=0x1000)

        self.assertEqual(result.virtual_address, 0x400123)

        self.assertEqual(result.physical_address, 0x9123)

        self.assertEqual(result.mapping_size, 4096)

        self.assertEqual(result.level, 3)

        self.assertFalse(result.is_block)

        self.assertEqual(result.table_paddr, 0x3000)

        self.assertEqual(result.descriptor_paddr, 0x3000)

    def test_read_crossing_virtual_page_boundary(self):
        actual = self.memory.read_virtual(
            0x400FF8,
            16,
            pgd_paddr=0x1000,
        )
        self.assertEqual(actual[:8], bytes(range(0xF8, 0x100)))
        self.assertEqual(actual[8:], b"Z" * 8)

    def test_unmapped_descriptor_is_explicit(self):
        empty_lime = FakeLime()
        memory = ARM64Memory(
            FakeProfile(),
            empty_lime,
            kernel_pgd_paddr=0x1000,
        )
        with self.assertRaises(UnmappedVirtualAddress):
            memory.translate(0x400000, pgd_paddr=0x1000)

    def test_noncanonical_address_is_explicit(self):
        with self.assertRaises(NonCanonicalVirtualAddress):
            self.memory.translate(1 << 40, pgd_paddr=0x1000)

    def test_user_top_byte_tag_is_removed_without_sign_extending(self):
        tagged = 0xAB00000000400123
        normalized, index = self.memory._normalize_virtual_address(
            tagged,
            True,
        )
        self.assertEqual(normalized, 0x400123)
        self.assertEqual(index, 0x400123)

    def test_tagged_address_with_bits_above_va_width_still_fails(self):
        # Stripping a top-byte tag must not turn an invalid lower-VA pointer
        # into a canonical kernel pointer by sign extension.
        tagged = 0xAB00800000400123
        with self.assertRaises(NonCanonicalVirtualAddress):
            self.memory._normalize_virtual_address(tagged, True)

    def test_block_mapping(self):
        lime = FakeLime()
        # With 39-bit VA and 4 KiB granules, depth 0 starts at architectural
        # level 1, where a 1 GiB block descriptor is allowed.
        lime.write_u64(0x1000 + 8, 0x40000000 | 0b01)
        memory = ARM64Memory(
            FakeProfile(),
            lime,
            kernel_pgd_paddr=0x1000,
        )
        result = memory.translate(0x40001234, pgd_paddr=0x1000)
        self.assertEqual(result.level, 1)
        self.assertEqual(result.physical_address, 0x40001234)
        self.assertTrue(result.is_block)
        self.assertEqual(result.mapping_size, 1 << 30)

    def test_level_2_block_mapping(self):
        lime = FakeLime()

        # L1 table -> L2 table.
        lime.write_u64(0x1000, 0x2000 | 0b11)

        # VA 0x00200000 uses L2 index 1.
        # Map it to PA 0x02000000 using an L2 block.
        lime.write_u64(0x2000 + 1 * 8, 0x02000000 | 0b01)

        memory = ARM64Memory(
            FakeProfile(),
            lime,
            kernel_pgd_paddr=0x1000,
        )

        result = memory.translate(0x00201234, pgd_paddr=0x1000)

        self.assertEqual(result.physical_address,0x02001234)
        self.assertEqual(result.level, 2)
        self.assertEqual(result.mapping_size, 2 * 1024 * 1024)
        self.assertTrue(result.is_block)

    def test_level_2_block_pfn(self):
        lime = FakeLime()

        lime.write_u64(0x1000, 0x2000 | 0b11)

        lime.write_u64(0x2000 + 1 * 8, 0x02000000 | 0b01)

        memory = ARM64Memory(
            FakeProfile(),
            lime,
            kernel_pgd_paddr=0x1000,
        )

        result = memory.translate(0x00201234, pgd_paddr=0x1000)

        self.assertEqual(result.physical_address >> 12, 0x2001)

    def test_level_2_block_preserves_offset(self):
        lime = FakeLime()

        lime.write_u64(0x1000, 0x2000 | 0b11)

        lime.write_u64(0x2000 + 1 * 8, 0x02000000 | 0b01)

        memory = ARM64Memory(
            FakeProfile(),
            lime,
            kernel_pgd_paddr=0x1000,
        )

        result = memory.translate(0x002ABCDE, pgd_paddr=0x1000)

        self.assertEqual(result.physical_address, 0x020ABCDE)

    def test_page_walks_for_all_supported_granules(self):
        for page_size, va_bits in (
            (4096, 39),
            (4096, 48),
            (16384, 48),
            (65536, 48),
        ):
            with self.subTest(page_size=page_size, va_bits=va_bits):
                geom = PageTableGeometry.create(
                    page_size=page_size,
                    va_bits=va_bits,
                    pa_bits=48,
                )
                lime = FakeLime(size=0x100000)
                root = 0x10000
                target_page = 0x80000
                va = 0x12345678
                va_index = va & ((1 << va_bits) - 1)
                table = root

                for depth in range(geom.levels):
                    index = geom.index_for(va_index, depth)
                    slot = table + index * 8
                    if depth == geom.levels - 1:
                        lime.write_u64(slot, target_page | 0b11)
                    else:
                        next_table = root + page_size * (depth + 1)
                        lime.write_u64(slot, next_table | 0b11)
                        table = next_table

                profile = type(
                    "GranuleProfile",
                    (),
                    {
                        "architecture": "arm64",
                        "page_size": page_size,
                        "va_bits": va_bits,
                        "pa_bits": 48,
                    },
                )()
                memory = ARM64Memory(
                    profile,
                    lime,
                    kernel_pgd_paddr=root,
                )
                result = memory.translate(va, pgd_paddr=root)
                expected_pa = target_page + (va & (page_size - 1))
                self.assertEqual(result.physical_address, expected_pa)
                self.assertEqual(result.mapping_size, page_size)
                self.assertFalse(result.is_block)


if __name__ == "__main__":
    unittest.main()
