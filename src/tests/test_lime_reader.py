import random
import struct
import tempfile
import unittest
from pathlib import Path

from lime_reader import (
    HEADER_SIZE,
    LIME_MAGIC,
    LIME_VERSION,
    LimeFormatError,
    LimeReader,
    LimeReaderError,
    PhysicalAddressNotCaptured,
)


def make_range(endian, start, payload, *, end=None, version=LIME_VERSION, magic=LIME_MAGIC):
    if end is None:
        end = start + len(payload) - 1
    header = struct.pack(
        endian + "IIQQ8s",
        magic,
        version,
        start,
        end,
        b"\0" * 8,
    )
    assert len(header) == HEADER_SIZE
    return header + payload


class LimeReaderTests(unittest.TestCase):
    def create_dump(self, content: bytes) -> Path:
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.write(content)
        tmp.close()
        self.addCleanup(lambda: Path(tmp.name).unlink(missing_ok=True))
        return Path(tmp.name)

    def test_read_single_range_and_integer(self):
        payload = bytes(range(16))
        path = self.create_dump(make_range("<", 0x1000, payload))
        with LimeReader(path) as reader:
            self.assertEqual(reader.read_phys(0x1000, 16), payload)
            self.assertEqual(reader.read_u32(0x1000), 0x03020100)
            self.assertTrue(reader.contains_phys(0x1004, 4))
            self.assertFalse(reader.contains_phys(0x100F, 2))

    def test_read_across_adjacent_physical_ranges(self):
        first = make_range("<", 0x1000, b"abcd")
        second = make_range("<", 0x1004, b"EFGH")
        path = self.create_dump(first + second)
        with LimeReader(path) as reader:
            self.assertEqual(reader.read_phys(0x1002, 4), b"cdEF")

    def test_hole_is_not_zero_filled(self):
        first = make_range("<", 0x1000, b"abcd")
        second = make_range("<", 0x1005, b"WXYZ")
        path = self.create_dump(first + second)
        with LimeReader(path) as reader:
            with self.assertRaises(PhysicalAddressNotCaptured):
                reader.read_phys(0x1002, 5)
            self.assertFalse(reader.contains_phys(0x1002, 5))

    def test_big_endian_header_and_integer(self):
        path = self.create_dump(make_range(">", 0x2000, b"\x01\x02\x03\x04"))
        with LimeReader(path) as reader:
            self.assertEqual(reader.endianness, ">")
            self.assertEqual(reader.read_u32(0x2000), 0x01020304)

    def test_reject_small_file(self):
        path = self.create_dump(b"short")
        with self.assertRaises(LimeFormatError):
            LimeReader(path)

    def test_reject_bad_magic(self):
        content = make_range("<", 0x1000, b"abc", magic=0x12345678)
        with self.assertRaises(LimeFormatError):
            LimeReader(self.create_dump(content))

    def test_reject_bad_version(self):
        with self.assertRaises(LimeFormatError):
            LimeReader(self.create_dump(make_range("<", 0x1000, b"abc", version=9)))

    def test_reject_truncated_payload(self):
        raw = make_range("<", 0x1000, b"abcd")[:-1]
        with self.assertRaises(LimeFormatError):
            LimeReader(self.create_dump(raw))

    def test_reject_trailing_partial_header(self):
        raw = make_range("<", 0x1000, b"abcd") + b"xyz"
        with self.assertRaises(LimeFormatError):
            LimeReader(self.create_dump(raw))

    def test_reject_overlapping_ranges(self):
        raw = (
            make_range("<", 0x1000, b"abcd")
            + make_range("<", 0x1002, b"WXYZ")
        )
        with self.assertRaises(LimeFormatError):
            LimeReader(self.create_dump(raw))

    def test_empty_read(self):
        path = self.create_dump(make_range("<", 0x1000, b"abc"))
        with LimeReader(path) as reader:
            self.assertEqual(reader.read_phys(0xDEADBEEF, 0), b"")

    def test_read_after_close_fails(self):
        path = self.create_dump(make_range("<", 0x1000, b"abc"))
        reader = LimeReader(path)
        reader.close()
        with self.assertRaises(LimeReaderError):
            reader.read_phys(0x1000, 1)

    def test_read_last_byte_of_range(self):
        path = self.create_dump(make_range("<", 0x1000, b"abcd"))

        with LimeReader(path) as reader:
            self.assertEqual(reader.read_phys(0x1003, 1,), b"d")

    def test_zero_size_span_is_contained(self):
        path = self.create_dump(make_range("<", 0x1000, b"abcd"))

        with LimeReader(path) as reader:
            self.assertTrue(reader.contains_phys(0xDEADBEEF, 0))

    def test_negative_physical_address_rejected(self):
        path = self.create_dump(make_range("<", 0x1000, b"abcd"))

        with LimeReader(path) as reader:
            with self.assertRaises(ValueError):
                reader.read_phys(-1, 1)

    def test_deterministic_random_reads_match_source_payload(self):
        rng = random.Random(20261029)
        physical_start = 0x10000
        payload = bytes(rng.randrange(256) for _ in range(8192))
        path = self.create_dump(make_range("<", physical_start, payload))
        with LimeReader(path) as reader:
            for _ in range(100):
                offset = rng.randrange(0, len(payload))
                size = rng.randrange(0, len(payload) - offset + 1)
                self.assertEqual(
                    reader.read_phys(physical_start + offset, size),
                    payload[offset:offset + size],
                )


if __name__ == "__main__":
    unittest.main()
