import unittest

from process_enumerator import ProcessEnumerator


class FakeProcessMemory:
    class Geometry:
        page_shift = 12
        page_size = 4096

    geometry = Geometry()

    def translate(
        self,
        address,
        *,
        pgd_paddr=None,
    ):
        self.last_address = address
        self.last_pgd = pgd_paddr

        return type(
            "Result",
            (),
            {
                "virtual_address": address,
                "physical_address": 0x9000,
                "mapping_size": 4096,
                "level": 3,
                "is_block": False,
            },
        )()


class ProcessTranslationTests(unittest.TestCase):
    def test_process_translation_uses_process_pgd(self):
        memory = FakeProcessMemory()

        enumerator = object.__new__(
            ProcessEnumerator
        )

        enumerator.memory = memory

        enumerator.process_pgd_paddr = (
            lambda pid: 0x123000
        )

        result = (
            enumerator.translate_process_address(
                1234,
                0x400000,
            )
        )

        self.assertEqual(
            memory.last_pgd,
            0x123000,
        )

        self.assertEqual(
            memory.last_address,
            0x400000,
        )

        self.assertEqual(
            result.physical_address,
            0x9000,
        )