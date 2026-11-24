"""Check compatibility with the unchanged shipped operator table after adding the new Yukawas."""
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from tex_inputs import TexInputError, read_inputs


class BenchmarkBindings(unittest.TestCase):
    def setUp(self):
        self.original = read_inputs(ROOT / 'input/sample_model_sextet.txt')

    def test_sample_still_builds_and_tampered_binding_is_rejected(self):
        self.original.build()
        self.original.assignments['_input_bindings']['coefficients']['Q^2 X†'] = '0' * 64
        with self.assertRaisesRegex(TexInputError, 'stale active operator bindings'):
            self.original.build()

    def test_all_paper_inputs_build_with_their_published_companions(self):
        from pipeline_config import load_config
        benchmark_dir = ROOT / 'input'
        for case in ('phantom_portal', 'inert_doublet', 'brw_leptoquark',
                     'hlm_diquark', 'mw_colour_octet'):
            with self.subTest(case=case):
                path = benchmark_dir / f'{case}_model.txt'
                read_inputs(path).build()
                self.assertEqual(load_config(model_file=path).output, ROOT / 'output')

    def test_unknown_compiler_revision_requires_review(self):
        with patch('tex_inputs._compiler_fingerprint', return_value='1' * 64):
            with self.assertRaisesRegex(TexInputError, 'stale active operator bindings'):
                self.original.build()
