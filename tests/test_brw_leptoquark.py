"""Regression checks for the BRW singlet's flavour basis and exact gauge charge."""
from pathlib import Path
import tempfile
import unittest

from feynrules_export import prepare_model
from model_builder import ModelDefinition

ROOT = Path(__file__).resolve().parents[1]
TRIPLET = dict(spin='scalar', su2=1, su3='3', hypercharge='-1/3', boson_kind='complex')


class BrwLeptoquarkTests(unittest.TestCase):
    def test_left_handed_operator_and_fractional_charge_in_feynrules(self):
        couplings = [0.] * 9
        couplings[5] = 0.31 + 0.47j  # Q2 L3, using the up-diagonal quark basis.
        coefficients = {'X X†': -200.**2, 'D(X) D(X†)': 1., 'L Q X†': couplings}
        definition = ModelDefinition(1, TRIPLET, selected_operators=list(coefficients))
        row = next(r for r in definition.operator_inputs() if r['key'] == 'L Q X†')
        self.assertEqual(row['kind'], 'triplet_lq_yukawa')
        self.assertEqual(row['invariant_labels'][5], 'lambda_L[2,3]')
        inputs = definition.required_inputs()
        inputs['coefficients'] = coefficients
        model = prepare_model(*definition.build(inputs))
        with tempfile.TemporaryDirectory() as name:
            files, _, _ = model.compiler.write_sources(model, Path(name), ROOT/'vendor'/'FeynRules')
            source = files[-1].read_text()
        self.assertIn('Q -> -1/3, LeptonNumber -> 1', source)
        self.assertIn('CC[uqbar][s1,2,c1].l[s2,3]', source)
        self.assertIn('-(CKM[2,1]', source)
        for k in range(1, 4):
            self.assertIn(f'CKM[2,{k}]*CC[dqbar][s1,{k},c1].vl[s2,3]', source)
        parameter = f'pmCYO{row["operator_id"][1:]}n6'
        self.assertIn('HC[' + parameter + '*', source)
        self.assertIn('ComplexParameter -> True', source)

    def test_unimplemented_diquark_still_has_an_obstruction(self):
        definition = ModelDefinition(1, TRIPLET, selected_operators=['Q^2 X'])
        self.assertIn('Q^2 X', definition.required_inputs()['unresolved'])


if __name__ == '__main__':
    unittest.main()
