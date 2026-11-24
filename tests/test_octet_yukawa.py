"""Check an off-diagonal octet Yukawa, which the diagonal width benchmark cannot probe."""
from fractions import Fraction
from types import SimpleNamespace
import unittest

from generic_colour_yukawa import basis_labels, classify, render
from generic_octet_yukawa import KIND


class OctetYukawaTests(unittest.TestCase):
    def test_complex_right_charm_left_top_entry(self):
        scalar = SimpleNamespace(name='Y', spin='scalar', kind='complex', su3=(1, 1),
                                 su2=2, hypercharge=Fraction(1, 2))
        factors = [('Q', 1), ('Y', 1), ('uc', 1)]
        row = {'derivatives': 0, 'multiplicity': 9,
               'factors': [dict(name=name, count=count) for name, count in factors]}
        self.assertEqual(classify(row, [scalar]), KIND)
        self.assertEqual(basis_labels(row, [scalar])[5], 'Y_u8[2,3]')
        coefficients = [0.] * 9
        coefficients[5] = .3 + .4j
        operator = SimpleNamespace(kind=KIND, factors=factors, coefficients=coefficients,
                                   operator_id='O0007')
        symbols = {('Y', 0): 'Splus[{index}]', ('Y', 1): '(SR[{index}]+I*SI[{index}])/Sqrt[2]'}
        [parameter], [term] = render(operator, symbols)
        self.assertIn('ComplexParameter -> True', parameter)
        self.assertIn('HC[pmCYO0007n6*', term)
        self.assertIn('T[c3,c1,c2]*ProjM[s1,s2]', term)
        self.assertIn('((SR[c3]+I*SI[c3])/Sqrt[2])*uqbar[s1,2,c1].uq[s2,3,c2]', term)
        self.assertIn('Splus[c3]', term)
        for k in range(1, 4):
            self.assertIn(f'CKM[3,{k}]*uqbar[s1,2,c1].dq[s2,{k},c2]', term)
        self.assertNotIn('CKM[2,', term)
