"""Recheck the triplet–sextet compiler with Python and the installed packages.

From Madgraph_step: python output/pdf_benchmark/check_compiler.py -v
These tests only generate source; the saved MadWidth and production runs
provide independent numerical checks of the exported UFO.
"""
from itertools import combinations_with_replacement, product
from pathlib import Path
import re
import sys
import tempfile
import unittest

import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))

from feynrules_export import ModelResolutionError, ResolvedOperator, prepare_model
from generic_colour_yukawa import lepton_labels, render
from model_builder import ModelDefinition
from model_initialization import initialize_model
from pipeline import Pipeline


SEXTET = dict(spin='scalar', su2=3, su3='6', hypercharge='1/3', boson_kind='complex')
TRIPLET = dict(spin='scalar', su2=1, su3='3', hypercharge='-1/3', boson_kind='complex')
QQ_COUPLING = 0.31 + 0.47j
EU_COUPLING = -0.21 + 0.13j


def single_entry(size, coupling):
    entries = [0.] * size
    entries[1] = coupling
    return entries


def prepare(fields, interactions):
    values = dict(interactions)
    for name, particle in zip(('X', 'Y'), fields):
        mass = 1500 if particle == SEXTET else 1200
        values[f'{name} {name}†'] = -(mass ** 2)
        values[f'D({name}) D({name}†)'] = 1.
    definition = ModelDefinition(len(fields), *fields, selected_operators=list(values))
    inputs = definition.required_inputs()
    inputs['coefficients'] = values
    return definition, prepare_model(*definition.build(inputs))


def sources(model):
    with tempfile.TemporaryDirectory() as folder:
        files, _, states = model.compiler.write_sources(
            model, Path(folder), PACKAGE / 'vendor' / 'FeynRules')
        return files[-1].read_text(), states


def scalar_declaration(text, particle_name):
    classes = re.split(r'\bS\[\d+\]\s*==\s*\{', text)
    name = re.compile(r'\bParticleName\s*->\s*"' + re.escape(particle_name) + r'"')
    return next(section for section in classes if name.search(section))


class ColourYukawaTests(unittest.TestCase):
    def test_families_alone_together_and_with_field_names_exchanged(self):
        for fields in ((SEXTET,), (TRIPLET,), (SEXTET, TRIPLET), (TRIPLET, SEXTET)):
            with self.subTest(fields=fields):
                values = {}
                for name, particle in zip(('X', 'Y'), fields):
                    if particle == SEXTET:
                        values[f'Q^2 {name}†'] = single_entry(6, QQ_COUPLING)
                    else:
                        values[f'{name} ec uc'] = single_entry(9, EU_COUPLING)
                definition, model = prepare(fields, values)
                self.assertFalse(definition.required_inputs()['unresolved'])
                descriptions = {row['key']: row for row in definition.operator_inputs()}
                operators = {definition.coefficient_key(op.operator_id): op for op in model.operators}
                for key, coefficients in values.items():
                    row = descriptions[key]
                    operator = operators[key]
                    expected_kind = 'sextet_qq_yukawa' if 'Q^2' in key else 'triplet_eu_yukawa'
                    self.assertEqual(row['kind'], expected_kind)
                    self.assertEqual(len(row['basis_fingerprint']), 64)
                    self.assertTrue(row['add_hc'])
                    self.assertFalse(row['real'])
                    self.assertEqual(list(operator.coefficients), coefficients)
                    self.assertTrue(operator.add_hc)
                    self.assertNotIn('zero_verified', model.coverage[operator.operator_id])

                text, states = sources(model)
                self.assertEqual('K6[' in text, SEXTET in fields)
                for state in states:
                    source = fields[('X', 'Y').index(state['source'])]
                    self.assertEqual(state['mass'], 1500. if source == SEXTET else 1200.)
                    if source == TRIPLET:
                        self.assertRegex(scalar_declaration(text, state['name']), r'LeptonNumber\s*->\s*1\b')

    def test_native_colour_mixing_transforms_lepton_numbers_and_fermion_vertices(self):
        _, model = prepare((TRIPLET, TRIPLET), {
            'X ec uc': single_entry(9, EU_COUPLING),
            'X Y†': -95000.,
        })
        self.assertEqual(type(model.compiler).__name__, 'GenericColorCompiler')
        text, states = sources(model)
        self.assertEqual(len(states), 2)
        couplings = next(op for op in model.operators if op.kind == 'triplet_eu_yukawa')
        symbol = f'pmCYO{couplings.operator_id[1:]}n2'
        yukawa_term = text[text.rindex(symbol + '*('):]
        for state in states:
            declaration = scalar_declaration(text, state['name'])
            self.assertRegex(declaration, r'LeptonNumber\s*->\s*1\b')
            class_name = re.search(r'ClassName\s*->\s*(\w+)', declaration).group(1)
            self.assertIn(class_name + '[', yukawa_term)

    def test_unrelated_contractions_keep_their_diagnostics(self):
        for label in ('Q^2 X', 'L Q X†'):
            with self.subTest(label=label):
                definition = ModelDefinition(1, TRIPLET, selected_operators=[label])
                self.assertTrue(definition.operator_inputs()[0]['obstruction'])
                self.assertIn(label, definition.required_inputs()['unresolved'])

    def test_conjugating_a_mass_eigenstate_reverses_the_lepton_label(self):
        _, model = prepare((TRIPLET,), {'X ec uc': single_entry(9, EU_COUPLING)})
        operator = next(op for op in model.operators if op.kind == 'triplet_eu_yukawa')
        scalar_maps = {('X', 0): {(0, False): .6, (1, True): .8j}}
        labels, note = lepton_labels([operator], scalar_maps)
        self.assertEqual(labels, {0: 1, 1: -1})
        self.assertNotIn('violate', note)
        breaking = ResolvedOperator('test_quartic', 'colour_scalar_potential',
                                    (('X', 2), ('Y', 2)), (0.02,))
        _, note = lepton_labels([operator, breaking], scalar_maps)
        self.assertIn('test_quartic', note)

    def test_invalid_arrays_and_complex_scalar_entries(self):
        for particle, label, size in ((TRIPLET, 'X ec uc', 9), (SEXTET, 'Q^2 X†', 6)):
            with self.subTest(label=label):
                definition = ModelDefinition(1, particle, selected_operators=[label])
                inputs = definition.required_inputs()
                for invalid in ([0.] * (size - 1), [0.] * (size - 1) + [float('nan')]):
                    inputs['coefficients'][label] = invalid
                    with self.assertRaises(ModelResolutionError):
                        definition.build(inputs)
                inputs['coefficients'][label] = [0.] * (size - 1) + [0.45j]
                _, built = definition.build(inputs)
                self.assertEqual(next(iter(built['coefficients'].values()))[-1], 0.45j)

    def test_file_edit_invalidates_ufo_and_widths_only_for_an_active_coefficient(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            particles = directory / 'particles.txt'
            particles.write_text('X = ' + repr(TRIPLET) + '\n')
            path = initialize_model(particles).path
            starter = path.read_text()
            line = next(line for line in starter.splitlines() if "'X ec uc':" in line)
            matrix = single_entry(9, EU_COUPLING)
            active = starter.replace(line, f"'X ec uc': {{'enabled': True, 'coefficient': {matrix!r}}},")
            path.write_text(active)
            project = Pipeline(model_file=path, output=directory / 'output')
            original = project.run_directory
            keys = {stage: project._key(stage) for stage in ('export', 'decays')}
            project._ufo = object()
            project._decays = object()
            edited = active.replace(repr(EU_COUPLING), repr(EU_COUPLING.conjugate()))
            path.write_text(edited)
            project.model()
            self.assertNotEqual(original, project.run_directory)
            self.assertIsNone(project._ufo)
            self.assertIsNone(project._decays)
            for stage in keys:
                self.assertNotEqual(project._key(stage), keys[stage])

            disabled = edited.replace("'X ec uc': {'enabled': True", "'X ec uc': {'enabled': False")
            path.write_text(disabled)
            project.model()
            original = project.run_directory
            path.write_text(disabled.replace(repr(EU_COUPLING.conjugate()), repr(2 * EU_COUPLING)))
            project.model()
            self.assertEqual(original, project.run_directory)

    def check_hermitian_term(self, declaration, term):
        self.assertIn('ComplexParameter -> True', declaration)
        normal, conjugate = term[1:-1].split('+HC[')
        self.assertEqual(conjugate, normal + ']')
        return normal

    def test_sextet_mass_basis_ckm_and_triplet_generators(self):
        # A complex unitary CKM detects transpositions and conjugations that
        # agreement at the real diagonal benchmark alone cannot distinguish.
        generations = np.arange(3)
        ckm = np.exp(2j * np.pi * np.outer(generations, generations) / 3) / np.sqrt(3)
        entries = [0.2+0.1j, 0.31+0.47j, 0.11-0.15j, 0.78j, -0.71j, -0.4+0.3j]
        definition, model = prepare((SEXTET,), {'Q^2 X†': entries})
        op = next(op for op in model.operators if op.kind == 'sextet_qq_yukawa')
        row = next(row for row in definition.operator_inputs() if row['kind'] == op.kind)
        expected_labels = [f'Y_QQ[{i},{j}]' for i, j in combinations_with_replacement(range(1, 4), 2)]
        self.assertEqual(list(row['invariant_labels']), expected_labels)
        params, terms = render(op, {('X†', i): f'W{i}[{{index}}]' for i in range(3)})
        monomial = re.compile(
            r'(Sqrt\[2\]|1)\*((?:CKM\[[1-3],[1-3]\]\*)*)W([0-2])\[c3\]'
            r'\*ProjM\[s1,s2\]\*CC\[(uq|dq)bar\]\[s1,([1-3]),c1\]\.(uq|dq)\[s2,([1-3]),c2\]')
        matrices = np.zeros((3, 6, 6), complex)
        for param, term, coupling in zip(params, terms, entries, strict=True):
            normal = self.check_hermitian_term(param, term)
            factor = re.search(r'\*\((1/2|1)\*K6\[c3,c1,c2\]\*\(', normal).group(1)
            symmetry = .5 if factor == '1/2' else 1.
            pieces = monomial.findall(normal)
            self.assertEqual(len(pieces), 16)
            for weak, factors, state, left, i, right, j in pieces:
                amplitude = symmetry * coupling * (np.sqrt(2) if weak == 'Sqrt[2]' else 1.)
                for a, b in re.findall(r'CKM\[([1-3]),([1-3])\]', factors):
                    amplitude *= ckm[int(a)-1, int(b)-1]
                left_index = (0 if left == 'uq' else 3) + int(i) - 1
                right_index = (0 if right == 'uq' else 3) + int(j) - 1
                matrices[int(state), left_index, right_index] += amplitude
        vertices = matrices + np.transpose(matrices, (0, 2, 1))
        yukawa = np.zeros((3, 3), complex)
        yukawa[np.triu_indices(3)] = entries
        yukawa += np.triu(yukawa, 1).T
        expected = np.zeros_like(vertices)
        expected[0, :3, :3] = np.sqrt(2) * yukawa
        expected[1, :3, 3:] = yukawa @ ckm
        expected[1, 3:, :3] = ckm.T @ yukawa
        expected[2, 3:, 3:] = np.sqrt(2) * ckm.T @ yukawa @ ckm
        np.testing.assert_allclose(vertices, expected, atol=1e-13, rtol=1e-13)

        # Return to an unmixed weak doublet and check both ladder generators.
        rotation = np.eye(6, dtype=complex)
        rotation[3:, 3:] = ckm.conj().T
        weak = np.stack([(rotation.T @ vertex @ rotation)[np.ix_([0, 3], [0, 3])]
                         for vertex in vertices])
        clebsch = weak.reshape(3, 4) / yukawa[0, 0]
        plus = np.array([[0., 1.], [0., 0.]])
        spin_one = np.diag([np.sqrt(2), np.sqrt(2)], 1)
        for fermion, scalar in ((plus, spin_one), (plus.T, spin_one.T)):
            generator = np.kron(np.eye(2), fermion) + np.kron(fermion, np.eye(2))
            np.testing.assert_allclose(clebsch @ generator, scalar @ clebsch, atol=1e-13)

    def test_leptoquark_all_nine_complex_flavours_with_hermitian_conjugates(self):
        entries = [k + .1j * (k + 2) for k in range(1, 10)]
        definition, model = prepare((TRIPLET,), {'X ec uc': entries})
        op = next(op for op in model.operators if op.kind == 'triplet_eu_yukawa')
        info = next(row for row in definition.operator_inputs() if row['kind'] == op.kind)
        flavours = list(product(range(1, 4), repeat=2))
        self.assertEqual(list(info['invariant_labels']), [f'lambda_R[{i},{j}]' for i, j in flavours])
        parameters, terms = render(op, {('X', 0): 'Ymass[{index}]'})
        for (i, j), parameter, term in zip(flavours, parameters, terms, strict=True):
            normal = self.check_hermitian_term(parameter, term)
            self.assertIn(f'Ymass[c1]*ProjM[s1,s2]*uqbar[s1,{i},c1].CC[l][s2,{j}]', normal)
            self.assertNotIn('CKM', normal)


if __name__ == '__main__':
    unittest.main()
