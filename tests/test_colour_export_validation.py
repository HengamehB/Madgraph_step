"""Independent component checks for native-colour tensors and retained UFOs.

The group checks run without FeynRules or MadGraph.  To additionally validate
the standard one-field export fixtures, set NATIVE_COLOUR_SEXTET_UFO and/or
NATIVE_COLOUR_REAL8_UFO to UFO directories.  The sextet fixture has charge one
and coefficients (-.01, -.01) for the norm and T6-current quartics.  The real
octet has d-cubic coefficient .1 and norm-quartic coefficient -.01.  All fields
have canonical kinetic normalization.  No generated model is executed or
imported, and the checks compare every colour component, not just vertex counts.
NATIVE_COLOUR_MIXED6_UFO additionally checks the two-field mixed_benchmark()
export, independently solving its generalized mass problem by spectral whitening.
"""
from __future__ import annotations

import ast
from functools import lru_cache
from itertools import combinations_with_replacement, permutations
import os
from pathlib import Path
import unittest

import numpy as np


@lru_cache(maxsize=1)
def reference_tensors():
    """Explicit fundamental matrices and the symmetric tensor-product rep."""
    t = np.array([
        [[0, 1, 0], [1, 0, 0], [0, 0, 0]],
        [[0, -1j, 0], [1j, 0, 0], [0, 0, 0]],
        [[1, 0, 0], [0, -1, 0], [0, 0, 0]],
        [[0, 0, 1], [0, 0, 0], [1, 0, 0]],
        [[0, 0, -1j], [0, 0, 0], [1j, 0, 0]],
        [[0, 0, 0], [0, 0, 1], [0, 1, 0]],
        [[0, 0, 0], [0, 0, -1j], [0, 1j, 0]],
        np.diag([1, 1, -2]) / np.sqrt(3),
    ], dtype=complex) / 2
    q = np.zeros((6, 9))
    q[0, 0] = q[1, 4] = q[2, 8] = 1
    q[3, [1, 3]] = q[4, [2, 6]] = q[5, [5, 7]] = 1 / np.sqrt(2)
    sextet = np.array([q @ (np.kron(a, np.eye(3)) + np.kron(np.eye(3), a)) @ q.T
                       for a in t])
    # Trace definitions are independent of the compiler's contraction lowering.
    triple = np.einsum('aij,bjk,cki->abc', t, t, t)
    f = (2j * (triple.swapaxes(0, 1) - triple)).real
    d = (2 * (triple + triple.swapaxes(0, 1))).real
    return {'T': t, 'T6': sextet, 'f': f, 'd': d,
            'K6': q.reshape(6, 3, 3), 'K6Bar': q.reshape(6, 3, 3)}


def mixed_sextet_reference():
    """Solve the benchmark via K^{-1/2}, without the production generalized eigh."""
    kinetic = np.array([[1.2, .1 - .05j], [.1 + .05j, .8]])
    mass = np.array([[40000, 2000 - 1000j], [2000 + 1000j, 90000]])
    kvalues, kvectors = np.linalg.eigh(kinetic)
    whitening = (kvectors / np.sqrt(kvalues)) @ kvectors.conj().T
    mass_squared, vectors = np.linalg.eigh(whitening @ mass @ whitening)
    rotation = whitening @ vectors
    # These well-separated benchmark states already have their largest
    # overlaps with X and Y, respectively. Fix only their arbitrary phases.
    assert tuple(np.argmax(abs(rotation), axis=0)) == (0, 1)
    for column in range(2):
        rotation[:, column] *= np.exp(-1j * np.angle(rotation[column, column]))
    return kinetic, mass, rotation, mass_squared


def mixed_sextet_quartic(particles, rotation, coefficient=-.02):
    """i times four exact field derivatives of c (X†X)(Y†Y).

    Enumerating assignments to the four original factors automatically retains
    multiplicities for repeated particles and complex conjugation of each leg.
    """
    flavours = {'x': 0, 'xbar': 0, 'y': 1, 'ybar': 1}
    # FeynRules' UFO writer calls MakeOutgoingParticles (PYIntVertices.m),
    # replacing each differentiated field by its antiparticle. Thus an outgoing
    # x is obtained by differentiating the conjugate field, and vice versa.
    barred = tuple(not name.endswith('bar') for name in particles)
    identity = np.eye(6)
    result = np.zeros((6,) * 4, complex)
    for assignment in permutations(range(4)):
        if tuple(barred[slot] for slot in assignment) != (False, True, False, True):
            continue
        value = 1j * coefficient
        for raw, bar, slot in zip((0, 0, 1, 1), (False, True, False, True), assignment):
            factor = rotation[raw, flavours[particles[slot]]]
            value *= factor.conjugate() if bar else factor
        result += value * np.einsum(identity, list(assignment[:2]), identity,
                                   list(assignment[2:]), list(range(4)))
    return result


def _constructor_records(path, constructor):
    """Read simple generated UFO declarations without executing model code."""
    records = {}
    for node in ast.parse(path.read_text()).body:
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == constructor):
            records[node.targets[0].id] = {kw.arg: kw.value for kw in node.value.keywords}
    return records


def _number(expression, names):
    """Evaluate the small numeric grammar used by the selected UFO couplings."""
    if isinstance(expression, str):
        return _number(ast.parse(expression, mode='eval').body, names)
    if isinstance(expression, ast.Constant):
        return expression.value
    if isinstance(expression, ast.Name):
        return names[expression.id]
    if isinstance(expression, ast.UnaryOp):
        value = _number(expression.operand, names)
        if isinstance(expression.op, ast.USub):
            return -value
        if isinstance(expression.op, ast.UAdd):
            return value
    if isinstance(expression, ast.BinOp):
        left, right = _number(expression.left, names), _number(expression.right, names)
        for kind, operation in ((ast.Add, lambda: left + right),
                                (ast.Sub, lambda: left - right),
                                (ast.Mult, lambda: left * right),
                                (ast.Div, lambda: left / right),
                                (ast.Pow, lambda: left ** right)):
            if isinstance(expression.op, kind):
                return operation()
    if (isinstance(expression, ast.Call) and isinstance(expression.func, ast.Name)
            and expression.func.id == 'complex' and not expression.keywords):
        return complex(*(_number(arg, names) for arg in expression.args))
    raise ValueError(f'Unsupported numeric expression: {ast.dump(expression)}')


class UFOComponents:
    """Reusable reader for scalar vertices in a native-tensor UFO directory."""

    def __init__(self, directory):
        root = Path(directory)
        self.vertices = _constructor_records(root / 'vertices.py', 'Vertex')
        self.couplings = _constructor_records(root / 'couplings.py', 'Coupling')
        self.lorentz = _constructor_records(root / 'lorentz.py', 'Lorentz')
        self.parameters = _constructor_records(root / 'parameters.py', 'Parameter')

    def parameter(self, name):
        value = ast.literal_eval(self.parameters[name]['value'])
        return _number(value, {}) if isinstance(value, str) else value

    def vertex(self, particles):
        matches = [v for v in self.vertices.values()
                   if tuple(p.attr for p in v['particles'].elts) == tuple(particles)]
        if len(matches) != 1:
            raise AssertionError(f'Expected one vertex for {particles}, found {len(matches)}')
        return matches[0]

    @staticmethod
    def colour(expression, dimensions):
        labels = {i + 1: i for i in range(len(dimensions))}
        operands = []

        def collect(node):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
                collect(node.left)
                collect(node.right)
                return
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                raise ValueError(f'Unsupported colour expression: {expression}')
            slots = [ast.literal_eval(arg) for arg in node.args]
            if node.func.id == 'Identity':
                external = next(i for i in slots if i > 0)
                tensor = np.eye(dimensions[external - 1])
            else:
                tensor = reference_tensors()[node.func.id]
            for slot in slots:
                if slot not in labels:
                    labels[slot] = len(labels)
            operands.extend((tensor, [labels[slot] for slot in slots]))

        collect(ast.parse(expression, mode='eval').body)
        used = {index for indices in operands[1::2] for index in indices}
        for axis, dimension in enumerate(dimensions):
            if axis not in used:
                if dimension != 1:
                    raise ValueError(f'Missing coloured external leg {axis + 1}')
                operands.extend((np.ones(1), [axis]))
        return np.einsum(*operands, list(range(len(dimensions))), optimize=True)

    def components(self, particles, dimensions, lorentz, parameters):
        vertex = self.vertex(particles)
        structures = [ast.literal_eval(self.lorentz[node.attr]['structure'])
                      for node in vertex['lorentz'].elts]
        if structures != [lorentz]:
            raise AssertionError(f'Unexpected Lorentz structures: {structures}')
        colours = ast.literal_eval(vertex['color'])
        result = np.zeros(dimensions, dtype=complex)
        for key, value in zip(vertex['couplings'].keys, vertex['couplings'].values):
            colour_index, lorentz_index = ast.literal_eval(key)
            if lorentz_index != 0:
                raise AssertionError('Unexpected multiple Lorentz structures')
            coefficient = _number(ast.literal_eval(self.couplings[value.attr]['value']), parameters)
            result += coefficient * self.colour(colours[colour_index], dimensions)
        return result


class NativeColourTensorTests(unittest.TestCase):
    def test_sextet_normalization_and_algebra(self):
        from generic_color_compiler import colour_tensors
        reference = reference_tensors()
        actual = colour_tensors()
        t = actual['T6']
        np.testing.assert_allclose(t, reference['T6'], atol=1e-14)
        np.testing.assert_allclose(np.einsum('aij,bji->ab', t, t), 2.5 * np.eye(8), atol=1e-14)
        np.testing.assert_allclose(np.einsum('aij,ajk->ik', t, t), (10 / 3) * np.eye(6), atol=1e-14)
        for a in range(8):
            for b in range(8):
                np.testing.assert_allclose(t[a] @ t[b] - t[b] @ t[a],
                    1j * np.einsum('c,cij->ij', reference['f'][a, b], t), atol=1e-14)

    def test_primitive_representation_orientations(self):
        from generic_color_compiler import _primitives
        reference = reference_tensors()
        generators = {3: reference['T'], 6: reference['T6'], 8: -1j * reference['f']}
        generators.update({-3: -generators[3].conj(), -6: -generators[6].conj()})
        for name, reps, tensor in _primitives():
            with self.subTest(tensor=name):
                for a in range(8):
                    variation = sum(np.moveaxis(np.tensordot(generators[rep][a].T,
                        tensor, axes=(1, axis)), 0, axis) for axis, rep in enumerate(reps))
                    np.testing.assert_allclose(variation, 0, atol=1e-14)


class _OptionalUFOTests(unittest.TestCase):
    fixture_variable = None

    @classmethod
    def setUpClass(cls):
        path = os.environ.get(cls.fixture_variable, '')
        if not path:
            raise unittest.SkipTest(f'Set {cls.fixture_variable} to validate a retained UFO')
        cls.ufo = UFOComponents(path)

    def check_gauge(self, dimension, scalar_names, generators):
        g = 1.17  # Non-unit probe also tests the power of the gauge coupling.
        triple = self.ufo.components(('g', *scalar_names), (8, dimension, dimension),
                                     'P(1,2) - P(1,3)', {'G': g})
        np.testing.assert_allclose(triple, -1j * g * generators.transpose(0, 2, 1), atol=1e-14)
        contact = self.ufo.components(('g', 'g', *scalar_names), (8, 8, dimension, dimension),
                                      'Metric(1,2)', {'G': g})
        expected = 1j * g ** 2 * (np.einsum('ail,blj->abji', generators, generators)
                                  + np.einsum('bil,alj->abji', generators, generators))
        np.testing.assert_allclose(contact, expected, atol=1e-14)


class SextetUFOTests(_OptionalUFOTests):
    fixture_variable = 'NATIVE_COLOUR_SEXTET_UFO'

    def test_quartic_feynman_rule(self):
        names = ('pmGCO0003n1', 'pmGCO0003n2')
        values = {name: self.ufo.parameter(name) for name in names}
        self.assertEqual(tuple(values.values()), (-.01, -.01))
        actual = self.ufo.components(('xbar', 'xbar', 'x', 'x'), (6,) * 4, '1', values)
        identity, t = np.eye(6), reference_tensors()['T6']
        expected = 2j * values[names[0]] * (np.einsum('ac,bd->abcd', identity, identity)
                                          + np.einsum('ad,bc->abcd', identity, identity))
        expected += 2j * values[names[1]] * (np.einsum('eca,edb->abcd', t, t)
                                            + np.einsum('ecb,eda->abcd', t, t))
        np.testing.assert_allclose(actual, expected, atol=1e-14)

    def test_gluon_vertices(self):
        self.check_gauge(6, ('xbar', 'x'), reference_tensors()['T6'])

    def test_electromagnetic_contacts(self):
        ee, g = .31, 1.17
        actual = self.ufo.components(('a', 'a', 'xbar', 'x'), (1, 1, 6, 6),
                                     'Metric(1,2)', {'ee': ee})
        np.testing.assert_allclose(actual, (2j * ee ** 2 * np.eye(6))[None, None], atol=1e-14)
        actual = self.ufo.components(('a', 'g', 'xbar', 'x'), (1, 8, 6, 6),
                                     'Metric(1,2)', {'ee': ee, 'G': g})
        expected = 2j * ee * g * reference_tensors()['T6'].transpose(0, 2, 1)
        np.testing.assert_allclose(actual, expected[None], atol=1e-14)


class RealOctetUFOTests(_OptionalUFOTests):
    fixture_variable = 'NATIVE_COLOUR_REAL8_UFO'

    def test_cubic_feynman_rule(self):
        coefficient = self.ufo.parameter('pmGCO0002n1')
        self.assertEqual(coefficient, .1)
        actual = self.ufo.components(('x',) * 3, (8,) * 3, '1', {'pmGCO0002n1': coefficient})
        np.testing.assert_allclose(actual, 6j * coefficient * reference_tensors()['d'], atol=1e-14)

    def test_quartic_feynman_rule(self):
        coefficient = self.ufo.parameter('pmGCO0004n1')
        self.assertEqual(coefficient, -.01)
        actual = self.ufo.components(('x',) * 4, (8,) * 4, '1', {'pmGCO0004n1': coefficient})
        identity = np.eye(8)
        expected = 8j * coefficient * (np.einsum('ab,cd->abcd', identity, identity)
                                       + np.einsum('ac,bd->abcd', identity, identity)
                                       + np.einsum('ad,bc->abcd', identity, identity))
        np.testing.assert_allclose(actual, expected, atol=1e-14)

    def test_gluon_vertices(self):
        self.check_gauge(8, ('x', 'x'), -1j * reference_tensors()['f'])


class MixedSextetUFOTests(_OptionalUFOTests):
    fixture_variable = 'NATIVE_COLOUR_MIXED6_UFO'

    def test_physical_mass_spectrum(self):
        kinetic, mass, rotation, expected = mixed_sextet_reference()
        np.testing.assert_allclose(rotation.conj().T @ kinetic @ rotation, np.eye(2), atol=1e-14)
        np.testing.assert_allclose(rotation.conj().T @ mass @ rotation, np.diag(expected), atol=1e-9)
        actual = [self.ufo.parameter('pmGCM1'), self.ufo.parameter('pmGCM2')]
        np.testing.assert_allclose(actual, np.sqrt(expected), rtol=1e-12)

    def test_all_rotated_quartic_components(self):
        _, _, rotation, _ = mixed_sextet_reference()
        values = {name: self.ufo.parameter(name) for name in self.ufo.parameters
                  if name.startswith(('pmGCR', 'pmGCO'))}
        # The same physical fixture may use a uniformly rescaled raw-field
        # normalization. Its quartic scales as the square of its mass term.
        scale = values['pmGCO0001n1'] / -40000.
        self.assertAlmostEqual(values['pmGCO0007n1'] / scale**2, -.02)
        names = ('x', 'xbar', 'y', 'ybar')
        expected_sets = {tuple(sorted(particles))
                         for particles in combinations_with_replacement(names, 4)
                         if sum(name.endswith('bar') for name in particles) == 2}
        seen = set()
        for vertex in self.ufo.vertices.values():
            particles = tuple(node.attr for node in vertex['particles'].elts)
            if len(particles) != 4 or not all(name in names for name in particles):
                continue
            with self.subTest(particles=particles):
                self.assertNotIn(tuple(sorted(particles)), seen)
                seen.add(tuple(sorted(particles)))
                actual = self.ufo.components(particles, (6,) * 4, '1', values)
                expected = mixed_sextet_quartic(particles, rotation)
                np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=1e-12)
        self.assertEqual(seen, expected_sets)

    def test_canonical_gauge_vertices(self):
        for scalar_names in (('xbar', 'x'), ('ybar', 'y')):
            self.check_gauge(6, scalar_names, reference_tensors()['T6'])
        for vertex in self.ufo.vertices.values():
            names = tuple(node.attr for node in vertex['particles'].elts)
            if names.count('g') not in (1, 2) or not all(name in ('g', 'x', 'xbar', 'y', 'ybar') for name in names):
                continue
            scalar_names = tuple(name for name in names if name != 'g')
            if scalar_names:
                self.assertIn(scalar_names, (('xbar', 'x'), ('ybar', 'y')))


if __name__ == '__main__':
    unittest.main()
