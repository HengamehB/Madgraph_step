import unittest
import errno
import sqlite3
from compiler_limits import normalize_limits, component_limits
from component_invariants import ComponentResourceLimit, InvariantVerificationError
from feynrules_export import ModelResolutionError, CompilerResolutionError, BackendCapabilityError
from workflow_errors import error_category


class WorkflowDiagnosticTests(unittest.TestCase):
    def test_disk_exhaustion_is_a_resource_failure_even_when_wrapped(self):
        sqlite_full = sqlite3.OperationalError('database or disk is full')
        sqlite_full.sqlite_errorcode = sqlite3.SQLITE_FULL
        for cause in (OSError(errno.ENOSPC,'disk full'), OSError(errno.EDQUOT,'quota'), sqlite_full):
            wrapper = RuntimeError('could not publish')
            wrapper.__cause__ = cause
            self.assertEqual(error_category(cause),'resource_limit')
            self.assertEqual(error_category(wrapper),'resource_limit')

    def test_physical_backend_compiler_and_resource_failures_stay_distinct(self):
        cases = [(ModelResolutionError('bad vacuum'), 'physical_input'),
                 (CompilerResolutionError('unlowered tensor'), 'compiler_failure'),
                 (BackendCapabilityError('colour decuplet'), 'not_madgraph_supported'),
                 (ComponentResourceLimit('budget'), 'resource_limit'),
                 (InvariantVerificationError('rank'), 'compiler_failure')]
        for error, expected in cases:
            self.assertEqual(error_category(error), expected)
            wrapper = RuntimeError('stage failed')
            wrapper.__cause__ = error
            self.assertEqual(error_category(wrapper), expected)

    def test_compiler_budgets_validate_and_filter_backend_specific_options(self):
        for bad in ({'unknown': 1}, {'max_output_terms': 0}, {'max_output_terms': True},
                    {'max_output_terms': 2.5}, []):
            with self.assertRaises(ValueError):
                normalize_limits(bad)
        limits = component_limits({'max_output_terms': 17, 'max_weak_generator_entries': 48})
        self.assertEqual(limits.max_output_terms, 17)

    def test_nonzero_weak_scalar_vev_is_a_physical_input_error(self):
        from lagrangian_maker import make_lagrangian
        from feynrules_export import prepare_model
        for dimension, kind, hypercharge in ((2, 'complex', '1/2'), (3, 'real', '0')):
            result = make_lagrangian(1, dict(spin='scalar', su2=dimension, su3='1',
                    hypercharge=hypercharge, boson_kind=kind), show=False)
            selected = result.select('all', show=False)
            values = {r['id']: 0. if r['multiplicity'] == 1 else [0.]*r['multiplicity'] for r in selected.operators}
            with self.subTest(dimension=dimension), self.assertRaises(ModelResolutionError) as caught:
                prepare_model(selected, dict(coefficients=values, vevs={'X': 1.}))
            self.assertEqual(error_category(caught.exception), 'physical_input')


if __name__ == '__main__':
    unittest.main()
