"""Actual Wolfram resource diagnostics must survive export failure wrapping."""
from pathlib import Path
import json
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from feynrules_export import PreparedModel, CompilerResolutionError, _python3_ufo
from workflow_errors import ResourceLimitError, error_category
from compiler_limits import DEFAULTS, normalize_limits
from workflow_errors import PhysicalInputError


class WolframResourceLimitTests(unittest.TestCase):
    def export_failure(self, message, expected, returncode=0, limits=None, trace=None):
        with TemporaryDirectory() as temporary:
            root=Path(temporary);kernel=root/'WolframKernel';kernel.touch()
            fr=root/'FeynRules';fr.mkdir();(fr/'FeynRules.m').touch()
            def sources(model,source,fr):
                path=source/'model.fr';path.write_text('LModel = 0;\n')
                return [path],'LModel',[]
            model=PreparedModel(SimpleNamespace(),{'model_name':'ResourceRegression','compiler_limits':limits or {}},(),(),None,
                                compiler=SimpleNamespace(write_sources=sources, masses={"H": 125.0}))
            def run(command,**kwargs):
                kwargs['stdout'].write(message)
                if trace is not None:
                    ufo = root / 'export' / (model.model_name + '_UFO')
                    ufo.mkdir()
                    (ufo / 'unknown_vertex_fields.json').write_text(trace)
                return SimpleNamespace(returncode=returncode)
            paths={'wolfram_kernel':str(kernel),'feynrules':str(fr)}
            with patch('portable_runtime.resolve_runtime',return_value=paths),patch('portable_runtime.environment',return_value={}),patch('feynrules_export.subprocess.run',side_effect=run):
                with self.assertRaises(expected) as caught:
                    model.export_ufo(root/'export',analytic_decays=False)
            self.assertEqual((root/'export/feynrules.log').read_text(),message)
            self.assertTrue((root/'export/feynrules_source/export.wl').is_file())
            self.assertIn(str(root/'export/feynrules.log'),str(caught.exception))
            driver=(root/'export/feynrules_source/export.wl').read_text()
            policy=json.loads((root/'export/export_policy.json').read_text())
            for name,symbol in [('max_wolfram_recursion','$RecursionLimit'),('max_wolfram_iterations','$IterationLimit')]:
                value=(limits or {}).get(name,DEFAULTS[name])
                assignment=f'{symbol} = {value};'
                self.assertIn(assignment,driver)
                self.assertLess(driver.index('Get[FileNameJoin'),driver.index(assignment))
                self.assertLess(driver.index(assignment),driver.index('LoadModel['))
                self.assertEqual(policy['effective_wolfram_limits'][name],value)
            return caught.exception

    def test_real_mixed_antitriplet_recursion_log_with_successful_process_exit(self):
        # Exact final lines retained from joint_mixed_antitriplet_87bd1f: the
        # kernel returned zero without generating a UFO or success marker.
        text=('"Checking for hermiticity by calculating the Feynman rules contained in L-HC[L]."\n'
              '"If the lagrangian is hermitian, then the number of vertices should be zero."\n\n'
              '$RecursionLimit::reclim: Recursion depth of 2500 exceeded.\n')
        error=self.export_failure(text,ResourceLimitError)
        self.assertEqual(error_category(error),'resource_limit')
        self.assertIn('hermiticity construction',str(error))
        self.assertIn('Recursion depth of 2500 exceeded',str(error))
        self.assertIn('compiler_limits.max_wolfram_recursion=2500',str(error))

    def test_custom_limits_are_applied_after_package_load_and_recorded(self):
        limits={'max_wolfram_recursion':5000,'max_wolfram_iterations':9000}
        error=self.export_failure('$IterationLimit::itlim: Iteration limit of 9000 exceeded.\n',ResourceLimitError,limits=limits)
        self.assertIn('compiler_limits.max_wolfram_iterations=9000',str(error))

    def test_wolfram_minima_prevent_silently_rejected_assignments(self):
        self.assertEqual((DEFAULTS['max_wolfram_recursion'],DEFAULTS['max_wolfram_iterations']),(2500,4096))
        for name in ('max_wolfram_recursion','max_wolfram_iterations'):
            for value in (0,1,19,True,20.5):
                with self.subTest(name=name,value=value),self.assertRaises(PhysicalInputError):normalize_limits({name:value})
            self.assertEqual(normalize_limits({name:20}),{name:20})

    def test_iteration_and_memory_failures_keep_actual_diagnostic(self):
        for text in ('$IterationLimit::itlim: Iteration limit of 4096 exceeded.\n',
                     'General::nomem: The current computation was aborted because there was insufficient memory available.\n',
                     'SystemException["MemoryAllocationFailure"]\n'):
            with self.subTest(diagnostic=text):
                error=self.export_failure(text,ResourceLimitError,returncode=18)
                self.assertIn(text.strip(),str(error))
                self.assertEqual(error_category(error),'resource_limit')

    def test_physical_export_check_failure_is_still_a_compiler_failure(self):
        error=self.export_failure('PORTABLE_HERMITICITY_FAILED\n',CompilerResolutionError,returncode=12)
        self.assertEqual(error_category(error),'compiler_failure')

    def test_unrecognized_external_failure_keeps_execution_category(self):
        error=self.export_failure('Kernel failed to load its license.\n',RuntimeError,returncode=1)
        self.assertEqual(error_category(error),'execution_failure')

    def test_export_surfaces_first_unclassified_leg_even_if_kernel_returns_zero(self):
        report = {'vertex_index': 32, 'legs': [
            {'outgoing_spin_code': 1, 'position': 1},
            {'outgoing_spin_code': 0, 'position': 2, 'incoming_field': 'Global`inputField',
             'outgoing_field': 'PRIVATE`outputField'}]}
        error = self.export_failure('PORTABLE_EXPORT_SUCCESS\n', CompilerResolutionError,
                                    trace=json.dumps(report))
        self.assertEqual(error_category(error), 'compiler_failure')
        self.assertIn('PORTABLE_UFO_UNKNOWN_FIELD vertex 32, leg 2: Global`inputField -> PRIVATE`outputField',
                      str(error))
        self.assertIn('unknown_vertex_fields.json', str(error))

    def test_export_preserves_raw_leg_from_log_if_trace_is_missing_or_broken(self):
        for trace in (None, '{'):
            with self.subTest(trace=trace):
                error = self.export_failure('PORTABLE_UFO_UNKNOWN_FIELD vertex 8, leg 2: raw -> outgoing\n',
                                            CompilerResolutionError, returncode=18, trace=trace)
                self.assertIn('vertex 8, leg 2: raw -> outgoing', str(error))
                self.assertIn('unreadable field trace', str(error))


class UFOExportDiagnosticsTests(unittest.TestCase):
    def test_legacy_python_syntax_conversion_is_preserved(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / 'object_library.py'
            path.write_text('raise ValueError, "invalid parameter"\n')
            _python3_ufo(path.parent)
            self.assertIn('raise ValueError("invalid parameter")', path.read_text())

    def test_broken_ufo_keeps_filename_line_and_source_before_and_after_legacy_print(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / 'vertices.py'
            for prefix in ('# UFO header\n', 'print "legacy header"\n'):
                with self.subTest(prefix=prefix):
                    original = prefix + 'lorentz = (ConvertSpinToString + )\n'
                    path.write_text(original)
                    with self.assertRaises(CompilerResolutionError) as caught:
                        _python3_ufo(path.parent)
                    self.assertIn(str(path), str(caught.exception))
                    self.assertIn('line 2, column', str(caught.exception))
                    self.assertIn('lorentz = (ConvertSpinToString + )', str(caught.exception))
                    self.assertEqual(path.read_text(), original)

    def test_feynrules_tool_fingerprint_includes_contraction_and_ufo_sources(self):
        from pipeline import Pipeline
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            names = ('FeynRulesPackage.m', 'Core/ExtractVertexTools.m',
                     'Interfaces/UFO/PYIntMain.m', 'Interfaces/UFO/PYIntVertices.m')
            pipeline = Pipeline.__new__(Pipeline)
            pipeline.config = SimpleNamespace(runtime={'feynrules': str(root)})
            pipeline._tool_digest_cache = {}
            for name in names:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('original')
            previous = pipeline._tool_fingerprint('feynrules')
            for name in names:
                with self.subTest(name=name):
                    (root / name).write_text('modified source')
                    current = pipeline._tool_fingerprint('feynrules')
                    self.assertNotEqual(current, previous)
                    previous = current


if __name__=='__main__':unittest.main()
