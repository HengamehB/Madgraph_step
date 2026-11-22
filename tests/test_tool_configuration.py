"""Host discovery and approved atomic configuration changes use no installers."""
from pathlib import Path
import configparser
import os
import stat
import sys
import tempfile
import unittest
from unittest.mock import patch

import portable_runtime
import tool_configuration as configuration


class ToolConfigurationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='tool paths with spaces ')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.path = self.root / '.config'
        self.original = '# retained comment\n[project]\nmodel = custom input.txt\noutput = output dir\n\n[tools]\n# retained tools comment\nmathematica = /old/missing\n\n[ranking]\nmin_relative_rate = 0.1\n'
        self.path.write_text(self.original)
        self.path.chmod(0o640)
        self.tool = self.root / 'kernel with spaces'
        self.tool.write_text('#!/bin/sh\nexit 0\n')
        self.tool.chmod(0o755)
        self.messages = []
        self.discovery = {'tools': {name: {'configured': None, 'selected': None, 'candidates': [], 'reason': 'no valid candidate found'}
                                   for name in configuration.CONFIG_KEYS}}
        self.discovery['tools']['wolfram_kernel'].update(selected=str(self.tool), candidates=[{'path': str(self.tool), 'origin': 'standard location'}], reason=None)
        patcher = patch.object(configuration, 'discover_tools', return_value=self.discovery)
        self.discover = patcher.start()
        self.addCleanup(patcher.stop)

    def configure(self, answers):
        return configuration.configure_tools(self.path, input_fn=lambda prompt: next(answers), output_fn=self.messages.append)

    def test_approval_preserves_comments_physics_permissions_without_backup(self):
        result = self.configure(iter(['yes']))
        self.assertTrue(result['saved'])
        self.assertIsNone(result['backup'])
        self.assertFalse(list(self.root.glob('.config.bak*')))
        self.assertFalse(any('backup' in item.lower() for item in self.messages))
        saved = self.path.read_text()
        self.assertIn('[project]\nmodel = custom input.txt\noutput = output dir', saved)
        self.assertIn('# retained tools comment', saved)
        self.assertIn('[ranking]\nmin_relative_rate = 0.1', saved)
        self.assertIn('mathematica = ' + str(self.tool), saved)
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o640)
        self.assertTrue(any('Exact values to save' in item for item in self.messages))
        self.assertEqual(self.discover.call_args.args[0]['wolfram_kernel'], '/old/missing')

    def test_approval_fills_shipped_blank_tool_settings_on_the_assignment_lines(self):
        original = ('# Empty paths mean auto-detect.\n[tools]\nmathematica =\n'
                    'madgraph = vendor/MG5_aMC\nfeynrules = vendor/FeynRules\n'
                    'fortran =\ncxx =\nmake =\nbash =\nperl =\n\n'
                    '[project]\nmodel = chosen model.txt\noutput = saved results\n\n'
                    '[collider]\nenergy_tev = 13.6\nmax_final = 3\n')
        self.path.write_text(original)
        for name, key in configuration.CONFIG_KEYS.items():
            selected = self.root / ('vendor/MG5_aMC' if name == 'madgraph' else
                                    'vendor/FeynRules' if name == 'feynrules' else
                                    f'bin/{key} with spaces')
            self.discovery['tools'][name].update(selected=str(selected), reason=None)
        result = self.configure(iter(['yes']))
        self.assertTrue(result['saved'])
        self.assertIsNone(result['backup'])
        self.assertFalse(list(self.root.glob('.config.bak*')))
        saved = self.path.read_text()
        expected = original
        for key, value in result['changes'].items():
            expected = expected.replace(f'{key} =\n', f'{key} ={value}\n')
        self.assertEqual(saved, expected)
        parser = configparser.ConfigParser(interpolation=None)
        parser.read_string(saved)
        self.assertEqual(parser['project']['model'], 'chosen model.txt')
        self.assertEqual(parser['collider']['energy_tev'], '13.6')
        for key, value in result['changes'].items():
            self.assertEqual(parser['tools'][key], value)

    def test_blank_assignment_update_preserves_line_endings_and_horizontal_spacing(self):
        for newline in ('\n', '\r\n'):
            for prefix in ('mathematica =', 'mathematica = ', 'mathematica:\t',
                           '\tmathematica\t=\t', 'MATHEMATICA ='):
                with self.subTest(newline=repr(newline), prefix=repr(prefix)):
                    original = newline.join(('[tools]', prefix, '# comment',
                                             '[project]', 'model = chosen model.txt', ''))
                    self.path.write_bytes(original.encode())
                    result = self.configure(iter(['yes']))
                    self.assertTrue(result['saved'])
                    self.assertEqual(self.path.read_bytes(),
                                     original.replace(prefix + newline,
                                                      prefix + str(self.tool) + newline).encode())
                    self.assertIsNone(result['backup'])
                    self.assertFalse(list(self.root.glob('.config.bak*')))

    def test_blank_final_assignment_without_newline_is_updated_in_place(self):
        original = '[tools]\nmathematica ='
        self.path.write_bytes(original.encode())
        result = self.configure(iter(['yes']))
        self.assertTrue(result['saved'])
        self.assertEqual(self.path.read_bytes(), (original + str(self.tool)).encode())
        self.assertIsNone(result['backup'])
        self.assertFalse(list(self.root.glob('.config.bak*')))

    def test_invalid_existing_config_has_a_readable_error_and_is_unchanged(self):
        original = '[tools]\nmathematica =\n/unattached/path\n'
        self.path.write_text(original)
        with self.assertRaisesRegex(ValueError, 'Invalid tool configuration.*file unchanged'):
            self.configure(iter(['yes']))
        self.discover.assert_not_called()
        self.assertEqual(self.path.read_text(), original)
        self.assertFalse(list(self.root.glob('.config.bak*')))

    def test_invalid_generated_config_is_rejected_before_write(self):
        with patch.object(configuration, '_replace_tools', return_value='[tools]\n/unattached/path\n'):
            with self.assertRaisesRegex(RuntimeError, 'Generated tool configuration is invalid; original file unchanged'):
                self.configure(iter(['yes']))
        self.assertEqual(self.path.read_text(), self.original)
        self.assertFalse(list(self.root.glob('.config.bak*')))

    def test_multiline_path_update_is_rejected_without_truncating_the_old_value(self):
        original = '[tools]\nmathematica =\n    /old/path\n'
        self.path.write_text(original)
        with self.assertRaisesRegex(ValueError, r'Cannot safely save \[tools\] mathematica'):
            self.configure(iter(['yes']))
        self.assertEqual(self.path.read_text(), original)
        self.assertFalse(list(self.root.glob('.config.bak*')))

    def test_no_enter_eof_and_interrupt_never_change_configuration_or_make_backup(self):
        for answer in ('', 'n', 'anything', EOFError(), KeyboardInterrupt()):
            with self.subTest(answer=repr(answer)):
                def input_fn(prompt):
                    if isinstance(answer, BaseException):
                        raise answer
                    return answer
                result = configuration.configure_tools(self.path, input_fn=input_fn, output_fn=lambda _: None)
                self.assertFalse(result['saved'])
                self.assertTrue(result['cancelled'])
                self.assertEqual(self.path.read_text(), self.original)
                self.assertFalse(list(self.root.glob('.config.bak*')))

    def test_edit_candidate_and_custom_path_then_reapprove(self):
        alternative = self.root / 'second kernel'
        alternative.write_text('#!/bin/sh\nexit 0\n')
        alternative.chmod(0o755)
        answers = iter(['edit', '/invalid/path', str(alternative), *([''] * 7), 'yes'])
        result = self.configure(answers)
        self.assertEqual(result['paths']['wolfram_kernel'], str(alternative))
        self.assertEqual(result['changes']['mathematica'], str(alternative))
        self.assertTrue(any('Invalid mathematica' in item for item in self.messages))

    def test_existing_current_values_need_no_prompt_or_backup(self):
        # This case covers a fully configured host. Missing tools have a
        # separate interactive path so users can enter custom locations.
        settings = ['[tools]']
        for name, key in configuration.CONFIG_KEYS.items():
            path = self.root / name
            self.discovery['tools'][name].update(selected=str(path), reason=None)
            value = name if name in {'madgraph', 'feynrules'} else str(path)
            settings.append(f'{key} = {value}')
        self.path.write_text('\n'.join(settings) + '\n')
        result = configuration.configure_tools(self.path, input_fn=lambda _: self.fail('No changes require no prompt'), output_fn=self.messages.append)
        self.assertFalse(result['saved'])
        self.assertFalse(result['cancelled'])
        self.assertEqual(result['changes'], {})

    def test_configuration_changed_during_approval_is_not_overwritten(self):
        replacement = '# updated concurrently\n' + self.original
        def approve(prompt):
            self.path.write_text(replacement)
            return 'yes'
        with self.assertRaisesRegex(RuntimeError, 'changed during approval'):
            configuration.configure_tools(self.path, input_fn=approve, output_fn=lambda _: None)
        self.assertEqual(self.path.read_text(), replacement)
        self.assertFalse(list(self.root.glob('.config.bak*')))

    def test_failed_final_replace_retains_original_without_backup(self):
        original_replace = os.replace
        def replace(source, target):
            if Path(target) == self.path:
                raise OSError('disk full fixture')
            return original_replace(source, target)
        with patch.object(configuration.os, 'replace', side_effect=replace):
            with self.assertRaisesRegex(OSError, 'disk full'):
                self.configure(iter(['yes']))
        self.assertEqual(self.path.read_text(), self.original)
        self.assertFalse(list(self.root.glob('.config.bak*')))
        self.assertEqual([p.name for p in self.root.iterdir() if p.name.startswith('..config.')], ['..config.lock'])

    def test_existing_backup_is_preserved_without_creating_another(self):
        backup = self.path.with_name('.config.bak')
        backup.write_text('older backup')
        result = self.configure(iter(['yes']))
        self.assertEqual(backup.read_text(), 'older backup')
        self.assertIsNone(result['backup'])
        self.assertEqual(list(self.root.glob('.config.bak*')), [backup])

    def test_creating_config_has_only_tools_and_relocatable_source_paths(self):
        self.path.unlink()
        bundled = self.root / 'vendor/MG5_aMC'
        self.discovery['tools']['madgraph'].update(selected=str(bundled))
        result = self.configure(iter(['yes']))
        self.assertIsNone(result['backup'])
        self.assertIn('madgraph = vendor/MG5_aMC', self.path.read_text())
        self.assertNotIn('[project]', self.path.read_text())
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o600)

    def test_default_config_and_dry_discovery_have_no_side_effects(self):
        with patch.object(configuration, 'ROOT', self.root):
            result = configuration.configure_tools(input_fn=lambda _: 'no', output_fn=lambda _: None)
        self.assertEqual(result['path'], self.path)
        self.assertEqual(self.path.read_text(), self.original)
        with patch.object(portable_runtime, '_probe', side_effect=AssertionError('Do not execute discovery')):
            report = portable_runtime.discover_tools({'fortran': sys.executable})
        self.assertEqual(report['tools']['fortran']['selected'], sys.executable)


if __name__ == '__main__':
    unittest.main()
