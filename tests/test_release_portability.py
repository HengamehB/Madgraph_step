"""Host-independent checks for the release launcher and tool configuration."""
import configparser
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import portable_runtime
import tool_configuration


def executable(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('#!/bin/sh\nexit 0\n')
    path.chmod(0o755)
    return path


class ToolDiscoveryTests(unittest.TestCase):
    def test_mac_engine_and_user_applications_with_spaces(self):
        with tempfile.TemporaryDirectory(prefix='tool discovery ') as directory:
            root = Path(directory)
            system, user = root / 'Applications', root / 'User Applications'
            engine = executable(system / 'Wolfram Engine.app/Contents/MacOS/WolframKernel')
            local = executable(user / 'Wolfram.app/Contents/MacOS/WolframKernel')
            candidates = portable_runtime._wolfram_candidates(
                application_roots=(system, user), linux_roots=())
            self.assertIn(engine, candidates)
            self.assertIn(local, candidates)
            self.assertEqual(portable_runtime.normalize_tool_path('wolfram_kernel', engine), str(engine))

    def test_linux_products_and_numeric_version_order(self):
        with tempfile.TemporaryDirectory(prefix='linux tools ') as directory:
            root = Path(directory)
            latest = executable(root / 'Wolfram/14.10/Executables/WolframKernel')
            older = executable(root / 'Wolfram/14.9/Executables/WolframKernel')
            engine = executable(root / 'WolframEngine/14.2/Executables/MathKernel')
            unversioned = executable(root / 'Mathematica/Executables/WolframKernel')
            candidates = portable_runtime._wolfram_candidates(
                application_roots=(), linux_roots=(root,))
            self.assertLess(candidates.index(latest), candidates.index(older))
            self.assertIn(engine, candidates)
            self.assertIn(unversioned, candidates)

    def test_explicit_stale_path_does_not_silently_select_alternative(self):
        with tempfile.TemporaryDirectory(prefix='explicit tools ') as directory:
            root = Path(directory)
            alternative = executable(root / 'Available Kernel')
            missing = root / 'Previous Machine/Kernel'
            with patch.object(portable_runtime, '_wolfram_candidates', return_value=[alternative]), \
                    patch.object(portable_runtime.shutil, 'which', return_value=None):
                report = portable_runtime.discover_tools({'wolfram_kernel': str(missing)})
            entry = report['tools']['wolfram_kernel']
            self.assertIsNone(entry['selected'])
            self.assertEqual(entry['configured'], str(missing))
            self.assertEqual(entry['reason'], 'configured path is invalid')
            self.assertEqual(entry['candidates'][0]['path'], str(alternative))

    def test_valid_explicit_path_has_precedence(self):
        with tempfile.TemporaryDirectory(prefix='preferred tools ') as directory:
            root = Path(directory)
            preferred = executable(root / 'Custom Kernel')
            alternative = executable(root / 'Standard Kernel')
            with patch.object(portable_runtime, '_wolfram_candidates', return_value=[alternative]), \
                    patch.object(portable_runtime.shutil, 'which', return_value=None):
                report = portable_runtime.discover_tools({'wolfram_kernel': str(preferred)})
            self.assertEqual(report['tools']['wolfram_kernel']['selected'], str(preferred))


class ConfigurationTests(unittest.TestCase):
    def test_can_edit_missing_tool_when_other_paths_are_unchanged(self):
        with tempfile.TemporaryDirectory(prefix='config with spaces ') as directory:
            root = Path(directory)
            kernel = executable(root / 'Custom Mathematica/Kernel')
            config = root / '.config'
            original = '[project]\nmodel = input/my scan.txt\n[tools]\n'
            entries = {}
            for name, key in tool_configuration.CONFIG_KEYS.items():
                value = None if name == 'wolfram_kernel' else str(root / name)
                original += f'{key} = {value or ""}\n'
                entries[name] = dict(selected=value, configured=value, candidates=[],
                                     reason='no valid candidate found' if value is None else None)
            config.write_text(original)
            answers = iter(['edit', str(kernel), *([''] * 7), 'y'])
            with patch.object(tool_configuration, 'discover_tools', return_value={'tools': entries}):
                result = tool_configuration.configure_tools(
                    config, input_fn=lambda _: next(answers), output_fn=lambda _: None)
            self.assertTrue(result['saved'])
            parser = configparser.ConfigParser(interpolation=None)
            parser.read(config)
            self.assertEqual(parser['tools']['mathematica'], str(kernel))
            self.assertEqual(parser['project']['model'], 'input/my scan.txt')
            self.assertIsNone(result['backup'])
            self.assertFalse(list(root.glob('*.bak*')))

    def test_cancel_missing_tools_preserves_original(self):
        with tempfile.TemporaryDirectory(prefix='cancel config ') as directory:
            config = Path(directory) / '.config'
            original = b'[tools]\nmathematica = /previous/machine/kernel\n'
            config.write_bytes(original)
            entries = {name: dict(selected=None, configured=None, candidates=[],
                                  reason='no valid candidate found')
                       for name in tool_configuration.CONFIG_KEYS}
            with patch.object(tool_configuration, 'discover_tools', return_value={'tools': entries}):
                result = tool_configuration.configure_tools(
                    config, input_fn=lambda _: 'n', output_fn=lambda _: None)
            self.assertTrue(result['cancelled'])
            self.assertEqual(config.read_bytes(), original)

    def test_empty_assignment_preserves_ini_line_boundaries(self):
        original = '[tools]\r\nmathematica =\r\nfortran =\r\n'
        result = tool_configuration._replace_tools(original, {
            'mathematica': '/Applications/Wolfram Engine.app/Contents/MacOS/WolframKernel',
            'fortran': '/usr/bin/gfortran',
        })
        parser = configparser.ConfigParser(interpolation=None)
        parser.read_string(result)
        self.assertEqual(parser['tools']['fortran'], '/usr/bin/gfortran')
        self.assertEqual(len(result.splitlines()), 3)


@unittest.skipUnless(os.name == 'posix' and Path('/bin/bash').is_file(), 'POSIX Bash launcher')
class LauncherTests(unittest.TestCase):
    def test_python3_fallback_and_arguments_with_spaces(self):
        with tempfile.TemporaryDirectory(prefix='launcher with spaces ') as directory:
            root = Path(directory)
            shutil.copy2(ROOT / 'run.sh', root / 'run.sh')
            binaries = root / 'bin'
            binaries.mkdir()
            (binaries / 'dirname').symlink_to(shutil.which('dirname'))
            python = executable(binaries / 'python3')
            python.write_text('#!/bin/sh\nprintf "<%s>\\n" "$@"\n')
            env = dict(os.environ, PATH=str(binaries))
            env.pop('PYTHON', None)
            result = subprocess.run(['/bin/bash', str(root / 'run.sh'), 'check', 'my model.txt'],
                                    env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.splitlines(),
                             [f'<{root / "pipeline_cli.py"}>', '<check>', '<my model.txt>'])

    def test_explicit_missing_python_is_not_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = str(Path(directory) / 'missing python')
            result = subprocess.run(['/bin/bash', str(ROOT / 'run.sh'), 'check'],
                                    env=dict(os.environ, PYTHON=missing), capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Python was not found', result.stderr)


@unittest.skipUnless(shutil.which('make'), 'make is required')
class MakefileTests(unittest.TestCase):
    def test_default_config_is_optional_but_explicit_config_is_forwarded(self):
        with tempfile.TemporaryDirectory(prefix='fresh clone ') as directory:
            root = Path(directory)
            shutil.copy2(ROOT / 'Makefile', root / 'Makefile')
            env = dict(os.environ)
            for name in ('CONFIG', 'MAKEFLAGS', 'MFLAGS', 'MAKEOVERRIDES'):
                env.pop(name, None)
            command = [shutil.which('make'), '-n', 'check', f'PYTHON={sys.executable}']
            default = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, check=True)
            self.assertNotIn('--config', default.stdout)
            explicit = subprocess.run(command + ['CONFIG=host settings.ini'], cwd=root, env=env,
                                      capture_output=True, text=True, check=True)
            self.assertIn('--config "host settings.ini"', explicit.stdout)


if __name__ == '__main__':
    unittest.main()
