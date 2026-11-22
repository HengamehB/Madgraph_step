"""Relocation must never probe saved original paths or escaped link targets."""
from contextlib import contextmanager
import builtins
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from run_paths import contained_path, portable_history_reference


class RunPathTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.root = self.base / 'copied package with spaces' / '63ae510ef90047a8657f'
        self.root.mkdir(parents=True)
        self.original = self.base / 'original must not be touched'
        self.original.mkdir()
        self.relative = Path('histories/e44f3bf51205287b5049/histories.sqlite')
        self.local = self.root / self.relative
        self.local.parent.mkdir(parents=True)
        self.local.write_bytes(b'local database placeholder')
        self.reference = dict(path=str(self.original / self.root.name / self.relative),
                              schema='madgraph-history-store-v2', fingerprint='abc',
                              revision=62, file_stat=dict(size_bytes=26, mtime_ns=1, ctime_ns=2))

    @contextmanager
    def prohibit_original_io(self):
        """Trap both target probes and accidental opening of the old tree."""
        prefix = str(self.original)
        def guard(function):
            def checked(path, *args, **kwargs):
                if isinstance(path, (str, bytes, os.PathLike)):
                    text = os.fsdecode(path)
                    if text == prefix or text.startswith(prefix + os.sep):
                        raise AssertionError('Original path was accessed: ' + text)
                return function(path, *args, **kwargs)
            return checked
        with patch('os.stat', guard(os.stat)), patch('os.lstat', guard(os.lstat)), \
                patch('os.readlink', guard(os.readlink)), patch('os.open', guard(os.open)), \
                patch('builtins.open', guard(builtins.open)), patch('io.open', guard(io.open)):
            yield

    def test_relative_paths_missing_descendants_and_unicode(self):
        with self.prohibit_original_io():
            self.assertEqual(contained_path(self.root, self.relative), self.local)
            self.assertEqual(contained_path(self.root, 'new results/μ.json'),
                             self.root / 'new results/μ.json')
            self.assertEqual(contained_path(self.root, '.'), self.root)

    def test_absolute_and_parent_traversal_inputs_are_rejected(self):
        for value in (self.original / 'secret', '../secret', 'histories/../secret',
                      'histories/../../original must not be touched'):
            with self.subTest(value=value), self.prohibit_original_io(), self.assertRaises(ValueError):
                contained_path(self.root, value)

    def test_metadata_and_artifact_symlinks_reject_outside_before_probe(self):
        for relative in ('manifest.json', '.checkpoints/export/record.json', 'ranking_exports.json',
                         'exports/bundle/ranking.json', 'histories/escaped.sqlite'):
            with self.subTest(relative=relative):
                link = self.root / relative
                link.parent.mkdir(parents=True, exist_ok=True)
                link.symlink_to(self.original / 'missing target')
                with self.prohibit_original_io(), self.assertRaisesRegex(ValueError, 'escapes'):
                    contained_path(self.root, relative)

    def test_parent_directory_symlink_rejects_before_original_access(self):
        (self.root / 'old').symlink_to(self.original, target_is_directory=True)
        with self.prohibit_original_io(), self.assertRaisesRegex(ValueError, 'escapes'):
            contained_path(self.root, 'old/nested/file.json')

    def test_relative_symlink_escape_rejects_even_when_target_is_missing(self):
        (self.root / 'escape').symlink_to('../../original must not be touched/missing')
        with self.prohibit_original_io(), self.assertRaisesRegex(ValueError, 'escapes'):
            contained_path(self.root, 'escape')

    def test_contained_relative_absolute_and_dangling_links(self):
        (self.root / 'relative').symlink_to(self.relative)
        (self.root / 'absolute').symlink_to(self.local)
        (self.root / 'histories' / 'parent').symlink_to('../' + self.relative.as_posix())
        (self.root / 'dangling').symlink_to('new/local.json')
        with self.prohibit_original_io():
            for relative in ('relative', 'absolute', 'histories/parent'):
                self.assertEqual(contained_path(self.root, relative), self.local)
            self.assertEqual(contained_path(self.root, 'dangling'), self.root / 'new/local.json')

    def test_symlink_parent_resolution_obeys_component_order(self):
        nested = self.root / 'nested' / 'deep'
        nested.mkdir(parents=True)
        (self.root / 'inner').symlink_to('nested/deep')
        (self.root / 'outer').symlink_to('inner/../result.json')
        self.assertEqual(contained_path(self.root, 'outer'), self.root / 'nested/result.json')
        (self.root / 'leave-and-return').symlink_to('../' + self.root.name + '/manifest.json')
        with self.assertRaisesRegex(ValueError, 'escapes'):
            contained_path(self.root, 'leave-and-return')

    def test_symlink_cycles_are_bounded(self):
        (self.root / 'first').symlink_to('second')
        (self.root / 'second').symlink_to('first')
        with self.assertRaisesRegex(ValueError, 'Too many symlinks'):
            contained_path(self.root, 'first')

    def test_legacy_reference_migrates_without_original_io_and_preserves_metadata(self):
        before = dict(self.reference)
        with self.prohibit_original_io():
            portable = portable_history_reference(self.reference, self.root)
            opened = portable_history_reference(self.reference, self.root, absolute=True)
        self.assertEqual(portable, {**self.reference, 'path': self.relative.as_posix(), 'path_base': 'run'})
        self.assertEqual(opened, {**portable, 'path': self.local})
        self.assertEqual(self.reference, before)
        self.assertIs(portable['file_stat'], self.reference['file_stat'])

    def test_legacy_migration_is_independent_of_old_tree_existence_or_contents(self):
        old = Path(self.reference['path'])
        old.parent.mkdir(parents=True)
        old.write_bytes(b'different original content')
        with self.prohibit_original_io():
            self.assertEqual(portable_history_reference(self.reference, self.root, absolute=True)['path'], self.local)

    def test_legacy_migration_checks_rebased_local_symlinks_before_opening(self):
        self.local.unlink()
        self.local.symlink_to(Path(self.reference['path']))
        with self.prohibit_original_io(), self.assertRaisesRegex(ValueError, 'escapes'):
            portable_history_reference(self.reference, self.root, absolute=True)

    def test_trusted_root_alias_accepts_current_logical_and_canonical_paths(self):
        alias = self.base / 'trusted root alias'
        alias.symlink_to(self.root, target_is_directory=True)
        for path in (alias / self.relative, self.local):
            with self.subTest(path=path), self.prohibit_original_io():
                result = portable_history_reference(dict(self.reference, path=path), alias,
                                                    absolute=True, allow_legacy=False)
                self.assertEqual(result['path'], self.local)

    def test_current_absolute_and_relative_reference_round_trip_without_legacy(self):
        for path in (self.local, self.relative, self.relative.as_posix()):
            with self.subTest(path=path), self.prohibit_original_io():
                relative = portable_history_reference(dict(self.reference, path=path), self.root, allow_legacy=False)
                self.assertEqual(relative['path'], self.relative.as_posix())
                opened = portable_history_reference(relative, self.root, absolute=True, allow_legacy=False)
                self.assertEqual(opened['path'], self.local)
                self.assertEqual(portable_history_reference(opened, self.root, allow_legacy=False), relative)

    def test_legacy_migration_requires_exact_suffix(self):
        candidates = [
            self.original / (self.root.name + '-sibling') / self.relative,
            self.original / self.root.name / 'histories/ABCDEF12345678901234/histories.sqlite',
            self.original / self.root.name / 'histories/abcdef/histories.sqlite',
            self.original / self.root.name / 'histories/e44f3bf51205287b5049/other.sqlite',
            self.original / self.root.name / self.relative / 'extra',
            self.original / self.root.name / 'other/e44f3bf51205287b5049/histories.sqlite',
            self.root.with_name(self.root.name + '-sibling') / 'other.sqlite',
            self.original / '..' / self.root.name / self.relative,
        ]
        for path in candidates:
            with self.subTest(path=path), self.prohibit_original_io(), self.assertRaises(ValueError):
                portable_history_reference(dict(self.reference, path=str(path)), self.root)
        with self.prohibit_original_io(), self.assertRaises(ValueError):
            portable_history_reference(self.reference, self.root, allow_legacy=False)

    def test_reference_rejects_unknown_base_traversal_and_contained_symlink_escape(self):
        candidates = [dict(self.reference, path_base='package'),
                      dict(self.reference, path='../histories/database.sqlite', path_base='run'),
                      dict(self.reference, path='escape', path_base='run')]
        (self.root / 'escape').symlink_to(self.original / 'database.sqlite')
        for reference in candidates:
            with self.subTest(reference=reference), self.prohibit_original_io(), self.assertRaises(ValueError):
                portable_history_reference(reference, self.root)
        for reference in (None, [], {}, {'path': None}):
            with self.subTest(reference=reference), self.assertRaises(ValueError):
                portable_history_reference(reference, self.root)

    def test_history_database_path_must_not_be_the_run_root(self):
        (self.root / 'self').symlink_to('.')
        for path in ('', '.', self.root, 'self'):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, 'must name a file'):
                portable_history_reference(dict(self.reference, path=path), self.root)


if __name__ == '__main__':
    unittest.main()
