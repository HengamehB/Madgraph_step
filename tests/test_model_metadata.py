"""Portable compact-model metadata is immutable, bounded and never executable."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import model_metadata as storage
from model_metadata import MetadataError, load_metadata, publish_metadata


class ModelMetadataTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='model metadata ')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.model = self.root / 'my model.txt'
        self.model.write_text('# Current editable input, never stored as metadata.\n')
        self.metadata = {
            '_input_bindings': {'schema': 'operator-table-bindings-v1', 'coefficients': {'X^2': 'abc'}},
            '_input_provenance': {'schema': 'model-input-origins-v1', 'coefficients': {
                'X^2': {'origin': 'explicit', 'value': -20000., 'binding': 'abc'}}},
        }

    def publish(self, metadata=None, reference=''):
        return publish_metadata(self.model, self.metadata if metadata is None else metadata,
                                reference=reference)

    def test_roundtrip_complex_arrays_tuples_and_integer_keys(self):
        value = [[1, 2j], (-.1 + .2j, 4), {1: 'one', '$tuple': [2]}]
        self.metadata['_input_provenance']['coefficients']['X^2']['value'] = value
        pointer = self.publish(reference='X^2: GeV^2; ordered entries [1, 2].\nHermitian convention.')
        recovered, raw, path = load_metadata(self.model, pointer)
        self.assertEqual(recovered, {**self.metadata, 'operator_reference':
                                    'X^2: GeV^2; ordered entries [1, 2].\nHermitian convention.'})
        self.assertEqual(raw, path.read_bytes())
        self.assertEqual(path, self.root / pointer)
        self.assertNotIn(self.model.read_bytes().strip(), raw)
        self.assertRegex(pointer, r'^\.model_metadata/[0-9a-f]{64}\.json$')

    def test_repeat_publication_reuses_immutable_record_without_backup(self):
        before = self.model.read_bytes()
        pointer = self.publish()
        artifact = self.root / pointer
        previous = artifact.stat()
        self.assertEqual(self.publish(), pointer)
        current = artifact.stat()
        self.assertEqual((previous.st_ino, previous.st_mtime_ns), (current.st_ino, current.st_mtime_ns))
        self.assertEqual(self.model.read_bytes(), before)
        self.assertEqual(len(list(artifact.parent.iterdir())), 1)
        self.assertFalse(list(self.root.rglob('*.bak*')))

    def test_changed_metadata_gets_distinct_record_and_preserves_previous(self):
        first = self.publish()
        old = (self.root / first).read_bytes()
        self.metadata['_input_provenance']['coefficients']['X^2']['value'] = -40000.
        second = self.publish()
        self.assertNotEqual(first, second)
        self.assertEqual((self.root / first).read_bytes(), old)
        self.assertEqual(load_metadata(self.model, second)[0], self.metadata)

    def test_reference_documentation_participates_in_identity(self):
        first, second = self.publish(reference='first convention'), self.publish(reference='second convention')
        self.assertNotEqual(first, second)
        self.metadata['operator_reference'] = 'first convention'
        self.assertEqual(self.publish(), first)
        with self.assertRaisesRegex(MetadataError, 'Conflicting'):
            self.publish(reference='different convention')

    def test_copied_input_with_companion_folder_needs_no_original(self):
        source = self.root / 'original'
        source.mkdir()
        self.model = source / 'my model.txt'
        self.model.write_text('model data')
        pointer = self.publish()
        destination = self.root / 'copied path with spaces'
        shutil.copytree(source, destination)
        shutil.rmtree(source)
        recovered, _, path = load_metadata(destination / self.model.name, pointer)
        self.assertEqual(recovered, self.metadata)
        self.assertEqual(path, destination / pointer)

    def test_missing_metadata_explains_required_companion(self):
        pointer = self.publish()
        (self.root / pointer).unlink()
        with self.assertRaisesRegex(MetadataError, 'keep the referenced .model_metadata folder'):
            load_metadata(self.model, pointer)

    def test_references_reject_absolute_traversal_other_folders_and_short_hashes(self):
        pointer = self.publish()
        for invalid in (str(self.root / pointer), '../' + pointer, './' + pointer,
                        '.model_metadata/../' + pointer, pointer.replace('/', '\\'),
                        '.metadata/' + Path(pointer).name, '.model_metadata/abc.json',
                        pointer.upper(), pointer + '\n', None, ['invalid']):
            with self.subTest(reference=invalid), self.assertRaisesRegex(MetadataError, 'relative'):
                load_metadata(self.model, invalid)

    def test_directory_symlink_is_rejected_for_load_and_publication(self):
        pointer = self.publish()
        directory = self.root / storage.DIRECTORY
        outside = self.root / 'other folder'
        directory.rename(outside)
        directory.symlink_to(outside, target_is_directory=True)
        for action in (lambda: load_metadata(self.model, pointer), self.publish):
            with self.assertRaisesRegex(MetadataError, 'safely access'):
                action()

    def test_file_symlink_is_rejected_even_to_contained_record(self):
        pointer = self.publish()
        path = self.root / pointer
        saved = path.parent / 'saved.json'
        path.rename(saved)
        path.symlink_to(saved)
        for action in (lambda: load_metadata(self.model, pointer), self.publish):
            with self.assertRaisesRegex(MetadataError, 'safely access'):
                action()

    def test_fifo_rejected_without_blocking(self):
        pointer = self.publish()
        path = self.root / pointer
        path.unlink()
        os.mkfifo(path)
        with self.assertRaisesRegex(MetadataError, 'regular file'):
            load_metadata(self.model, pointer)

    def test_altered_record_fails_closed_without_replacing_it(self):
        pointer = self.publish()
        path = self.root / pointer
        raw = path.read_bytes().replace(b'explicit', b'invented')
        path.write_bytes(raw)
        for action in (lambda: load_metadata(self.model, pointer), self.publish):
            with self.assertRaisesRegex(MetadataError, 'checksum mismatch'):
                action()
        self.assertEqual(path.read_bytes(), raw)
        self.assertEqual(list(path.parent.glob('.publish-*')), [])

    def test_duplicate_json_key_and_nonfinite_constants_rejected(self):
        pointer = self.publish()
        path = self.root / pointer
        raw = path.read_bytes()
        for invalid in (raw.replace(b'{', b'{"schema": "duplicate",', 1),
                        raw.replace(b'-20000.0', b'NaN'), raw.replace(b'-20000.0', b'Infinity'),
                        raw.replace(b'-20000.0', b'1e999')):
            path.write_bytes(invalid)
            with self.subTest(raw=invalid[:50]), self.assertRaises(MetadataError):
                load_metadata(self.model, pointer)

    def _unchecked(self, encoded):
        digest = hashlib.sha256(storage._canonical(encoded)).hexdigest()
        pointer = storage.DIRECTORY + '/' + digest + '.json'
        directory = self.root / storage.DIRECTORY
        directory.mkdir(exist_ok=True)
        (self.root / pointer).write_text(json.dumps(dict(schema=storage.SCHEMA, sha256=digest,
                                                       metadata=encoded)))
        return pointer

    def test_duplicate_encoded_dictionary_keys_rejected(self):
        encoded = storage._encode(self.metadata, storage._Budget())
        encoded['$dict'].append(deepcopy(encoded['$dict'][0]))
        with self.assertRaisesRegex(MetadataError, 'Duplicate model metadata'):
            load_metadata(self.model, self._unchecked(encoded))

    def test_arbitrary_checkpoint_types_and_paths_are_not_decoded(self):
        for dangerous in ({'$path': '/tmp/arbitrary'}, {'$type': 'UFOModel', 'fields': {}},
                          {'$history_rows': {}}):
            encoded = storage._encode(self.metadata, storage._Budget())
            encoded['$dict'][0][1] = dangerous
            with self.subTest(value=dangerous), self.assertRaisesRegex(MetadataError, 'encoded value'):
                load_metadata(self.model, self._unchecked(encoded))

    def test_schema_and_payload_validation(self):
        cases = [None, {}, {'source_text': 'old model input'}, {'_input_bindings': 7},
                 {'_input_bindings': {'schema': 'future schema'}},
                 {'_input_provenance': {'schema': 'future schema'}},
                 {**self.metadata, 'operator_reference': []}]
        for invalid in cases:
            with self.subTest(value=invalid), self.assertRaises(MetadataError):
                publish_metadata(self.model, invalid)
        with self.assertRaisesRegex(MetadataError, 'reference documentation'):
            self.publish(reference=[])
        # Legacy binding-only inputs remain supported.
        legacy = {'_input_bindings': {'schema': 'tex-input-bindings-v1'}}
        self.assertEqual(load_metadata(self.model, self.publish(legacy))[0], legacy)

    def test_nonfinite_and_unsupported_values_rejected_before_publication(self):
        for value in (float('nan'), float('inf'), 1j * float('inf'), object(), Path('a')):
            invalid = deepcopy(self.metadata)
            invalid['_input_provenance']['value'] = value
            with self.subTest(value=value), self.assertRaises(MetadataError):
                self.publish(invalid)
        self.assertFalse((self.root / storage.DIRECTORY).exists())

    def test_bounded_size_depth_and_node_counts(self):
        with patch.object(storage, 'MAX_BYTES', 5), self.assertRaisesRegex(MetadataError, '8 MB'):
            self.publish()
        with patch.object(storage, 'MAX_DEPTH', 1), self.assertRaisesRegex(MetadataError, 'budget'):
            self.publish()
        pointer = self.publish()
        for field, value, expected in (('MAX_BYTES', 5, '8 MB'), ('MAX_DEPTH', 1, 'budget'),
                                       ('MAX_NODES', 1, 'budget')):
            with self.subTest(limit=field), patch.object(storage, field, value), self.assertRaisesRegex(MetadataError, expected):
                load_metadata(self.model, pointer)

    def test_failed_or_cancelled_publication_leaves_model_and_previous_records_intact(self):
        old_model = self.model.read_bytes()
        first = self.publish()
        old_record = (self.root / first).read_bytes()
        self.metadata['_input_provenance']['new'] = 'value'
        for error in (OSError('simulated disk failure'), KeyboardInterrupt()):
            with patch.object(storage.os, 'link', side_effect=error):
                with self.assertRaises((MetadataError, KeyboardInterrupt)):
                    self.publish()
            self.assertEqual(self.model.read_bytes(), old_model)
            self.assertEqual((self.root / first).read_bytes(), old_record)
            self.assertEqual(list((self.root / storage.DIRECTORY).iterdir()), [self.root / first])

    def test_concurrent_publishers_reuse_one_complete_record(self):
        with ThreadPoolExecutor(max_workers=8) as executor:
            pointers = list(executor.map(lambda _: self.publish(), range(24)))
        self.assertEqual(len(set(pointers)), 1)
        self.assertEqual(load_metadata(self.model, pointers[0])[0], self.metadata)
        self.assertEqual(len(list((self.root / storage.DIRECTORY).iterdir())), 1)

    def test_temporary_link_cleanup_retries_one_complete_read(self):
        pointer = self.publish()
        path = self.root / pointer
        raw = path.read_bytes()
        inode = path.stat().st_ino
        original_fstat = storage.os.fstat
        for remove_after in (1, 2):
            temporary = path.parent / '.publisher.tmp'
            os.link(path, temporary)
            snapshots = []

            def fstat(descriptor):
                snapshot = original_fstat(descriptor)
                if snapshot.st_ino == inode:
                    snapshots.append(snapshot.st_nlink)
                    if len(snapshots) == remove_after:
                        temporary.unlink()
                return snapshot

            with self.subTest(remove_after=remove_after), patch.object(storage.os, 'fstat', side_effect=fstat):
                recovered, exact, _ = load_metadata(self.model, pointer)
            self.assertEqual(recovered, self.metadata)
            self.assertEqual(exact, raw)
            self.assertEqual(snapshots, [2, 1, 1, 1] if remove_after == 1 else [2, 2, 1, 1])

    def test_cleanup_retry_still_rejects_changed_content_and_replaced_records(self):
        pointer = self.publish()
        path = self.root / pointer
        raw = path.read_bytes()
        original_fstat = storage.os.fstat
        for action in ('mutate', 'replace'):
            path.write_bytes(raw)
            temporary = path.parent / '.publisher.tmp'
            os.link(path, temporary)
            inode = path.stat().st_ino
            snapshots = []

            def fstat(descriptor):
                snapshot = original_fstat(descriptor)
                if snapshot.st_ino == inode:
                    snapshots.append(snapshot)
                    if len(snapshots) == 1:
                        temporary.unlink()
                        if action == 'mutate':
                            path.write_bytes(raw.replace(b'explicit', b'invented'))
                        else:
                            replacement = path.parent / '.replacement.tmp'
                            replacement.write_bytes(raw)
                            os.replace(replacement, path)
                return snapshot

            with self.subTest(action=action), patch.object(storage.os, 'fstat', side_effect=fstat), \
                    self.assertRaisesRegex(MetadataError, 'changed during access'):
                load_metadata(self.model, pointer)
            self.assertEqual(len(snapshots), 2)

    def test_cleanup_retry_still_checks_checksum(self):
        pointer = self.publish()
        path = self.root / pointer
        path.write_bytes(path.read_bytes().replace(b'explicit', b'invented'))
        temporary = path.parent / '.publisher.tmp'
        os.link(path, temporary)
        inode = path.stat().st_ino
        original_fstat = storage.os.fstat
        snapshots = []

        def fstat(descriptor):
            snapshot = original_fstat(descriptor)
            if snapshot.st_ino == inode:
                snapshots.append(snapshot)
                if len(snapshots) == 1:
                    temporary.unlink()
            return snapshot

        with patch.object(storage.os, 'fstat', side_effect=fstat), \
                self.assertRaisesRegex(MetadataError, 'checksum mismatch'):
            load_metadata(self.model, pointer)
        self.assertEqual(len(snapshots), 4)

    def test_ctime_change_without_link_cleanup_is_not_retried(self):
        from types import SimpleNamespace
        pointer = self.publish()
        path = self.root / pointer
        inode = path.stat().st_ino
        original_fstat = storage.os.fstat
        snapshots = []

        def fstat(descriptor):
            snapshot = original_fstat(descriptor)
            if snapshot.st_ino == inode:
                snapshots.append(snapshot)
                if len(snapshots) == 1:
                    values = {key: getattr(snapshot, key) for key in
                              ('st_dev', 'st_ino', 'st_size', 'st_mode', 'st_mtime_ns', 'st_ctime_ns', 'st_nlink')}
                    values['st_ctime_ns'] -= 1
                    return SimpleNamespace(**values)
            return snapshot

        with patch.object(storage.os, 'fstat', side_effect=fstat), \
                self.assertRaisesRegex(MetadataError, 'changed during access'):
            load_metadata(self.model, pointer)
        self.assertEqual(len(snapshots), 2)

    def test_swapped_metadata_directory_is_detected(self):
        pointer = self.publish()
        directory = self.root / storage.DIRECTORY
        original = storage._read

        def swap(*args):
            result = original(*args)
            directory.rename(self.root / 'saved directory')
            directory.mkdir()
            return result

        with patch.object(storage, '_read', side_effect=swap), self.assertRaisesRegex(MetadataError, 'directory changed'):
            load_metadata(self.model, pointer)


if __name__ == '__main__':
    unittest.main()
