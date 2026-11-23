"""Check the published archive rules and Make targets with Git's ZIP writer."""
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which('git') and shutil.which('make'), 'requires git and make')
class SourceArchiveTests(unittest.TestCase):
    def test_archive_keeps_runtime_sources_and_exposes_only_available_targets(self):
        sampler = 'vendor/MG5_aMC/vendor/DiscreteSampler/'
        stdhep = 'vendor/MG5_aMC/vendor/StdHEP/'
        keep = {'Makefile', sampler + 'makefile', sampler + 'DiscreteSampler.f', sampler + 'tester.f',
                stdhep + 'mcfio/src/mcfio.h', 'sample_output/compare.py', 'input/sample_model_sextet.txt'}
        exclude = {'.gitattributes', 'tests/test_example.py', 'ci/run_benchmark.py', sampler + 'check',
                   stdhep + 'CVS/Entries', stdhep + 'mcfio/src/CVS/Root'}
        with tempfile.TemporaryDirectory(prefix='release test ') as tmp:
            root = Path(tmp)
            checkout = root / 'checkout'
            checkout.mkdir()
            for name in keep | exclude:
                path = checkout / name
                path.parent.mkdir(parents=True, exist_ok=True)
                if name in {'.gitattributes', 'Makefile'}:
                    shutil.copyfile(ROOT / name, path)
                else:
                    path.touch()
            subprocess.run(['git', 'init', '-q'], cwd=checkout, check=True)
            subprocess.run(['git', 'add', '--force', '.'], cwd=checkout, check=True)
            tree = subprocess.check_output(['git', 'write-tree'], cwd=checkout, text=True).strip()
            archive = root / 'source.zip'
            subprocess.run(['git', 'archive', '--format=zip', '-o', archive, tree], cwd=checkout, check=True)
            download = root / 'extracted source'
            with ZipFile(archive) as zipped:
                self.assertEqual({p for p in zipped.namelist() if not p.endswith('/')}, keep)
                zipped.extractall(download)

            for directory, has_tests in ((checkout, True), (download, False)):
                with self.subTest(has_tests=has_tests):
                    help_text = subprocess.check_output(['make', 'help'], cwd=directory, text=True)
                    self.assertEqual('make test' in help_text, has_tests)
                    self.assertIn('make rates', help_text)
                    test = subprocess.run(['make', '-n', 'test'], cwd=directory, capture_output=True, text=True)
                    self.assertEqual(test.returncode == 0, has_tests)
                    if has_tests:
                        self.assertIn('unittest discover', test.stdout)
                        self.assertIn('ci/run_benchmark.py --mode fast', test.stdout)
