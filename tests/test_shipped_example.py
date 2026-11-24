"""Check the model a new user receives against the licensed two-body benchmark."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

from numpy.testing import assert_allclose

from ci.run_benchmark import COEFFICIENTS, FREE, ROOT, read_json
from model_initialization import DEFAULT_PARTICLES, initialize_model
from pipeline_config import load_config
from tex_inputs import read_inputs


EXAMPLE = ROOT / "input" / "sample_model_sextet.txt"
REFERENCE = ROOT / "output" / "triplet_sextet"


class ShippedExampleTests(unittest.TestCase):
    def test_particle_file_names_and_starter_coefficients(self):
        with TemporaryDirectory(prefix="sample particle files ") as tmp:
            directory = Path(tmp).resolve()
            particles = directory / "sample_particle_sextet.txt"
            self.assertEqual(DEFAULT_PARTICLES, ROOT / "input" / particles.name)
            shutil.copyfile(DEFAULT_PARTICLES, particles)
            with patch("model_initialization.DEFAULT_PARTICLES", particles):
                result = initialize_model()
            self.assertEqual(result.path, directory / EXAMPLE.name)
            fresh = read_inputs(result.path).assignments
            sample = read_inputs(EXAMPLE).assignments
            for field in ("X", "Y"):
                self.assertEqual(fresh[field], sample[field])
            self.assertEqual({k for k, v in fresh["operators"].items() if v["enabled"]}, FREE)

            before = result.path.read_bytes()
            custom = directory / "another scan.txt"
            init = subprocess.run(
                [sys.executable, str(ROOT / "pipeline_cli.py"), "init", str(particles), "--model", str(custom)],
                cwd=directory, capture_output=True, text=True, timeout=90,
            )
            self.assertEqual(init.returncode, 0, init.stderr)
            self.assertTrue(custom.is_file())
            self.assertEqual(result.path.read_bytes(), before)
            legacy_source = directory / "particles.txt"
            particles.rename(legacy_source)
            self.assertEqual(initialize_model(legacy_source).path, directory / "my_model.txt")

    def test_configured_and_explicit_names_override_bundled_example(self):
        with TemporaryDirectory() as tmp:
            directory = Path(tmp).resolve()
            config = directory / "tools.config"
            config.write_text("[tools]\n", encoding="utf-8")
            self.assertEqual(load_config(config, read_model=False).model_file, EXAMPLE)
            old = directory / "input" / "my_model.txt"
            old.parent.mkdir()
            old.touch()
            self.assertEqual(load_config(config, read_model=False).model_file, old)
            configured = directory / "my scan.txt"
            config.write_text("[project]\nmodel = my scan.txt\n", encoding="utf-8")
            self.assertEqual(load_config(config, read_model=False).model_file, configured)
            self.assertEqual(load_config(config, model_file=EXAMPLE, read_model=False).model_file, EXAMPLE)

    def test_validate_distributed_sample_against_licensed_benchmark(self):
        with TemporaryDirectory(prefix="shipped model test ") as tmp:
            scratch = Path(tmp)
            config = scratch / "tools.config"
            config.write_text("[tools]\n", encoding="utf-8")
            output = scratch / "sample results"
            result = subprocess.run(
                ["bash", str(ROOT / "run.sh"), "validate", str(EXAMPLE),
                 "--config", str(config), "--output", str(output)],
                cwd=scratch, env={**os.environ, "PYTHON": sys.executable},
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", timeout=180, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout)

            validations = list(output.glob("*/*/validation.json"))
            self.assertEqual(len(validations), 1, result.stdout)
            run = validations[0].parent
            self.assertEqual(read_json(validations[0])["status"], "physical_inputs_validated")
            inputs = read_json(run / "resolved_inputs.json")
            reference = read_json(REFERENCE / "resolved_inputs.json")["coefficients"]
            self.assertEqual(set(inputs["coefficients"]), set(reference))
            for key, value in reference.items():
                assert_allclose(inputs["coefficients"][key], value, rtol=1e-12, atol=1e-14)
            definition = read_inputs(EXAMPLE).definition
            active = {definition.coefficient_key(key): value for key, value in inputs["coefficients"].items()}
            self.assertEqual(set(active), set(COEFFICIENTS))
            for key, expected in COEFFICIENTS.items():
                with self.subTest(operator=key):
                    self.assertEqual(type(active[key]) is list, isinstance(expected, (tuple, list)))
                    assert_allclose(active[key], expected, rtol=1e-12, atol=1e-14)
            self.assertEqual(inputs["sm_inputs"], {"Gf": 1.16637e-5, "MZ": 91.1876, "aEWM1": 127.9})
            self.assertEqual(inputs["stable_particles"], [])
            masses = read_json(run / "spectrum.json")["masses"]
            target = {"H": 125., "X1": 1500., "X2": 1500., "X3": 1500., "Y1": 1200.}
            self.assertEqual(set(masses), set(target))
            for name, expected in target.items():
                self.assertAlmostEqual(masses[name], expected, places=7, msg=name)
            reference_masses = read_json(REFERENCE / "spectrum.json")["masses"]
            self.assertEqual(set(masses), set(reference_masses))
            for key, value in reference_masses.items():
                self.assertAlmostEqual(masses[key], value, places=7)
            reference_settings = read_json(REFERENCE / "manifest.json")["settings"]
            settings = read_json(run / "settings.json")
            for section in ("collider", "decays"):
                for key, value in reference_settings[section].items():
                    self.assertEqual(settings[section][key], value, (section, key))
            collider = {"energy_tev": 13.6, "pdf": "nn23lo1", "dynamical_scale_choice": 3,
                        "seed": 1, "accuracy": .01, "max_final": 2, "new_particles_only": True,
                        "ptj": 20., "pta": 20., "ptl": 10., "etaj": 5., "etaa": 2.5, "etal": 2.5,
                        "drjj": .4, "drjl": .4, "draa": .4, "draj": .4, "dral": .4, "drll": .4}
            self.assertEqual({k: settings["collider"][k] for k in collider}, collider)
            self.assertEqual(settings["decays"], {"max_body": 2, "precision": .01, "min_br": 1e-8})
            self.assertEqual(settings["explicit_decays"], {"history_ids": [], "bwcutoff": 15., "cut_decays": False})
