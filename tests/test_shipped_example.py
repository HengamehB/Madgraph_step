"""Check the model a new user receives against the licensed two-body benchmark."""
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from numpy.testing import assert_allclose

from ci.run_benchmark import COEFFICIENTS, ROOT, read_json
from model_metadata import load_metadata, publish_metadata
from tex_inputs import operator_bindings, read_inputs


EXAMPLE = ROOT / "input" / "my_model.txt"


def binding_diagnostic(model, scratch):
    """Describe a candidate refresh for review. Never modify a shipped input."""
    document = read_inputs(model)
    expected = operator_bindings(document.definition)
    current = document.assignments["_metadata"]
    metadata, _, _ = load_metadata(model, current)
    metadata["_input_bindings"] = expected
    proposed = publish_metadata(scratch / model.name, metadata)
    lines = [f"SAMPLE_METADATA current {current}", f"SAMPLE_METADATA proposed {proposed}"]
    for key, value in expected.items():
        values = ([key, k, v] for k, v in value.items()) if isinstance(value, dict) else [[key, value]]
        for entry in values:
            lines.append("SAMPLE_BINDING " + json.dumps(entry, ensure_ascii=False))
    return "\n".join(lines)


class ShippedExampleTests(unittest.TestCase):
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
            if result.returncode:
                reason = result.stdout
                if "stale active operator bindings" in reason:
                    reason += "\n" + binding_diagnostic(EXAMPLE, scratch)
                self.fail(reason)

            validations = list(output.glob("*/*/validation.json"))
            self.assertEqual(len(validations), 1, result.stdout)
            run = validations[0].parent
            self.assertEqual(read_json(validations[0])["status"], "physical_inputs_validated")
            inputs = read_json(run / "resolved_inputs.json")
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
            settings = read_json(run / "settings.json")
            collider = {"energy_tev": 13.6, "pdf": "nn23lo1", "dynamical_scale_choice": 3,
                        "seed": 1, "accuracy": .01, "max_final": 2, "new_particles_only": True,
                        "ptj": 20., "pta": 20., "ptl": 10., "etaj": 5., "etaa": 2.5, "etal": 2.5,
                        "drjj": .4, "drjl": .4, "draa": .4, "draj": .4, "dral": .4, "drll": .4}
            self.assertEqual({k: settings["collider"][k] for k in collider}, collider)
            self.assertEqual(settings["decays"], {"max_body": 2, "precision": .01, "min_br": 1e-8})
            self.assertEqual(settings["explicit_decays"], {"history_ids": [], "bwcutoff": 15., "cut_decays": False})
