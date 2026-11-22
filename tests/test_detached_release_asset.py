"""Verify the exact downloadable text in isolation before attaching it to the release."""
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import test_shipped_example as shipped

MODEL = r"""# Madgraph_step v0.1.0 coloured-scalar sample. This copy can be moved on its own.
# From the installed package directory, substitute the path to this file:
#   ./run.sh rates /path/to/triplet_sextet.txt --selected C0004 --page-size 5
# Results: output/triplet_sextet_particles/<run-id>/ next to this file.

X = {'spin': 'scalar', 'su2': 3, 'su3': '6', 'hypercharge': '1/3', 'boson_kind': 'complex'}
Y = {'spin': 'scalar', 'su2': 1, 'su3': '3', 'hypercharge': '-1/3', 'boson_kind': 'complex'}
scenario = 1
sm_generations = 3
compiler_limits = {}

# Portal contribution to each physical mass squared at the SM vacuum.
Gf = 1.16637e-5
portal = 0.1
portal_mass_squared = portal / (2 * 2**0.5 * Gf)

operators = {
    # Mass and kinetic terms
    'X X†': {'enabled': True, 'coefficient': -1500**2 + portal_mass_squared},  # GeV^2
    'Y Y†': {'enabled': True, 'coefficient': -1200**2 + portal_mass_squared},  # GeV^2
    'D(X) D(X†)': {'enabled': True, 'coefficient': 1.0},  # dimensionless
    'D(Y) D(Y†)': {'enabled': True, 'coefficient': 1.0},  # dimensionless

    # Higgs interactions
    'H H† X X†': {'enabled': True, 'coefficient': [-portal, 0.0]},  # dimensionless
    'H H† Y Y†': {'enabled': True, 'coefficient': -portal},  # dimensionless

    # Scalar quartic interactions
    'X X† Y Y†': {'enabled': True, 'coefficient': [-0.05 + 0.02/3, -0.02]},  # dimensionless; 2 ordered entries: see reference
    'X X†^2 Y†': {'enabled': False, 'coefficient': None},  # dimensionless; limitations: see reference
    'X^2 X†^2': {'enabled': True, 'coefficient': [0.0, 0.0, -0.1, 0.0, 0.0]},  # dimensionless; 5 ordered entries: see reference
    'X^2 Y^2': {'enabled': False, 'coefficient': -0.01},  # dimensionless
    'Y^2 Y†^2': {'enabled': True, 'coefficient': -0.1},  # dimensionless

    # Fermion interactions
    'L Q Y†': {'enabled': False, 'coefficient': None},  # dimensionless; 9 ordered entries: see reference; limitations: see reference
    'Q^2 X†': {'enabled': True, 'coefficient': [0.1, 0, 0, 0.1, 0, 0.1]},  # symmetric, order 11, 12, 13, 22, 23, 33
    'Y ec uc': {'enabled': True, 'coefficient': [0.1, 0, 0, 0, 0.1, 0, 0, 0, 0.1]},  # rows u, c, t; columns e, mu, tau
    'Q^2 Y': {'enabled': False, 'coefficient': None},  # dimensionless; 6 ordered entries: see reference; limitations: see reference

    # Other interactions
    'Y dc† uc†': {'enabled': False, 'coefficient': None},  # dimensionless; 9 ordered entries: see reference; limitations: see reference

}

# Fixed SM vacuum and particle stability
# Incompatible tadpoles or Higgs mixing must be removed; coefficients are never silently changed.
model_name = 'triplet_sextet_particles'
sm_inputs = {'Gf': Gf, 'MZ': 91.1876, 'aEWM1': 127.9}
stable_particles = []

# Collider, decays and saved results
collider = {
    'energy_tev': 13.6, 'max_final': 2, 'max_tree_signatures': 250000, 'new_particles_only': True,
    'pdf': 'nn23lo1', 'seed': 1, 'accuracy': 0.01, 'ptj': 20.0, 'pta': 20.0, 'ptl': 10.0, 'etaj': 5.0,
    'etaa': 2.5, 'etal': 2.5, 'drjj': 0.4, 'drjl': 0.4, 'draa': 0.4, 'draj': 0.4, 'dral': 0.4, 'drll': 0.4,
    'dynamical_scale_choice': 3,
}
decays = {'max_body': 2, 'precision': 0.01, 'min_br': 1e-08}
ranking = {
    'min_relative_rate': 0.0001, 'max_width_mass_ratio': 0.01, 'threshold_widths': 5.0, 'max_depth': 12,
    'batch_size': 1000, 'memory_mb': 64, 'max_render_nodes': 10000,
}
explicit_decays = {'history_ids': [], 'bwcutoff': 15.0, 'cut_decays': False}
output = 'output'

# Managed identities for the coefficient conventions used above.
_input_bindings = {'schema': 'operator-table-bindings-v1',
 'context': 'd8026fdd8f8774e010e8523a8da4c2f593ee2e262f3d59fcb22343553d06d851',
 'compiler_fingerprint': '7938716385fbb1cf7aa95377f9241a0f424fcac9b7900c7f293f2ea4a8a5b7fd',
 'catalogue_fingerprint': 'e4bcc582959c69886d10d083eb9eae751d1488d5d67c04b8012c8ce9cb42fda6',
 'operator_ids': {'O0001': 'X X†',
                  'O0002': 'Y Y†',
                  'O0003': 'H H† X X†',
                  'O0004': 'H H† Y Y†',
                  'O0005': 'L Q Y†',
                  'O0006': 'Q^2 X†',
                  'O0007': 'Q^2 Y',
                  'O0008': 'X X† Y Y†',
                  'O0009': 'X X†^2 Y†',
                  'O0010': 'X^2 X†^2',
                  'O0011': 'X^2 Y^2',
                  'O0012': 'Y dc† uc†',
                  'O0013': 'Y ec uc',
                  'O0014': 'Y^2 Y†^2',
                  'O0015': 'D(X) D(X†)',
                  'O0016': 'D(Y) D(Y†)'},
 'coefficients': {'X X†': 'fb206c7c3f29898d91972a61e1d635041d527c246e98e9f7f9ed1b98ee7b4090',
                  'Y Y†': 'cb2753ed87ddc5c1b5f2f05937830a74367ec12a2b7a68e2456fccb528ac875d',
                  'H H† X X†': 'fa604b1dcd2e063d0092ede92faac4f2d3ed2dc29bbce1f016d07e0bfac97727',
                  'H H† Y Y†': '37b2f58fb63bea9c7ec26aa99bfa3e41542546d1e1a024be077ab8c9158ab495',
                  'L Q Y†': '007578ab42635ae0a45820c28e397aecee27edb8a61c401aebef916c30e6a1e3',
                  'Q^2 X†': '7f6a4afdfe28f7157cf020c82e10cef3fab04522ecce43209414395ccac8d37a',
                  'Q^2 Y': '4040f38dde3184d27f0bdd408af5cb292e0485f697828073724e03f828efd388',
                  'X X† Y Y†': '2f0a3f14a4c3e0afd74ac37907de6a372baf9b2e70233f46ac836e2ddf60ae0d',
                  'X X†^2 Y†': 'b089c0f6ea948c98dca224bfc3814fd83fb15f7400ce066df90398d1e77d29d5',
                  'X^2 X†^2': 'e2661351f7daaaa1cd2d3cfa05b0cedb73560c1b3b485c028c275ef21e58c1e8',
                  'X^2 Y^2': '6b53eee7ad975cec7b7764194ab111df9001e8750346a58dd79706231d537611',
                  'Y dc† uc†': 'cc688262a7bff98355acac017f801662262c0d794d266e187de58e4ceeaf844d',
                  'Y ec uc': '8a1974b7010fdb5914e33ae815e1b7e5e9f2a331a4a19395a6d145487d63e1e6',
                  'Y^2 Y†^2': '441b1de9e38781f5b4fdb5b1257d11710b3c62dd8ac333529f8d12d8d407a9b2',
                  'D(X) D(X†)': 'adcd2fbbdd1713058a3951a5f7f36c5d3a1b083c48302f59309115d3510c4e1e',
                  'D(Y) D(Y†)': 'c8805fc7197f025c0dcf502e612655c38000dfabfdf254183d11c196fbbcfabd'}}
"""


class DetachedReleaseTests(unittest.TestCase):
    def test_standalone_file(self):
        raw = MODEL.encode("utf-8")
        self.assertEqual(hashlib.sha256(raw).hexdigest(), "3307d9b7535cca147ee01376d53129b5cf7b0067b54729f03fa402aaeab61ae3")
        with TemporaryDirectory(prefix="detached release example ") as folder:
            sample = Path(folder) / "triplet_sextet.txt"
            sample.write_bytes(raw)
            self.assertFalse((sample.parent / ".model_metadata").exists())
            with patch.object(shipped, "EXAMPLE", sample):
                case = shipped.ShippedExampleTests("test_validate_distributed_sample_against_licensed_benchmark")
                case.test_validate_distributed_sample_against_licensed_benchmark()
