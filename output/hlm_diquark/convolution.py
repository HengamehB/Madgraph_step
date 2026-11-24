"""Han, Lewis and McElmurry Eq. (3.4) with the shipped member-zero NNPDF grid.

Usage: python convolution.py path/to/NNPDF23_lo_as_0130_qed_mem0.grid
Dependencies are NumPy and SciPy, as in the physics package.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad


def read_up_distribution(grid_path, scale):
    """Read xf_u. Follow NNPDF's four-x-node/two-log-Q²-node prescription."""
    with grid_path.open(encoding='ascii') as stream:
        for line in stream:
            if line.strip() == "'NNPDF20intqed'":
                break
        else:
            raise ValueError('Expected a member-zero NNPDF QED grid.')
        stream.readline()  # member descriptor
        nx = int(stream.readline())
        xs = np.array([float(stream.readline()) for _ in range(nx)])
        nq = int(stream.readline())
        stream.readline()  # reference scale
        qs_squared = np.array([float(stream.readline()) for _ in range(nq)])
        if int(stream.readline()) != 0:
            raise ValueError('Expected member zero.')
        flavours = np.loadtxt(stream, max_rows=nx*nq).reshape((nx, nq, 14))
    # The table stores xf in PDG-flavour order -6 to +7.
    iu = 2 + 6
    q2 = scale**2
    if not qs_squared[0] <= q2 <= qs_squared[-1]:
        raise ValueError('Scale outside the PDF grid.')
    low = min(np.searchsorted(qs_squared, q2, side='right') - 1, nq - 2)
    fraction = np.log(q2 / qs_squared[low]) / np.log(qs_squared[low+1] / qs_squared[low])
    x_up = (1-fraction)*flavours[:,low,iu] + fraction*flavours[:,low+1,iu]
    logs = np.log(xs)

    def up(x):
        lower = min(max(np.searchsorted(xs, x, side='right') - 2, 0), nx - 4)
        if x < 0.1:
            target = math.log(x)
            nodes = logs[lower:lower+4]
        else:
            target = x
            nodes = xs[lower:lower+4]
        weights = []
        for j in range(4):
            weights.append(math.prod((target - nodes[k]) / (nodes[j] - nodes[k]) for k in range(4) if j != k))
        return np.dot(weights, x_up[lower:lower+4]) / x

    return xs, up


def prediction(grid_path, mass, beam_energy, coupling):
    # Symmetric proton beams and on-shell 2→1: H_T/2 = m_D/2.
    scale = mass / 2
    s = beam_energy**2
    tau = mass**2 / s
    nodes, up = read_up_distribution(grid_path, scale)
    # y = ln(x/sqrt(tau)); dx/x = dy. Split quadrature at interpolation boundaries.
    root_tau = math.sqrt(tau)
    rapidity_max = -math.log(root_tau)
    breaks = sorted({abs(math.log(x/root_tau)) for x in [*nodes, 0.1] if tau < x < 1})
    half_luminosity, numerical_error = quad(lambda y: up(root_tau*math.exp(y)) * up(root_tau*math.exp(-y)),
                                            0, rapidity_max, points=breaks, limit=250, epsrel=2e-10)
    normalization = 2*math.pi*coupling**2 / s * 6/9 * 0.3893793721e9
    return dict(mass_GeV=mass, pp_energy_GeV=beam_energy, lambda_L=coupling, scale_GeV=scale,
                source='arXiv:0909.2666, Eq. (3.4); N_D=6 and N_c=3', pdf=grid_path.name,
                pdf_sha256=hashlib.sha256(grid_path.read_bytes()).hexdigest(),
                uu_luminosity=2*half_luminosity, sigma_pb=2*half_luminosity*normalization,
                quadrature_error_pb=2*numerical_error*normalization)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('grid', type=Path)
    cli.add_argument('--mass', type=float, default=1000.)
    cli.add_argument('--pp-energy', type=float, default=13600.)
    cli.add_argument('--lambda-left', type=float, default=0.1)
    args = cli.parse_args()
    print(json.dumps(prediction(args.grid, args.mass, args.pp_energy, args.lambda_left), indent=2))
