"""Catalogue-normalized F+ V V tensors for colour-singlet vector multiplets.

The declared Lorentz convention is F+ = (F + i dual(F))/2, with
dual(F)_{mu nu} = epsilon_{mu nu rho sigma} F^{rho sigma}/2.
Both terms are retained in the emitted source and the conjugate catalogue
operator is supplied by HC. Weak tensors and their Bose-symmetrized physical
polynomials are rank checked against the catalogue, before any user coefficient
is applied. The real Cartesian SM adjoint is explicitly intertwined with the
catalogue's highest-weight basis.
"""
from functools import lru_cache
from itertools import product, permutations
import math
import numpy as np

from component_invariants import component_invariants, ComponentResourceLimit, _su2_conjugation
from compiler_limits import component_limits
from generic_scalar_compiler import coordinates, add, mul
from generic_vector_invariants import _reduce
from generic_fermion_compiler import weak_generators
from lagrangian_engine import make_letters

BASIS_VERSION = 'vector-selfdual-field-strength-weak-rref-v1'


def classify(row, fields):
    if row['derivatives']:
        return None
    names = tuple(p['name'] for p in row['factors'] for _ in range(p['count']))
    strengths = [n for n in names if n in ('B+', 'B-', 'W+', 'W-')]
    vectors = {n for f in fields for n in (f.name, f.name+'†')}
    if len(names) == 3 and len(strengths) == 1 and all(n in vectors or n == strengths[0] for n in names):
        return strengths[0], tuple(n for n in names if n != strengths[0])
    return None


def adjoint_intertwiner():
    """S maps real Cartesian adjoint components to C-real spherical ones."""
    s = np.array([[-1j, -1, 0], [0, 0, math.sqrt(2)*1j], [1j, -1, 0]], complex)/math.sqrt(2)
    epsilon = np.zeros((3, 3, 3))
    for indices in permutations(range(3)):
        epsilon[indices] = (-1)**sum(indices[a] > indices[b] for a in range(3) for b in range(a+1,3))
    for j, cartesian in zip(weak_generators(3), -1j*epsilon):
        if not np.allclose(j@s, s@cartesian, atol=1e-14):
            raise ValueError('SM adjoint/highest-weight intertwiner failed its generator identity')
    if not np.allclose(s@s.conj().T, np.eye(3), atol=1e-14):
        raise ValueError('SM adjoint intertwiner is not unitary')
    if not np.allclose(_su2_conjugation(2).toarray()@s.conj(), s, atol=1e-14):
        raise ValueError('SM adjoint intertwiner does not satisfy the catalogue reality convention')
    return s


def _field_strength_polynomial(base, mu, nu, chirality):
    """Formal lower-index F+ with six independent real F_{mu nu} symbols."""
    if mu == nu:
        return {}
    pairs = tuple((a, b) for a in range(4) for b in range(a+1,4))
    pair = tuple(sorted((mu, nu)))
    result = {(base+pairs.index(pair),): .5 if mu < nu else -.5}
    metric = (1, -1, -1, -1)
    for rho, sigma in pairs:
        indices = (mu, nu, rho, sigma)
        if len(set(indices)) != 4:
            continue
        epsilon = (-1)**sum(indices[a] > indices[b] for a in range(4) for b in range(a+1,4))
        # epsilon_lower=-epsilon_upper; the two antisymmetric F terms cancel
        # the 1/2 in dual(F), leaving the outer self-dual factor 1/2.
        result[(base+pairs.index((rho, sigma)),)] = -.5j*chirality*epsilon*metric[rho]*metric[sigma]
    return result


@lru_cache(maxsize=128)
def _basis_cached(fields, strength, names, multiplicity, limits):
    c = coordinates(fields)
    letters = {l.name:l for l in make_letters(fields)}
    gauge_names = (strength,)+names
    descriptors = [dict(name=f'slot{i}', count=1, lorentz=(0,0), weak=letters[name].weak,
                        colour=(0,0), nf=1, fermion=False) for i,name in enumerate(gauge_names)]
    raw = component_invariants(descriptors, limits=limits)
    mapping = adjoint_intertwiner() if strength.startswith('W') else np.ones((1,1), complex)
    tensors = []
    polynomials = []
    chirality = 1 if strength.endswith('+') else -1
    for basis in raw['basis']:
        tensor = {}
        for term in basis['terms']:
            indices = tuple(raw['components'][f'slot{i}'][next(p['components'][0] for p in term['factors'] if p['letter']==f'slot{i}')-1]['indices']['weak']-1 for i in range(3))
            k,i,j = indices
            for a,value in enumerate(mapping[k]):
                if value:
                    key = (a,i,j)
                    tensor[key] = tensor.get(key,0)+complex(term['coefficient'])*value
        polynomial = {}
        for (a,i,j),z in tensor.items():
            for mu,nu in product(range(4),repeat=2):
                left = {tuple(4*k+mu for k in m):v for m,v in c.components[names[0]][i].items()}
                right = {tuple(4*k+nu for k in m):v for m,v in c.components[names[1]][j].items()}
                fs = _field_strength_polynomial(4*len(c.labels)+6*a,mu,nu,chirality)
                polynomial = add(polynomial,{m:z*v for m,v in mul(fs,mul(left,right)).items()})
                if len(polynomial)>limits.max_power_monomials:
                    raise ComponentResourceLimit('Vector field-strength polynomial exceeds max_power_monomials')
        tensors.append(tensor)
        polynomials.append(polynomial)
        if sum(map(len,polynomials))>limits.max_output_terms:
            raise ComponentResourceLimit('Vector field-strength basis exceeds max_output_terms')
    basis,transform = _reduce(polynomials,multiplicity)
    return dict(strength=strength,names=names,tensors=tuple(tensors),basis=basis,transform=transform,
                coordinate_count=len(c.labels),chirality=chirality)


def invariant_basis(row,fields,limits=None):
    match = classify(row,fields)
    if match is None:
        raise ValueError('Not a colour-singlet-vector FVV operator')
    limits = component_limits(limits) if limits is None or isinstance(limits,dict) else limits
    return _basis_cached(tuple(fields),*match,row['multiplicity'],limits)


def basis_labels(row,fields,limits=None):
    result = invariant_basis(row,fields,limits)
    c = coordinates(tuple(fields));offset=4*len(c.labels)
    pairs = tuple((a,b) for a in range(4) for b in range(a+1,4))
    def label(i):
        if i<offset:
            return f'{c.labels[i//4]}[{i%4}]'
        a,pair=divmod(i-offset,6);mu,nu=pairs[pair]
        return f'{result["strength"][0]}F[{a+1},{mu},{nu}]'
    return tuple(' + '.join(f'({z.real:g}{z.imag:+g}j)*'+'*'.join(map(label,m)) for m,z in sorted(p.items())) for p in result['basis'])


def source_candidates(basis, literal, gauge_component):
    """Emit the same tensor/Lorentz candidates used by the verified rank test."""
    out=[];strength=basis['strength'];names=basis['names']
    for tensor in basis['tensors']:
        pieces=[]
        for (a,i,j),z in sorted(tensor.items()):
            def fs(mu,nu):
                return f'FS[Wi,{mu},{nu},{a+1}]' if strength.startswith('W') else f'FS[B,{mu},{nu}]'
            dual=f'Eps[mu,nu,rho,sigma]*{fs("rho","sigma")}'
            sign='+' if basis['chirality']==1 else '-'
            contraction=f'({fs("mu","nu")} {sign} I/2*{dual})/2'
            pieces.append('('+literal(z)+')*('+contraction+')*'+gauge_component(names[0],i,'mu')+'*'+gauge_component(names[1],j,'nu'))
        out.append('('+' + '.join(pieces)+')')
    return tuple(out)
