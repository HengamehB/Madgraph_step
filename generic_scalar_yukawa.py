"""Component-verified same-chirality SM-lepton/scalar Yukawa contractions.

The displayed basis is defined by four-component scalar bilinears, not by an
unstated conversion between dotted spinor conventions.  A full Lorentz/weak/
flavour invariant construction fixes exchange symmetry and multiplicity. Its
spinor epsilon is then removed with an explicit reconstruction check.
"""
from functools import lru_cache
from collections import Counter
import hashlib
import json
import numpy as np

from component_invariants import component_invariants
from lagrangian_engine import make_letters

BASIS_VERSION='lepton-scalar-bilinears-rref-v1'


def classify(row,fields):
    if row['derivatives']:
        return None
    counts=Counter({p['name']:p['count'] for p in row['factors']})
    for fermion in ('L','ec','ec†'):
        if counts[fermion]!=2 or sum(counts.values())!=3:
            continue
        scalar=next((n for n in counts if n!=fermion),None)
        field=next((f for f in fields if scalar in (f.name,f.name+'†')),None)
        if field is not None and field.spin=='scalar' and field.su3==(0,0):
            return fermion,scalar
    return None


def _rref(polynomials, expected):
    keys=sorted(set().union(*(p.keys() for p in polynomials)))
    matrix=np.array([[p.get(k,0) for k in keys] for p in polynomials],complex)
    pivot=0
    for column in range(len(keys)):
        candidates=np.flatnonzero(abs(matrix[pivot:,column])>1e-9)
        if not len(candidates):continue
        index=pivot+int(candidates[np.argmax(abs(matrix[pivot+candidates,column]))])
        matrix[[pivot,index]]=matrix[[index,pivot]];matrix[pivot]/=matrix[pivot,column]
        for i in range(len(matrix)):
            if i!=pivot:matrix[i]-=matrix[i,column]*matrix[pivot]
        pivot+=1
        if pivot==len(matrix):break
    if pivot!=expected:raise ValueError(f'Spinor-bilinear basis rank {pivot} differs from catalogue multiplicity {expected}')
    return tuple({k:complex(0. if abs(z.real)<1e-11 else z.real,0. if abs(z.imag)<1e-11 else z.imag)
                  for k,z in zip(keys,r) if abs(z)>1e-9} for r in matrix[:pivot])


@lru_cache(maxsize=128)
def _basis_cached(fields,factors,multiplicity,limits):
    from generic_scalar_compiler import coordinates
    coords=coordinates(fields);letters={l.name:l for l in make_letters(fields)}
    descriptors=[]
    for name,count in factors:
        l=letters[name]
        descriptors.append(dict(name=name,count=count,lorentz=l.lorentz,weak=l.weak,
                                colour=l.colour,nf=l.nf,fermion=l.fermion,hypercharge=str(l.charge)))
    result=component_invariants(descriptors,expected_multiplicity=multiplicity,limits=limits)
    fermion=next(n for n,c in factors if letters[n].fermion)
    scalar=next(n for n,c in factors if not letters[n].fermion)
    spinor_index='lorentz_left' if letters[fermion].lorentz==(1,0) else 'lorentz_right'
    reduced=[]
    for invariant in result['basis']:
        raw={}
        for term in invariant['terms']:
            fs={p['letter']:p['components'] for p in term['factors']}
            a,b=[result['components'][fermion][i-1]['indices'] for i in fs[fermion]]
            if (a[spinor_index],b[spinor_index])!=(1,2):
                raise ValueError('Unexpected Lorentz components in scalar spinor contraction')
            ia=(a['weak'],a['flavour']);ib=(b['weak'],b['flavour'])
            weak=result['components'][scalar][fs[scalar][0]-1]['indices']['weak']-1
            for monomial,z in coords.components[scalar][weak].items():
                if len(monomial)!=1:raise ValueError('Yukawa scalar is not linear in physical coordinates')
                key=ia+ib+(monomial[0],)
                raw[key]=raw.get(key,0)+term['coefficient']*z
        polynomial={}
        for key,value in raw.items():
            a,b=key[:2],key[2:4];swapped=b+a+key[4:]
            if not np.isclose(value,raw.get(swapped,0),atol=1e-9,rtol=1e-9):
                raise ValueError('Scalar Weyl bilinear exchange identity failed')
            if a<=b:
                polynomial[key]=value/(2 if a==b else 1)
        reconstructed={}
        for key,value in polynomial.items():
            a,b=key[:2],key[2:4]
            reconstructed[key]=value*(2 if a==b else 1)
            if a!=b:reconstructed[b+a+key[4:]]=value
        if any(abs(raw.get(k,0)-reconstructed.get(k,0))>1e-8 for k in set(raw)|set(reconstructed)):
            raise ValueError('Lorentz epsilon reconstruction residual failed')
        reduced.append(polynomial)
    return _rref(reduced,multiplicity)


def invariant_basis(row,fields,limits=None):
    from compiler_limits import component_limits
    if isinstance(limits,dict) or limits is None:limits=component_limits(limits)
    if classify(row,fields) is None:raise ValueError('Not a same-chirality lepton/scalar Yukawa row')
    return _basis_cached(tuple(fields),tuple((p['name'],p['count']) for p in row['factors']),row['multiplicity'],limits)


def basis_labels(row,fields,limits=None):
    from generic_scalar_compiler import coordinates
    fermion,_=classify(row,fields);c=coordinates(tuple(fields))
    return tuple(' + '.join(f'({z.real:g}{z.imag:+g}j)*B({fermion}[{a},{f}],{fermion}[{b},{g}])*{c.labels[k]}'
                             for (a,f,b,g,k),z in sorted(p.items())) for p in invariant_basis(row,fields,limits))


def basis_fingerprint(row,fields,limits=None):
    data=dict(version=BASIS_VERSION,basis=[[(list(k),round(z.real,10),round(z.imag,10))
                for k,z in sorted(p.items())] for p in invariant_basis(row,fields,limits)])
    return hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()


def bilinear(fermion,a,f,b,g):
    """The convention defining B in the displayed coefficient basis."""
    if fermion=='L':
        return f'HC[ProjP[s1,s2]*LLbar[s1,{a},{f}].CC[LL][s2,{b},{g}]]'
    expression=f'ProjM[s1,s2]*lRbar[s1,{f}].CC[lR][s2,{g}]'
    return f'HC[{expression}]' if fermion=='ec†' else expression
