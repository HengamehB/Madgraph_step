"""Gauge-verified SM flavour currents coupled to colour-singlet vectors."""
from functools import lru_cache
from collections import Counter
import math
import numpy as np
from component_invariants import component_invariants,_su2_conjugation
from lagrangian_engine import make_letters
from generic_scalar_compiler import coordinates
from compiler_limits import component_limits

BASIS_VERSION='vector-sm-hermitian-currents-v1'


def classify(row,fields):
    if row['derivatives'] or row['multiplicity']!=9:return None
    counts=Counter({p['name']:p['count'] for p in row['factors']})
    for species in ('Q','L','uc','dc','ec'):
        for f in fields:
            if f.hypercharge==0 and f.su2 in ((1,3) if species in ('Q','L') else (1,)):
                for token in (f.name,f.name+'†'):
                    if counts=={species:1,species+'†':1,token:1}:return species,token
    return None


@lru_cache(maxsize=64)
def _gauge(fields,species,token,limits):
    c=coordinates(fields);letters={l.name:l for l in make_letters(fields)}
    weak=letters[species].weak
    descriptors=[dict(name='left',count=1,lorentz=(0,0),weak=weak,colour=(0,0),nf=1,fermion=False),
                 dict(name='right',count=1,lorentz=(0,0),weak=weak,colour=(0,0),nf=1,fermion=False),
                 dict(name='vector',count=1,lorentz=(0,0),weak=letters[token].weak,colour=(0,0),nf=1,fermion=False)]
    result=component_invariants(descriptors,expected_multiplicity=1,limits=limits)
    p={};conj=_su2_conjugation(weak).toarray()
    for term in result['basis'][0]['terms']:
        indices={part['letter']:result['components'][part['letter']][part['components'][0]-1]['indices']['weak']-1 for part in term['factors']}
        a,b,k=indices['left'],indices['right'],indices['vector']
        actual=weak-a;factor=conj[a,actual]
        for m,z in c.components[token][k].items():p[(actual+1,b+1,m[0])]=term['coefficient']*factor*z
    field=next(f for f in fields if token in (f.name,f.name+'†'))
    if field.kind=='real':
        hc={(b,a,c.bars[i]):z.conjugate() for (a,b,i),z in p.items()}
        candidate={k:p.get(k,0)+hc.get(k,0) for k in set(p)|set(hc)}
        if max(map(abs,candidate.values()),default=0)<1e-9:
            candidate={k:1j*(p.get(k,0)-hc.get(k,0)) for k in set(p)|set(hc)}
        p={k:z for k,z in candidate.items() if abs(z)>1e-9}
    first=next(z for k,z in sorted(p.items()) if abs(z)>1e-9)
    if field.kind=='real' and abs(first.imag)>1e-9:
        # A real multiplier preserves Hermiticity; the displayed leading
        # coefficient can be i when the spherical-field phase requires it.
        scale=abs(first)
    else:scale=first
    return {k:complex(z/scale) for k,z in p.items()}


def gauge_basis(row,fields,limits=None):
    match=classify(row,fields)
    if match is None:raise ValueError('Not a supported SM flavour vector current')
    return _gauge(tuple(fields),*match,component_limits(limits))


def flavour_basis(row):
    if row['field_content']!=row['conjugate_content']:
        return tuple(((f,g,1),) for f in range(1,4) for g in range(1,4))
    return tuple([((f,f,1),) for f in range(1,4)]+
                 [((f,g,1),(g,f,1)) for f in range(1,4) for g in range(f+1,4)]+
                 [((f,g,1j),(g,f,-1j)) for f in range(1,4) for g in range(f+1,4)])


def basis_labels(row,fields,limits=None):
    c=coordinates(tuple(fields));species,_=classify(row,fields);gauge=gauge_basis(row,fields,limits)
    return tuple(' + '.join(f'({(z*phase).real:g}{(z*phase).imag:+g}j)*J({species}[{a},{f}],{species}[{b},{g}])*{c.labels[i]}'
                           for f,g,phase in flavour for (a,b,i),z in sorted(gauge.items())) for flavour in flavour_basis(row))


def current(species,a,f,b,g):
    if species in ('Q','L'):
        field='QL' if species=='Q' else 'LL';colour=',cc' if species=='Q' else ''
        return f'Ga[mu,s1,s2]*ProjM[s2,s3]*{field}bar[s1,{a},{f}{colour}].{field}[s3,{b},{g}{colour}]'
    field={'uc':'uR','dc':'dR','ec':'lR'}[species];colour=',cc' if species!='ec' else ''
    # A current of a charge-conjugate left-handed Weyl field equals minus
    # the right-handed current with the flavour indices exchanged.
    return f'-Ga[mu,s1,s2]*ProjP[s2,s3]*{field}bar[s1,{g}{colour}].{field}[s3,{f}{colour}]'
