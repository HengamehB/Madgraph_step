"""Gauge tensors times explicit Lorentz contractions for vector monomials.

A numerical polynomial is used only to establish identities/rank. Emission
retains abstract Lorentz indices so no fixed spacetime component enters UFOs.
"""
from functools import lru_cache
from itertools import product,permutations
import numpy as np

from component_invariants import component_invariants, ComponentResourceLimit
from generic_scalar_compiler import coordinates,add,mul,conjugate
from lagrangian_engine import make_letters
from compiler_limits import component_limits

BASIS_VERSION='vector-gauge-tensors-lorentz-pairings-v1'


def pairings(slots):
    if not slots:return ((),)
    first=slots[0];out=[]
    for position in range(1,len(slots)):
        rest=slots[1:position]+slots[position+1:]
        for tail in pairings(rest):out.append(((first,slots[position]),)+tail)
    return tuple(out)


def _reduce(rows,expected=None,hermitian=False):
    keys=sorted(set().union(*(r.keys() for r in rows)))
    matrix=np.array([[r.get(k,0) for k in keys] for r in rows],complex)
    if hermitian:matrix=np.concatenate((matrix.real,matrix.imag),axis=1)
    transform=np.eye(len(rows),dtype=matrix.dtype);pivot=0
    for col in range(matrix.shape[1]):
        candidates=np.flatnonzero(abs(matrix[pivot:,col])>1e-9)
        if not len(candidates):continue
        selected=pivot+int(candidates[np.argmax(abs(matrix[pivot+candidates,col]))])
        matrix[[pivot,selected]]=matrix[[selected,pivot]];transform[[pivot,selected]]=transform[[selected,pivot]]
        value=matrix[pivot,col];matrix[pivot]/=value;transform[pivot]/=value
        for i in range(len(matrix)):
            if i==pivot:continue
            value=matrix[i,col];matrix[i]-=value*matrix[pivot];transform[i]-=value*transform[pivot]
        pivot+=1
        if pivot==len(matrix):break
    if expected is not None and pivot!=expected:
        raise ValueError(f'Lorentz/gauge polynomial rank {pivot} differs from catalogue multiplicity {expected}')
    result=matrix[:pivot]
    if hermitian:result=result[:,:len(keys)]+1j*result[:,len(keys):]
    return tuple({k:complex(z) for k,z in zip(keys,r) if abs(z)>1e-9} for r in result),transform[:pivot]


def _lorentz_terms(contraction):
    if contraction=='epsilon':
        return [(p,(-1)**sum(p[i]>p[j] for i in range(4) for j in range(i+1,4))) for p in permutations(range(4))]
    terms=[]
    size=2*len(contraction)
    for values in product(range(4),repeat=len(contraction)):
        indices=[None]*size;factor=1
        for (a,b),mu in zip(contraction,values):
            indices[a]=indices[b]=mu;factor*=1 if mu==0 else -1
        terms.append((tuple(indices),factor))
    return terms


@lru_cache(maxsize=128)
def _basis_cached(fields,factors,multiplicity,selfconjugate,limits):
    c=coordinates(fields);letters={l.name:l for l in make_letters(fields)}
    tokens=tuple(name for name,count in factors for _ in range(count))
    derivative_slots=tuple(i for i,n in enumerate(tokens) if n.startswith('D('))
    names=tuple(n[2:-1] if n.startswith('D(') else n for n in tokens)
    vectors=[i for i,n in enumerate(names) if n not in ('H','H†')]
    if len(derivative_slots)>1 or len(vectors)+len(derivative_slots) not in (2,4):
        raise ValueError('This vector invariant needs two or four Lorentz slots and at most one derivative')
    descriptors=[dict(name=f'slot{i}',count=1,lorentz=(0,0),weak=letters[name].weak,
                      colour=(0,0),nf=1,fermion=False,hypercharge=str(letters[name].charge)) for i,name in enumerate(names)]
    raw=component_invariants(descriptors,limits=limits)
    weak=[]
    for basis in raw['basis']:
        weak.append({tuple(raw['components'][f'slot{i}'][next(p['components'][0] for p in term['factors'] if p['letter']==f'slot{i}')-1]['indices']['weak']-1
                           for i in range(len(names))):complex(term['coefficient']) for term in basis['terms']})
    weak,_=_reduce(weak)
    n_lorentz=len(vectors)+len(derivative_slots)
    contractions=list(pairings(tuple(range(n_lorentz))))
    if n_lorentz==4:contractions.append('epsilon')
    candidates=[];polys=[]
    encoded_bars=[0]*(20*len(c.labels))
    for i in range(len(c.labels)):
        for mu in range(20):encoded_bars[20*i+mu]=20*c.bars[i]+mu
    for tensor in weak:
        for contraction in contractions:
            polynomial={}
            for indices,weakvalue in tensor.items():
                pieces=[c.components[name][index] for name,index in zip(names,indices)]
                for lorentz,metric in _lorentz_terms(contraction):
                    term={():weakvalue*metric}
                    for slot,p in enumerate(pieces):
                        mu=lorentz[vectors.index(slot)] if slot in vectors else None
                        derivative=lorentz[len(vectors)+derivative_slots.index(slot)]+1 if slot in derivative_slots else 0
                        mapped={tuple(20*i+4*derivative+mu if mu is not None else 0 for i in m):z for m,z in p.items()}
                        term=mul(term,mapped)
                    polynomial=add(polynomial,term)
                    if len(polynomial)>limits.max_power_monomials:
                        raise ComponentResourceLimit('Vector physical polynomial exceeds max_power_monomials')
            if polynomial:
                candidates.append((tensor,contraction));polys.append(polynomial)
    # Keep explicit HC operations in the source representation, with the
    # corresponding polynomial conjugation used to establish independence.
    hermitian_candidates=[];hermitian_polys=[]
    for index,p in enumerate(polys):
        if selfconjugate:
            hermitian_candidates.extend(((index,1),(index,1j)))
            hermitian_polys.extend((add(p,conjugate(p,encoded_bars)),
                                    {m:1j*z for m,z in add(p,conjugate(p,encoded_bars),-1).items()}))
        else:
            hermitian_candidates.append((index,0));hermitian_polys.append(p)
    basis,transform=_reduce(hermitian_polys,multiplicity,selfconjugate)
    return dict(names=names,vectors=tuple(vectors),derivative_slots=derivative_slots,stride=20,candidates=tuple(candidates),
                hermitian_candidates=tuple(hermitian_candidates),transform=transform,basis=basis)


def invariant_basis(row,fields,limits=None):
    limits=component_limits(limits) if isinstance(limits,dict) or limits is None else limits
    return _basis_cached(tuple(fields),tuple((p['name'],p['count']) for p in row['factors']),
                         row['multiplicity'],row['field_content']==row['conjugate_content'],limits)
