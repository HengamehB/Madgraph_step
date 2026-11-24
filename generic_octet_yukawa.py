"""Colour-octet weak-doublet Yukawa with a general up-type flavour matrix."""

from fractions import Fraction


KIND = 'octet_up_yukawa'
LABEL = 'Y_u8'
NORMALIZATION = (
    'Rows label right-handed up-quark generation i, columns label left-handed quark '
    'doublet generation j in the up-diagonal basis. The coefficient multiplies '
    'ubar_R[i] T^A (Splus[A] sum_k V[j,k] d_L[k] - Szero[A] u_L[j]), plus HC. '
    'The generators obey Tr(T^A T^B)=delta^AB/2.'
)


def classify(row, fields):
    if row['derivatives']:
        return None
    counts = {item['name']: item['count'] for item in row['factors']}
    for field in fields:
        if (field.spin == 'scalar' and field.kind == 'complex' and
                (field.su3, field.su2, field.hypercharge) == ((1, 1), 2, Fraction(1, 2)) and
                counts == {field.name: 1, 'Q': 1, 'uc': 1}):
            return KIND
    return None


def term(scalar, symbols, i, j):
    charged = symbols[scalar, 0].replace('{index}', 'c3')
    neutral = symbols[scalar, 1].replace('{index}', 'c3')
    quark_bar = f'uqbar[s1,{i},c1]'
    # The anti-fundamental index sits on ubar; preserve its order in T^A.
    common = 'T[c3,c1,c2]*ProjM[s1,s2]'
    down = '+'.join(f'CKM[{j},{k}]*{quark_bar}.dq[s2,{k},c2]' for k in range(1, 4))
    return f'{common}*(({charged})*({down})-({neutral})*{quark_bar}.uq[s2,{j},c2])'
