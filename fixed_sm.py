"""A declared SM electroweak vacuum, without potential minimization.

This prescription fixes the Higgs state as well as the expansion point. It
does not establish boundedness, a global minimum, or a vacuum lifetime.
Ordinary compiler checks still reject inconsistent quadratic forms/tadpoles.
"""
from collections.abc import Mapping
from copy import deepcopy
import math

import numpy as np


SCHEMA = 'fixed-sm-v1'
HIGGS_MASS = 125.0


def normalize_fixed_sm(inputs, fields):
    """Return independent inputs with a fixed 125-GeV, zero-new-VEV vacuum.

    No coefficient is changed, no counterterm is supplied, and no potential
    solver is invoked. Conflicting old vacuum options are errors,
    so a declared nonzero VEV cannot silently become a different theory.
    """
    from coefficient_defaults import SM_DEFAULTS
    from feynrules_export import ModelResolutionError, _number
    from sm_parameters import SM_EXTERNAL_INPUTS

    if not isinstance(inputs, Mapping):
        raise ModelResolutionError('Physical inputs must be a dictionary.')
    result = deepcopy(dict(inputs))
    if result.get('sm_scheme', 'fixed_sm') != 'fixed_sm':
        raise ModelResolutionError(
            'The package always uses the SM electroweak vacuum. Remove sm_scheme and any '
            'vacuum search or selection settings to migrate this input.')
    for key in ('vacuum', '_resolved_vacuum', '_vacuum_report'):
        if key in result:
            raise ModelResolutionError(
                f'{key} is incompatible with the fixed electroweak vacuum; remove old vacuum '
                'search/selection settings. This prescription assumes zero new-scalar VEVs without minimization.')
    for key in ('higgs_mass', 'observed_higgs_mass'):
        if key in result and _number(result[key], key, real=True) != HIGGS_MASS:
            raise ModelResolutionError(f'{key} must be 125 GeV in the fixed electroweak vacuum.')
    result.pop('observed_higgs_mass', None)
    result['higgs_mass'] = HIGGS_MASS
    result['sm_scheme'] = 'fixed_sm'

    scalar_names = {field.name for field in fields if field.spin == 'scalar'}
    vevs = result.get('vevs', {})
    if not isinstance(vevs, Mapping):
        raise ModelResolutionError('vevs must be a dictionary.')
    unknown = set(vevs) - scalar_names
    if unknown:
        raise ModelResolutionError('Unknown scalar VEV fields: ' + ', '.join(sorted(map(str, unknown))) + '.')
    for name, value in vevs.items():
        if _number(value, 'vevs.' + name, real=True) != 0:
            raise ModelResolutionError(
                f'vevs.{name} must be zero in the fixed electroweak vacuum.')
    result['vevs'] = {name: 0.0 for name in sorted(scalar_names)}

    sm = result.get('sm_inputs', {})
    if not isinstance(sm, Mapping):
        raise ModelResolutionError('sm_inputs must be a dictionary.')
    unknown = set(sm) - SM_EXTERNAL_INPUTS
    if unknown:
        raise ModelResolutionError(
            'Fixed SM electroweak parameters are derived; these are not independent sm_inputs: '
            + ', '.join(sorted(map(str, unknown))) + '.')
    sm = {**SM_DEFAULTS, **sm}
    for name, value in sm.items():
        sm[name] = _number(value, 'sm_inputs.' + name, real=True)
    if any(sm[name] <= 0 for name in ('Gf', 'MZ', 'aEWM1')):
        raise ModelResolutionError('Gf, MZ and aEWM1 must be positive.')
    # The bundled SM derives MW/sin(theta_W) from these three measurements.
    # Reject an imaginary electroweak solution before any external export.
    discriminant = sm['MZ'] ** 4 / 4 - math.pi * sm['MZ'] ** 2 / (math.sqrt(2) * sm['Gf'] * sm['aEWM1'])
    if discriminant < 0:
        raise ModelResolutionError('Gf, MZ and aEWM1 do not define real SM electroweak masses/couplings.')
    result['sm_inputs'] = sm
    return result


def _mixing_error():
    from feynrules_export import ModelResolutionError
    return ModelResolutionError(
        'The selected operators mix the Higgs with new scalars, conflicting with '
        'the required unmixed 125 GeV SM Higgs. Disable or consistently '
        'cancel those mixing terms. No coupling or mass is changed automatically.')


def _check_higgs_basis(basis):
    """The automatic scalar basis always lists the Higgs first."""
    from feynrules_export import ModelResolutionError
    transform = np.asarray(basis['transformation'], dtype=complex)
    if transform.ndim != 2 or transform.shape[0] != transform.shape[1] or not transform.size:
        raise ModelResolutionError('The fixed SM Higgs basis could not be verified.')
    # Check the original quadratic form when supplied. This also catches tiny
    # physical mixings that round away in the eigensolver's mass differences.
    for key in ('hessian', 'kinetic'):
        if key in basis:
            matrix = np.asarray(basis[key], dtype=complex)
            if np.any(matrix[0, 1:] != 0) or np.any(matrix[1:, 0] != 0):
                raise _mixing_error()
    if (not np.all(np.isfinite(transform)) or
            not math.isclose(abs(transform[0, 0]), 1., rel_tol=1e-12, abs_tol=1e-12) or
            np.any(abs(transform[0, 1:]) > 1e-12)):
        raise _mixing_error()


def check_fixed_sm_model(prepared):
    """Verify that a compiled automatic model actually obeys the assumption.

    This inspects scalar transformations only: colour, fermion and vector
    transformations do not identify a Higgs row and must not be conflated.
    Existing compiler consistency checks have already run at this point.
    """
    from feynrules_export import CompilerResolutionError, ModelResolutionError
    from generic_scalar_compiler import GenericScalarCompiler, _shift
    if prepared.manual:
        raise CompilerResolutionError('The compiler cannot verify the 125 GeV Higgs, its mixing '
            'and zero new-scalar VEVs in manual FeynRules source files. Use automatic model declarations.')
    compiler = prepared.compiler
    scalar_basis = getattr(compiler, 'scalar_basis', None)
    if isinstance(scalar_basis, Mapping) and 'transformation' in scalar_basis:
        _check_higgs_basis(scalar_basis)
    elif isinstance(prepared.basis, Mapping) and 'transformation' in prepared.basis:
        _check_higgs_basis(prepared.basis)
    elif isinstance(compiler, GenericScalarCompiler):
        # Generic weak scalar components include the Higgs; native colour
        # transformation matrices contain only new fields and bypass this.
        shifted = _shift(compiler.potential, compiler.vacuum, len(compiler.coordinates.labels))
        if any(len(monomial) == 2 and 0 in monomial and monomial != (0, 0) and value != 0
               for monomial, value in shifted.items()):
            raise _mixing_error()
        _check_higgs_basis({'transformation': compiler.transformation})
    # The specialized doublet compiler does not retain its original Hessian.
    # Its odd quartic creates Higgs mixing at zero new VEV, even after an odd
    # quadratic cancels the tadpole. Catch arbitrarily small coefficients too.
    for operator in prepared.operators:
        if operator.kind == 'doublet_potential':
            counts = dict(operator.factors)
            higgs_count = counts.get('H', 0) + counts.get('H†', 0)
            if higgs_count == 3 and any(value != 0 for value in operator.coefficients):
                raise _mixing_error()
    masses = prepared.masses
    mass = masses.get('H', masses.get('h'))
    if mass is None or not math.isclose(float(mass), HIGGS_MASS, rel_tol=1e-11, abs_tol=1e-9):
        raise ModelResolutionError(
            'The compiled Higgs mass is inconsistent with the assumed 125 GeV SM Higgs; '
            'review Higgs mixing/kinetic operators. The exported mass was not overwritten.')
    return prepared


def fixed_sm_report(inputs):
    """Small, explicit assumption record; this is never a stability proof."""
    gf = inputs.get('sm_inputs', {}).get('Gf', 1.16637e-5)
    vev = (math.sqrt(2.) * gf) ** -.5
    return dict(schema=SCHEMA, status='assumed_fixed_sm', analysis='fixed_sm',
                search_complete=False, potential_minimized=False,
                global_minimum=None, boundedness='not_evaluated',
                loop_stability='not_evaluated', vacuum_lifetime='not_evaluated',
                assumptions=dict(higgs_mass_gev=HIGGS_MASS, higgs_vev_gev=vev,
                                 scalar_vevs=dict(inputs.get('vevs', {})), higgs_mixing=False,
                                 lambda_h=HIGGS_MASS ** 2 / (2 * vev ** 2),
                                 mu_h_squared=HIGGS_MASS ** 2 / 2),
                warnings=['The SM electroweak vacuum and zero new-scalar VEVs are assumed; '
                          'the scalar potential is not minimized. Global stability, '
                          'boundedness and vacuum lifetime are not established.'])
