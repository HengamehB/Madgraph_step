"""Validated construction budgets, independent of numerical physics inputs."""
from collections.abc import Mapping
from dataclasses import fields

from component_invariants import Limits
from workflow_errors import PhysicalInputError


DEFAULTS = {field.name: field.default for field in fields(Limits)}
# Verified with the bundled FeynRules 2.3.49 and local Wolfram kernel:
# FeynRulesPackage.m sets recursion to2500; the kernel iteration default is4096.
DEFAULTS.update(max_weak_generator_entries=100000, max_colour_candidate_networks=10000,
                max_wolfram_recursion=2500, max_wolfram_iterations=4096)


def normalize_limits(values=None):
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise PhysicalInputError('compiler_limits must be a dictionary of positive integer construction budgets')
    unknown = set(values) - set(DEFAULTS)
    if unknown:
        raise PhysicalInputError('Unknown compiler_limits: ' + ', '.join(sorted(map(str, unknown))))
    result = {}
    for key, value in values.items():
        if type(value) is not int or value < 1:
            raise PhysicalInputError(f'compiler_limits.{key} must be a positive integer')
        if key in ('max_wolfram_recursion', 'max_wolfram_iterations') and value < 20:
            raise PhysicalInputError(f'compiler_limits.{key} must be at least 20, the Wolfram kernel minimum')
        result[key] = value
    return result


def component_limits(values=None):
    values = normalize_limits(values)
    return Limits(**{key: value for key, value in values.items() if key in Limits.__dataclass_fields__})
