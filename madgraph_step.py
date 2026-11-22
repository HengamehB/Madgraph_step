"""Python API for the portable Lagrangian → FeynRules → MadGraph workflow."""
from lagrangian_maker import make_lagrangian
from model_builder import ModelDefinition
from portable_runtime import check_environment
from feynrules_export import (
    ModelResolutionError, required_inputs, prepare_model, scalar_portal_example,
)
from madgraph_runner import (
    MadGraphError, DEFAULT_COLLIDER, list_production_channels,
    compute_decays, compute_cross_sections,
)
from pipeline_config import load_config
from pipeline import Pipeline, load_model_definition
from cascade_ranking import rank_final_states
from tex_inputs import read_inputs, read_tex, setup as setup_inputs, TexInputError

__all__ = [
    "make_lagrangian", "check_environment", "required_inputs", "prepare_model",
    "scalar_portal_example", "list_production_channels", "compute_decays",
    "compute_cross_sections", "DEFAULT_COLLIDER", "ModelResolutionError",
    "MadGraphError",
    "Pipeline", "load_config", "rank_final_states", "ModelDefinition", "load_model_definition",
    "read_inputs", "read_tex", "setup_inputs", "TexInputError",
]
