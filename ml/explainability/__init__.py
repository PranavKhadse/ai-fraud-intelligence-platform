"""
Phase 7: Explainability & Reason Codes Package.

Provides production TreeSHAP feature attributions, plain-English reason code generation,
and mathematically complete margin waterfalls for fraud risk intelligence.
"""

from ml.explainability.schemas import (
    AttributionDirection,
    ReasonSource,
    ReasonSeverity,
    FeatureAttribution,
    ReasonCodeDetail,
    WaterfallStep,
    TransactionExplanation,
)
from ml.explainability.config import (
    FEATURE_REGISTRY,
    FEATURE_METADATA_REGISTRY,
    get_feature_metadata,
)
from ml.explainability.explainer import (
    TreeSHAPExplainer,
    DEFAULT_MODEL_PATH,
    DEFAULT_PREPROCESSOR_PATH,
)
from ml.explainability.reason_codes import (
    ReasonCodeGenerator,
)

__all__ = [
    "AttributionDirection",
    "ReasonSource",
    "ReasonSeverity",
    "FeatureAttribution",
    "ReasonCodeDetail",
    "WaterfallStep",
    "TransactionExplanation",
    "FEATURE_REGISTRY",
    "FEATURE_METADATA_REGISTRY",
    "get_feature_metadata",
    "TreeSHAPExplainer",
    "ReasonCodeGenerator",
    "DEFAULT_MODEL_PATH",
    "DEFAULT_PREPROCESSOR_PATH",
]
