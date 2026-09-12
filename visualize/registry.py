from .methods.attention_heatmap import AttentionHeatmapVisualizer
from .methods.loss_landscape import LossLandscapeVisualizer

from registry.base import Registry

canonical_map = {
    "attentionheatmap": AttentionHeatmapVisualizer,
    "losslandscape": LossLandscapeVisualizer,
}

lookup_map = {
    "attentionheatmap": "attentionheatmap",
    "attention_heatmap": "attentionheatmap",
    "losslandscape": "losslandscape",
    "loss_landscape": "losslandscape",
}

VISUALIZER_REGISTRY = Registry(
    category="visualizer",
    canonical_map=canonical_map,
    lookup_map=lookup_map,
)

SUPPORTED_VISUALIZERS = set(VISUALIZER_REGISTRY.keys())