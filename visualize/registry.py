from .methods.attention_heatmap import AttentionHeatmapVisualizer

from registry.base import Registry

canonical_map = {
    "attentionheatmap": AttentionHeatmapVisualizer,
}

lookup_map = {
    "attentionheatmap": "attentionheatmap",
    "attention_heatmap": "attentionheatmap",
}

VISUALIZER_REGISTRY = Registry(
    category="visualizer",
    canonical_map=canonical_map,
    lookup_map=lookup_map,
)

SUPPORTED_VISUALIZERS = set(VISUALIZER_REGISTRY.keys())