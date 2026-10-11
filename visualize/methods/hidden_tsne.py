
class HiddenTSNEVisualizer:
    def __init__(self, layer=-1, max_samples=100, perplexity=10):
        self.layer = layer
        self.max_samples = max_samples
        self.perplexity = perplexity