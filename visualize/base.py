import pathlib

from abc import ABC, abstractmethod

class BaseVisualizer(ABC):
    def __init__(self, **params):
        self.params = params

    @abstractmethod
    def visualize(self, model, tokenizer, dataloader, save_dir, device, logger):
        raise NotImplementedError()