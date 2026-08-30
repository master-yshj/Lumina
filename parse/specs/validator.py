from dataclasses import dataclass
from collections.abc import Callable

@dataclass(frozen=True)
class ValidatorSpec:
    name: str
    phase: str
    priority: int
    validator: Callable
    applies_to: Callable | None
    source: str

    def __post_init__(self):
        if not isinstance(self.name, str):
            raise TypeError(f"name: {self.name} 必须为str类型的喵! 现在的type: {type(self.name)}")
        if not self.name.strip():
            raise ValueError(f"name 不能为空喵!")
        if not isinstance(self.phase, str):
            raise TypeError(f"phase: {self.phase} 必须为str类型的喵! 现在的type: {type(self.phase)}")
        if not self.phase.strip():
            raise ValueError(f"phase 不能为空喵!")
        if not type(self.priority) is int:
            raise TypeError(f"priority: {self.priority} 必须为int类型的喵! 现在的type: {type(self.priority)}")
        if not callable(self.validator):
            raise TypeError(f"validator: {self.validator} 必须为可调用对象喵! 现在的type: {type(self.validator)}")
        if not self.applies_to is None and not callable(self.applies_to):
            raise TypeError(f"applies_to: {self.applies_to} 必须为None或者可调用对象喵! 现在的type: {type(self.applies_to)}")
        if not isinstance(self.source, str):
            raise TypeError(f"source: {self.source} 必须为str类型的喵! 现在的type: {type(self.source)}")
        if not self.source.strip():
            raise ValueError(f"source 不能为空喵!")

    def should_run(self, context, config):
        if self.applies_to is None:
            return True
        return self.applies_to(context, config)

    def run(self, context, config) -> None:
        self.validator(context, config)