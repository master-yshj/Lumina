class Metric:
    def __init__(self):
        pass

    def __call__(self, **context):
        self.update(**context)

    def process(self, **context):
        raise NotImplementedError()

    def update(self, **context):
        raise NotImplementedError()

    def compute(self):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    def output_fields(self):
        raise NotImplementedError()

    def distributed_state(self):
        raise NotImplementedError(f"{self.__class__.__name__} 尚未实现 distributed_state 方法喵!")

    def compute_from_distributed_state(self):
        raise NotImplementedError(f"{self.__class__.__name__} 尚未实现 compute_from_distributed_state 方法喵!")
