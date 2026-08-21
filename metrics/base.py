class Metric:
    def __init__(self):
        pass

    def __call__(self, **context):
        self.update(**context)

    def update(self, **context):
        raise NotImplementedError

    def compute(self):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    def output_fields(self):
        raise NotImplementedError()