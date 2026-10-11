
class BuildError(RuntimeError):
    """
    :param args: tuple, 传给 RuntimeError 构造器的异常参数，通常是一条含任务、模式、操作和字段位置的说明字符串。

    :return: BuildError, 表示运行组件组装失败的异常实例。

    该类继承 RuntimeError，供 Builder._call 统一包装组装错误并保留原始异常原因，使调用方能定位失败的组件。
    """
