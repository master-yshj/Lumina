

class ConfigError(ValueError):
    """
    :param args: tuple, 传给 ValueError 的异常参数，项目通常传入一条错误信息字符串

    :return: ConfigError, 创建的配置异常实例

    该类继承 ValueError，统一表示配置解析和校验错误，供调用方捕获并补充字段来源
    """
    pass

class UnknownRegistryKeyError(ConfigError):
    """
    :param args: tuple, 传给父类的异常参数，通常为无法识别组件名称的说明字符串

    :return: UnknownRegistryKeyError, 创建的注册名称异常实例

    该类继承 ConfigError，表示组件名称或别名无法在对应 Registry 中找到
    """
    pass