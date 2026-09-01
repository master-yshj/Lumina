
class ConfigError(Exception):
    def __init__(self, message, actual_value=None, path=None, expected=None, source=None):
        """
        报错的基类
        :param message: 人为补充的错误信息
        :param actual_value: config中实际拿到的值，即用户实际填写的值
        :param path: 错误所在的yaml路径，描述“哪里错误”
        :param expected: 预期的规则
        :param source: 错误来自于哪一个模块或扩展
        """

        if not isinstance(message, str):
            raise TypeError(f"message: {message} 必须为str类型的喵! 现在的type: {type(message)}")
        if isinstance(source, str):
            if not source.strip():
                raise ValueError(f"source 在为字符串类型时不能为空喵!")
        elif source is None:
            pass
        else:
            raise TypeError(f"source: {source} 只能为str类型或None喵! 现在的type: {type(source)}")

        super(ConfigError, self).__init__(message)
        self.message = message
        self.actual_value = actual_value
        self.path = path
        self.expected = expected
        self.source = source

class YamlLoadError(ConfigError):
    """
    继承父类ConfigError
    加载yaml的报错
    """

class BootstrapError(ConfigError):
    """
    继承父类ConfigError
    读取配置字段的报错
    """

class SchemaError(ConfigError):
    """
    继承父类ConfigError
    判别字段时的报错
    """

class ParameterNormalizationError(ConfigError):
    """
    继承父类ConfigError
    规范化参数时的报错
    """

class PathResolutionError(ConfigError):
    """
    继承父类ConfigError
    路径绝对化时的报错
    """

class ConfigValidationError(ConfigError):
    """
    继承父类ConfigError
    参数校验时的报错
    """

class RuntimeValidationError(ConfigError):
    """
    继承父类ConfigError
    运行时的报错
    """

class UnknownRegistryKeyError(BootstrapError):
    """
    继承父类BootstrapError
    未知注册表键的报错
    """

class ModeTaskMismatchError(BootstrapError):
    """
    继承父类BootstrapError
    Mode与Task类型不匹配的报错
    """

class SchemaConflictError(SchemaError):
    """
    继承父类SchemaError
    配置冲突时的报错
    """

class MissingFieldError(SchemaError):
    """
    继承父类SchemaError
    缺失配置字段的报错
    """

class UnknownFieldError(SchemaError):
    """
    继承父类SchemaError
    未知配置字段的报错
    """

class TypeMismatchError(SchemaError):
    """
    继承父类SchemaError
    类型不匹配时的报错
    """

class NullNotAllowedError(SchemaError):
    """
    继承父类SchemaError
    空值不被允许时的报错
    """

class PathDependencyError(PathResolutionError):
    """
    继承父类PathResolutionError
    路径依赖的报错
    """
