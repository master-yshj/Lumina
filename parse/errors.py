

class ConfigError(Exception):
    def __init__(self, message, actual_value, path=None, expected=None, source=None):
        """
        报错的基类
        :param message: 人为补充的错误信息
        :param actual_value: config中实际拿到的值，即用户实际填写的值
        :param path: 错误所在的yaml路径，描述“哪里错误”
        :param expected: 预期的规则
        :param source: 错误来自于哪一个模块或扩展
        """
        super(ConfigError, self).__init__(message)

class YamlLoadError(ConfigError):
    def __init__(self, message, actual_value, path=None, expected=None, source=None):
        """
        继承父类ConfigError
        加载yaml的报错
        :param message: 人为补充的错误信息
        :param actual_value: config中实际拿到的值，即用户实际填写的值
        :param path: 错误所在的yaml路径，描述“哪里错误”
        :param expected: 预期的规则
        :param source: 错误来自于哪一个模块或扩展
        """
        super(YamlLoadError, self).__init__(message, actual_value, path, expected, source)

class BootstrapError(ConfigError):
    def __init__(self, message, actual_value, path=None, expected=None, source=None):
        """
        继承父类ConfigError
        读取配置字段的报错
        :param message: 人为补充的错误信息
        :param actual_value: config中实际拿到的值，即用户实际填写的值
        :param path: 错误所在的yaml路径，描述“哪里错误”
        :param expected: 预期的规则
        :param source: 错误来自于哪一个模块或扩展
        """
        super(BootstrapError, self).__init__(message, actual_value, path, expected, source)

class SchemaError(ConfigError):
    def __init__(self, message, actual_value, path=None, expected=None, source=None):
        """
        继承父类ConfigError
        判别字段时的报错
        :param message: 人为补充的错误信息
        :param actual_value: config中实际拿到的值，即用户实际填写的值
        :param path: 错误所在的yaml路径，描述“哪里错误”
        :param expected: 预期的规则
        :param source: 错误来自于哪一个模块或扩展
        """
        super(SchemaError, self).__init__(message, actual_value, path, expected, source)

class ParameterNormalizationError(ConfigError):
    def __init__(self, message, actual_value, path=None, expected=None, source=None):
        """
        继承父类ConfigError
        规范化参数时的报错
        :param message: 人为补充的错误信息
        :param actual_value: config中实际拿到的值，即用户实际填写的值
        :param path: 错误所在的yaml路径，描述“哪里错误”
        :param expected: 预期的规则
        :param source: 错误来自于哪一个模块或扩展
        """
        super(ParameterNormalizationError, self).__init__(message, actual_value, path, expected, source)

class PathResolutionError(ConfigError):
    def __init__(self, message, actual_value, path=None, expected=None, source=None):
        """
        继承父类ConfigError
        路径绝对化时的报错
        :param message: 人为补充的错误信息
        :param actual_value: config中实际拿到的值，即用户实际填写的值
        :param path: 错误所在的yaml路径，描述“哪里错误”
        :param expected: 预期的规则
        :param source: 错误来自于哪一个模块或扩展
        """
        super(PathResolutionError, self).__init__(message, actual_value, path, expected, source)

class ConfigValidationError(ConfigError):
    def __init__(self, message, actual_value, path=None, expected=None, source=None):
        """
        继承父类ConfigError
        参数校验时的报错
        :param message: 人为补充的错误信息
        :param actual_value: config中实际拿到的值，即用户实际填写的值
        :param path: 错误所在的yaml路径，描述“哪里错误”
        :param expected: 预期的规则
        :param source: 错误来自于哪一个模块或扩展
        """
        super(ConfigValidationError, self).__init__(message, actual_value, path, expected, source)

class RuntimeValidationError(ConfigError):
    def __init__(self, message, actual_value, path=None, expected=None, source=None):
        """
        继承父类ConfigError
        运行时的报错
        :param message: 人为补充的错误信息
        :param actual_value: config中实际拿到的值，即用户实际填写的值
        :param path: 错误所在的yaml路径，描述“哪里错误”
        :param expected: 预期的规则
        :param source: 错误来自于哪一个模块或扩展
        """
        super(RuntimeValidationError, self).__init__(message, actual_value, path, expected, source)

class UnknownRegistryKeyError(ConfigError):
    def __init__(self, message, actual_value, path=None, expected=None, source=None):
        """
        继承父类BootstrapError
        未知注册表键的报错
        :param message: 人为补充的错误信息
        :param actual_value: config中实际拿到的值，即用户实际填写的值
        :param path: 错误所在的yaml路径，描述“哪里错误”
        :param expected: 预期的规则
        :param source: 错误来自于哪一个模块或扩展
        """
        super(UnknownRegistryKeyError, self).__init__(message, actual_value, path, expected, source)

class ModeTaskMismatchError(ConfigError):
    def __init__(self, message, actual_value, path=None, expected=None, source=None):
        """
        继承父类BootstrapError
        Mode与Task类型不匹配的报错
        :param message: 人为补充的错误信息
        :param actual_value: config中实际拿到的值，即用户实际填写的值
        :param path: 错误所在的yaml路径，描述“哪里错误”
        :param expected: 预期的规则
        :param source: 错误来自于哪一个模块或扩展
        """
        super(ModeTaskMismatchError, self).__init__(message, actual_value, path, expected, source)

class SchemaConflictError(ConfigError):
    def __init__(self, message, actual_value, path=None, expected=None, source=None):
        """
        继承父类SchemaError
        配置冲突时的报错
        :param message: 人为补充的错误信息
        :param actual_value: config中实际拿到的值，即用户实际填写的值
        :param path: 错误所在的yaml路径，描述“哪里错误”
        :param expected: 预期的规则
        :param source: 错误来自于哪一个模块或扩展
        """
        super(SchemaConflictError, self).__init__(message, actual_value, path, expected, source)

class MissingFieldError(ConfigError):
    def __init__(self, message, actual_value, path=None, expected=None, source=None):
        """
        继承父类SchemaError
        缺失配置字段的报错
        :param message: 人为补充的错误信息
        :param actual_value: config中实际拿到的值，即用户实际填写的值
        :param path: 错误所在的yaml路径，描述“哪里错误”
        :param expected: 预期的规则
        :param source: 错误来自于哪一个模块或扩展
        """
        super(MissingFieldError, self).__init__(message, actual_value, path, expected, source)

class UnknownFieldError(ConfigError):
    def __init__(self, message, actual_value, path=None, expected=None, source=None):
        """
        继承父类SchemaError
        未知配置字段的报错
        :param message: 人为补充的错误信息
        :param actual_value: config中实际拿到的值，即用户实际填写的值
        :param path: 错误所在的yaml路径，描述“哪里错误”
        :param expected: 预期的规则
        :param source: 错误来自于哪一个模块或扩展
        """
        super(UnknownFieldError, self).__init__(message, actual_value, path, expected, source)

class TypeMismatchError(ConfigError):
    def __init__(self, message, actual_value, path=None, expected=None, source=None):
        """
        继承父类SchemaError
        类型不匹配时的报错
        :param message: 人为补充的错误信息
        :param actual_value: config中实际拿到的值，即用户实际填写的值
        :param path: 错误所在的yaml路径，描述“哪里错误”
        :param expected: 预期的规则
        :param source: 错误来自于哪一个模块或扩展
        """
        super(TypeMismatchError, self).__init__(message, actual_value, path, expected, source)

class NullNotAllowedError(ConfigError):
    def __init__(self, message, actual_value, path=None, expected=None, source=None):
        """
        继承父类SchemaError
        空值不被允许时的报错
        :param message: 人为补充的错误信息
        :param actual_value: config中实际拿到的值，即用户实际填写的值
        :param path: 错误所在的yaml路径，描述“哪里错误”
        :param expected: 预期的规则
        :param source: 错误来自于哪一个模块或扩展
        """
        super(NullNotAllowedError, self).__init__(message, actual_value, path, expected, source)

class PathDependencyError(ConfigError):
    def __init__(self, message, actual_value, path=None, expected=None, source=None):
        """
        继承父类PathResolutionError
        路径依赖的报错
        :param message: 人为补充的错误信息
        :param actual_value: config中实际拿到的值，即用户实际填写的值
        :param path: 错误所在的yaml路径，描述“哪里错误”
        :param expected: 预期的规则
        :param source: 错误来自于哪一个模块或扩展
        """
        super(PathDependencyError, self).__init__(message, actual_value, path, expected, source)