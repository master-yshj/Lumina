from collections.abc import Callable, Mapping

from parse.schema.nodes import SchemaNode, MappingSchema, ScalarSchema, SequenceSchema, EnumSchema, RegistryKeySchema, PathSchema, PolymorphicSchema
from parse.context import ConfigPath, MISSING
from parse.errors import SchemaConflictError

def merge_schema(left, right, path, left_source, right_source):
    """
    :param left: SchemaNode
    :param right: SchemaNode
    :param path: ConfigPath
    :param left_source: str
    :param right_source: str
    """
    if not isinstance(left, SchemaNode):
        raise TypeError(f"left: {left} 必须为SchemaNode类型的喵! 现在的type: {type(left)}")
    if not isinstance(right, SchemaNode):
        raise TypeError(f"right: {right} 必须为SchemaNode类型的喵! 现在的type: {type(right)}")
    if not isinstance(path, ConfigPath):
        raise TypeError(f"path: {path} 必须为ConfigPath类型的喵! 现在的type: {type(path)}")
    if not isinstance(left_source, str):
        raise TypeError(f"left_source: {left_source} 必须为str类型的喵! 现在的type: {type(left_source)}")
    if not isinstance(right_source, str):
        raise TypeError(f"right_source: {right_source} 必须为str类型的喵! 现在的type: {type(right_source)}")

    if not type(left) is type(right):
        raise SchemaConflictError(f"left与right的schema规则不一致喵!\n"
                                  f"path: {path}\n"
                                  f"left_type: {type(left)}\n"
                                  f"right_type: {type(right)}\n"
                                  f"left_source: {left_source}\n"
                                  f"right_source: {right_source}")

    if isinstance(left, MappingSchema):
        merged_fields = {}
        for field_name, left_field_schema in left.fields.items():
            if not field_name in right.fields:
                merged_fields[field_name] = left_field_schema
            else:
                field_path = path.child(field_name)
                merged_fields[field_name] = merge_schema(
                    left_field_schema,
                    right.fields[field_name],
                    field_path,
                    left_source,
                    right_source,
                )
        for field_name, right_field_schema in right.fields.items():
            if not field_name in left.fields:
                merged_fields[field_name] = right_field_schema
            else:
                pass
        merged_allow_none = left.allow_none and right.allow_none
        merged_allow_unknown = left.allow_unknown and right.allow_unknown
        return MappingSchema(
            fields=merged_fields,
            allow_unknown=merged_allow_unknown,
            allow_none=merged_allow_none,
        )
    elif isinstance(left, ScalarSchema):
        left_python_types = left.python_types
        right_python_types = right.python_types
        if not left_python_types == right_python_types:
            raise SchemaConflictError(f"left与right的python_types不一致喵!\n"
                                      f"path: {path}\n"
                                      f"left_python_types: {left_python_types}\n"
                                      f"right_python_types: {right_python_types}\n"
                                      f"left_source: {left_source}\n"
                                      f"right_source: {right_source}")
        left_required = left.required
        right_required = right.required
        merged_required = left_required or right_required
        merged_allow_none = left.allow_none and right.allow_none
        left_default = left.default
        right_default = right.default
        if left_default is MISSING and right_default is MISSING:
            merged_default = MISSING
        else:
            if left_default is MISSING:
                merged_default = right_default
            elif right_default is MISSING:
                merged_default = left_default
            else:
                if left_default == right_default:
                    merged_default = left_default
                else:
                    raise SchemaConflictError(f"left与right的default不一致喵!\n"
                                              f"path: {path}\n"
                                              f"left_default: {left_default}\n"
                                              f"right_default: {right_default}\n"
                                              f"left_source: {left_source}\n"
                                              f"right_source: {right_source}")
        left_normalizer = left.normalizer
        right_normalizer = right.normalizer
        if left_normalizer is None and right_normalizer is None:
            merged_normalizer = None
        else:
            if left_normalizer is None:
                merged_normalizer = right_normalizer
            elif right_normalizer is None:
                merged_normalizer = left_normalizer
            else:
                if left_normalizer is right_normalizer:
                    merged_normalizer = left_normalizer
                else:
                    raise SchemaConflictError(f"left与right的normalizer不一致喵!\n"
                                              f"path: {path}\n"
                                              f"left_normalizer: {left_normalizer}\n"
                                              f"right_normalizer: {right_normalizer}\n"
                                              f"left_source: {left_source}\n"
                                              f"right_source: {right_source}")
        if not merged_default is MISSING:
            if merged_required:
                raise SchemaConflictError(f"合并后required为True时, default必须为MISSING喵!\n"
                                          f"path: {path}\n"
                                          f"merged_required: {merged_required}\n"
                                          f"merged_default: {merged_default}\n"
                                          f"left_required: {left_required}\n"
                                          f"right_required: {right_required}\n"
                                          f"left_default: {left_default}\n"
                                          f"right_default: {right_default}\n"
                                          f"left_source: {left_source}\n"
                                          f"right_source: {right_source}")
        return ScalarSchema(
            python_types=left_python_types,
            required=merged_required,
            default=merged_default,
            allow_none=merged_allow_none,
            normalizer=merged_normalizer,
        )
    elif isinstance(left, SequenceSchema):
        left_item_schema = left.item_schema
        right_item_schema = right.item_schema
        merged_item_schema = merge_schema(
            left_item_schema,
            right_item_schema,
            path.child("<item>"),
            left_source,
            right_source,
        )
        left_min_items = left.min_items
        left_max_items = left.max_items
        right_min_items = right.min_items
        right_max_items = right.max_items
        merged_min_items = max(left_min_items, right_min_items)
        if left_max_items is None and right_max_items is None:
            merged_max_items = None
        else:
            if left_max_items is None:
                merged_max_items = right_max_items
            elif right_max_items is None:
                merged_max_items = left_max_items
            else:
                merged_max_items = min(left_max_items, right_max_items)
        if merged_max_items is not None and not merged_min_items <= merged_max_items:
            raise SchemaConflictError(f"合并后的min_items必须小于等于max_items喵!\n"
                                      f"path: {path}\n"
                                      f"merged_min_items: {merged_min_items}\n"
                                      f"merged_max_items: {merged_max_items}\n"
                                      f"left_min_items: {left_min_items}\n"
                                      f"right_min_items: {right_min_items}\n"
                                      f"left_max_items: {left_max_items}\n"
                                      f"right_max_items: {right_max_items}\n"
                                      f"left_source: {left_source}\n"
                                      f"right_source: {right_source}")
        left_allow_none = left.allow_none
        right_allow_none = right.allow_none
        merged_allow_none = left_allow_none and right_allow_none
        left_required = left.required
        right_required = right.required
        left_default = left.default
        right_default = right.default
        merged_required = left.required or right_required
        if left_default is MISSING and right_default is MISSING:
            merged_default = MISSING
        else:
            if left_default is MISSING:
                merged_default = right_default
            elif right_default is MISSING:
                merged_default = left_default
            else:
                if left_default == right_default:
                    merged_default = left_default
                else:
                    raise SchemaConflictError(f"left与right的default不一致喵!\n"
                                              f"path: {path}\n"
                                              f"left_default: {left_default}\n"
                                              f"right_default: {right_default}\n"
                                              f"left_source: {left_source}\n"
                                              f"right_source: {right_source}")
        if merged_required:
            if not merged_default is MISSING:
                raise SchemaConflictError(f"合并后required为True时, default必须为MISSING喵!\n"
                                          f"path: {path}\n"
                                          f"merged_required: {merged_required}\n"
                                          f"merged_default: {merged_default}\n"
                                          f"left_required: {left_required}\n"
                                          f"right_required: {right_required}\n"
                                          f"left_default: {left_default}\n"
                                          f"right_default: {right_default}\n"
                                          f"left_source: {left_source}\n"
                                          f"right_source: {right_source}")
        if merged_default is not MISSING:
            merged_default_length = len(merged_default)
            if not merged_default_length >= merged_min_items:
                raise SchemaConflictError(f"合并后default的长度必须大于等于min_items喵!\n"
                                          f"path: {path}\n"
                                          f"merged_min_items: {merged_min_items}\n"
                                          f"merged_default: {merged_default}\n"
                                          f"left_min_items: {left_min_items}\n"
                                          f"right_min_items: {right_min_items}\n"
                                          f"left_default: {left_default}\n"
                                          f"right_default: {right_default}\n"
                                          f"left_source: {left_source}\n"
                                          f"right_source: {right_source}\n")
            else:
                if merged_max_items is not None and merged_default_length > merged_max_items:
                    raise SchemaConflictError(f"合并后default的长度必须小于等于max_items喵!\n"
                                              f"path: {path}\n"
                                              f"merged_max_items: {merged_max_items}\n"
                                              f"merged_default: {merged_default}\n"
                                              f"left_max_items: {left_max_items}\n"
                                              f"right_max_items: {right_max_items}\n"
                                              f"left_default: {left_default}\n"
                                              f"right_default: {right_default}\n"
                                              f"left_source: {left_source}\n"
                                              f"right_source: {right_source}\n")
        return SequenceSchema(
            item_schema=merged_item_schema,
            min_items=merged_min_items,
            max_items=merged_max_items,
            allow_none=merged_allow_none,
            required=merged_required,
            default=merged_default,
        )

    elif isinstance(left, EnumSchema):
        left_case_sensitive = left.case_sensitive
        right_case_sensitive = right.case_sensitive
        if not left_case_sensitive == right_case_sensitive:
            raise SchemaConflictError(f"left_case_sensitive 与 right_case_sensitive 必须一致喵!\n"
                                      f"path: {path}\n"
                                      f"left_case_sensitive: {left_case_sensitive}\n"
                                      f"right_case_sensitive: {right_case_sensitive}\n"
                                      f"left_source: {left_source}\n"
                                      f"right_source: {right_source}\n")
        else:
            merged_case_sensitive = left_case_sensitive
        left_choices = left.choices
        right_choices = right.choices
        left_choices_type = type(left_choices)
        right_choices_type = type(right_choices)
        if not left_choices_type is right_choices_type:
            raise SchemaConflictError(f"left_choices_type 与 right_choices_type 必须类型一致喵!\n"
                                      f"path: {path}\n"
                                      f"left_choices: {left_choices}\n"
                                      f"right_choices: {right_choices}\n"
                                      f"left_choices_type: {left_choices_type}\n"
                                      f"right_choices_type: {right_choices_type}\n"
                                      f"left_source: {left_source}\n"
                                      f"right_source: {right_source}\n")
        if isinstance(left_choices, set):
            merged_choices = left_choices & right_choices
            if not merged_choices:
                raise SchemaConflictError(f"合并后的 choices 不能为空喵!\n"
                                          f"path: {path}\n"
                                          f"left_choices: {left_choices}\n"
                                          f"right_choices: {right_choices}\n"
                                          f"left_source: {left_source}\n"
                                          f"right_source: {right_source}\n")
        elif isinstance(left_choices, Mapping):
            left_choices_keys = set(left_choices.keys())
            right_choices_keys = set(right_choices.keys())
            common_choices_keys = left_choices_keys & right_choices_keys
            merged_choices = {}
            for common_choices_key in common_choices_keys:
                left_common_choices_value = left_choices[common_choices_key]
                right_common_choices_value = right_choices[common_choices_key]
                if left_common_choices_value != right_common_choices_value:
                    raise SchemaConflictError(f"left_choices与right_choices相同键的值必须相同喵!\n"
                                              f"path: {path}\n"
                                              f"left_choices: {left_choices}\n"
                                              f"right_choices: {right_choices}\n"
                                              f"common_choices_keys: {common_choices_keys}\n"
                                              f"left_common_choices_value: {left_common_choices_value}\n"
                                              f"right_common_choices_value: {right_common_choices_value}\n"
                                              f"left_source: {left_source}\n"
                                              f"right_source: {right_source}\n")
                else:
                    merged_choices[common_choices_key] = left_common_choices_value
            if not merged_choices:
                raise SchemaConflictError(f"合并后的 choices 不能为空喵!\n"
                                          f"path: {path}\n"
                                          f"left_choices: {left_choices}\n"
                                          f"right_choices: {right_choices}\n"
                                          f"left_source: {left_source}\n"
                                          f"right_source: {right_source}\n")
        left_required = left.required
        right_required = right.required
        merged_required = left_required or right_required
        left_allow_none = left.allow_none
        right_allow_none = right.allow_none
        merged_allow_none = left_allow_none and right_allow_none
        left_default = left.default
        right_default = right.default
        if left_default is MISSING and right_default is MISSING:
            merged_default = MISSING
        else:
            if left_default is MISSING:
                merged_default = right_default
            elif right_default is MISSING:
                merged_default = left_default
            else:
                if left_default == right_default:
                    merged_default = left_default
                else:
                    raise SchemaConflictError(f"left与right的default不一致喵!\n"
                                              f"path: {path}\n"
                                              f"left_default: {left_default}\n"
                                              f"right_default: {right_default}\n"
                                              f"left_source: {left_source}\n"
                                              f"right_source: {right_source}")
        if not merged_default is MISSING:
            if merged_required:
                raise SchemaConflictError(f"合并后required为True时, default必须为MISSING喵!\n"
                                          f"path: {path}\n"
                                          f"merged_required: {merged_required}\n"
                                          f"merged_default: {merged_default}\n"
                                          f"left_required: {left_required}\n"
                                          f"right_required: {right_required}\n"
                                          f"left_default: {left_default}\n"
                                          f"right_default: {right_default}\n"
                                          f"left_source: {left_source}\n"
                                          f"right_source: {right_source}")
            if isinstance(merged_choices, set):
                if not merged_default in merged_choices:
                    raise SchemaConflictError(f"合并后的default必须在合并后的choices中喵!\n"
                                              f"path: {path}\n"
                                              f"merged_default: {merged_default}\n"
                                              f"left_default: {left_default}\n"
                                              f"right_default: {right_default}\n"
                                              f"merged_choices: {merged_choices}\n"
                                              f"left_choices: {left_choices}\n"
                                              f"right_choices: {right_choices}\n"
                                              f"left_source: {left_source}\n"
                                              f"right_source: {right_source}\n")
            if isinstance(merged_choices, Mapping):
                if not merged_default in merged_choices.values():
                    raise SchemaConflictError(f"合并后的default必须在合并后的choices的values中喵!\n"
                                              f"path: {path}\n"
                                              f"merged_default: {merged_default}\n"
                                              f"left_default: {left_default}\n"
                                              f"right_default: {right_default}\n"
                                              f"merged_choices: {merged_choices}\n"
                                              f"left_choices: {left_choices}\n"
                                              f"right_choices: {right_choices}\n"
                                              f"left_source: {left_source}\n"
                                              f"right_source: {right_source}\n")
        return EnumSchema(
            choices=merged_choices,
            case_sensitive=merged_case_sensitive,
            required=merged_required,
            allow_none=merged_allow_none,
            default=merged_default,
        )

    elif isinstance(left, RegistryKeySchema):
        left_registry = left.registry
        right_registry = right.registry
        if not left_registry is right_registry:
            raise SchemaConflictError(f"left_registry与right_registry的类型必须一致喵!\n"
                                      f"path: {path}\n"
                                      f"left_registry: {left_registry}\n"
                                      f"right_registry: {right_registry}\n"
                                      f"left_source: {left_source}\n"
                                      f"right_source: {right_source}\n")
        merged_registry = left_registry
        merged_required = left.required or right.required
        merged_allow_none = left.allow_none and right.allow_none
        return RegistryKeySchema(
            registry=merged_registry,
            required=merged_required,
            allow_none=merged_allow_none,
        )

    elif isinstance(left, PathSchema):
        left_base = left.base
        right_base = right.base
        if not left_base == right_base:
            raise SchemaConflictError(f"left_base与right_base必须一致喵!\n"
                                      f"path: {path}\n"
                                      f"left_base: {left_base}\n"
                                      f"right_base: {right_base}\n"
                                      f"left_source: {left_source}\n"
                                      f"right_source: {right_source}\n")
        merged_base = left_base
        left_kind = left.kind
        right_kind = right.kind
        if not left_kind == right_kind:
            raise SchemaConflictError(f"left_kind与right_kind必须一致喵!\n"
                                      f"path: {path}\n"
                                      f"left_kind: {left_kind}\n"
                                      f"right_kind: {right_kind}\n"
                                      f"left_source: {left_source}\n"
                                      f"right_source: {right_source}\n")
        merged_kind = left_kind
        left_resolver_name = left.resolver_name
        right_resolver_name = right.resolver_name
        if not left_resolver_name == right_resolver_name:
            raise SchemaConflictError(f"left_resolver_name与right_resolver_name必须一致喵!\n"
                                      f"path: {path}\n"
                                      f"left_resolver_name: {left_resolver_name}\n"
                                      f"right_resolver_name: {right_resolver_name}\n"
                                      f"left_source: {left_source}\n"
                                      f"right_source: {right_source}\n")
        merged_resolver_name = left_resolver_name
        left_required = left.required
        right_required = right.required
        merged_required = left_required or right_required
        left_allow_none = left.allow_none
        right_allow_none = right.allow_none
        merged_allow_none = left_allow_none and right_allow_none
        left_must_exist = left.must_exist
        right_must_exist = right.must_exist
        merged_must_exist = left_must_exist or right_must_exist
        left_non_empty = left.non_empty
        right_non_empty = right.non_empty
        merged_non_empty = left_non_empty or right_non_empty
        left_allowed_suffixes = left.allowed_suffixes
        right_allowed_suffixes = right.allowed_suffixes
        if not left_allowed_suffixes and not right_allowed_suffixes:
            merged_allowed_suffixes = ()
        else:
            if not left_allowed_suffixes:
                merged_allowed_suffixes = right_allowed_suffixes
            elif not right_allowed_suffixes:
                merged_allowed_suffixes = left_allowed_suffixes
            else:
                merged_allowed_suffixes = tuple((set(left_allowed_suffixes) & set(right_allowed_suffixes)))
            if not merged_allowed_suffixes:
                raise SchemaConflictError(f"合并后的left与right的suffix不能为空喵!\n"
                                          f"path: {path}\n"
                                          f"left_allowed_suffixes: {left_allowed_suffixes}\n"
                                          f"right_allowed_suffixes: {right_allowed_suffixes}\n"
                                          f"merged_allowed_suffixes: {merged_allowed_suffixes}\n"
                                          f"left_source: {left_source}\n"
                                          f"right_source: {right_source}\n")
        left_default = left.default
        right_default = right.default
        if left_default is MISSING and right_default is MISSING:
            merged_default = MISSING
        else:
            if left_default is MISSING:
                merged_default = right_default
            elif right_default is MISSING:
                merged_default = left_default
            else:
                if left_default == right_default:
                    merged_default = left_default
                else:
                    raise SchemaConflictError(f"left与right的default不一致喵!\n"
                                              f"path: {path}\n"
                                              f"left_default: {left_default}\n"
                                              f"right_default: {right_default}\n"
                                              f"left_source: {left_source}\n"
                                              f"right_source: {right_source}")
        if not merged_default is MISSING:
            if merged_required:
                raise SchemaConflictError(f"合并后required为True时, default必须为MISSING喵!\n"
                                          f"path: {path}\n"
                                          f"merged_required: {merged_required}\n"
                                          f"merged_default: {merged_default}\n"
                                          f"left_required: {left_required}\n"
                                          f"right_required: {right_required}\n"
                                          f"left_default: {left_default}\n"
                                          f"right_default: {right_default}\n"
                                          f"left_source: {left_source}\n"
                                          f"right_source: {right_source}")

        return PathSchema(
            base=merged_base,
            kind=merged_kind,
            required=merged_required,
            default=merged_default,
            allow_none=merged_allow_none,
            allowed_suffixes=merged_allowed_suffixes,
            must_exist=merged_must_exist,
            non_empty=merged_non_empty,
            resolver_name=merged_resolver_name,
        )

    elif isinstance(left, PolymorphicSchema):
        left_category = left.category
        right_category = right.category
        if not left_category == right_category:
            raise SchemaConflictError(f"left与right的category不一致喵!\n"
                                      f"path: {path}\n"
                                      f"left_category: {left_category}\n"
                                      f"right_category: {right_category}\n"
                                      f"left_source: {left_source}\n"
                                      f"right_source: {right_source}\n")
        merged_category = left_category
        left_discriminator = left.discriminator
        right_discriminator = right.discriminator
        if not left_discriminator == right_discriminator:
            raise SchemaConflictError(f"left与right的discriminator不一致喵!\n"
                                      f"path: {path}\n"
                                      f"left_discriminator: {left_discriminator}\n"
                                      f"right_discriminator: {right_discriminator}\n"
                                      f"left_source: {left_source}\n"
                                      f"right_source: {right_source}\n")
        merged_discriminator = left_discriminator
        left_params_field = left.params_field
        right_params_field = right.params_field
        if not left_params_field == right_params_field:
            raise SchemaConflictError(f"left与right的params_field不一致喵!\n"
                                      f"path: {path}\n"
                                      f"left_params_field: {left_params_field}\n"
                                      f"right_params_field: {right_params_field}\n"
                                      f"left_source: {left_source}\n"
                                      f"right_source: {right_source}\n")
        merged_params_field = left.params_field
        left_common_schema = left.common_schema
        right_common_schema = right.common_schema
        merged_common_schema = merge_schema(
            left_common_schema,
            right_common_schema,
            path,
            left_source,
            right_source,
        )
        if not isinstance(merged_common_schema, MappingSchema):
            raise SchemaConflictError(f"合并后的common_schema必须为MappingSchema喵!\n"
                                      f"path: {path}\n"
                                      f"left_common_schema: {left_common_schema}\n"
                                      f"right_common_schema: {right_common_schema}\n"
                                      f"left_source: {left_source}\n"
                                      f"right_source: {right_source}\n")
        return PolymorphicSchema(
            category=merged_category,
            common_schema=merged_common_schema,
            discriminator=merged_discriminator,
            params_field=merged_params_field,
        )

    else:
        raise NotImplementedError