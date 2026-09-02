from collections.abc import Mapping
from copy import deepcopy

from parse.errors import UnknownRegistryKeyError


class Registry:
    def __init__(self, category, canonical_map, lookup_map):

        if not isinstance(category, str):
            raise TypeError(f"category: {category} 必须为str类型的喵! 现在的type: {type(category)}")
        normalized_category = category.strip().casefold()
        if not normalized_category:
            raise ValueError(f"category 不能为空喵!")

        if not isinstance(canonical_map, Mapping):
            raise TypeError(f"canonical_map: {canonical_map} 必须为Mapping类型的喵! 现在的type: {type(canonical_map)}")
        if not canonical_map:
            raise ValueError(f"canonical_map 不能为空喵!")
        if not isinstance(lookup_map, Mapping):
            raise TypeError(f"lookup_map: {lookup_map} 必须为Mapping类型的喵! 现在的type: {type(lookup_map)}")
        if not lookup_map:
            raise ValueError(f"lookup_map 不能为空喵!")
        for canonical_key, canonical_value in canonical_map.items():
            if not isinstance(canonical_key, str):
                raise TypeError(f"canonical_map: {canonical_map} canonical_key: {canonical_key} 必须为str类型的喵! 现在的type: {type(canonical_key)}")
            if not canonical_key.strip():
                raise ValueError(f"canonical_map: {canonical_map} 里面的canonical_key都不能为空喵!")
            if canonical_value is None:
                raise TypeError(f"canonical_map: {canonical_map} canonical_key: {canonical_key} 的值: {canonical_value} 不能为None喵!")
            if not canonical_key in set(lookup_map.keys()):
                raise ValueError(f"canonical_key: {canonical_key} 必须存在于lookup_map: {lookup_map} 的键中喵!")

        normalized_lookup_keys = []
        for lookup_key, lookup_value in lookup_map.items():
            if not isinstance(lookup_key, str):
                raise TypeError(f"lookup_map: {lookup_map} lookup_key: {lookup_key} 必须为str类型的喵! 现在的type: {type(lookup_key)}")
            if not lookup_key.strip():
                raise ValueError(f"lookup_map: {lookup_map} 中的lookup_key都不能为空喵!")
            if not isinstance(lookup_value, str):
                raise TypeError(f"lookup_map: {lookup_map} lookup_value: {lookup_value} 必须为str类型的喵! 现在的type: {type(lookup_value)}")
            if not lookup_value.strip():
                raise ValueError(f"lookup_map: {lookup_map} 中的lookup_value都不能为空喵!")
            normalized_lookup_key = lookup_key.strip().casefold()
            if not normalized_lookup_key in normalized_lookup_keys:
                normalized_lookup_keys.append(normalized_lookup_key)
            else:
                raise ValueError(f"lookup_key: {lookup_key} 键规范化后不能重复存在字段喵! normalized_lookup_keys: {normalized_lookup_keys}")
            if not lookup_value.strip().casefold() in set(canonical_map.keys()):
                raise ValueError(f"lookup_value: {lookup_value} 必须存在于 canonical_map: {canonical_map} 的值中喵!")

        self.category = category
        self.canonical_map = deepcopy(canonical_map)
        self.lookup_map = deepcopy(lookup_map)

    def register(self, canonical_key, aliases, registered_object):
        if not isinstance(canonical_key, str):
            raise TypeError(f"canonical_key: {canonical_key} 必须为str类型的喵! 现在的type: {type(canonical_key)}")
        if not canonical_key.strip():
            raise ValueError(f"canonical_key 不能为空喵!")
        normalized_canonical_key = canonical_key.strip().casefold()
        normalized_aliases = []
        for alias in aliases:
            if not isinstance(alias, str):
                raise TypeError(f"aliases: {aliases} alias: {alias} 必须为str类型的喵! 现在的type: {type(alias)}")
            if not alias.strip():
                raise ValueError(f"aliases: {aliases} alias: {alias} 中的元素不能为空喵!")
            normalized_alias = alias.strip().casefold()
            if not normalized_alias in normalized_aliases:
                normalized_aliases.append(normalized_alias)
            else:
                pass
        if normalized_canonical_key in normalized_aliases:
            raise ValueError(f"规范化名称normalized_canonical_key: {normalized_canonical_key} 不能和规范化别名normalized_aliases: {normalized_aliases} 冲突喵!")
        if normalized_canonical_key in self.lookup_map:
            raise ValueError(f"规范化名称normalized_canonical_key: {normalized_canonical_key} 不能与lookup_map: {self.lookup_map} 冲突喵!")
        if registered_object is None:
            raise TypeError(f"registered_object: {registered_object} 不能为None喵!")
        for normalized_alias in normalized_aliases:
            if not normalized_alias in set(self.lookup_map.keys()):
                pass
            elif self.lookup_map[normalized_alias] == normalized_canonical_key:
                pass
            else:
                raise ValueError(f"normalized_alias: {normalized_alias} 指向的是: {self.lookup_map[normalized_alias]}\n"
                                 f"normalized_canonical_key: {normalized_canonical_key}\n"
                                 f"normalized_alias 在 lookup_map 中必须指向 normalized_canonical_key 喵!\n")
        canonical_map = deepcopy(self.canonical_map)
        lookup_map = deepcopy(self.lookup_map)
        canonical_map.update({
            normalized_canonical_key: registered_object,
        })
        lookup_map.update({
            normalized_canonical_key: normalized_canonical_key,
        })
        for normalized_alias in normalized_aliases:
            lookup_map.update({
                normalized_alias: normalized_canonical_key,
            })
        self.canonical_map = canonical_map
        self.lookup_map = lookup_map

    def resolve_key(self, raw_key):
        if not isinstance(raw_key, str):
            raise TypeError(f"raw_key: {raw_key} 必须为str类型的喵! 现在的type: {type(raw_key)}")
        if not raw_key.strip():
            raise ValueError(f"raw_key 不能为空喵!")
        normalized_raw_key = raw_key.strip().casefold()
        try:
            canonical_key = self.lookup_map[normalized_raw_key]
        except KeyError:
            raise UnknownRegistryKeyError(f"lookup_map: {self.lookup_map} 中没有找到normalized_raw_key: {normalized_raw_key} 对应的 canonical_key 喵!")
        return canonical_key

    def get(self, canonical_key):
        if not isinstance(canonical_key, str):
            raise TypeError(f"canonical_key: {canonical_key} 必须为str类型的喵! 现在的type: {type(canonical_key)}")
        if not canonical_key.strip():
            raise ValueError(f"canonical_key 不能为空喵!")
        normalized_canonical_key = canonical_key.strip().casefold()
        try:
            registered_object = self.canonical_map[normalized_canonical_key]
        except KeyError:
            raise UnknownRegistryKeyError(f"canonical_map: {self.canonical_map} 中没有找到normalized_canonical_key: {normalized_canonical_key} 对应的 registered_object 喵!")
        return registered_object

    def keys(self):
        canonical_keys = self.canonical_map.keys()
        canonical_tuple = tuple(canonical_keys)
        return canonical_tuple