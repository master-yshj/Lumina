"""构建与 YAML 结构一致的配置 Schema 树。

PARSE_REFACTOR_DEMO:
    本模块采用正式 Schema Builder 的职责边界，但当前只覆盖已经注册的
    MLM train/test/inference 与 QA train/test。它不创建任何运行时对象，
    也不调用 tasks 目录中的 Builder。
"""

from parse.schema.nodes import EnumSchema, MappingSchema, PathSchema, PolymorphicSchema, RegistryKeySchema, ScalarSchema, SequenceSchema


def _plain_mapping_schema(allow_none=False):
    """创建尚未细化字段的普通 Mapping 节点。

    PARSE_REFACTOR_DEMO:
        该宽松节点只用于尚未完成精确字段定义的普通配置区，不用于动态组件
        params；动态组件 params 始终由 ComponentSpec 处理。
    """
    return MappingSchema(fields={}, allow_unknown=True, allow_none=allow_none)


def _polymorphic_schema(category, registry, extra_fields=None):
    """创建动态组件的公共外壳，具体 params 交给 ComponentSpec。"""
    fields = {
        "type": RegistryKeySchema(registry=registry),
        "params": MappingSchema(fields={}, allow_unknown=True, allow_none=True),
    }
    if extra_fields is not None:
        fields.update(extra_fields)
    return PolymorphicSchema(
        category=category,
        common_schema=MappingSchema(
            fields=fields,
            allow_unknown=False,
            allow_none=False,
        ),
    )


def _path_value_schema(kind="file", allow_none=False, required=True):
    """构建路径字段 Schema，统一让旧版路径语义进入新 Parse。

    PARSE_REFACTOR_DEMO: 只有声明为 PathSchema 的字段才会在 normalize 阶段
    转为 pathlib.Path，随后由 resolve_paths() 以 YAML 目录为基准绝对化。
    """
    return PathSchema(
        base="config_dir",
        kind=kind,
        required=required,
        allow_none=allow_none,
    )


def _dataset_schema(task_key):
    """按任务构建 Dataset Schema；Dataset 不参与 Registry 或 Bootstrap。"""
    if task_key.startswith("mlmpretrain_"):
        dataset_params_schema = MappingSchema(
            fields={
                "max_seq_length": ScalarSchema(python_types=(int,), required=True),
                "mlm_prob": ScalarSchema(python_types=(int, float), required=True),
            },
            allow_unknown=False,
            allow_none=False,
        )
        dataset_choices = {"tiny_zhwiki", "base_zhwiki", "full_zhwiki"}
    elif task_key.startswith("qasft_"):
        dataset_params_schema = MappingSchema(fields={}, allow_unknown=False, allow_none=False)
        dataset_choices = {"tiny_cmrc2018", "base_cmrc2018", "full_cmrc2018", "cmrc2018"}
    else:
        raise ValueError(f"task_key: {task_key} 暂无Dataset Schema喵!")

    # PARSE_REFACTOR_DEMO: Dataset type 是任务约束字段，不是 Registry 判别字段。
    # split 的必填性由 task mode 决定：Train 必须提供 train，Test 必须提供
    # test；其余 split 可以省略，不能让 Test 配置被迫声明训练数据。
    is_train_task = task_key.endswith("_train")
    is_test_task = task_key.endswith("_test")
    return MappingSchema(
        fields={
            "type": EnumSchema(choices=dataset_choices, required=True),
            "root": _path_value_schema(kind="directory"),
            "splits": MappingSchema(
                fields={
                    "train": _path_value_schema(required=is_train_task),
                    "valid": _path_value_schema(required=False),
                    "test": _path_value_schema(required=is_test_task),
                },
                allow_unknown=False,
                allow_none=False,
            ),
            "params": dataset_params_schema,
        },
        allow_unknown=False,
        allow_none=False,
    )


def _runtime_schema():
    """构建 Runtime 的静态结构；真实设备可用性由验证阶段负责。"""
    return MappingSchema(
        fields={
            "device": MappingSchema(
                fields={
                    "type": EnumSchema(choices={"cpu", "cuda", "mps"}, required=True),
                    "index": ScalarSchema(python_types=(int,), allow_none=True, default=None),
                    "count": ScalarSchema(python_types=(int,), default=1),
                },
                allow_unknown=False,
                allow_none=False,
            ),
        },
        allow_unknown=False,
        allow_none=False,
    )


def build_schema(context, registries):
    """根据 ParseContext 和运行 Registry 构建完整根 Schema。

    Args:
        context: Bootstrap 已生成的 ParseContext。
        registries: category 到运行 Registry 的映射。

    Returns:
        与当前 YAML 顶层结构一致的 MappingSchema。
    """
    tokenizer_schema = _polymorphic_schema("tokenizer", registries["tokenizer"])
    model_schema = _polymorphic_schema(
        "model",
        registries["model"],
        extra_fields={
            "init_from": MappingSchema(
                fields={
                    "kind": ScalarSchema(python_types=(str,), required=True),
                    "path": _path_value_schema(),
                },
                allow_unknown=False,
                allow_none=True,
            ),
        },
    )
    loss_schema = _polymorphic_schema(
        "loss",
        registries["loss"],
        extra_fields={
            "name": ScalarSchema(python_types=(str,), required=True),
            "weight": ScalarSchema(python_types=(int, float), default=1.0),
        },
    )
    optimizer_schema = _polymorphic_schema("optimizer", registries["optimizer"])
    scheduler_schema = _polymorphic_schema(
        "scheduler",
        registries["scheduler"],
        extra_fields={"interval": EnumSchema(choices={"step", "epoch"}, default="step")},
    )
    metric_schema = _polymorphic_schema("metric", registries["metric"])
    decoding_schema = _polymorphic_schema("decoding", registries["decoding"])

    common_fields = {
        "version": ScalarSchema(python_types=(int,), required=True),
        "experiment": _plain_mapping_schema(),
        "task": RegistryKeySchema(registry=registries["task"]),
        "run": MappingSchema(
            fields={
                "resume_from": _path_value_schema(
                    allow_none=True,
                    required=False,
                ),
                "checkpoint": MappingSchema(
                    fields={
                        "path": _path_value_schema(),
                        "train_config": _path_value_schema(),
                    },
                    allow_unknown=False,
                    allow_none=False,
                ),
            },
            allow_unknown=True,
            allow_none=False,
        ),
        "runtime": _runtime_schema(),
    }
    task_key = context.task_key
    if task_key.endswith("_train"):
        common_fields.update({
            "tokenizer": tokenizer_schema,
            "model": model_schema,
            "losses": SequenceSchema(item_schema=loss_schema, min_items=1, required=True),
            "data": MappingSchema(
                fields={
                    "dataset": _dataset_schema(task_key),
                    "loaders": _plain_mapping_schema(),
                },
                allow_unknown=False,
                allow_none=False,
            ),
            "optimization": MappingSchema(
                fields={
                    "optimizer": optimizer_schema,
                    "scheduler": scheduler_schema,
                    "loop": _plain_mapping_schema(),
                    "validation": _plain_mapping_schema(),
                },
                allow_unknown=False,
                allow_none=False,
            ),
            "evaluation": MappingSchema(
                fields={
                    "test_after_train": ScalarSchema(python_types=(bool,), default=False),
                    "metrics": MappingSchema(
                        fields={
                            "train": SequenceSchema(item_schema=metric_schema, min_items=1),
                            "valid": SequenceSchema(item_schema=metric_schema, min_items=1),
                            "test": SequenceSchema(item_schema=metric_schema, min_items=1),
                        },
                        allow_unknown=False,
                        allow_none=False,
                    ),
                },
                allow_unknown=False,
                allow_none=False,
            ),
        })
    elif task_key.endswith("_test"):
        common_fields.update({
            # PARSE_REFACTOR_DEMO: Test 配置可以声明 losses；旧版语义中
            # Test 仍需使用训练损失的结构信息，具体训练配置仍从 checkpoint
            # train_config 读取。
            "losses": SequenceSchema(
                item_schema=loss_schema,
                min_items=1,
                required=False,
            ),
            "data": MappingSchema(
                fields={
                    "dataset": _dataset_schema(task_key),
                    "loaders": _plain_mapping_schema(),
                },
                allow_unknown=False,
                allow_none=False,
            ),
            "evaluation": MappingSchema(
                fields={
                    "metrics": MappingSchema(
                        fields={"test": SequenceSchema(item_schema=metric_schema, min_items=1, required=True)},
                        allow_unknown=False,
                        allow_none=False,
                    ),
                },
                allow_unknown=False,
                allow_none=False,
            ),
        })
    elif task_key.endswith("_inference"):
        common_fields["inference"] = MappingSchema(
            fields={
                "max_seq_length": ScalarSchema(python_types=(int,), required=True),
                "decoding": decoding_schema,
            },
            allow_unknown=False,
            allow_none=False,
        )
    else:
        raise ValueError(f"task_key: {task_key} 暂无Schema构建规则喵!")

    return MappingSchema(fields=common_fields, allow_unknown=False, allow_none=False)
