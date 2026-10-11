import re


def build_metric_fieldnames(split, loss_names, metric_fields):
    """
    :param split: str, 指标所属划分，只允许 train、valid 或 test。
    :param loss_names: list[str], components.losses 中各损失项的 name，按配置顺序传入。
    :param metric_fields: list[str], 当前 MetricCollection.output_fields() 返回的指标列名。

    :return: list[str], 按基础列、各损失统计列、指标列排列的新表头列表。

    该方法训练基础列为 global_step、lr、avg_loss、interval_avg_loss，验证和测试为 global_step、avg_loss；追加各损失累计列及训练区间列，再追加指标列，检查非空和重名后交给 MetricWriter。
    """
    if split not in ('train', 'valid', 'test'):
        raise ValueError(f'不支持写入指标的划分：{split}')
    fields = (['global_step', 'lr', 'avg_loss', 'interval_avg_loss']
              if split == 'train' else ['global_step', 'avg_loss'])
    for name in loss_names:
        fields.extend([f'{name}_avg_raw', f'{name}_avg_weighted_loss'])
        if split == 'train':
            fields.extend([f'{name}_interval_avg_raw', f'{name}_interval_avg_weighted_loss'])
    fields.extend(metric_fields)
    seen = set()
    for field in fields:
        if not isinstance(field, str) or not field.strip():
            raise ValueError(f'CSV 字段必须为非空字符串：{field!r}')
        if field in seen:
            raise ValueError(f'CSV 字段重复：{field}；请检查 Loss 名称与指标输出字段')
        seen.add(field)
    return fields


def visualizer_output_name(index, canonical_type):
    """
    :param index: int, 可视化方法在配置列表中的编号，从 1 开始。
    :param canonical_type: str, 方法的规范名称，必须以小写英文字母开头，后续只允许小写字母、数字或下划线。

    :return: str, 至少两位的编号与规范名拼成的子目录名，例如 01_attentionheatmap。

    该方法为每个可视化方法生成独立输出名称，校验名称可安全作为单个目录名，此处不创建目录。
    """
    if type(index) is not int or index < 1:
        raise ValueError('可视化编号必须为从 1 开始的整数')
    if not isinstance(canonical_type, str) or not re.fullmatch(r'[a-z][a-z0-9_]*', canonical_type):
        raise ValueError(f'不安全的可视化规范名称：{canonical_type!r}')
    return f'{index:02d}_{canonical_type}'
