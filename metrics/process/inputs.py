import torch

def _calculate_qa_predictions(start_logits, end_logits, token_type_ids, offset_mapping_list, context_list):
    if not isinstance(start_logits, torch.Tensor):
        raise ValueError(f"start_logits 必须为一个Tensor张量喵! 现在的type: {type(start_logits)}")
    if not isinstance(end_logits, torch.Tensor):
        raise ValueError(f"end_logits 必须为一个Tensor张量喵! 现在的type: {type(end_logits)}")
    if not isinstance(token_type_ids, torch.Tensor):
        raise ValueError(f"token_type_ids 必须为一个Tensor张量喵! 现在的type: {type(token_type_ids)}")
    if not isinstance(offset_mapping_list, list):
        raise ValueError(f"offset_mapping_list 必须为list类型的喵! 现在的type: {type(offset_mapping_list)}")
    if not isinstance(context_list, list):
        raise ValueError(f"context_list 必须为list类型的喵! 现在的type: {type(context_list)}")

    if not start_logits.dim() == 2:
        raise ValueError(f"start_logits 的维度必须为2喵! 现在的维度: {start_logits.dim()}")
    if not end_logits.dim() == 2:
        raise ValueError(f"end_logits 的维度必须为2喵! 现在的维度: {end_logits.dim()}")

    if not start_logits.shape == end_logits.shape:
        raise ValueError(f"start_logits 与 end_logits 的shape必须一致喵!\n"
                         f"start_logits.shape: {start_logits.shape}\n"
                         f"end_logits.shape: {end_logits.shape}\n")

    if not token_type_ids.dim() == 2:
        raise ValueError(f"token_type_ids 的维度必须为2喵! 现在的维度: {token_type_ids.dim()}")
    if not start_logits.shape == token_type_ids.shape:
        raise ValueError(f"start_logits 与 token_type_ids 的shape必须一致喵!\n"
                         f"start_logits.shape: {start_logits.shape}\n"
                         f"token_type_ids.shape: {token_type_ids.shape}\n")

    batch_size = start_logits.shape[0]

    if not len(offset_mapping_list) == len(context_list) == batch_size:
        raise ValueError(f"offset_mapping_list 和 context_list 的长度必须与batch_size一致喵!\n"
                         f"offset_mapping_list: {len(offset_mapping_list)}\n"
                         f"context_list: {len(context_list)}\n")

    seq_length = start_logits.shape[1]
    if not all(len(offset_mapping) <= seq_length for offset_mapping in offset_mapping_list):
        raise ValueError(f"offset_mapping_list中的每一个子offset_mapping的长度必须小于seq_length喵!")

    predictions = []
    for sample_index in range(batch_size):
        start_index = start_logits[sample_index].argmax()
        end_index = end_logits[sample_index].argmax()
        offset_mapping = offset_mapping_list[sample_index]
        context = context_list[sample_index]

        if start_index > end_index:
            predictions.append("")
            continue
        if start_index >= len(offset_mapping):
            predictions.append("")
            continue
        if end_index >= len(offset_mapping):
            predictions.append("")
            continue
        if token_type_ids[sample_index][start_index] != 1 or token_type_ids[sample_index][end_index] != 1:
            predictions.append("")
            continue

        start_offset = offset_mapping[start_index]
        end_offset = offset_mapping[end_index]

        start_position = start_offset[0]
        end_position = end_offset[1]

        prediction = context[start_position:end_position]
        predictions.append(prediction)
    return predictions

def process_logits_labels(**context):
    if "labels" in context:
        labels = context["labels"]
    else:
        if not "batch" in context:
            raise ValueError(f"context中不包含batch这个键喵!")
        if not "outputs" in context:
            raise ValueError(f"context中不包含outputs这个键喵!")
        batch = context["batch"]
        outputs = context["outputs"]
        if not isinstance(batch, dict):
            raise ValueError(f"context.batch 必须为字典类型的喵! 现在的type: {type(batch)}")
        if not batch:
            raise ValueError(f"context.batch 不能为空喵!")
        if not isinstance(outputs, dict):
            raise ValueError(f"context.outputs 必须为字典类型的喵! 现在的type: {type(outputs)}")
        if not outputs:
            raise ValueError(f"context.outputs 不能为空喵!")
        if not "labels" in batch:
            raise ValueError(f"context.batch 中必须包含名为labels的键喵!")
        labels = batch["labels"]
    if "logits" in context:
        logits = context["logits"]
    else:
        if not "batch" in context:
            raise ValueError(f"context中不包含batch这个键喵!")
        if not "outputs" in context:
            raise ValueError(f"context中不包含outputs这个键喵!")
        batch = context["batch"]
        outputs = context["outputs"]
        if not isinstance(batch, dict):
            raise ValueError(f"context.batch 必须为字典类型的喵! 现在的type: {type(batch)}")
        if not batch:
            raise ValueError(f"context.batch 不能为空喵!")
        if not isinstance(outputs, dict):
            raise ValueError(f"context.outputs 必须为字典类型的喵! 现在的type: {type(outputs)}")
        if not outputs:
            raise ValueError(f"context.outputs 不能为空喵!")
        if not "logits" in outputs:
            raise ValueError(f"context.outputs 中必须包含名为logits的键喵!")
        logits = outputs["logits"]
    if not isinstance(labels, torch.Tensor):
        raise ValueError(f"context.batch.labels 必须为Tensor类型的喵! 现在的type: {type(labels)}")
    if not labels.dim() == 2:
        raise ValueError(f"context.batch.labels 必须为2维的喵! 现在的维度: {labels.dim()}")
    if not isinstance(logits, torch.Tensor):
        raise ValueError(f"context.outputs.logits 必须为Tensor类型的喵! 现在的type: {type(logits)}")
    if not logits.dim() == 3:
        raise ValueError(f"context.outputs.logits 必须为3维的喵! 现在的维度: {logits.dim()}")

    if not logits.shape[:2] == labels.shape:
        raise ValueError(f"labels的前两维必须和logits保持一致喵!")
    if not logits.size(-1) > 0:
        raise ValueError(f"logits的最后一维必须大于0喵!")

    if not logits.is_floating_point():
        raise ValueError(f"logits必须为浮点类型的Tensor喵! 现在的dtype: {logits.dtype}")
    if not labels.dtype == torch.long:
        raise ValueError(f"labels必须为long类型的Tensor喵! 现在的dtype: {labels.dtype}")

    if not logits.device == labels.device:
        raise ValueError(
            f"logits和labels的设备必须保持一致喵! logits的device: {logits.device} labels的device: {labels.device}")
    process_dict = {
        "labels": labels,
        "logits": logits,
    }
    return process_dict

def process_predictions_references(**context):
    if "predictions" in context:
        predictions = context["predictions"]
    else:
        if not "batch" in context:
            raise ValueError(f"context中不包含batch这个键喵!")
        if not "outputs" in context:
            raise ValueError(f"context中不包含outputs这个键喵!")
        batch = context["batch"]
        outputs = context["outputs"]
        if not isinstance(batch, dict):
            raise ValueError(f"context.batch 必须为字典类型的喵! 现在的type: {type(batch)}")
        if not batch:
            raise ValueError(f"context.batch 不能为空喵!")
        if not isinstance(outputs, dict):
            raise ValueError(f"context.outputs 必须为字典类型的喵! 现在的type: {type(outputs)}")
        if not outputs:
            raise ValueError(f"context.outputs 不能为空喵!")

        start_logits = outputs["logits"]["start_logits"]
        end_logits = outputs["logits"]["end_logits"]
        token_type_ids = batch["token_type_ids"]
        offset_mapping_list = batch["offset_mapping_list"]
        context_list = batch["context_list"]

        predictions = _calculate_qa_predictions(
            start_logits=start_logits,
            end_logits=end_logits,
            token_type_ids=token_type_ids,
            offset_mapping_list=offset_mapping_list,
            context_list=context_list,
        )
    if "references" in context:
        references = context["references"]
    else:
        if not "batch" in context:
            raise ValueError(f"context中不包含batch这个键喵!")
        batch = context["batch"]
        if not isinstance(batch, dict):
            raise ValueError(f"context.batch 必须为字典类型的喵! 现在的type: {type(batch)}")
        if not batch:
            raise ValueError(f"context.batch 不能为空喵!")
        if not "answers_list" in batch:
            raise ValueError(f"context.batch 必须包含answers_list这一字段喵!")
        answers_list = batch["answers_list"]
        references = [
            [answer["text"] for answer in answers]
            for answers in answers_list
        ]
    process_dict = {
        "predictions": predictions,
        "references": references,
    }
    return process_dict

