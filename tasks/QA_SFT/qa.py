
import torch.nn as nn

class QATask(nn.Module):
    def __init__(self, model, loss_fn):
        super(QATask, self).__init__()
        self.model = model
        self.loss_fn = loss_fn

    def forward(self, batch):
        if not isinstance(batch, dict):
            raise ValueError(f"传入的batch必须为dict类型的喵! 现在的type: {type(batch)}")
        if not batch:
            raise ValueError(f"传入的batch不能为空喵!")
        if not "input_ids" in batch:
            raise ValueError(f"batch里面没有叫 'input_ids' 的字段喵! ")
        if not "attention_mask" in batch:
            raise ValueError(f"batch里面没有叫 'attention_mask' 的字段喵! ")
        if not "token_type_ids" in batch:
            raise ValueError(f"batch里面没有叫 'token_type_ids' 的字段喵! ")

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]

        logits_dict = self.model(input_ids=input_ids, attention_mask=attention_mask,token_type_ids=token_type_ids)  # [batch_size, seq_length]
        if not isinstance(logits_dict, dict):
            raise ValueError(f"模型的输出必须为dict类型的喵! 现在的type: {type(logits_dict)}")
        if not "start_logits" in logits_dict:
            raise ValueError(f"logits_dict 必须包含 start_logits 字段喵!")
        if not "end_logits" in logits_dict:
            raise ValueError(f"logits_dict 必须包含 end_logits 字段喵!")
        start_logits = logits_dict["start_logits"]
        end_logits = logits_dict["end_logits"]
        output_dict = {
            "logits": {
                "start_logits": start_logits,
                "end_logits": end_logits,
            }
        }
        if self.loss_fn is not None:
            if not "start_positions" in batch:
                raise ValueError(f"batch 必须包含 start_positions 字段喵!")
            if not "end_positions" in batch:
                raise ValueError(f"batch 必须包含 end_positions 字段喵!")
            start_positions = batch["start_positions"]
            end_positions = batch["end_positions"]
            start_result = self.loss_fn(start_logits, start_positions)
            end_result = self.loss_fn(end_logits, end_positions)
            start_losses_dict = start_result["losses"]
            end_losses_dict = end_result["losses"]
            output_dict["losses"] = {}
            start_loss_names = set(start_losses_dict)
            end_loss_names = set(end_losses_dict)
            if start_loss_names != end_loss_names:
                raise ValueError(
                    "start 和 end 的 Loss 名称必须一致喵！"
                    f"start_loss_names: {start_loss_names}, "
                    f"end_loss_names: {end_loss_names}"
                )
            for loss_name, start_loss_info in start_losses_dict.items():
                end_loss_info = end_losses_dict[loss_name]
                start_loss_raw = start_loss_info["raw"]
                end_loss_raw = end_loss_info["raw"]
                start_loss_weight = start_loss_info["weight"]
                end_loss_weight = end_loss_info["weight"]
                if start_loss_weight != end_loss_weight:
                    raise ValueError(f"start和loss对应loss的权重必须一致喵! loss_name: {loss_name} start_loss_weight: {start_loss_weight} end_loss_weight: {end_loss_weight}")
                weight = start_loss_weight
                start_loss_weighted = start_loss_info["weighted"]
                end_loss_weighted = end_loss_info["weighted"]
                output_dict["losses"][loss_name] = {
                    "raw": (start_loss_raw+end_loss_raw)/2,
                    "weight": weight,
                    "weighted": (start_loss_weighted+end_loss_weighted)/2,
                }
            total_loss = sum(
                loss["weighted"] for loss in output_dict["losses"].values()
            )
            output_dict["loss"] = total_loss
            output_dict["details"] = {
                "start_loss_details": start_result,
                "end_loss_details": end_result,
            }

        return output_dict