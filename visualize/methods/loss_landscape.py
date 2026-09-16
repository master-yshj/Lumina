import pathlib
import torch

import torch.nn as nn

from visualize.base import BaseVisualizer
from math import isfinite

class LossLandscapeVisualizer(BaseVisualizer):
    def __init__(self, normalization, output_format, grid_size=11, alpha_min=-1.0, alpha_max=1.0, beta_min=-1.0, beta_max=1.0, batch_count=1, loss_fn=None):
        if not type(grid_size) is int:
            raise TypeError(f"grid_size: {grid_size} 必须为int类型的喵! 现在的type: {type(grid_size)}")
        if grid_size < 2:
            raise ValueError(f"grid_size: {grid_size} 必须至少为2喵!")
        if not isinstance(alpha_min, float) and not type(alpha_min) is int:
            raise TypeError(f"alpha_min: {alpha_min} 必须为float类型的喵! 现在的type: {type(alpha_min)}")
        if not isinstance(alpha_max, float) and not type(alpha_max) is int:
            raise TypeError(f"alpha_max: {alpha_max} 必须为float类型的喵! 现在的type: {type(alpha_max)}")
        if alpha_min >= alpha_max:
            raise ValueError(f"alpha_min: {alpha_min} 必须小于 alpha_max: {alpha_max} 喵!")
        if not isinstance(beta_min, float) and not type(beta_min) is int:
            raise TypeError(f"beta_min: {beta_min} 必须为float类型的喵! 现在的type: {type(beta_min)}")
        if not isinstance(beta_max, float) and not type(beta_max) is int:
            raise TypeError(f"beta_max: {beta_max} 必须为float类型的喵! 现在的type: {type(beta_max)}")
        if beta_min >= beta_max:
            raise ValueError(f"beta_min: {beta_min} 必须小于 beta_max: {beta_max} 喵!")
        if not type(batch_count) is int:
            raise TypeError(f"batch_count: {batch_count} 必须为int类型的喵! 现在的type: {type(batch_count)}")
        if loss_fn is not None and not isinstance(loss_fn, nn.Module):
            raise TypeError(f"loss_fn: {loss_fn} 必须为nn.Module类型或None喵! 现在的type: {type(loss_fn)}")
        if batch_count < 1:
            raise ValueError(f"batch_count: {batch_count} 必须至少为1喵!")
        if not type(normalization) is str:
            raise TypeError(f"normalization: {normalization} 必须为str类型的喵! 现在的type: {type(normalization)}")
        if not normalization.strip():
            raise ValueError(f"normalization 不能为空喵!")
        normalized_normalization = normalization.strip().casefold()
        allowed_normalizations = ("per_parameter", "global")
        if normalized_normalization not in allowed_normalizations:
            raise ValueError(f"normalization: {normalization} 必须为 {allowed_normalizations} 中的一个喵!")
        if not type(output_format) is str:
            raise TypeError(f"output_format: {output_format} 必须为str类型的喵! 现在的type: {type(output_format)}")
        if not output_format.strip():
            raise ValueError(f"output_format 不能为空喵!")
        normalized_output_format = output_format.strip().casefold()
        allowed_output_formats = ("png", "html", "both")
        if normalized_output_format not in allowed_output_formats:
            raise ValueError(f"output_format: {output_format} 必须为 {allowed_output_formats} 中的一个喵!")

        super().__init__(
            normalization=normalized_normalization,
            output_format=normalized_output_format,
            grid_size=grid_size,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            beta_min=beta_min,
            beta_max=beta_max,
            batch_count=batch_count,
            loss_fn=loss_fn,
        )

        self.loss_fn = loss_fn
        self.normalization = normalized_normalization
        self.output_format = normalized_output_format
        self.grid_size = grid_size
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.batch_count = batch_count

    def _validate_batch(self, batch):
        required_keys = (
            "input_ids",
            "attention_mask",
            "token_type_ids",
        )
        if not isinstance(batch, dict):
            raise TypeError(
                f"batch: {batch} 必须为dict类型的喵! 现在的type: {type(batch)}"
            )
        for batch_key in required_keys:
            if batch_key not in batch:
                raise KeyError(
                    f"batch 中缺少必需字段: {batch_key} 喵!"
                )
        for batch_key in required_keys:
            batch_value = batch[batch_key]
            if not isinstance(batch_value, torch.Tensor):
                raise TypeError(
                    f"batch['{batch_key}'] 必须为torch.Tensor类型的喵! "
                    f"现在的type: {type(batch_value)}"
                )
            if batch_value.dim() != 2:
                raise ValueError(
                    f"batch['{batch_key}'] 必须为二维Tensor喵! "
                    f"现在的dim: {batch_value.dim()}"
                )
        expected_shape = batch["input_ids"].shape
        for batch_key in required_keys:
            if batch[batch_key].shape != expected_shape:
                raise ValueError(
                    f"batch['{batch_key}'] 的shape: {batch[batch_key].shape} "
                    f"必须与input_ids的shape: {expected_shape}一致喵!"
                )

        if "labels" in batch:
            if not isinstance(batch["labels"], torch.Tensor):
                raise TypeError("batch['labels'] 必须为torch.Tensor类型的喵!")
            if batch["labels"].shape != expected_shape:
                raise ValueError("labels的shape必须与input_ids一致喵!")
        else:
            for label_key in ("start_positions", "end_positions"):
                if label_key not in batch:
                    raise KeyError(f"QA batch 中缺少必需字段: {label_key} 喵!")
                if not isinstance(batch[label_key], torch.Tensor):
                    raise TypeError(f"batch['{label_key}'] 必须为torch.Tensor类型的喵!")
                if batch[label_key].dim() != 1:
                    raise ValueError(f"batch['{label_key}'] 必须为一维Tensor喵!")
                if batch[label_key].shape[0] != expected_shape[0]:
                    raise ValueError(f"batch['{label_key}'] 的batch维度必须与input_ids一致喵!")
        batch_size, seq_length = expected_shape
        if batch_size <= 0:
            raise ValueError(
                f"batch_size: {batch_size} 必须为正整数喵!"
            )
        if seq_length <= 0:
            raise ValueError(
                f"seq_length: {seq_length} 必须为正整数喵!"
            )

    def _collect_fixed_batches(self, dataloader):
        batch_iterator = iter(dataloader)
        fixed_batches = []
        actual_batch_count = 0

        while actual_batch_count < self.batch_count:
            try:
                batch = next(batch_iterator)
            except StopIteration:
                break

            self._validate_batch(batch)

            fixed_batch = {}
            for batch_key, batch_value in batch.items():
                if isinstance(batch_value, torch.Tensor):
                    fixed_batch[batch_key] = (
                        batch_value.detach()
                        .cpu()
                        .clone()
                    )
                else:
                    fixed_batch[batch_key] = batch_value

            fixed_batches.append(fixed_batch)
            actual_batch_count += 1

        if actual_batch_count < self.batch_count:
            raise ValueError(
                f"dataloader中的批次数量不足喵! "
                f"需要: {self.batch_count}, "
                f"实际获取: {actual_batch_count}"
            )

        return fixed_batches

    def _snapshot_parameters(self, model):
        if not isinstance(model, nn.Module):
            raise TypeError(f"model 的type必须为nn.Module喵! 现在的type: {type(model)}")
        base_parameters = {}
        for parameter_name, parameter in model.named_parameters():
            if not isinstance(parameter_name, str):
                raise TypeError(f"parameter_name: {parameter_name} 必须为str类型的喵! 现在的type: {type(parameter_name)}")
            if not parameter_name.strip():
                raise ValueError(f"parameter_name 不能为空喵!")
            if not isinstance(parameter, torch.Tensor):
                raise TypeError(f"parameter: {parameter} 必须为torch.Tensor类型的喵! 现在的type: {type(parameter)}")
            if not parameter.is_floating_point():
                raise TypeError(f"parameter: {parameter} 必须为浮点类型的张量喵! 现在的dtype: {parameter.dtype}")
            if not torch.isfinite(parameter).all():
                raise ValueError(f"parameter: {parameter} 的值非法喵!")
            copy_parameter = parameter.clone().detach().cpu()
            base_parameters[parameter_name] = copy_parameter
        if not base_parameters:
            raise ValueError(f"model 中没有任何参数喵!")
        return base_parameters

    def _generate_directions(self, base_parameters):
        if not isinstance(base_parameters, dict):
            raise TypeError(
                f"base_parameters: {base_parameters} 必须为dict类型的喵! "
                f"现在的type: {type(base_parameters)}"
            )
        if not base_parameters:
            raise ValueError("base_parameters 不能为空喵!")

        direction_1 = {}
        direction_2 = {}

        for parameter_name, base_parameter in base_parameters.items():
            if not isinstance(parameter_name, str):
                raise TypeError(
                    f"parameter_name: {parameter_name} 必须为str类型的喵! "
                    f"现在的type: {type(parameter_name)}"
                )
            if not parameter_name.strip():
                raise ValueError("parameter_name 不能为空喵!")
            if not isinstance(base_parameter, torch.Tensor):
                raise TypeError(
                    f"base_parameters['{parameter_name}'] 必须为torch.Tensor类型的喵! "
                    f"现在的type: {type(base_parameter)}"
                )
            if not base_parameter.is_floating_point():
                raise TypeError(
                    f"base_parameters['{parameter_name}'] 必须为浮点Tensor喵! "
                    f"现在的dtype: {base_parameter.dtype}"
                )
            if not torch.isfinite(base_parameter).all().item():
                raise ValueError(
                    f"base_parameters['{parameter_name}'] 中包含非法值喵!"
                )

            random_direction_1 = torch.randn_like(base_parameter)
            random_direction_2 = torch.randn_like(base_parameter)

            direction_norm_1 = torch.linalg.vector_norm(random_direction_1)
            direction_norm_2 = torch.linalg.vector_norm(random_direction_2)

            if not torch.isfinite(direction_norm_1).item():
                raise ValueError(
                    f"direction_1['{parameter_name}'] 的范数包含非法值喵!"
                )
            if not torch.isfinite(direction_norm_2).item():
                raise ValueError(
                    f"direction_2['{parameter_name}'] 的范数包含非法值喵!"
                )
            if direction_norm_1.item() == 0.0:
                raise ValueError(
                    f"direction_1['{parameter_name}'] 的范数不能为0喵!"
                )
            if direction_norm_2.item() == 0.0:
                raise ValueError(
                    f"direction_2['{parameter_name}'] 的范数不能为0喵!"
                )

            direction_1[parameter_name] = random_direction_1
            direction_2[parameter_name] = random_direction_2

        if self.normalization == "per_parameter":
            for parameter_name, base_parameter in base_parameters.items():
                parameter_norm = torch.linalg.vector_norm(base_parameter)

                if not torch.isfinite(parameter_norm).item():
                    raise ValueError(
                        f"base_parameters['{parameter_name}'] 的范数包含非法值喵!"
                    )

                if parameter_norm.item() == 0.0:
                    direction_1[parameter_name].zero_()
                    direction_2[parameter_name].zero_()
                    continue

                direction_norm_1 = torch.linalg.vector_norm(
                    direction_1[parameter_name]
                )
                direction_norm_2 = torch.linalg.vector_norm(
                    direction_2[parameter_name]
                )

                direction_1[parameter_name].mul_(
                    parameter_norm / direction_norm_1
                )
                direction_2[parameter_name].mul_(
                    parameter_norm / direction_norm_2
                )

        elif self.normalization == "global":
            base_squared_norm = torch.zeros(
                (),
                dtype=torch.float64,
            )
            direction_1_squared_norm = torch.zeros(
                (),
                dtype=torch.float64,
            )
            direction_2_squared_norm = torch.zeros(
                (),
                dtype=torch.float64,
            )

            for parameter_name, base_parameter in base_parameters.items():
                base_parameter_float64 = base_parameter.to(dtype=torch.float64)
                direction_1_float64 = direction_1[parameter_name].to(dtype=torch.float64)
                direction_2_float64 = direction_2[parameter_name].to(dtype=torch.float64)

                base_squared_norm += torch.sum(
                    base_parameter_float64.square()
                )
                direction_1_squared_norm += torch.sum(
                    direction_1_float64.square()
                )
                direction_2_squared_norm += torch.sum(
                    direction_2_float64.square()
                )

            base_norm = torch.sqrt(base_squared_norm)
            direction_1_norm = torch.sqrt(direction_1_squared_norm)
            direction_2_norm = torch.sqrt(direction_2_squared_norm)

            if not torch.isfinite(base_norm).item():
                raise ValueError("base_parameters 的全局范数包含非法值喵!")
            if not torch.isfinite(direction_1_norm).item():
                raise ValueError("direction_1 的全局范数包含非法值喵!")
            if not torch.isfinite(direction_2_norm).item():
                raise ValueError("direction_2 的全局范数包含非法值喵!")
            if direction_1_norm.item() == 0.0:
                raise ValueError("direction_1 的全局范数不能为0喵!")
            if direction_2_norm.item() == 0.0:
                raise ValueError("direction_2 的全局范数不能为0喵!")

            direction_1_scale = base_norm / direction_1_norm
            direction_2_scale = base_norm / direction_2_norm

            for parameter_name in base_parameters:
                direction_1[parameter_name].mul_(
                    direction_1_scale.to(
                        dtype=direction_1[parameter_name].dtype
                    )
                )
                direction_2[parameter_name].mul_(
                    direction_2_scale.to(
                        dtype=direction_2[parameter_name].dtype
                    )
                )

        else:
            raise ValueError(
                f"normalization: {self.normalization} "
                f"不是受支持的归一化方式喵!"
            )

        for parameter_name, base_parameter in base_parameters.items():
            current_direction_1 = direction_1[parameter_name]
            current_direction_2 = direction_2[parameter_name]

            if current_direction_1.shape != base_parameter.shape:
                raise ValueError(
                    f"direction_1['{parameter_name}'] 的shape: "
                    f"{current_direction_1.shape} 必须与base_parameters"
                    f"['{parameter_name}'] 的shape: {base_parameter.shape}一致喵!"
                )
            if current_direction_2.shape != base_parameter.shape:
                raise ValueError(
                    f"direction_2['{parameter_name}'] 的shape: "
                    f"{current_direction_2.shape} 必须与base_parameters"
                    f"['{parameter_name}'] 的shape: {base_parameter.shape}一致喵!"
                )
            if not current_direction_1.is_floating_point():
                raise TypeError(
                    f"direction_1['{parameter_name}'] 必须为浮点Tensor喵!"
                )
            if not current_direction_2.is_floating_point():
                raise TypeError(
                    f"direction_2['{parameter_name}'] 必须为浮点Tensor喵!"
                )
            if not torch.isfinite(current_direction_1).all().item():
                raise ValueError(
                    f"direction_1['{parameter_name}'] 中包含非法值喵!"
                )
            if not torch.isfinite(current_direction_2).all().item():
                raise ValueError(
                    f"direction_2['{parameter_name}'] 中包含非法值喵!"
                )

        return direction_1, direction_2

    def _build_coordinates(self):
        x = torch.linspace(
            start=self.alpha_min,
            end=self.alpha_max,
            steps=self.grid_size,
        )
        y = torch.linspace(
            start=self.beta_min,
            end=self.beta_max,
            steps=self.grid_size,
        )

        alpha_min_tensor = torch.as_tensor(
            self.alpha_min,
            dtype=x.dtype,
            device=x.device,
        )
        alpha_max_tensor = torch.as_tensor(
            self.alpha_max,
            dtype=x.dtype,
            device=x.device,
        )
        beta_min_tensor = torch.as_tensor(
            self.beta_min,
            dtype=y.dtype,
            device=y.device,
        )
        beta_max_tensor = torch.as_tensor(
            self.beta_max,
            dtype=y.dtype,
            device=y.device,
        )

        if not len(x) == len(y) == self.grid_size:
            raise ValueError(
                f"x的长度: {len(x)} 与 y的长度: {len(y)} "
                f"必须与grid_size: {self.grid_size} 一样喵!"
            )
        if not torch.isclose(x[0], alpha_min_tensor).item():
            raise ValueError(
                f"x[0]: {x[0]} 与 alpha_min: {self.alpha_min} "
                f"必须一致喵!"
            )
        if not torch.isclose(x[-1], alpha_max_tensor).item():
            raise ValueError(
                f"x[-1]: {x[-1]} 与 alpha_max: {self.alpha_max} "
                f"必须一致喵!"
            )
        if not torch.isclose(y[0], beta_min_tensor).item():
            raise ValueError(
                f"y[0]: {y[0]} 与 beta_min: {self.beta_min} "
                f"必须一致喵!"
            )
        if not torch.isclose(y[-1], beta_max_tensor).item():
            raise ValueError(
                f"y[-1]: {y[-1]} 与 beta_max: {self.beta_max} "
                f"必须一致喵!"
            )

        return x, y

    def _apply_perturbation(self, base_parameters, direction_1, direction_2, alpha, beta):
        if not isinstance(base_parameters, dict):
            raise TypeError(f"base_parameters: {base_parameters} 必须为dict类型的喵! 现在的type: {type(base_parameters)}")
        if not isinstance(direction_1, dict):
            raise TypeError(f"direction_1: {direction_1} 必须为dict类型的喵! 现在的type: {type(direction_1)}")
        if not isinstance(direction_2, dict):
            raise TypeError(f"direction_2: {direction_2} 必须为dict类型的喵! 现在的type: {type(direction_2)}")
        if not set(base_parameters.keys()) == set(direction_1.keys()) == set(direction_2.keys()):
            raise ValueError(f"base_parameters: {base_parameters}\n"
                             f"direction_1: {direction_1}\n"
                             f"direction_2: {direction_2}\n"
                             f"三者key的集合必须一致喵!")
        if isinstance(alpha, float) or type(alpha) is int:
            if not isfinite(alpha):
                raise ValueError(f"alpha: {alpha} 的值不合法喵!")
        elif isinstance(alpha, torch.Tensor):
            if not alpha.is_floating_point():
                raise TypeError(f"alpha: {alpha} 为torch.Tensor类型时必须为浮点类型喵! 现在的dtype: {alpha.dtype}")
            if not torch.isfinite(alpha).all().item():
                raise ValueError(f"alpha: {alpha} 的值不合法喵!")
            if not alpha.dim() == 0:
                raise ValueError(f"alpha.dim: {alpha.dim()} 必须为0喵!")
        else:
            raise TypeError(f"alpha: {alpha} 必须为float或int或浮点类型torch.Tensor类型的喵! 现在的type: {type(alpha)}")
        if isinstance(beta, float) or type(beta) is int:
            if not isfinite(beta):
                raise ValueError(f"beta: {beta} 的值不合法喵!")
        elif isinstance(beta, torch.Tensor):
            if not beta.is_floating_point():
                raise TypeError(f"beta: {beta} 为torch.Tensor类型时必须为浮点类型喵! 现在的dtype: {beta.dtype}")
            if not torch.isfinite(beta).all().item():
                raise ValueError(f"beta: {beta} 的值不合法喵!")
            if not beta.dim() == 0:
                raise ValueError(f"beta.dim: {beta.dim()} 必须为0喵!")
        else:
            raise TypeError(f"beta: {beta} 必须为float或int或浮点类型torch.Tensor类型的喵! 现在的type: {type(beta)}")
        perturbed_parameters = {}
        for parameter_name, base_parameter in base_parameters.items():
            direction_1_parameter = direction_1[parameter_name]
            direction_2_parameter = direction_2[parameter_name]
            if not isinstance(base_parameter, torch.Tensor):
                raise TypeError(f"base_parameter: {base_parameter} 必须为torch.Tensor张量类型的喵! 现在的type: {type(base_parameter)}")
            if not isinstance(direction_1_parameter, torch.Tensor):
                raise TypeError(f"direction_1_parameter: {direction_1_parameter} 必须为torch.Tensor张量类型的喵! 现在的type: {type(direction_1_parameter)}")
            if not isinstance(direction_2_parameter, torch.Tensor):
                raise TypeError(f"direction_2_parameter: {direction_2_parameter} 必须为torch.Tensor张量类型的喵! 现在的type: {type(direction_2_parameter)}")
            if not base_parameter.is_floating_point():
                raise TypeError(f"base_parameter: {base_parameter} 必须为浮点类型张量喵! 现在的dtype: {base_parameter.dtype}")
            if not direction_1_parameter.is_floating_point():
                raise TypeError(f"direction_1_parameter: {direction_1_parameter} 必须为浮点类型张量喵! 现在的dtype: {direction_1_parameter.dtype}")
            if not direction_2_parameter.is_floating_point():
                raise TypeError(f"direction_2_parameter: {direction_2_parameter} 必须为浮点类型张量喵! 现在的dtype: {direction_2_parameter.dtype}")
            if not direction_1_parameter.shape == base_parameter.shape:
                raise ValueError(f"direction_1[{parameter_name}] 的形状: {direction_1_parameter.shape}\n"
                                 f"base_parameters[{parameter_name}] 的形状: {base_parameter.shape}\n"
                                 f"两者必须一致喵!")
            if not direction_2_parameter.shape == base_parameter.shape:
                raise ValueError(f"direction_2[{parameter_name}] 的形状: {direction_2_parameter.shape}\n"
                                 f"base_parameters[{parameter_name}] 的形状: {base_parameter.shape}\n"
                                 f"两者必须一致喵!")
            if not torch.isfinite(base_parameter).all().item():
                raise ValueError(f"base_parameter: {base_parameter} 的值不合法喵!")
            if not torch.isfinite(direction_1_parameter).all().item():
                raise ValueError(f"direction_1_parameter: {direction_1_parameter} 的值不合法喵!")
            if not torch.isfinite(direction_2_parameter).all().item():
                raise ValueError(f"direction_2_parameter: {direction_2_parameter} 的值不合法喵!")
            if isinstance(alpha, torch.Tensor):
                alpha_value = alpha.to(dtype=base_parameter.dtype, device=base_parameter.device)
            else:
                alpha_value = torch.as_tensor(alpha, dtype=base_parameter.dtype, device=base_parameter.device)
            if isinstance(beta, torch.Tensor):
                beta_value = beta.to(dtype=base_parameter.dtype, device=base_parameter.device)
            else:
                beta_value = torch.as_tensor(beta, dtype=base_parameter.dtype, device=base_parameter.device)
            perturbed_parameter = base_parameter + alpha_value * direction_1_parameter + beta_value * direction_2_parameter
            if not perturbed_parameter.shape == base_parameter.shape:
                raise ValueError(f"perturbed_parameter的形状: {perturbed_parameter.shape}\n"
                                 f"base_parameter的形状: {base_parameter.shape}\n"
                                 f"两者必须一致喵!")
            if not perturbed_parameter.is_floating_point():
                raise TypeError(f"perturbed_parameter: {perturbed_parameter} 必须为浮点类型张量喵! 现在的dtype: {perturbed_parameter.dtype}")
            if not torch.isfinite(perturbed_parameter).all().item():
                raise ValueError(f"perturbed_parameter: {perturbed_parameter} 的值不合法喵!")
            perturbed_parameters[parameter_name] = perturbed_parameter
        if not set(perturbed_parameters.keys())==set(base_parameters.keys()):
            raise ValueError(f"perturbed_parameters 的key集合: {set(perturbed_parameters.keys())} 与 base_parameters 的key集合: {set(base_parameters.keys())} 不一致喵!")
        return perturbed_parameters

    def _evaluate_single_loss(self, model, fixed_batches, device):
        if self.loss_fn is None:
            raise ValueError(
                "loss_fn 不能为空，loss landscape 必须提供损失函数喵!"
            )
        if not isinstance(self.loss_fn, nn.Module):
            raise TypeError(
                f"loss_fn: {self.loss_fn} 必须为nn.Module类型的喵! "
                f"现在的type: {type(self.loss_fn)}"
            )
        if not isinstance(model, nn.Module):
            raise TypeError(
                f"model: {model} 必须为nn.Module类型的喵! "
                f"现在的type: {type(model)}"
            )
        if not isinstance(fixed_batches, list):
            raise TypeError(
                f"fixed_batches: {fixed_batches} 必须为list类型的喵! "
                f"现在的type: {type(fixed_batches)}"
            )
        if not fixed_batches:
            raise ValueError("fixed_batches 不能为空喵!")

        device = torch.device(device)
        total_loss = 0.0

        with torch.no_grad():
            for batch_index, fixed_batch in enumerate(fixed_batches):
                self._validate_batch(fixed_batch)

                model_batch = {}
                for batch_key, batch_value in fixed_batch.items():
                    if isinstance(batch_value, torch.Tensor):
                        model_batch[batch_key] = batch_value.to(device=device)
                    else:
                        model_batch[batch_key] = batch_value

                logits = model(
                    input_ids=model_batch["input_ids"],
                    attention_mask=model_batch["attention_mask"],
                    token_type_ids=model_batch["token_type_ids"],
                )

                if "labels" in model_batch:
                    if not isinstance(logits, torch.Tensor):
                        raise TypeError("MLM model输出必须为torch.Tensor类型的喵!")
                    model_logits = logits
                    labels = model_batch["labels"]
                    if model_logits.ndim != 3 or labels.ndim != 2:
                        raise ValueError("MLM logits必须为三维、labels必须为二维Tensor喵!")
                    flattened_logits = model_logits.reshape(-1, model_logits.shape[-1])
                    flattened_labels = labels.reshape(-1)
                    loss_result = self.loss_fn(logits=flattened_logits, labels=flattened_labels)
                else:
                    if not isinstance(logits, dict):
                        raise TypeError("QA model输出必须为dict类型的喵!")
                    if "start_logits" not in logits or "end_logits" not in logits:
                        raise KeyError("QA model输出必须包含start_logits和end_logits字段喵!")
                    start_positions = model_batch["start_positions"]
                    end_positions = model_batch["end_positions"]
                    start_result = self.loss_fn(logits=logits["start_logits"], labels=start_positions)
                    end_result = self.loss_fn(logits=logits["end_logits"], labels=end_positions)
                    if not isinstance(start_result, dict) or not isinstance(end_result, dict):
                        raise TypeError("loss_fn返回值必须为dict类型的喵!")
                    if "loss" not in start_result or "loss" not in end_result:
                        raise KeyError("loss_fn返回值中缺少loss字段喵!")
                    batch_loss = (start_result["loss"] + end_result["loss"]) / 2

                if "labels" in model_batch:
                    if not isinstance(loss_result, dict) or "loss" not in loss_result:
                        raise KeyError("loss_fn返回值中缺少loss字段喵!")
                    batch_loss = loss_result["loss"]

                if not isinstance(batch_loss, torch.Tensor):
                    raise TypeError(
                        f"loss_result['loss']必须为torch.Tensor类型的喵! "
                        f"现在的type: {type(batch_loss)}"
                    )
                if batch_loss.ndim != 0:
                    raise ValueError(
                        f"batch_loss必须为标量Tensor喵! "
                        f"现在的shape: {tuple(batch_loss.shape)}"
                    )
                if not torch.isfinite(batch_loss).item():
                    raise ValueError(
                        f"第{batch_index}个batch的loss值非法喵!"
                    )

                total_loss += batch_loss.detach().item()

        mean_loss = total_loss / len(fixed_batches)

        if not isfinite(mean_loss):
            raise ValueError(
                f"mean_loss: {mean_loss} 的值非法喵!"
            )

        return mean_loss

    def _apply_parameters_to_model(self, model, perturbed_parameters):
        if not isinstance(model, nn.Module):
            raise TypeError(
                f"model: {model} 必须为nn.Module类型的喵! "
                f"现在的type: {type(model)}"
            )
        if not isinstance(perturbed_parameters, dict):
            raise TypeError(
                f"perturbed_parameters: {perturbed_parameters} "
                f"必须为dict类型的喵! "
                f"现在的type: {type(perturbed_parameters)}"
            )
        if not perturbed_parameters:
            raise ValueError(
                "perturbed_parameters 不能为空喵!"
            )

        model_parameters = dict(model.named_parameters())
        model_parameter_names = set(model_parameters.keys())
        perturbed_parameter_names = set(perturbed_parameters.keys())

        if model_parameter_names != perturbed_parameter_names:
            missing_names = model_parameter_names - perturbed_parameter_names
            extra_names = perturbed_parameter_names - model_parameter_names
            raise ValueError(
                f"perturbed_parameters 的参数名集合与model不一致喵! "
                f"缺少参数: {missing_names}, "
                f"多余参数: {extra_names}"
            )

        with torch.no_grad():
            for parameter_name, parameter in model_parameters.items():
                perturbed_parameter = perturbed_parameters[parameter_name]

                if not isinstance(parameter, nn.Parameter):
                    raise TypeError(
                        f"model参数 {parameter_name} 必须为nn.Parameter类型喵! "
                        f"现在的type: {type(parameter)}"
                    )
                if not isinstance(perturbed_parameter, torch.Tensor):
                    raise TypeError(
                        f"perturbed_parameters['{parameter_name}'] "
                        f"必须为torch.Tensor类型喵! "
                        f"现在的type: {type(perturbed_parameter)}"
                    )
                if not perturbed_parameter.is_floating_point():
                    raise TypeError(
                        f"perturbed_parameters['{parameter_name}'] "
                        f"必须为浮点Tensor喵! "
                        f"现在的dtype: {perturbed_parameter.dtype}"
                    )
                if parameter.shape != perturbed_parameter.shape:
                    raise ValueError(
                        f"模型参数 {parameter_name} 的shape: {parameter.shape} "
                        f"与扰动参数的shape: {perturbed_parameter.shape} "
                        f"不一致喵!"
                    )
                if not torch.isfinite(perturbed_parameter).all().item():
                    raise ValueError(
                        f"perturbed_parameters['{parameter_name}'] "
                        f"包含非法值喵!"
                    )

                parameter.copy_(
                    perturbed_parameter.to(
                        device=parameter.device,
                        dtype=parameter.dtype,
                    )
                )

    def _evaluate_loss_grid(
        self,
        model,
        fixed_batches,
        device,
        base_parameters,
        direction_1,
        direction_2,
        alpha_values,
        beta_values,
    ):
        if not isinstance(model, nn.Module):
            raise TypeError(
                f"model: {model} 必须为nn.Module类型的喵! "
                f"现在的type: {type(model)}"
            )
        if not isinstance(alpha_values, torch.Tensor):
            raise TypeError(
                f"alpha_values: {alpha_values} 必须为torch.Tensor类型的喵! "
                f"现在的type: {type(alpha_values)}"
            )
        if not isinstance(beta_values, torch.Tensor):
            raise TypeError(
                f"beta_values: {beta_values} 必须为torch.Tensor类型的喵! "
                f"现在的type: {type(beta_values)}"
            )
        if alpha_values.ndim != 1:
            raise ValueError(
                f"alpha_values必须为一维Tensor喵! "
                f"现在的shape: {tuple(alpha_values.shape)}"
            )
        if beta_values.ndim != 1:
            raise ValueError(
                f"beta_values必须为一维Tensor喵! "
                f"现在的shape: {tuple(beta_values.shape)}"
            )
        if alpha_values.numel() != self.grid_size:
            raise ValueError(
                f"alpha_values的数量: {alpha_values.numel()} "
                f"必须等于grid_size: {self.grid_size} 喵!"
            )
        if beta_values.numel() != self.grid_size:
            raise ValueError(
                f"beta_values的数量: {beta_values.numel()} "
                f"必须等于grid_size: {self.grid_size} 喵!"
            )
        if not torch.isfinite(alpha_values).all().item():
            raise ValueError("alpha_values中包含非法值喵!")
        if not torch.isfinite(beta_values).all().item():
            raise ValueError("beta_values中包含非法值喵!")

        loss_grid = torch.empty(
            (alpha_values.numel(), beta_values.numel()),
            dtype=torch.float64,
        )

        model.eval()

        for row_index, alpha in enumerate(alpha_values):
            for column_index, beta in enumerate(beta_values):
                perturbed_parameters = self._apply_perturbation(
                    base_parameters=base_parameters,
                    direction_1=direction_1,
                    direction_2=direction_2,
                    alpha=alpha,
                    beta=beta,
                )

                self._apply_parameters_to_model(
                    model=model,
                    perturbed_parameters=perturbed_parameters,
                )

                point_loss = self._evaluate_single_loss(
                    model=model,
                    fixed_batches=fixed_batches,
                    device=device,
                )

                if not isfinite(point_loss):
                    raise ValueError(
                        f"网格点loss非法喵! "
                        f"row_index: {row_index}, "
                        f"column_index: {column_index}, "
                        f"alpha: {alpha.item()}, "
                        f"beta: {beta.item()}, "
                        f"loss: {point_loss}"
                    )

                loss_grid[row_index, column_index] = point_loss

        if not torch.isfinite(loss_grid).all().item():
            raise ValueError("loss_grid中包含非法值喵!")

        return loss_grid

    def _restore_parameters(self, model, parameter_snapshot):
        if not isinstance(model, nn.Module):
            raise TypeError(
                f"model: {model} 必须为nn.Module类型的喵! 现在的type: {type(model)}"
            )
        if not isinstance(parameter_snapshot, dict) or not parameter_snapshot:
            raise ValueError("parameter_snapshot必须为非空dict喵!")

        model_parameters = dict(model.named_parameters())
        if set(model_parameters) != set(parameter_snapshot):
            raise ValueError("parameter_snapshot与model的参数名集合不一致喵!")

        with torch.no_grad():
            for parameter_name, parameter in model_parameters.items():
                snapshot = parameter_snapshot[parameter_name]
                if not isinstance(snapshot, torch.Tensor):
                    raise TypeError(
                        f"parameter_snapshot['{parameter_name}']必须为Tensor喵!"
                    )
                if snapshot.shape != parameter.shape:
                    raise ValueError(
                        f"parameter_snapshot['{parameter_name}']的shape与模型不一致喵!"
                    )
                parameter.copy_(snapshot.to(device=parameter.device, dtype=parameter.dtype))

    def _save_loss_values(self, save_dir, alpha_values, beta_values, loss_grid):
        save_dir = pathlib.Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        if not isinstance(loss_grid, torch.Tensor) or loss_grid.ndim != 2:
            raise ValueError("loss_grid必须为二维Tensor喵!")
        if loss_grid.shape != (alpha_values.numel(), beta_values.numel()):
            raise ValueError("loss_grid形状与坐标数量不一致喵!")

        values_path = save_dir / "loss_landscape_values.npz"
        import numpy as np
        np.savez(
            values_path,
            alpha=alpha_values.detach().cpu().numpy(),
            beta=beta_values.detach().cpu().numpy(),
            loss=loss_grid.detach().cpu().numpy(),
        )
        return values_path

    def _render_static_surface(self, save_dir, alpha_values, beta_values, loss_grid):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        alpha_grid, beta_grid = np.meshgrid(
            alpha_values.detach().cpu().numpy(),
            beta_values.detach().cpu().numpy(),
            indexing="ij",
        )
        loss_array = loss_grid.detach().cpu().numpy()
        figure = plt.figure(figsize=(10, 8))
        try:
            axis = figure.add_subplot(111, projection="3d")
            surface = axis.plot_surface(
                alpha_grid,
                beta_grid,
                loss_array,
                cmap="viridis",
                linewidth=0,
                antialiased=True,
            )
            axis.set_xlabel("alpha")
            axis.set_ylabel("beta")
            axis.set_zlabel("loss")
            axis.set_title("Loss Landscape")
            figure.colorbar(surface, ax=axis, shrink=0.6, pad=0.1, label="loss")
            figure.tight_layout()
            output_path = pathlib.Path(save_dir) / "loss_landscape.png"
            figure.savefig(output_path, dpi=150)
        finally:
            plt.close(figure)
        return output_path

    def _render_interactive_surface(self, save_dir, alpha_values, beta_values, loss_grid):
        import plotly.graph_objects as go

        alpha_array = alpha_values.detach().cpu().tolist()
        beta_array = beta_values.detach().cpu().tolist()
        loss_array = loss_grid.detach().cpu().transpose(0, 1).tolist()
        figure = go.Figure(
            data=[
                go.Surface(
                    x=alpha_array,
                    y=beta_array,
                    z=loss_array,
                    colorscale="Viridis",
                    colorbar={"title": "loss"},
                )
            ]
        )
        figure.update_layout(
            title="Loss Landscape",
            scene={
                "xaxis_title": "alpha",
                "yaxis_title": "beta",
                "zaxis_title": "loss",
            },
        )
        output_path = pathlib.Path(save_dir) / "loss_landscape.html"
        figure.write_html(output_path, include_plotlyjs="cdn")
        return output_path

    def visualize(self, model, tokenizer, dataloader, save_dir, device, logger):
        if isinstance(save_dir, str):
            if not save_dir:
                raise ValueError(f"save_dir: {save_dir} 为str类型时不能为空喵!")
        save_dir = pathlib.Path(save_dir)
        if not save_dir.exists():
            save_dir.mkdir(parents=True, exist_ok=True)
        if not hasattr(tokenizer, "convert_ids_to_tokens"):
            raise ValueError(f"tokenizer 必须具有convert_ids_to_tokens方法喵!")
        device = torch.device(device)
        was_training = model.training
        parameter_snapshot = self._snapshot_parameters(model)
        model.eval()
        try:
            fixed_batches = self._collect_fixed_batches(dataloader)
            direction_1, direction_2 = self._generate_directions(parameter_snapshot)
            alpha_values, beta_values = self._build_coordinates()
            loss_grid = self._evaluate_loss_grid(
                model=model,
                fixed_batches=fixed_batches,
                device=device,
                base_parameters=parameter_snapshot,
                direction_1=direction_1,
                direction_2=direction_2,
                alpha_values=alpha_values,
                beta_values=beta_values,
            )
            output_paths = []
            if self.output_format in ("png", "both"):
                output_paths.append(
                    self._render_static_surface(
                        save_dir, alpha_values, beta_values, loss_grid
                    )
                )
            if self.output_format in ("html", "both"):
                output_paths.append(
                    self._render_interactive_surface(
                        save_dir, alpha_values, beta_values, loss_grid
                    )
                )
            self._save_loss_values(
                save_dir, alpha_values, beta_values, loss_grid
            )
            if logger is not None:
                logger.info(f"loss landscape 已完成，输出文件: {output_paths}")
            return output_paths
        finally:
            self._restore_parameters(model, parameter_snapshot)
            if was_training:
                model.train()
            else:
                model.eval()