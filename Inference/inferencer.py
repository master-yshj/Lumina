import torch

class Inferencer:
    def __init__(self, task, task_inference, device, logger=None):
        if not isinstance(task, torch.nn.Module):
            raise ValueError(f"task 必须为torch.nn.Module类型的喵! 现在的type: {type(task)}")
        if not (hasattr(task_inference, "infer") and callable(task_inference.infer)):
            raise ValueError(f"task_inference 必须具有infer方法且可调用喵!")


        self.task = task
        self.task_inference = task_inference
        self.device = torch.device(device)
        self.logger = logger

    def _log_info(self, message):
        if self.logger is None:
            print(message)
        else:
            self.logger.info(message)

    def infer(self, input_data):
        self.task.to(self.device)
        self.task.eval()
        self._log_info(f"Inference input: {input_data}")

        with torch.inference_mode():
            prediction = self.task_inference.infer(input_data)

        if not isinstance(prediction, dict):
            raise ValueError(f"prediction 必须为dict类型的喵! 现在的type: {type(prediction)}")
        if not prediction:
            raise ValueError(f"prediction 不能为空喵!")

        self._log_info(f"Inference output: {prediction}")

        return prediction