import torch
class AdamW(torch.optim.AdamW):
    def __init__(self, params, lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8):
        if not type(lr) in (float, int) or lr <= 0:
            raise ValueError(f"lr必须为大于0的浮点数或整数喵！现在lr是: {lr}")
        if not type(weight_decay) in (float, int) or weight_decay < 0:
            raise ValueError(f"weight_decay必须为非负的浮点数或整数喵！现在weight_decay是: {weight_decay}")
        if not isinstance(betas, tuple) or len(betas) != 2:
            raise ValueError(f"betas必须为长度为2的元组喵！betas: {betas} betas.type: {type(betas)}")
        if not all(type(num) in (int, float) and 0<=num<1 for num in betas):
            raise ValueError(f"betas中的值必须为[0,1)之间的整数或浮点数喵！betas: {betas}")
        if not type(eps) in (float, int) or eps <= 0:
            raise ValueError(f"eps必须为大于0的浮点数或整数喵！现在eps是: {eps}")
        super(AdamW, self).__init__(params, lr=lr,weight_decay=weight_decay, betas=betas, eps=eps)