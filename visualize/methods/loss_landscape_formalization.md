# `_generate_directions` 形式化设计

本文形式化描述 `LossLandscapeVisualizer._generate_directions` 的职责、输入、输出、随机方向生成、两种归一化方式、边界条件以及后续使用方式。

本文是设计说明，不代表函数已经实现，也不修改函数接口之外的其他模块。

## 1. 函数职责

`_generate_directions` 接收模型当前检查点参数的快照，在完整模型参数空间中生成两个方向集合：

- 第一个方向集合：`direction_1`，对应数学符号 \(d_1\)；
- 第二个方向集合：`direction_2`，对应数学符号 \(d_2\)。

这两个方向用于把高维模型参数空间限制到一个二维参数平面：

\[
\theta(\alpha, \beta)
=
\theta_0 + \alpha d_1 + \beta d_2
\]

其中：

- \(\theta_0\)：checkpoint 中的原始模型参数；
- \(d_1\)：第一个参数空间方向；
- \(d_2\)：第二个参数空间方向；
- \(\alpha\)：沿第一个方向的移动坐标；
- \(\beta\)：沿第二个方向的移动坐标。

该函数只负责生成方向，不负责：

- 修改模型参数；
- 计算 loss；
- 生成 \(\alpha\)、\(\beta\) 坐标网格；
- 固定可视化数据；
- 绘制三维曲面；
- 保存图片或网页。

## 2. 输入数据结构

输入参数为：

```text
base_parameters
```

它是一个非空字典：

\[
\text{base\_parameters}[p] = \theta_{0,p}
\]

其中：

- \(p\)：参数名，例如 `bert.encoder.layers.0.self_attention.query.weight`；
- \(\theta_{0,p}\)：该参数名对应的参数张量。

抽象结构如下：

```text
base_parameters = {
    parameter_name_1: parameter_tensor_1,
    parameter_name_2: parameter_tensor_2,
    ...,
}
```

每个输入参数必须满足：

1. `base_parameters` 是非空字典；
2. 参数名是非空字符串；
3. 参数值是 `torch.Tensor`；
4. 参数值是浮点类型；
5. 参数值不包含 `NaN` 或无穷值。

`base_parameters` 来源于 `_snapshot_parameters`。该快照已经是独立的、脱离计算图的 CPU 张量，因此 `_generate_directions` 不应重新读取模型，也不应修改模型。

## 3. 输出数据结构

函数返回两个字典：

```text
(direction_1, direction_2)
```

数学上：

\[
\text{direction\_1}[p] = d_{1,p}
\]

\[
\text{direction\_2}[p] = d_{2,p}
\]

输出必须满足：

\[
\operatorname{keys}(\text{direction\_1})
=
\operatorname{keys}(\text{direction\_2})
=
\operatorname{keys}(\text{base\_parameters})
\]

对于每个参数名 \(p\)，必须满足：

\[
\operatorname{shape}(d_{1,p})
=
\operatorname{shape}(d_{2,p})
=
\operatorname{shape}(\theta_{0,p})
\]

两个方向中的每个张量都必须是浮点 Tensor，并且不能包含非法值。

## 4. 为什么需要两个方向

模型所有参数可以概念上拼接为一个高维向量：

\[
\theta_0 \in \mathbb{R}^{D}
\]

其中 \(D\) 是模型的总参数量。

完整 loss 是高维函数：

\[
L(\theta_1, \theta_2, \ldots, \theta_D)
\]

无法直接绘制。因此从 \(\theta_0\) 出发，在参数空间中选取两个方向 \(d_1\)、\(d_2\)，只观察：

\[
L(\theta_0 + \alpha d_1 + \beta d_2)
\]

这样可以用两个坐标描述一个二维参数切面：

- 横轴是 \(\alpha\)；
- 纵轴是 \(\beta\)；
- 高度是 loss。

随机张量用于提供方向候选，不是新的模型，也不会永久替换模型参数。

## 5. 原始随机方向

对于每个参数张量 \(\theta_{0,p}\)，生成两个独立的随机 Tensor：

\[
r_{1,p} \sim \mathcal{N}(0, I)
\]

\[
r_{2,p} \sim \mathcal{N}(0, I)
\]

它们的形状必须与当前参数相同：

\[
\operatorname{shape}(r_{1,p})
=
\operatorname{shape}(r_{2,p})
=
\operatorname{shape}(\theta_{0,p})
\]

代码层面通常使用：

```python
torch.randn_like(base_parameter)
```

`randn_like` 的作用是：

- 复制输入张量的形状；
- 复制输入张量的数据类型；
- 在相同设备上生成随机张量；
- 从标准正态分布采样。

随机生成决定方向的大致朝向，但不决定最终尺度。最终尺度由归一化步骤决定。

## 6. 范数定义

对任意张量 \(x\)，使用 L2 范数：

\[
\|x\|_2 = \sqrt{\sum_i x_i^2}
\]

对于当前参数和两个随机方向，分别定义：

\[
n_{\theta,p} = \|\theta_{0,p}\|_2
\]

\[
n_{1,p} = \|r_{1,p}\|_2
\]

\[
n_{2,p} = \|r_{2,p}\|_2
\]

对应的计算方式是：

```python
parameter_norm = torch.linalg.vector_norm(base_parameter)
direction_norm_1 = torch.linalg.vector_norm(random_direction_1)
direction_norm_2 = torch.linalg.vector_norm(random_direction_2)
```

范数只表示张量的整体大小，不改变张量方向。

## 7. `per_parameter` 归一化

当：

```text
self.normalization == "per_parameter"
```

每个参数张量独立归一化。

### 7.1 非零参数

如果：

\[
n_{\theta,p} > 0
\]

则定义：

\[
d_{1,p}
=
r_{1,p}\frac{n_{\theta,p}}{n_{1,p}}
\]

\[
d_{2,p}
=
r_{2,p}\frac{n_{\theta,p}}{n_{2,p}}
\]

这保证：

\[
\|d_{1,p}\|_2 = \|\theta_{0,p}\|_2
\]

\[
\|d_{2,p}\|_2 = \|\theta_{0,p}\|_2
\]

证明如下：

\[
\|d_{1,p}\|_2
=
\left\|r_{1,p}\frac{n_{\theta,p}}{n_{1,p}}\right\|_2
=
\|r_{1,p}\|_2\frac{n_{\theta,p}}{n_{1,p}}
=
 n_{1,p}\frac{n_{\theta,p}}{n_{1,p}}
=
 n_{\theta,p}
\]

第二个方向同理。

### 7.2 参数范数为零

如果：

\[
\|\theta_{0,p}\|_2 = 0
\]

说明该参数张量所有元素都是零。此时将对应方向置零：

\[
d_{1,p}=0
\]

\[
d_{2,p}=0
\]

原因是：

1. 目标方向尺度应与参数尺度一致；
2. 参数尺度是零；
3. 这样可以避免无意义的零尺度扰动；
4. 这样可以避免后续不必要的除零问题。

### 7.3 随机方向范数为零

如果：

\[
n_{1,p}=0
\]

则比例：

\[
\frac{n_{\theta,p}}{n_{1,p}}
\]

没有定义，应报错或重新生成方向。

同理，如果：

\[
n_{2,p}=0
\]

也不能继续缩放。

## 8. `global` 归一化

当：

```text
self.normalization == "global"
```

所有参数被视为一个整体。

概念上，可以将所有参数拼接成一个高维向量：

\[
\Theta_0
=
\operatorname{concat}(\theta_{0,1}, \theta_{0,2}, \ldots, \theta_{0,P})
\]

两个原始方向拼接为：

\[
R_1
=
\operatorname{concat}(r_{1,1}, r_{1,2}, \ldots, r_{1,P})
\]

\[
R_2
=
\operatorname{concat}(r_{2,1}, r_{2,2}, \ldots, r_{2,P})
\]

实际代码不需要真的拼接，可以累计平方和。

### 8.1 全局参数范数

\[
N_\theta
=
\|\Theta_0\|_2
=
\sqrt{\sum_{p=1}^{P}\|\theta_{0,p}\|_2^2}
\]

### 8.2 全局方向范数

\[
N_1
=
\|R_1\|_2
=
\sqrt{\sum_{p=1}^{P}\|r_{1,p}\|_2^2}
\]

\[
N_2
=
\|R_2\|_2
=
\sqrt{\sum_{p=1}^{P}\|r_{2,p}\|_2^2}
\]

### 8.3 统一缩放

定义缩放因子：

\[
s_1=\frac{N_\theta}{N_1}
\]

\[
s_2=\frac{N_\theta}{N_2}
\]

对所有参数使用同一个方向缩放因子：

\[
d_{1,p}=r_{1,p}s_1
\]

\[
d_{2,p}=r_{2,p}s_2
\]

因此：

\[
\|D_1\|_2=\|\Theta_0\|_2
\]

\[
\|D_2\|_2=\|\Theta_0\|_2
\]

但不保证每个参数块分别满足：

\[
\|d_{i,p}\|_2=\|\theta_{0,p}\|_2
\]

这正是 `global` 与 `per_parameter` 的区别。

## 9. 两种归一化方式对比

### `per_parameter`

对每个参数分别使用比例：

\[
s_{1,p}=\frac{\|\theta_{0,p}\|}{\|r_{1,p}\|}
\]

\[
s_{2,p}=\frac{\|\theta_{0,p}\|}{\|r_{2,p}\|}
\]

每个参数块都有自己的缩放因子。

### `global`

每个方向分别使用一个全局比例：

\[
s_1=\frac{\|\Theta_0\|}{\|R_1\|}
\]

\[
s_2=\frac{\|\Theta_0\|}{\|R_2\|}
\]

同一个方向中的所有参数块共用同一个缩放因子。

## 10. 函数的形式化执行流程

整个函数可以形式化为以下流程：

### 输入校验

\[
\text{base\_parameters}\neq \varnothing
\]

并且对所有参数名 \(p\)：

\[
\text{name}_p \in \text{NonEmptyString}
\]

\[
\theta_{0,p}\in\text{FloatingTensor}
\]

\[
\forall x\in\theta_{0,p},\quad x\in\mathbb{R}_{\text{finite}}
\]

### 方向生成

对所有参数名 \(p\)：

\[
r_{1,p},r_{2,p}
\leftarrow
\operatorname{RandomNormalLike}(\theta_{0,p})
\]

### 归一化

若选择 `per_parameter`：

\[
d_{i,p}
\leftarrow
r_{i,p}
\frac{\|\theta_{0,p}\|}{\|r_{i,p}\|}
\]

若选择 `global`：

\[
d_{i,p}
\leftarrow
r_{i,p}
\frac{\|\Theta_0\|}{\|R_i\|}
\]

其中 \(i\in\{1,2\}\)。

### 输出校验

对所有参数名 \(p\) 检查：

\[
\operatorname{shape}(d_{i,p})
=
\operatorname{shape}(\theta_{0,p})
\]

\[
d_{i,p}\in\text{FloatingTensor}
\]

\[
\forall x\in d_{i,p},\quad x\in\mathbb{R}_{\text{finite}}
\]

最后返回：

\[
(d_1,d_2)
\]

## 11. 伪代码级结构

```text
输入 base_parameters

校验 base_parameters

创建空 direction_1
创建空 direction_2

对于每个 parameter_name, base_parameter：
    校验参数名和参数张量
    生成 random_direction_1
    生成 random_direction_2
    检查两个原始方向的范数
    保存两个原始方向

如果 normalization == per_parameter：
    对每个参数：
        计算参数范数
        计算两个方向范数
        如果参数范数为零：
            两个方向置零
        否则：
            direction_1 按当前参数范数缩放
            direction_2 按当前参数范数缩放

否则如果 normalization == global：
    计算所有参数的总范数
    计算 direction_1 的总范数
    计算 direction_2 的总范数
    计算两个全局缩放因子
    对每个参数：
        direction_1 使用第一个全局缩放因子
        direction_2 使用第二个全局缩放因子

否则：
    报告不支持的归一化方式

校验两个方向的键、形状、类型和有限值

返回 direction_1, direction_2
```

## 12. 与后续参数扰动的关系

后续在任意坐标 \((\alpha,\beta)\) 处，对每个参数名 \(p\) 使用：

\[
\theta_p(\alpha,\beta)
=
\theta_{0,p}
+
\alpha d_{1,p}
+
\beta d_{2,p}
\]

所有参数都完成该计算后，组合成一个完整的模型参数状态。

因此 `_generate_directions` 必须保证：

1. 两个方向覆盖模型的全部参数；
2. 参数名完全对应；
3. 方向形状完全对应；
4. 方向数值有限；
5. 方向不会修改 `base_parameters`；
6. 方向不会修改模型本身。

## 13. 随机性与可复现性

由于 \(r_1\)、\(r_2\) 是随机生成的，不同随机种子可能得到不同的二维切面：

\[
L(\theta_0+\alpha d_1+\beta d_2)
\]

因此随机方向会影响曲面的具体形状，这是 loss landscape 降维可视化的固有性质，不是计算错误。

如果需要复现结果，必须在方向生成前固定随机种子。相同的：

- checkpoint；
- 固定批次；
- 归一化方式；
- 网格范围；
- 随机种子；

应生成相同的方向和相同的损失曲面。

如果要判断模型在更广泛方向上的平坦性，应使用多个随机种子生成多组方向，而不能把单个二维切面解释成完整高维 loss landscape。

## 14. 当前实现的完成标准

`_generate_directions` 完成后应满足：

- 接收非空的 `base_parameters`；
- 生成两个完整方向字典；
- 两个方向拥有相同的参数名集合；
- 每个方向张量与对应参数形状相同；
- 支持 `per_parameter`；
- 支持 `global`；
- 正确处理参数范数为零；
- 防止随机方向范数为零；
- 检查结果有限性；
- 不修改模型；
- 不修改基准参数；
- 不计算 loss；
- 不生成坐标；
- 不执行绘图。
