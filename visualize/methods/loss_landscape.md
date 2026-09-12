# Loss Landscape 可视化方法说明

## 1. 方法概述

`LossLandscapeVisualizer` 用于观察模型在参数空间局部区域内的损失变化。

深度学习模型通常有非常多的参数：

\[
\theta=(\theta_1,\theta_2,\ldots,\theta_D)
\]

其中 \(D\) 可能达到几十万、几百万甚至更多。模型损失可以写成高维函数：

\[
L(\theta_1,\theta_2,\ldots,\theta_D)
\]

这个高维函数无法直接绘制。因此，本方法从 checkpoint 的参数 \(\theta_0\) 出发，在参数空间中选择两个方向 \(d_1\) 和 \(d_2\)，只观察由这两个方向张成的二维切面：

\[
\theta(\alpha,\beta)
=
\theta_0+\alpha d_1+\beta d_2
\]

然后在每个 \((\alpha,\beta)\) 坐标处计算模型损失：

\[
Z(\alpha,\beta)
=
L\left(\theta_0+\alpha d_1+\beta d_2\right)
\]

最终得到一个二维 loss 矩阵，并将它渲染为：

- PNG 静态三维曲面图；
- HTML 交互式三维曲面图；
- NPZ 原始数值文件。

## 2. 适用场景

这个方法适合用于：

- 观察 checkpoint 附近的局部损失形状；
- 对比不同训练阶段的模型局部平坦程度；
- 观察参数沿随机方向移动时的敏感性；
- 检查模型附近是否存在尖锐区域、平坦区域或明显谷底；
- 为实验报告或论文提供损失曲面图；
- 保存原始曲面数值，供后续分析或重新绘图。

它不表示完整的高维损失地形，而是完整参数空间中的一个二维切面。

## 3. 运行入口

可视化程序入口是：

```bash
python main/start_visualize.py --config <visualize_config.yaml>
```

具体参数以项目当前 `parse_args()` 的定义为准。

可视化配置中的任务必须是 MLM 可视化任务，例如：

```yaml
task: mlmpretrain_visualize
```

运行时需要通过 checkpoint 配置找到：

- checkpoint 权重文件；
- 训练配置文件；
- 可视化数据集；
- 可视化 DataLoader。

## 4. 配置示例

一个完整的 `loss_landscape` 方法配置如下：

```yaml
visualize:
  methods:
    - type: loss_landscape
      params:
        normalization: per_parameter
        output_format: both
        grid_size: 11
        alpha_min: -1.0
        alpha_max: 1.0
        beta_min: -1.0
        beta_max: 1.0
        batch_count: 1
        losses:
          - name: mlm_cross_entropy
            type: cross_entropy
            weight: 1.0
            params:
              ignore_index: -100
```

同一个可视化配置中也可以同时使用不需要 loss 的方法：

```yaml
visualize:
  methods:
    - type: attention_heatmap
      params:
        layer: all
        head: all
        max_samples: 2

    - type: loss_landscape
      params:
        normalization: per_parameter
        output_format: both
        grid_size: 11
        alpha_min: -1.0
        alpha_max: 1.0
        beta_min: -1.0
        beta_max: 1.0
        batch_count: 1
        losses:
          - name: mlm_cross_entropy
            type: cross_entropy
            weight: 1.0
            params:
              ignore_index: -100
```

`losses` 是方法级参数：

- `loss_landscape` 需要它，因为 z 轴是 loss；
- `attention_heatmap` 不需要它，因此不应写入 `attention_heatmap.params`；
- 其他方法是否拥有 `losses`，由各自的参数 Schema 决定。

## 5. 配置参数

### 5.1 `normalization`

类型：字符串。

可选值：

```text
per_parameter
global
```

默认值：

```text
per_parameter
```

用于控制随机参数方向的尺度。

#### `per_parameter`

每个参数张量独立归一化：

\[
\|d_{i,p}\|_2
=
\|\theta_{0,p}\|_2
\]

其中 \(p\) 表示一个参数张量，\(i\in\{1,2\}\) 表示两个方向之一。

优点：

- 每个参数块按照自身尺度受到扰动；
- 大参数张量不会因为绝对数值较大而完全主导所有扰动；
- 更容易保持不同层之间的相对扰动规模。

#### `global`

把模型所有参数视为一个整体向量：

\[
\|D_i\|_2
=
\|\Theta_0\|_2
\]

其中：

- \(\Theta_0\) 是所有模型参数拼接后的整体向量；
- \(D_i\) 是所有方向参数拼接后的整体向量。

优点：

- 更接近把整个模型参数视为一个向量后进行归一化；
- 保证整个方向的总尺度与模型参数总尺度一致。

注意：`global` 不保证每个参数张量单独满足方向范数等于参数范数。

### 5.2 `output_format`

类型：字符串。

可选值：

```text
png
html
both
```

默认值：

```text
both
```

含义：

- `png`：生成静态 PNG 曲面图；
- `html`：生成交互式 HTML 曲面图；
- `both`：同时生成 PNG 和 HTML。

原始数值文件会随计算结果保存，不受该字段限制。

### 5.3 `grid_size`

类型：整数。

约束：

```text
grid_size >= 2
```

默认值：

```text
11
```

表示每条坐标轴包含多少个采样点。

如果：

```text
grid_size = 11
```

则总参数点数量为：

\[
11\times11=121
\]

也就是说，模型需要进行 121 个网格点的损失评估。

计算量大致与下面的数量成正比：

\[
grid\_size^2\times batch\_count
\]

因此增大 `grid_size` 会显著增加运行时间。

### 5.4 `alpha_min` 与 `alpha_max`

控制第一个方向的坐标范围：

\[
\alpha\in[\alpha_{min},\alpha_{max}]
\]

默认值：

```text
alpha_min = -1.0
alpha_max = 1.0
```

约束：

```text
alpha_min < alpha_max
```

### 5.5 `beta_min` 与 `beta_max`

控制第二个方向的坐标范围：

\[
\beta\in[\beta_{min},\beta_{max}]
\]

默认值：

```text
beta_min = -1.0
beta_max = 1.0
```

约束：

```text
beta_min < beta_max
```

### 5.6 `batch_count`

类型：整数。

约束：

```text
batch_count >= 1
```

默认值：

```text
1
```

表示从可视化 DataLoader 中固定并使用多少个 batch。

它不代表训练 batch size，也不代表单个 batch 中的样本数。

最终一个参数点的 loss 是这些固定 batch loss 的平均值：

\[
L_{point}
=
\frac{1}{N}\sum_{k=1}^{N}L_k
\]

其中 \(N=\text{batch\_count}\)。

### 5.7 `losses`

类型：非空列表。

这是 `loss_landscape` 的方法级损失配置。每一项结构为：

```yaml
- name: mlm_cross_entropy
  type: cross_entropy
  weight: 1.0
  params:
    ignore_index: -100
```

字段说明：

- `name`：该损失项的名称；
- `type`：通过 `LOSS_REGISTRY` 动态解析的损失类型；
- `weight`：该损失项的权重；
- `params`：该具体损失的参数。

`losses` 的整体计算方式由项目的 `WeightedLoss` 负责：

\[
L_{weighted}
=
\sum_{m=1}^{M}w_mL_m
\]

其中：

- \(M\)：损失项数量；
- \(w_m\)：第 \(m\) 项权重；
- \(L_m\)：第 \(m\) 项具体损失。

可视化器不知道具体使用的是交叉熵还是其他损失，只调用统一接口：

```python
loss_result = self.loss_fn(
    logits=flattened_logits,
    labels=flattened_labels,
)
```

并读取统一结果字段：

```python
batch_loss = loss_result["loss"]
```

## 6. Loss 的动态解析流程

可视化方法中的 loss 配置不会直接作为字典交给模型，而是经过以下流程：

```text
visualize.methods[*].params.losses
    ↓
Parse Schema 检查外层结构
    ↓
根据 losses[*].type 选择 Loss Spec
    ↓
校验对应 loss 的 params
    ↓
visualize_builder 读取规范化配置
    ↓
LOSS_REGISTRY.resolve_key(type)
    ↓
实例化具体 Loss
    ↓
组成 loss_list
    ↓
WeightedLoss(loss_list)
    ↓
LossLandscapeVisualizer(loss_fn=...)
```

因此：

- YAML 负责声明使用什么损失；
- Parse 负责校验配置；
- Registry 负责类型解析；
- Builder 负责实例化；
- `WeightedLoss` 负责组合；
- Visualizer 只消费统一的 `loss_fn`。

## 7. 运行时数据流

### 7.1 固定数据

可视化开始时，从 DataLoader 取出前 `batch_count` 个 batch，并复制到内存：

```text
DataLoader batch
    ↓
独立字典 fixed_batch
    ↓
detach + cpu + clone
    ↓
fixed_batches
```

固定数据的原因是 MLM Collator 通常包含随机 mask。如果每个参数网格点重新遍历 DataLoader，输入和标签可能发生变化，最终的 loss 差异就同时包含：

- 参数变化造成的差异；
- 随机 mask 造成的差异。

固定 batch 后，网格中每个点使用完全相同的输入和标签，loss 变化主要反映参数变化。

### 7.2 保存参数起点

从当前 checkpoint 模型中复制：

```text
base_parameters
```

它表示：

\[
\theta_0
\]

这个快照用于保证每个网格点都从同一个模型参数起点计算。

### 7.3 生成方向

为每个参数生成两个随机 Tensor：

```text
random_direction_1
random_direction_2
```

它们与对应参数形状一致，然后根据 `normalization` 进行缩放，得到：

```text
direction_1
direction_2
```

### 7.4 生成坐标

使用 `torch.linspace` 生成两组等间隔坐标：

```text
alpha_values
beta_values
```

如果 `grid_size=3`，则可能得到：

```text
alpha_values = [-1.0, 0.0, 1.0]
beta_values  = [-1.0, 0.0, 1.0]
```

### 7.5 计算一个网格点

对于一个坐标 \((\alpha_i,\beta_j)\)，每个参数张量按下面公式计算：

\[
\theta_p^{(i,j)}
=
\theta_{0,p}
+
\alpha_i d_{1,p}
+
\beta_j d_{2,p}
\]

然后把整个参数集合写回模型。

### 7.6 计算单点 loss

模型前向得到 MLM logits：

\[
\text{logits}\in\mathbb{R}^{B\times S\times V}
\]

其中：

- \(B\)：batch size；
- \(S\)：序列长度；
- \(V\)：词表大小。

labels 形状为：

\[
\text{labels}\in\mathbb{R}^{B\times S}
\]

随后整理为 Loss 接口需要的形状：

\[
\text{logits}\rightarrow[B\times S,V]
\]

\[
\text{labels}\rightarrow[B\times S]
\]

再调用配置得到的 `self.loss_fn`。

### 7.7 填充损失矩阵

将当前点的平均 loss 写入：

```text
loss_grid[row_index, column_index]
```

最终：

```text
loss_grid.shape == (grid_size, grid_size)
```

## 8. 函数模块职责

### `_validate_batch`

验证 MLM batch 是否具有：

```text
input_ids
attention_mask
token_type_ids
labels
```

并检查它们都是二维 Tensor 且形状一致。

### `_collect_fixed_batches`

从 DataLoader 固定前 `batch_count` 个 batch，并将 Tensor 复制到 CPU。

### `_snapshot_parameters`

复制模型参数，形成基准参数快照 \(\theta_0\)。

### `_generate_directions`

生成两个随机方向，并按 `per_parameter` 或 `global` 方式归一化。

### `_build_coordinates`

生成 alpha 和 beta 两条坐标轴。

### `_apply_perturbation`

根据：

\[
\theta_0+\ alpha d_1+\u0000beta d_2
\]

生成一个网格点对应的参数字典。

### `_apply_parameters_to_model`

将当前网格点的参数逐个复制到模型参数中。

### `_evaluate_single_loss`

在当前模型状态下，使用固定 batch 和动态 `loss_fn` 计算平均 loss。

### `_evaluate_loss_grid`

遍历所有 alpha/beta 坐标，生成二维 loss 矩阵。

### `_restore_parameters`

在可视化完成或异常退出时恢复模型参数。

### `_save_loss_values`

将坐标和 loss 矩阵保存为 NPZ。

### `_render_static_surface`

使用 Matplotlib 生成 PNG 静态三维曲面。

### `_render_interactive_surface`

使用 Plotly 生成 HTML 交互式三维曲面。

## 9. 输出文件

输出目录由调用方传入，通常是：

```text
<visualize_dir>/losslandscape/
```

可能生成以下文件：

```text
loss_landscape.png
loss_landscape.html
loss_landscape_values.npz
```

### 9.1 PNG 文件

文件名：

```text
loss_landscape.png
```

特点：

- 静态三维图；
- 适合报告、论文和图片预览；
- 不需要浏览器交互；
- 不能旋转或查看悬停信息。

### 9.2 HTML 文件

文件名：

```text
loss_landscape.html
```

特点：

- 支持旋转；
- 支持缩放和平移；
- 可以悬停查看坐标和 loss；
- 适合交互式探索。

当前实现使用 Plotly 生成 HTML。运行环境需要安装 Plotly：

```bash
python3 -m pip install -r main/requirements.txt
```

### 9.3 NPZ 文件

文件名：

```text
loss_landscape_values.npz
```

包含：

```text
alpha
beta
loss
```

读取示例：

```python
import numpy as np

values = np.load("loss_landscape_values.npz")
alpha_values = values["alpha"]
beta_values = values["beta"]
loss_grid = values["loss"]
```

NPZ 是后续数值分析的推荐数据源。

## 10. PNG 和 HTML 的区别

| 输出 | 库 | 交互性 | 适用场景 |
|---|---|---:|---|
| PNG | Matplotlib | 无 | 报告、论文、静态分享 |
| HTML | Plotly | 有 | 调试、探索、交互分析 |
| NPZ | NumPy | 不适用 | 保存原始数据、后续分析 |

PNG 和 HTML 必须使用同一份 `loss_grid`，不能分别重新计算模型 loss。

## 11. 依赖

当前相关依赖位于：

```text
main/requirements.txt
```

Loss landscape 输出需要：

```text
torch
numpy
matplotlib
plotly
PyYAML
```

其中：

- `torch`：模型、Tensor 和前向计算；
- `numpy`：保存 NPZ 和辅助处理数组；
- `matplotlib`：生成 PNG；
- `plotly`：生成 HTML；
- `PyYAML`：读取 YAML 配置。

只生成 PNG 时不需要执行 Plotly 的渲染代码；生成 HTML 或 `both` 时需要安装 Plotly。

## 12. 计算复杂度与资源消耗

如果：

```text
grid_size = G
batch_count = N
```

则需要评估：

\[
G^2
\]

个参数点，每个参数点使用 \(N\) 个固定 batch，因此总 batch 前向次数约为：

\[
G^2N
\]

例如：

```text
grid_size = 11
batch_count = 2
```

需要：

```text
11 × 11 × 2 = 242
```

次 batch 前向。

注意：

- `grid_size` 增加一倍，网格点数量增加约四倍；
- 模型参数会在每个网格点被重新复制；
- 固定 batch 本身占用 CPU 内存；
- 模型前向时会占用目标设备显存；
- `torch.no_grad()` 可避免保存反向计算图。

建议初次运行使用：

```yaml
grid_size: 3
batch_count: 1
```

确认流程正确后，再逐步增加网格密度和 batch 数量。

## 13. 随机方向与可复现性

方向由随机 Tensor 生成：

\[
r_1,r_2\sim\mathcal{N}(0,I)
\]

因此不同运行可能得到不同的二维参数切面，即使 checkpoint、数据和配置相同，也可能出现不同曲面形状。

归一化只能控制方向长度，不能消除方向本身的差异。

如果需要复现结果，应在生成方向前固定随机种子：

```python
torch.manual_seed(seed)
```

更严谨的实验应使用多个随机种子，比较不同切面下的：

- 中心点 loss；
- 周围区域平均 loss；
- loss 最大值；
- loss 增长速度；
- 局部曲率；
- 不同方向结果的一致性。

单个随机曲面只能说明模型在一组方向上的表现，不能直接代表整个高维参数空间的平坦性。

## 14. 如何解释图像

### 中心点

如果坐标范围包含 0，那么：

\[
(\alpha,\beta)=(0,0)
\]

对应：

\[
\theta(0,0)=\theta_0
\]

因此中心点应对应原始 checkpoint 参数的 loss。

### 平坦区域

如果周围较大范围内的 loss 变化较小，说明沿当前两个方向移动时模型较不敏感。

### 尖锐区域

如果参数仅发生小幅移动，loss 就快速增大，说明沿当前方向的局部敏感性较强。

### 谷底

如果中心附近存在较低 loss 区域，可能说明 checkpoint 附近仍有沿所选方向下降的空间。

但不要只根据一张图判断模型整体优劣，因为：

- 图只覆盖二维切面；
- 方向是随机的；
- 使用的 batch 数量有限；
- loss 配置会影响曲面；
- 不同归一化方式会改变参数尺度。

## 15. 常见问题

### 15.1 为什么每个网格点都要使用相同 batch

因为 MLM Collator 可能随机生成 mask。重新生成 batch 会让 loss 同时受到数据随机性的影响，无法只观察参数扰动。

### 15.2 为什么需要两个随机方向

一个方向只能形成一条 loss 曲线。两个方向才能构造：

\[
\theta_0+\alpha d_1+\beta d_2
\]

对应二维 loss 曲面。

### 15.3 归一化是不是投影

不是。

随机生成决定方向的朝向，归一化只调整方向长度。它不会把模型参数压缩成二维，也不会把参数投影到某个低维向量。

### 15.4 为什么不能直接把随机 Tensor 加到参数上

不同参数张量的尺度不同。未归一化的随机方向可能导致某些层受到过大的相对扰动，另一些层几乎不受影响。

### 15.5 为什么不能用上一个网格点继续加扰动

每个网格点都必须满足：

\[
\theta(\alpha_i,\beta_j)
=
\theta_0+
\alpha_i d_1+
\beta_j d_2
\]

如果从上一个点继续修改，就会产生累积扰动，破坏坐标公式。

### 15.6 为什么 loss landscape 必须配置 losses

因为 loss 是曲面的 z 轴。没有损失函数，就无法计算：

\[
Z(\alpha,\beta)
\]

`losses` 由该可视化方法的 `params` 声明，经过 Parse 动态校验，再由 Builder 实例化。

### 15.7 为什么 attention heatmap 可以没有 losses

因为 attention heatmap 的目标是展示注意力矩阵，不需要计算损失。因此它的参数 Schema 不包含 `losses`，也不会构造 `loss_fn`。

## 16. 推荐调试顺序

建议按照以下顺序验证：

1. 使用 `grid_size: 3`、`batch_count: 1`；
2. 只配置 `loss_landscape`；
3. 使用 `output_format: png` 验证 Matplotlib 输出；
4. 使用 `output_format: html` 验证 Plotly 输出；
5. 使用 `output_format: both` 验证两种输出；
6. 检查 `loss_landscape_values.npz` 的数组形状；
7. 确认 `loss_grid.shape == (grid_size, grid_size)`；
8. 确认模型可正常恢复；
9. 再增加 `grid_size`；
10. 最后增加 `batch_count`。

## 17. 结果合理性检查

完成一次运行后，建议检查：

```python
import numpy as np

values = np.load("loss_landscape_values.npz")

assert values["alpha"].ndim == 1
assert values["beta"].ndim == 1
assert values["loss"].shape == (
    len(values["alpha"]),
    len(values["beta"]),
)
assert np.isfinite(values["loss"]).all()
```

还应检查：

- `(0, 0)` 是否位于坐标网格中；
- 中心点 loss 是否与单独评估 checkpoint 得到的 loss 接近；
- PNG 和 HTML 的曲面形状是否一致；
- 输出目录中是否出现预期文件；
- 可视化完成后模型参数是否恢复；
- 可视化完成后模型的训练/评估模式是否恢复。

## 18. 设计边界

当前实现面向 MLM 模型和 MLM batch，模型需要支持：

```python
model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    token_type_ids=token_type_ids,
)
```

并返回三维 logits：

```text
[batch_size, seq_length, vocab_size]
```

当前固定 batch 要求包含：

```text
input_ids
attention_mask
token_type_ids
labels
```

如果未来支持 QA 或其他任务，需要重新设计：

- batch Schema；
- 模型输入组织；
- logits 和 labels 的形状适配；
- loss 输入适配；
- 可视化方法与任务之间的接口。

不应简单地把当前 MLM 的形状逻辑复制到所有任务。

## 19. 总结

`LossLandscapeVisualizer` 的核心公式是：

\[
\theta(\alpha,\beta)
=
\theta_0+
\alpha d_1+
\beta d_2
\]

其完整数据流是：

```text
checkpoint 模型
    ↓
保存 base_parameters
    ↓
生成 direction_1、direction_2
    ↓
固定 MLM batches
    ↓
生成 alpha、beta 坐标
    ↓
遍历每个坐标
    ↓
生成当前扰动参数
    ↓
写入模型
    ↓
调用动态 loss_fn
    ↓
生成 loss_grid
    ↓
保存 NPZ
    ↓
渲染 PNG 和/或 HTML
    ↓
恢复模型参数与训练状态
```

最重要的实验解释边界是：

> 该方法展示的是模型在两个选定参数方向上的二维损失切面，而不是整个高维参数空间的完整 loss landscape。

要进行可靠比较，必须同时固定或记录：

- checkpoint；
- 可视化 batch；
- loss 配置；
- normalization；
- 坐标范围；
- grid size；
- 随机方向 seed。
