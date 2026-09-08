# `train_preference_rm.py` 训练逻辑分析

该脚本训练基于 Bradley–Terry 的偏好奖励模型。输入是同一个 prompt 对应的 `chosen`（人类偏好回答）和 `rejected`（非偏好回答），目标是让：

```text
reward(chosen) > reward(rejected)
```

默认启动命令：

```bash
uv run python -m reward_models.train_preference_rm \
  --config reward_models/configs/preference_rm.yaml
```

## 1. 数据预处理

### `format_conversation`

第 42–49 行遍历消息列表，读取每条消息的 `role` 和 `content`，拼成 `user: ...`、`assistant: ...` 的多行字符串。它只在 tokenizer 没有 `chat_template` 时作为后备方案使用。

### `tokenize_messages`

第 52–80 行优先调用 `tokenizer.apply_chat_template`，使用模型规定的聊天格式，并设置 `tokenize=True`、`add_generation_prompt=False`、`max_length=max_length` 和 `truncation=True`。返回 `input_ids` 与 `attention_mask`。没有聊天模板时，先调用 `format_conversation`，再使用普通 tokenizer。

### `build_preference_dataset`

第 84–171 行负责加载、筛选和 tokenize 数据。默认配置为：

```yaml
dataset_name: argilla/ultrafeedback-binarized-preferences-cleaned
dataset_split: train
samples: 5000
max_length: 512
```

如果 `dataset_name` 是本地路径，使用 `load_from_disk`；否则使用：

```python
load_dataset(dataset_name, split=dataset_split)
```

这会由 Hugging Face `datasets` 自动下载并缓存数据。之后调用 `shuffle(seed=config.seed)` 打乱数据，再通过 `select` 最多保留 `config.samples` 条。

每条样本读取 `prompt`、`chosen` 和 `rejected`。如果 `chosen` 是消息列表，直接使用；如果是字符串，则自动组装为 user/assistant 消息。如果格式不支持，就跳过样本。

chosen 和 rejected 分别 tokenize 为完整序列：

```text
prompt + chosen response
prompt + rejected response
```

如果截断后两者的 `input_ids` 完全相同，就丢弃该样本。通常这是因为 prompt 过长，两个回答都被截掉；此时 reward 差为 0，损失固定为 `log(2) ≈ 0.693`，没有有效偏好信号。

有效记录包含 `chosen_ids`、`chosen_mask`、`rejected_ids` 和 `rejected_mask`，最后用 `Dataset.from_list(records)` 构造内存中的 Dataset。脚本不会自动调用 `save_to_disk`。

## 2. Batch 组织

### `collate_fn`

第 174–181 行在 DataLoader 组 batch 时动态 padding：token ID 使用 `tokenizer.pad_token_id`，attention mask 使用 0。典型形状为：

```text
chosen_ids:     [batch_size, chosen_seq_len]
chosen_mask:    [batch_size, chosen_seq_len]
rejected_ids:   [batch_size, rejected_seq_len]
rejected_mask:  [batch_size, rejected_seq_len]
```

## 3. 模型结构和奖励计算

### `PreferenceRewardModel`

第 189–203 行继承 `BaseRewardModel`，设置 `head_dim=1`。基类创建 `nn.Linear(hidden_size, 1)`，因此每条完整序列输出一个 scalar reward。

### `get_reward`

第 205–219 行调用基类的 `get_hidden_states`。基类使用 `self.model.base_model`，返回形状为 `[batch_size, sequence_length, hidden_size]` 的最后隐藏状态，而不是 causal language modeling 输出。

```python
seq_lengths = attention_mask.sum(dim=1) - 1
batch_indices = torch.arange(hidden.size(0), device=hidden.device)
last_hidden = hidden[batch_indices, seq_lengths]
reward = self.head(last_hidden).squeeze(-1)
```

这些代码找到每条序列最后一个非 padding token，取其隐藏向量，再通过线性奖励头得到形状为 `[batch_size]` 的 reward。

## 4. Bradley–Terry 损失

### `forward`

第 221–248 行分别计算 chosen 和 rejected 的 reward，然后计算：

```python
r_chosen = self.get_reward(chosen_ids, chosen_mask)
r_rejected = self.get_reward(rejected_ids, rejected_mask)
loss = -F.logsigmoid(r_chosen - r_rejected).mean()
```

数学形式为：

$$\mathcal{L}=-\log\sigma(r_{chosen}-r_{rejected})$$

chosen reward 越高，损失越小；rejected reward 越高，损失越大。该损失只依赖 reward 差值，因此 reward 的绝对值没有固定意义。

## 5. 验证逻辑

`evaluate_preference_rm`（第 255–285 行）使用 `@torch.no_grad()` 和 `model.eval()`，不计算梯度并关闭 dropout。每个 batch 计算：

```python
margin = r_chosen - r_rejected
total_correct += (margin > 0).sum().item()
```

最后返回 `val/loss`、`val/accuracy`、`val/r_chosen_mean`、`val/r_rejected_mean` 和 `val/reward_margin`。其中 accuracy 是 chosen 得分高于 rejected 的比例。

## 6. 训练初始化

`train_preference_rm`（第 287 行起）首先设置 Python/PyTorch 随机种子，并通过 `config.get_device()` 选择设备。随后初始化 W&B、加载 tokenizer、调用 `build_preference_dataset`。

默认 `val_ratio=0.1`，调用 `data.train_test_split` 得到 90% 训练集和 10% 验证集。这里的 split 名称 `test` 实际表示验证集。

训练 DataLoader 默认 `batch_size=2`、`shuffle=True`，使用动态 padding；验证 DataLoader 不打乱，也不丢弃最后一个 batch。

## 7. 加载模型、优化器和调度器

默认模型为 `Qwen/Qwen3-0.6B-Base`，`freeze_backbone=false`。基类通过 `AutoModelForCausalLM.from_pretrained` 加载模型，使用 FP32 参数并关闭 `use_cache`。默认情况下基座模型和奖励头都参与全参数微调；若 `freeze_backbone=true`，只训练奖励头。

优化器是 AdamW，只接收 `requires_grad=True` 的参数，默认学习率为 `5e-5`。

```python
total_optimizer_steps = (
    -(-len(train_loader) // config.grad_accum_steps)
    * config.epochs
)
warmup_steps = int(total_optimizer_steps * config.warmup_ratio)
```

第一段用整数向上取整计算优化器更新次数，第二段计算 warmup 更新数。`linear_decay` 会在 warmup 后线性衰减到 0；`warmup_only` 在 warmup 后保持学习率。CUDA 可用时前向和损失计算使用 BF16 autocast，参数仍为 FP32。

## 8. 训练循环

每个 epoch 开始时调用 `model.train()` 和 `optimizer.zero_grad()`。每个 microbatch 移动到设备后执行：

```python
loss, r_chosen, r_rejected = model(**batch)
(loss / grad_accum_steps).backward()
```

默认 `grad_accum_steps=8`，每 8 个 microbatch 更新一次参数。batch size 为 2 时，有效 batch size 约为 16。

当累计满一个窗口，或到达 epoch 最后一个 microbatch 时：

1. `optimizer.step()` 更新参数；
2. `scheduler.step()` 更新学习率；
3. `optimizer.zero_grad()` 清空梯度；
4. `global_step += 1`。

最后不足完整累积窗口的 batch 仍除以固定的 `grad_accum_steps`，因此最后一次更新可能略小。每次优化器更新后记录 loss、排序准确率、chosen/rejected 平均 reward、reward margin 和学习率。

## 9. 周期性验证和 epoch 结束

当存在验证集、`eval_interval > 0` 且 `global_step % eval_interval == 0` 时调用验证函数；验证结束后重新执行 `model.train()`。

每个 epoch 结束时计算训练集平均 loss 和 accuracy。如果该 epoch 结束时没有刚好做过验证，则额外验证一次，避免重复验证同一个 step。

## 10. 训练结束和 Demo

训练结束后调用 `finish_wandb()` 并返回模型。当前脚本没有调用 `save_pretrained` 或 `torch.save`，不会自动生成 checkpoint；模型只存在当前 Python 进程中。

当 `skip_demo=false` 时，`demo_scoring` 对固定的“解释量子计算” prompt 准备一个较好回答和一个较差回答，分别计算 reward，并检查 `good_reward > bad_reward`。这只是训练后的 sanity check，不会继续更新参数。

## 11. 核心结论

模型不直接学习“标准答案”，而是学习回答之间的排序关系：

$$
r_{chosen} > r_{rejected}
$$

完整数据流为：

```text
UltraFeedback 偏好对
        ↓
chosen/rejected 对话 tokenization
        ↓
基座语言模型提取最后有效 token 的隐藏状态
        ↓
线性奖励头输出两个 scalar reward
        ↓
-log sigmoid(r_chosen-r_rejected)
        ↓
反向传播、梯度累积和参数更新
```

## 12. Prompt、attention mask 与 batch 详解

### Prompt 是否包含在 token 序列中

包含。对于字符串格式的数据，脚本会把同一个 prompt 分别和 chosen、rejected 组成两段完整对话：

```python
chosen_messages = [
    {"role": "user", "content": prompt},
    {"role": "assistant", "content": chosen},
]
rejected_messages = [
    {"role": "user", "content": prompt},
    {"role": "assistant", "content": rejected},
]
```

因此两条序列分别是：

```text
chosen_ids   = tokenize(prompt + chosen response)
rejected_ids = tokenize(prompt + rejected response)
```

如果 tokenizer 有 chat template，实际序列还会包含 user/assistant 的角色标记和其他模板标记。脚本没有单独保存 `prompt_ids` 或 `prompt_mask`，而是让模型对 prompt 和回答的完整序列进行 attention，最后使用整条序列最后一个有效 token 的隐藏状态产生 reward。

当 prompt 很长时，`max_length=512` 的截断可能保留共同的 prompt、截掉 chosen 和 rejected 的差异，使两条 `input_ids` 完全相同。此时样本会被丢弃，因为 Bradley–Terry 损失没有有效的排序信号。

### `attention_mask` 的形式和作用

`tokenize_messages` 返回 `input_ids` 和 `attention_mask`。mask 与 token 序列等长，是由 0 和 1 组成的一维列表：

```text
input_ids:      [101, 205, 87, 42, 2, 0, 0]
attention_mask: [  1,   1,  1,  1, 1, 0, 0]
```

其中 `1` 表示真实 token，`0` 表示为了组成 batch 而补入的 padding token。`collate_fn` 使用 `tokenizer.pad_token_id` 补齐 `input_ids`，使用 `0` 补齐 mask。

mask 在两个地方起作用。首先，它传给 Transformer：

```python
self.model.base_model(
    input_ids=input_ids,
    attention_mask=attention_mask,
)
```

这样 padding 位置不会参与 attention。其次，reward 计算使用 mask 找到每条序列最后一个真实 token：

```python
seq_lengths = attention_mask.sum(dim=1) - 1
last_hidden = hidden[batch_indices, seq_lengths]
```

因此 reward 不会错误地取 padding 位置。Transformer 内部的 causal mask 由语言模型自动生成，脚本没有显式创建它；`attention_mask` 主要负责区分有效 token 和 padding。

### Dataset 和 DataLoader 的层次

`Dataset.from_list(records)` 创建的是一个内存中的 Hugging Face Dataset。每条记录是一个偏好对，字段形式为：

```python
{
    "chosen_ids": list[int],
    "chosen_mask": list[int],
    "rejected_ids": list[int],
    "rejected_mask": list[int],
}
```

单条记录中的四个字段仍然是长度可能不同的 Python 列表，不能直接堆叠成矩阵。DataLoader 取出若干条记录后，调用 `collate_fn` 动态 padding，再将它们组成 Tensor batch。训练集默认随机打乱，并在数据量足够大时丢弃最后一个不完整 batch；验证集不打乱，也不丢弃最后一个 batch。

### Batch 的形式

chosen 和 rejected 分别 padding，因此 `chosen_seq_len` 与 `rejected_seq_len` 不要求相同。一个 batch 是包含四个 Tensor 的字典：

```python
{
    "chosen_ids": tensor(...),
    "chosen_mask": tensor(...),
    "rejected_ids": tensor(...),
    "rejected_mask": tensor(...),
}
```

实际形状为：

```text
chosen_ids:     [B, L_chosen]
chosen_mask:    [B, L_chosen]
rejected_ids:   [B, L_rejected]
rejected_mask:  [B, L_rejected]
```

其中 `B` 是物理 batch size，`L_chosen` 和 `L_rejected` 是当前 batch 内各自的最大长度。例如 `batch_size=2` 时，chosen 的形状可能是 `[2, 7]`，rejected 的形状可能是 `[2, 6]`；padding 位置的 mask 为 0。进入训练循环后，四个 Tensor 都被移动到目标设备，并通过 `model(**batch)` 传入模型。模型返回一个标量 `loss`，以及形状为 `[B]` 的 `r_chosen` 和 `r_rejected`。

默认配置的物理 batch size 是 2。由于 `grad_accum_steps=8`，连续 8 个这样的 microbatch 才执行一次 `optimizer.step()`，所以有效 batch size 约为 `2 × 8 = 16` 个偏好对。若显存不足改为 `batch_size=1`，可以用 `grad_accum_steps=16` 维持相同的有效 batch size。

## 13. Batch 到损失的完整计算过程

训练循环先把 batch 中的四个 Tensor 移到目标设备，然后在 CUDA 上用 BF16 autocast 执行前向计算：

```python
batch = {k: v.to(device) for k, v in batch.items()}

with torch.amp.autocast(
    "cuda",
    dtype=torch.bfloat16,
    enabled=autocast_enabled,
):
    loss, r_chosen, r_rejected = model(**batch)
```

假设物理 batch size 为 `B=2`，动态 padding 后可能有：

```text
chosen_ids:    [2, 420]
chosen_mask:   [2, 420]
rejected_ids:  [2, 380]
rejected_mask: [2, 380]
```

### `model(**batch)` 展开参数

`model(**batch)` 等价于：

```python
model(
    chosen_ids=batch["chosen_ids"],
    chosen_mask=batch["chosen_mask"],
    rejected_ids=batch["rejected_ids"],
    rejected_mask=batch["rejected_mask"],
)
```

这会调用 `PreferenceRewardModel.forward()`。一个 batch 中的 chosen 和 rejected 是两条独立序列，二者不会互相做 attention。

### 基座模型提取 token 隐藏状态

`forward()` 先分别计算 chosen 和 rejected 的 reward：

```python
r_chosen = self.get_reward(chosen_ids, chosen_mask)
r_rejected = self.get_reward(rejected_ids, rejected_mask)
```

`get_reward()` 调用基类的 `get_hidden_states()`，后者把输入传给基座语言模型：

```python
outputs = self.model.base_model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    use_cache=False,
    return_dict=True,
)
hidden = outputs.last_hidden_state
```

对于长度为 `L` 的一组序列，输入形状是 `[B, L]`，模型输出的 `last_hidden_state` 形状是：

```text
[B, L, H]
```

其中 `H` 是模型隐藏层维度。这里使用的是 `base_model` 的 hidden states，而不是 causal language modeling head 输出的词表 logits。prompt 和回答都在输入中，模型会对完整序列进行计算；`attention_mask` 为 0 的 padding 位置不会参与 attention。

### 取每条序列最后一个有效 token

padding 后不能直接取 `hidden[:, -1, :]`，因为最后位置可能是 padding。代码根据 mask 计算每条序列最后一个真实 token 的位置：

```python
seq_lengths = attention_mask.sum(dim=1) - 1
batch_indices = torch.arange(hidden.size(0), device=hidden.device)
last_hidden = hidden[batch_indices, seq_lengths]
```

如果某条 mask 是：

```text
[1, 1, 1, 1, 1, 0, 0]
```

则最后一个有效位置是 `5 - 1 = 4`。对整个 batch，`seq_lengths` 的形状为 `[B]`，索引后得到：

```text
hidden:      [B, L, H]
last_hidden: [B, H]
```

因此，每个 chosen 或 rejected 序列最终只保留一个代表整条序列的 hidden vector。

### Reward head 输出标量奖励

最后一个有效 token 的 hidden state 经过线性 reward head：

```python
reward = self.head(last_hidden).squeeze(-1)
```

reward head 是一个 `nn.Linear(hidden_size, 1)`，数学形式为：

$$
r(x)=w^\top h_{\text{last}}+b
$$

形状变化为：

```text
last_hidden: [B, H]
线性层输出:  [B, 1]
squeeze(-1): [B]
```

所以 forward 得到：

```python
r_chosen   # [B]
r_rejected # [B]
```

例如：

```python
r_chosen   = tensor([1.20, 0.40])
r_rejected = tensor([0.30, 0.80])
```

### Bradley–Terry reward margin

模型先计算每个偏好对的 reward margin：

```python
margin = r_chosen - r_rejected
```

数学形式为：

$$
\Delta_i=r_{\text{chosen},i}-r_{\text{rejected},i}
$$

上面的例子得到：

```text
margin = [0.90, -0.40]
```

margin 大于 0 表示模型把 chosen 排在 rejected 前面；margin 小于 0 表示排序错误。

### Bradley–Terry 概率和损失

Bradley–Terry 模型把 chosen 胜出的概率定义为：

$$
P(\text{chosen}\succ\text{rejected})
=\sigma(r_{\text{chosen}}-r_{\text{rejected}})
$$

代码使用数值稳定的 `F.logsigmoid`，而不是先计算 sigmoid 再取对数：

```python
loss = -F.logsigmoid(r_chosen - r_rejected).mean()
```

完整的 batch 损失为：

$$
\mathcal{L}
=-\frac{1}{B}\sum_{i=1}^{B}
\log\sigma\left(
 r_{\text{chosen},i}-r_{\text{rejected},i}
\right)
$$

chosen reward 越高，margin 越大，损失越小；如果 rejected reward 更高，损失会变大。损失只约束两个 reward 的差值，因此 reward 的绝对数值本身没有固定意义。

最终 `forward()` 返回：

```python
return loss, r_chosen, r_rejected
```

其中 `loss` 是标量，供反向传播使用；两个 reward 向量用于计算训练准确率、平均 reward 和 reward margin 等日志指标。训练循环随后执行：

```python
(loss / grad_accum_steps).backward()
```

损失除以梯度累积步数后才反向传播。连续多个 microbatch 的梯度累积完成后，才调用 `optimizer.step()` 更新参数。这个过程不是 token-level 交叉熵训练，也没有单独的回答 loss mask；它对包含 prompt 和回答的完整序列产生一个 reward，再通过 chosen/rejected reward 的差值学习偏好排序。
