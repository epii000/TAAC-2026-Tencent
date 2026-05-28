# TAAC 2026 Tencent Academic Track (0.831984开源方案，学术组66名)

晚上八点压哨测试出分：**AUC = 0.831984**

<img width="1335" height="276" alt="325a9778b75fd8b82f4cf774689516d2" src="https://github.com/user-attachments/assets/1a5a9be4-d0f5-4949-a2a0-4736512fe620" />



## 核心模块

### 1. DenseGroupProjector

baseline 中 dense 特征基本是整体投影成一个 token。这里将部分用户 dense 特征按照语义分成几组，例如 embedding 类、统计类和分位类特征，然后分别编码成额外的 NS token。

这样做的目的，是让不同类型的 dense 特征可以以更清晰的形式参与后续交互，而不是全部混在一个 dense 表示里。

---

### 2. 时间特征建模

final 版本加入了当前样本时间和历史序列时间相关的特征。

当前样本侧主要使用 Beijing hour 和 weekday；序列侧则构造了历史行为与当前样本之间的时间差、相邻行为 gap、历史行为发生的 hour / weekday，以及每个序列域的时间 profile。

这些时间信息会被后续的时间上下文编码、序列调制、动态 query 和 DIN 读取模块使用。主要目的是让模型在读取用户历史行为时，能够考虑行为发生时间和当前预测时间之间的关系。

---

### 3. TimeContextEncoder

`TimeContextEncoder` 用来聚合不同序列域中的时间信息，得到一个全局时间上下文表示。

由于不是每个样本的每个序列域都有有效历史，所以聚合时会过滤掉空序列 domain，只对有效序列进行汇总。

这个模块主要给后续 query 生成和序列读取提供统一的时间上下文。

---

### 4. SeqTimeFiLM

`SeqTimeFiLM` 用于根据时间上下文对序列 token 做调制。

简单来说，它会根据时间信息对序列表示做一层动态调整，使序列 token 在进入后续读取模块前带上时间相关的信息。

这个模块主要是为了让序列表示不只包含行为内容，也能反映这些行为在时间上的变化。

---

### 5. DualQueryTargetTimeGenerator

final 版本中，每个序列域使用两个 query：一个 long query，一个 target query。

long query 更偏向读取用户长期兴趣；target query 则会结合候选物料、当前时间上下文、近期序列表示和序列时间 profile，用来读取和当前预测目标更相关的历史行为。

query 生成时采用 residual delta 的方式，在原始 base query 上叠加动态生成的增量，避免直接替换原有 query 导致结构不稳定。

---

### 6. TargetTimeDINReader

`TargetTimeDINReader` 是一个 DIN-style 的序列读取模块，主要作用在 target query 上。

它会结合候选物料、目标时间和历史序列 token，计算 target-aware 的注意力权重，并对历史序列进行加权聚合。最终结果作为 residual 信息融合回模型输出。

这个模块主要用于加强候选物料和用户历史行为之间的匹配关系，同时保留原有 HyFormer 的序列建模能力。

---

### 7. NS token cross 与输出融合

final 版本在 NS token 上加入了类似 DCN-v2 的低秩交叉模块，用于增强用户侧、物料侧以及 dense group token 之间的组合表达。

更新后的 NS token 还会在最终输出前再次融合回来，作为非序列特征的 residual 信息。

这个部分主要是为了避免最终预测过度依赖序列表示，同时补充非序列特征之间的高阶组合信息。

---

### 8. HashEmbedding fallback

对于高基数 sparse 特征，final 版本保留了 HashEmbedding fallback。

当某些 sparse 特征的基数超过设定阈值时，不直接跳过或置零，而是通过 hash embedding 的方式继续保留这部分信息。

该逻辑覆盖用户 sparse、物料 sparse 和序列 sparse 特征，主要是为了减少高基数特征被完全丢弃带来的信息损失。

---

## 模块汇总

| 模块                             | 主要作用                         |
| ------------------------------ | ---------------------------- |
| `DenseGroupProjector`          | 将不同类型 dense 特征编码为额外 NS token |
| 时间特征建模                         | 引入当前时间和历史行为时间信息              |
| `TimeContextEncoder`           | 聚合不同序列域的时间上下文                |
| `SeqTimeFiLM`                  | 用时间信息调制序列 token              |
| `DualQueryTargetTimeGenerator` | 生成长期兴趣 query 和目标时间 query     |
| `TargetTimeDINReader`          | 进行候选物料感知的 DIN-style 序列读取     |
| NS token cross                 | 增强非序列特征之间的交叉                 |
| NS residual fusion             | 将更新后的非序列信息融合到最终输出            |
| HashEmbedding fallback         | 保留高基数 sparse 特征信息            |

---

## Result

最终线上 AUC：

```text
0.831984
```

