# Bulk Transcriptomics Autoencoder:Latent Space 中的 Batch 与 Biological Signal

用 PyTorch 建立一个 bulk RNA-seq 的 autoencoder,**量化**观察 latent space 里 batch effect 和生物信号各自占多少,并在此基础上尝试用 **batch discriminator(对抗)** 与 **triplet loss** 做 cross-cohort 对齐。

这个仓库和大多数 batch correction 代码不一样的地方:它不只给你一个调好的成品,而是**如实记录了对抗训练有多难调、在哪些超参下会崩**(见下方「对抗训练踩坑实录」)。因为在 small-n / high-p 的 bulk 场景里,方法能不能复现、值不值得上,比方法本身更重要。

---

## 一、文件说明

| 文件 | 内容 |
|---|---|
| `ae_bulk.py` | 数据(合成 demo + 真实数据接口)、preprocessing、plain autoencoder、**latent 诊断**(线性探针 + silhouette)、**线性 baseline**(PCA / removeBatchEffect)、latent 可视化 |
| `ae_adv_triplet.py` | 在 `ae_bulk.py` 基础上加 **batch discriminator(双优化器对抗)** + **跨 batch triplet loss**,复用同一套诊断以便直接对照 |

## 二、快速开始

```bash
pip install torch scikit-learn numpy matplotlib

python ae_bulk.py          # plain AE + 两条线性 baseline + 诊断 + latent 图
python ae_adv_triplet.py   # 对抗 + triplet 版本,与 plain AE 对照
```

接自己的数据:填 `ae_bulk.py` 里的 `load_real()` 桩,返回 `(X, batch, bio)`——
`X` 为已 log-normalized 的表达矩阵(log1p CPM / VST,限制到共享基因),
`batch` 为 cohort id(如 0 = GSE79362,1 = GSE94438),
`bio` 为生物标签(progressor=1 / non-progressor=0,或 active / latent,自选对比)。

### 跑完你会看到什么(复现步骤)

所有随机种子固定(`SEED = 0`),结果可复现。

**`python ae_bulk.py`** 依次打印:
1. 合成数据规模(400 样本 × 2000 基因,两队列 220/180,progressor 54/24——刻意让两队列比例不同,制造 batch × 生物混杂);
2. AE 训练的 reconstruction MSE(epoch 50→200:约 0.65 → 0.56,单调下降 = 收敛正常);
3. 三种表示的诊断(见结果表第 4.2 节):plain AE / PCA(raw) / PCA+removeBatchEffect;
4. 一张 `ae_latent.png`(latent 的 2D 投影,左按 batch、右按 biology 着色);
5. 一个 summary,可直接读出"线性 baseline 的 batch 探针从 1.00 掉到 0.37"。

**`python ae_adv_triplet.py`** 打印对抗 + triplet 版本的训练日志(每 75 epoch 一行:`lam_adv` / `rec` / `D_acc` / `tri`)与同一套诊断,并存 `ae_adv_latent.png`。关注两个信号:训练中 **D_acc 应从 ~0.88 降到 ~0.50**(判别器被骗到随机),但**事后独立探针的 batch_acc 仍可能 = 1.00**——这个反差正是第五节要讲的核心陷阱。

---

## 三、设计理念(第一性原理)

bulk RNA-seq 跨队列场景是典型的 **small-n / high-p**:两个 cohort、~14k 共享基因、真正独立个体数不多。这决定了三条设计约束:

1. **深度 AE 极易过拟合,也极易把 batch 直接背下来当"信号"。** 所以 AE 不是唯一工具,而是一个被诊断、被约束的探索对象——内建 HVG 筛选 + L2 正则 + Dropout。
2. **"观察 batch / biological signal" 必须量化,不能只看 UMAP/PCA。** 2D 投影的簇很会骗人。这里用两个指标:
   - **线性探针准确率**(在 latent 上跑 logistic 回归):batch 探针越低越好(混得开),bio 探针越高越好(信号还在);
   - **silhouette**:batch silhouette 越低越好,bio silhouette 越高越好。
3. **必须有线性 baseline 作对照。** 如果 AE 的 latent 在 batch mixing 上打不过一行 `removeBatchEffect`,那对抗 / triplet 就是在给一个错的架构打补丁。

---

## 四、结果

### 4.1 Plain AE 根本不会"自动"去 batch

一个朴素 autoencoder 会很乐意把 batch 当成主要可压缩结构编码进 latent。下图:左边按 cohort 着色两个队列几乎完全分开,右边按生物标签反而没那么干净。

![plain AE latent](ae_latent.png)

### 4.2 量化对照

| 表示 | batch 探针(↓好) | bio 探针(↑好) | 训练稳定性 | 备注 |
|---|---|---|---|---|
| Plain AE latent | 1.00(sil +0.22) | 1.00 | 稳 | batch 被原样编码,未做任何校正 |
| PCA(raw) | 1.00(sil +0.63) | 1.00 | 确定性 | 无校正,batch 主导 |
| **PCA + removeBatchEffect** | **0.37**(sil +0.00) | 1.00 | 确定性 | 线性 baseline,在这个设定里直接赢 |
| AE + discriminator + triplet | 见下方实录 | — | 依超参 | 需仔细调,且要用独立探针验证 |

> 合成数据里生物信号本身太易分,所以多数方案 bio 探针都是 1.00;真实 TB 数据里这一列才有区分度。

**核心结论:在这个设定下,朴素 AE 在 batch mixing 上打不过一行 `removeBatchEffect`。** 因此 batch discriminator 和 triplet loss 不是锦上添花,它们要证明自己存在的价值——价值在于以下三种**线性方法失效**的场景:
- batch effect 是**非线性**的(线性回归扣不掉);
- batch 与生物标签**混杂**(直接扣 batch 均值会连生物信号一起扣掉;TB 队列里 progressor 比例在两个 cohort 不同,正是这种情况);
- 你想要一个**可复用的 encoder**,把未来的新 cohort 直接投到同一 latent。

---

## 五、对抗训练踩坑实录

对抗式 batch correction 在实践中出了名的难调。这里如实记录三次尝试,而不是靠调参数把 demo 做漂亮——因为「它有多容易崩」本身就是你需要知道的结论。

### 尝试 1:单优化器 GRL —— `batch_acc` 没掉(0.997),`tri` 一开始就为 0

这不是 bug,它恰好暴露了两个问题:
1. 合成数据里的 batch effect 又强又低秩,弱 λ 的 GRL 推不动;
2. 生物信号在合成数据里太容易分,triplet 一上来就满足了 margin 所以失效(真实 TB 数据里生物信号弱,triplet 才会真正起作用)。

于是改用更稳的标准配方重做:**双优化器交替更新**(先把判别器 D 训到位,再让 encoder 去骗它)、**去掉 encoder 里会泄漏 batch 统计量的 BatchNorm**、加大 λ_adv。

### 尝试 2:双优化器 + λ_adv=8 —— 又崩了(rec 爆到 6.1,batch silhouette 0.97)

`λ_adv=8` + `k_disc=5` 让判别器永远赢,confusion 梯度过大,直接把重构炸掉;triplet 在 0↔88 之间震荡,是因为 `z` 没归一化、距离尺度失控。

**这就是对抗训练的真实面貌:它在 min-max 的钢丝上,超参一重就发散。**

三个标准稳定化修正:λ_adv 降到 1.5、`k_disc=1`、加梯度裁剪;triplet 改在 **L2 归一化的 z** 上算(把距离限制在 [0,2],度量学习的惯例)。

### 尝试 3:稳定化后 —— 训练稳了,但揭示了一个更深的陷阱

训练过程中 D_acc 从 0.88 掉到 0.50(判别器被骗到随机水平),重构稳定(~0.76),triplet 也生效了(bio silhouette 0.59)。

**但——注意这个关键现象:训练时把那个判别器骗到 chance 了,可事后拿一个全新的 logistic 回归探针去测,`batch_acc` 依然 = 1.00。**

这不是失败,而是对抗式 batch correction 一个著名且深刻的陷阱:**骗过某一个判别器 ≠ 把 batch 信息从表示里去掉。** encoder 只是找到了让「那个特定 D」抓不到的编码方式,而一个独立探针照样能把 batch 挖出来(16 维 latent 空间大得很,藏得下)。这正是领域后来转向用 **iLISI / kBET** 这类更严格的指标、并把对抗与显式分布匹配(MMD)结合起来的原因。

### 踩坑清单(直接可用的经验)

- **encoder 别用 BatchNorm,用 LayerNorm。** BN 在 minibatch 内归一化,当一个 batch 里两个 cohort 比例失衡时会把 batch 统计量重新注入 `z`,和判别器对着干——最隐蔽的坑。
- **λ_adv 用 sigmoid 从 0 缓升**(DANN schedule),让重构先稳住;`k_disc` 保持 1~2,配梯度裁剪。
- **triplet 前先 L2 归一化 `z`**,否则距离尺度失控、loss 震荡。
- **收敛判据不是「D_acc 掉到 chance」,而是独立探针 + iLISI。** 别信 loss 曲线。

---

## 六、下一步:如何加 discriminator 与 triplet

### Batch discriminator

原理是 encoder 与判别器 D 的 min-max。推荐**双优化器**方案(比一次性 GRL 更可控):
- `opt_D` 只更新 D,用 `z.detach()` 分类 batch;
- `opt_G` 更新 encoder+decoder,让 D 的输出趋向 uniform(batch 不可分)。

完整实现见 `ae_adv_triplet.py` 的 `train_adv()`。

### Triplet loss —— 灵魂是「跨 batch 取 positive」

普通 triplet 只让同类靠近;对 cross-cohort 有用的那一步是:**anchor 的 positive 优先取自另一个 cohort 的同类样本**。这等价于把「同生物、不同 batch → 拉近」写进目标,才是它对齐 cohort 的机制。见 `triplet_loss()`。

两个现实问题:
- **progressor 稀有 → 很多 minibatch 里没有正样本**,triplet 静默失效。用 **PK-sampling**(每个 minibatch 保证 P 个类 × K 个样本,Hermans et al. batch-hard 标配)或退化到 online semi-hard。
- **`z` 必须 L2 归一化**再算距离。

### 推荐路径(第一性原理,不是最短但最稳)

1. **先跑两条 baseline**:plain AE 和 `removeBatchEffect`,用探针 + iLISI/kBET 量化。**如果线性方法已经 batch 混得好、生物又没丢,就到此为止**——cross-cohort reproducibility 越简单越可辩护。
2. **只有当 batch × 生物标签混杂让线性方法连信号一起扣掉时**,才上有监督的对齐。这时**优先试 MMD 惩罚而不是对抗**——它是确定性的、没有 min-max 钢丝,在 small-n 上稳得多:

```python
import torch

def mmd_rbf(za, zb, sigmas=(1., 2., 4., 8.)):
    """batch a 与 batch b 在 latent 里的分布差异;加进 loss 惩罚它。"""
    def k(x, y):
        d = torch.cdist(x, y) ** 2
        return sum(torch.exp(-d / (2 * s ** 2)) for s in sigmas)
    return k(za, za).mean() + k(zb, zb).mean() - 2 * k(za, zb).mean()

# L = recon + lam_trip * triplet + lam_mmd * mmd_rbf(z[batch == 0], z[batch == 1])
```

MMD + triplet(分布对齐 + 生物结构保持)通常比对抗 + triplet 更容易复现,也更好向审稿人解释。对抗留作「MMD 也压不下去」时的重武器。

3. **评估协议固定**:独立 logistic 探针(batch↓ / bio↑)+ scib 的 iLISI / cLISI + kBET;探针的 train/test **按 subject 划分而非 sample**——有重复测量时,同一个人的不同时间点跨进 train/test 会让「生物可解码性」虚高。

### 可选:更贴合 count 分布

把重构损失从 MSE 换成**负二项(NB)似然**、直接在 raw counts 上建模(scVI 的做法),对 bulk count 的均值-方差关系更忠实。

---

## 依赖

```
torch >= 2.0
scikit-learn
numpy
matplotlib
```

## 参考

- Ganin & Lempitsky, *Domain-Adversarial Training of Neural Networks*(GRL / DANN)
- Hermans et al., *In Defense of the Triplet Loss for Person Re-Identification*(batch-hard mining)
- Luecken et al., *Benchmarking atlas-level data integration*(iLISI / kBET / scIB 指标)
- Lopez et al., *scVI*(NB likelihood decoder)
