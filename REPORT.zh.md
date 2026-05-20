# 使用机器学习预测交友 App 沟通中断风险

**课程：** WIA1006/WID3006 Machine Learning  
**小组成员：** 提交前请将此处替换为所有组员姓名。  
**Notebook：** `RESULT_dating_dropoff_risk.ipynb`  
**数据集：** `data/dating_app_behavior_dataset.csv`

## 摘要

本项目研究是否可以利用虚构交友 App 的用户行为数据预测沟通中断风险。原始数据集包含 50,000 条记录和 19 个特征，涵盖用户人口统计信息、App 使用习惯、滑动行为、消息行为以及配对结果。本项目根据原始 `match_outcome` 字段构建了二分类目标变量 `dropoff_risk`。其中，`Ghosted` 和 `Chat Ignored` 被定义为正类，即存在沟通中断风险；其他结果被定义为负类。

项目训练、调优并比较了 5 个作业允许的机器学习模型：Logistic Regression、Decision Tree、Random Forest、Support Vector Machine 和 Multilayer Perceptron。同时，项目使用 Dummy Classifier 作为基准模型，并在本地 WSL/Linux 环境中运行 auto-sklearn 进行对比。最佳手动模型是 Support Vector Machine，其 F1 score 为 0.2900。auto-sklearn 获得了更高的 F1 score，为 0.3313，但主要原因是它的 recall 很高。整体来看，大多数模型的 balanced accuracy 和 ROC-AUC 都接近 0.50，说明当前特征对该二分类任务的区分能力较弱。

## 1. 问题与目标

数字交友平台会产生大量用户行为信号，例如 App 使用时长、右滑比例、收到的点赞数、双向匹配数、个人资料完整度、发送消息数量和 emoji 使用率等。这些信号可能反映用户在互动结果出现前的行为模式。本项目关注的问题是：

**交友 App 的用户行为数据能否预测一次互动是否会以沟通中断结束？**

本项目的目标是建立并比较多个机器学习模型，用于预测一条记录是否属于沟通中断风险类别。在本项目中，沟通中断被定义为最终 `match_outcome` 为 `Ghosted` 或 `Chat Ignored` 的情况。

项目目标包括：

- 根据指定数据集构建二分类机器学习问题。
- 进行探索性数据分析，理解数据分布和潜在模式。
- 对数值特征、类别特征和多标签兴趣特征进行预处理。
- 训练并调优至少 5 个机器学习模型。
- 使用合适的分类指标比较模型表现。
- 将手动选择的模型与 auto-sklearn 进行比较。
- 解释结果，并讨论局限性和伦理问题。

## 2. 数据集与目标变量定义

该数据集是一个虚构交友 App 用户行为的合成数据集，包含：

- 50,000 条记录
- 19 个原始字段
- 无缺失值
- 数值型、类别型和类似文本的多标签特征
- 一个标签字段：`match_outcome`

原始目标字段 `match_outcome` 包含 10 种结果类别。本项目将其转换为二分类目标：

| 原始 `match_outcome` 值 | 新目标变量 |
|---|---|
| `Ghosted` | `dropoff_risk = 1` |
| `Chat Ignored` | `dropoff_risk = 1` |
| 其他所有结果 | `dropoff_risk = 0` |

最终目标变量分布如下：

| 类别 | 数量 | 比例 |
|---|---:|---:|
| 无沟通中断 | 40,022 | 80.04% |
| 沟通中断 | 9,978 | 19.96% |

该分布说明数据存在类别不平衡。由于正类只占约五分之一，accuracy 不能作为唯一或主要评价指标。

## 3. 探索性数据分析

探索性数据分析显示，二分类目标存在不平衡，但正类并不是极少数类别。正类比例约为 20%。

数值特征在沟通中断和非沟通中断样本之间的均值差异很小。例如：

| 特征 | 无沟通中断均值 | 沟通中断均值 | 差异 |
|---|---:|---:|---:|
| `likes_received` | 99.4172 | 99.9625 | 0.5453 |
| `app_usage_time_min` | 150.0067 | 149.5342 | -0.4725 |
| `bio_length` | 250.1117 | 250.4257 | 0.3140 |
| `mutual_matches` | 13.8416 | 13.9854 | 0.1438 |
| `message_sent_count` | 50.0984 | 49.9656 | -0.1328 |

类别特征也没有表现出明显区分度。大部分类别下的沟通中断率都接近整体 20% 的基准比例。例如，`app_usage_time_label` 不同类别的沟通中断率约在 19.54% 到 20.64% 之间，`location_type` 不同类别的沟通中断率约在 19.57% 到 20.21% 之间。

这些 EDA 结果说明，根据当前特征预测沟通中断风险是一个较困难的任务。数据中能够区分正类与负类的信号较弱。

建议从 Notebook 中加入以下图表：

- 图 1：二分类目标变量分布
- 图 2：原始 `match_outcome` 分布
- 图 3：不同 `dropoff_risk` 下的数值特征分布
- 图 4：数值特征相关性热力图
- 图 5：不同类别特征下的沟通中断率表格

## 4. 方法与模型说明

### 4.1 数据预处理

原始 `match_outcome` 字段只用于创建二分类目标变量。创建 `dropoff_risk` 后，`match_outcome` 被从特征矩阵中删除，以避免目标泄漏。

预处理流程分为三类特征：

1. **数值特征**
   - 使用中位数填补缺失值
   - 使用 StandardScaler 标准化

2. **类别特征**
   - 使用众数填补缺失值
   - 使用 one-hot encoding 编码

3. **兴趣标签**
   - `interest_tags` 字段包含逗号分隔的兴趣标签。
   - 使用标签拆分和 `CountVectorizer` 转换为多标签特征。

数据集使用 80:20 的 stratified train/test split：

- 训练集：40,000 条记录
- 测试集：10,000 条记录
- 训练集正类比例：19.96%
- 测试集正类比例：19.96%

使用 stratified split 是为了确保训练集和测试集保持相同的类别比例。

### 4.2 使用的模型

本项目训练并调优了以下 5 个作业允许的模型：

1. **Logistic Regression**
   - 线性分类模型，作为简单且较易解释的基准模型。
   - 使用 `class_weight="balanced"` 处理类别不平衡。

2. **Decision Tree**
   - 非线性树模型，可以捕捉特征之间的交互关系。
   - 调优了树深度和叶节点最小样本数等参数。

3. **Random Forest**
   - 由多棵决策树组成的集成模型，用于降低过拟合并提高泛化能力。
   - 调优了树数量、最大深度和叶节点最小样本数。

4. **Support Vector Machine**
   - 使用 `LinearSVC` 实现，以提高训练效率。
   - 使用 `class_weight="balanced"`。
   - 调优了正则化参数 `C`。

5. **Multilayer Perceptron / Artificial Neural Network**
   - 使用 `MLPClassifier` 实现。
   - 调优了隐藏层结构和正则化强度。

此外，项目还加入了 **Dummy Classifier** 作为基准模型，但它不计入作业要求的 5 个模型。

### 4.3 超参数调优

模型超参数调优使用 `GridSearchCV` 和 3-fold stratified cross-validation。调优指标为正类 F1 score，因为本项目重点是识别沟通中断风险样本。

各模型的最佳参数如下：

| 模型 | 最佳超参数 |
|---|---|
| Logistic Regression | `C = 1.0` |
| Decision Tree | `max_depth = None`, `min_samples_leaf = 50` |
| Random Forest | `n_estimators = 100`, `max_depth = 10`, `min_samples_leaf = 10` |
| Support Vector Machine | `C = 0.1` |
| MLP / ANN | `hidden_layer_sizes = (64,)`, `alpha = 0.0001` |

## 5. 结果与可视化

### 5.1 手动模型比较

最佳手动模型是 Support Vector Machine，其测试集 F1 score 为 0.2900。

| 模型 | Accuracy | Balanced Accuracy | Precision | Recall | F1 | ROC-AUC | Average Precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| Support Vector Machine | 0.5142 | 0.5077 | 0.2047 | 0.4970 | 0.2900 | 0.5065 | 0.2002 |
| Logistic Regression | 0.5141 | 0.5077 | 0.2047 | 0.4970 | 0.2899 | 0.5065 | 0.2002 |
| Decision Tree | 0.5240 | 0.4952 | 0.1963 | 0.4474 | 0.2728 | 0.4890 | 0.1939 |
| Random Forest | 0.6992 | 0.4966 | 0.1930 | 0.1593 | 0.1745 | 0.4879 | 0.1950 |
| MLP / ANN | 0.8004 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.5128 | 0.2040 |
| Dummy Baseline | 0.8004 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |

虽然 Dummy Classifier 的 accuracy 达到 80.04%，但它的 F1 score 为 0.0000，因为它只预测多数类。这说明在类别不平衡任务中，accuracy 具有误导性。

SVM 和 Logistic Regression 的结果几乎相同。它们都能识别约一半真实的沟通中断样本，但 precision 只有约 20%。这意味着模型预测为沟通中断的样本中有大量误报。

Random Forest 的 accuracy 较高，但正类 recall 很低，因此不适合本项目目标。MLP 模型在默认阈值下没有预测任何正类，因此 F1 score 为 0.0000。

建议加入以下图表：

- 图 6：模型比较柱状图
- 图 7：最佳手动模型的混淆矩阵

### 5.2 最佳手动模型：Support Vector Machine

最佳手动模型的分类报告如下：

| 类别 | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| 无沟通中断 | 0.81 | 0.52 | 0.63 | 8,004 |
| 沟通中断 | 0.20 | 0.50 | 0.29 | 1,996 |

该模型整体表现为：

- Accuracy：0.51
- Macro average F1：0.46
- Weighted average F1：0.56

最重要的解释是：SVM 能够识别约一半真实沟通中断样本，但同时产生了大量 false positives。因此，它更适合作为弱风险筛查模型，而不是可靠的预测系统。

### 5.3 auto-sklearn 比较

由于 Google Colab 默认 Python 环境无法成功安装 auto-sklearn，本项目在本地 WSL/Linux Python 3.9 环境中运行 auto-sklearn。

auto-sklearn 运行结果如下：

| 指标 | 数值 |
|---|---:|
| Accuracy | 0.2209 |
| Balanced accuracy | 0.5009 |
| Precision | 0.1999 |
| Recall | 0.9669 |
| F1 | 0.3313 |
| ROC-AUC | 0.4997 |
| Average precision | 0.2008 |

auto-sklearn 在所有方法中取得了最高的 F1 score。然而，这一提升主要来自非常高的 recall。0.9669 的 recall 表示 auto-sklearn 几乎识别了所有真实沟通中断样本，但 0.1999 的 precision 表示大多数被预测为沟通中断的样本实际上是误报。

auto-sklearn 的 ROC-AUC 和 balanced accuracy 都接近 0.50。这说明即使 auto-sklearn 获得了更高的 F1 score，它也没有学到强区分性的模式，而是倾向于频繁预测正类。

auto-sklearn 的运行统计如下：

- 目标算法运行次数：282
- 成功运行次数：156
- 崩溃运行次数：87
- 超时次数：9
- 超出内存限制次数：30
- 最佳验证分数：0.333484

运行过程中出现了 OpenBLAS 内存分配警告，并且部分候选模型超过了内存限制。但最终 auto-sklearn 仍然完成运行并输出了有效指标。

### 5.4 包含 auto-sklearn 的完整模型比较

| 模型 | F1 | Precision | Recall | Balanced Accuracy | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| auto-sklearn | 0.3313 | 0.1999 | 0.9669 | 0.5009 | 0.4997 |
| Support Vector Machine | 0.2900 | 0.2047 | 0.4970 | 0.5077 | 0.5065 |
| Logistic Regression | 0.2899 | 0.2047 | 0.4970 | 0.5077 | 0.5065 |
| Decision Tree | 0.2728 | 0.1963 | 0.4474 | 0.4952 | 0.4890 |
| Random Forest | 0.1745 | 0.1930 | 0.1593 | 0.4966 | 0.4879 |
| MLP / ANN | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.5128 |

如果以 F1 score 作为主要指标，auto-sklearn 是表现最好的方法。但如果从 balanced accuracy 或 ROC-AUC 角度看，所有模型都没有明显优于随机分类。

## 6. 洞察与结果解释

### 6.1 主要洞察

本项目最重要的发现是：当前数据特征对所选二分类目标的预测能力有限。无论是手动模型还是 auto-sklearn，都难以有效区分沟通中断和非沟通中断样本。

这一结论由以下现象支持：

- EDA 中不同类别的特征差异很小。
- 各类别下的沟通中断率接近整体基准比例。
- ROC-AUC 接近 0.50。
- Balanced accuracy 接近 0.50。
- Average precision 接近正类比例。

### 6.2 最佳模型解释

最佳手动模型 SVM 的 recall 为 0.4970，precision 为 0.2047。也就是说，它能识别约一半真实沟通中断样本，但大多数正类预测是误报。

auto-sklearn 的 F1 score 更高，为 0.3313，但这是通过更激进地预测正类实现的。它的 recall 高达 0.9669，但 precision 仍然只有 0.1999。这说明 auto-sklearn 更倾向于“尽量不漏掉沟通中断样本”，代价是产生大量误报。

从实际应用角度看，这些模型都不足以用于真实决策。它们适合用于课程项目中的建模流程展示和结果讨论，但不应被视为可靠的个体预测工具。

### 6.3 特征重要性解释

在最佳手动模型中，较重要的特征包括部分兴趣标签，例如 `MMA`、`History`、`Binge-Watching`、`Clubbing`，以及部分 App 使用标签，例如 `Extreme User` 和 `Very Low`。

但是，由于模型整体性能较弱，这些特征重要性应谨慎解释。它们可能反映的是较弱的统计模式或噪声，而不是稳定的预测关系，更不能被解释为因果关系。

## 7. 伦理考量与局限性

### 7.1 合成数据局限

该数据集是程序生成的合成数据。因此，项目结论不能直接推广到真实交友 App 用户。该分析主要展示如何将机器学习应用于行为数据，而不是证明真实世界中的社交规律。

### 7.2 敏感属性

数据集中包含 gender、sexual orientation、income bracket 和 education level 等敏感或个人属性。如果不谨慎使用，这些特征可能导致偏见或不公平预测。

项目通过消融实验比较了保留和移除部分敏感特征后的 Logistic Regression 表现：

| 模型变体 | F1 | Precision | Recall | Balanced Accuracy | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| 保留敏感特征 | 0.2899 | 0.2047 | 0.4970 | 0.5077 | 0.5065 |
| 移除部分敏感特征 | 0.2875 | 0.2026 | 0.4950 | 0.5046 | 0.5023 |

移除敏感特征后，F1 score 只出现了很小下降。这说明所选敏感特征并不是模型整体表现的核心来源。

然而，按 gender 分组检查时可以看到，即使真实沟通中断率相近，模型预测的沟通中断率仍存在明显差异。例如，各 gender 组的真实沟通中断率约为 18.8% 到 20.7%，但预测沟通中断率约为 41.8% 到 60.0%。这说明即使模型整体表现较弱，也可能在不同群体之间产生不均衡预测。

### 7.3 实际局限

本项目存在以下局限：

- 二分类目标可能过于宽泛，因为 `Ghosted` 和 `Chat Ignored` 相关但并不完全相同。
- 合成数据中可能不存在足够强的特征与目标关系。
- 模型区分能力较弱。
- auto-sklearn 运行过程中出现内存警告，并且多个候选模型运行失败。
- 模型不应被用于真实用户评估、排序、审核或关系预测。

## 8. 结论

本项目构建了一个用于预测交友 App 沟通中断风险的二分类机器学习流程。项目将 `match_outcome` 为 `Ghosted` 或 `Chat Ignored` 的记录定义为正类。

项目训练并比较了 5 个作业允许的模型。最佳手动模型是 Support Vector Machine，F1 score 为 0.2900。auto-sklearn 获得了更高的 F1 score，为 0.3313，但主要原因是它具有很高的 recall 和较低的 precision。无论是 auto-sklearn 还是手动模型，其 balanced accuracy 和 ROC-AUC 都接近 0.50。

最终结论是：当前数据特征对沟通中断风险的预测信号较弱。尽管如此，本项目完整展示了机器学习流程，包括目标变量构建、数据预处理、模型选择、超参数调优、模型比较、auto-sklearn 对比、结果解释和伦理评估。但这些结果应谨慎解释，模型不应被用于真实场景中的决策。

## 参考资料

1. 作业说明：`guideline.md`
2. 数据集来源：https://www.kaggle.com/datasets/keyushnisar/dating-app-behavior-dataset
3. auto-sklearn 文档：https://automl.github.io/auto-sklearn/master/
4. scikit-learn 文档：https://scikit-learn.org/stable/

