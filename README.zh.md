# 交友 App 断联风险预测

本项目是一个基于 Google Colab 的机器学习作业，用于预测虚构交友 App 用户互动中的“沟通中断 / 断联风险”。

Notebook 会从原始的 `match_outcome` 字段构建一个二分类任务：

- `dropoff_risk = 1`：`match_outcome` 是 `Ghosted` 或 `Chat Ignored`
- `dropoff_risk = 0`：其他所有结果

主要 Notebook：

- `dating_dropoff_risk.ipynb`

数据集：

- `data/dating_app_behavior_dataset.csv`

## 使用了哪些模型

Notebook 训练并比较了 5 个作业允许的机器学习模型：

1. **Logistic Regression（逻辑回归）**
2. **Decision Tree（决策树）**
3. **Random Forest（随机森林）**
4. **Support Vector Machine（支持向量机）**
   - 在 Notebook 中使用 `LinearSVC` 实现，适合在 Google Colab 中更快训练。
5. **Artificial Neural Network / Multilayer Perceptron（人工神经网络 / 多层感知机）**
   - 在 Notebook 中使用 `MLPClassifier` 实现。

Notebook 还包含：

- **DummyClassifier baseline**
  - 只作为基准模型使用。
  - 不计入作业要求的 5 个模型。
- **可选 auto-sklearn 部分**
  - 默认关闭：`RUN_AUTOSKLEARN = False`
  - 如果 auto-sklearn 无法安装或运行，Notebook 会保留环境限制说明，方便写进报告。

## 评估指标

主要比较指标是正类的 **F1 score**，因为本项目重点是识别断联风险样本。

Notebook 还会输出：

- Accuracy（准确率）
- Balanced accuracy（平衡准确率）
- Precision（精确率）
- Recall（召回率）
- ROC-AUC
- Average precision / PR-AUC
- Classification report（分类报告）
- Confusion matrix（混淆矩阵）

## 数据预处理

Notebook 使用 sklearn pipeline 完成预处理：

- 数值特征：
  - 中位数填补
  - 标准化
- 类别特征：
  - 众数填补
  - One-hot encoding
- `interest_tags`：
  - 按逗号拆分标签
  - 使用 `CountVectorizer` 编码成多标签特征

在构建 `dropoff_risk` 之后，原始的 `match_outcome` 字段会从训练特征中删除，避免标签泄漏。

## 如何在 Google Colab 运行

1. 打开 Google Colab：<https://colab.research.google.com/>
2. 上传 `dating_dropoff_risk.ipynb`。
3. 上传数据集文件：
   - `dating_app_behavior_dataset.csv`
4. 确保 CSV 在 Colab 中的路径是：
   - `/content/dating_app_behavior_dataset.csv`
5. 在 Colab 菜单点击：
   - `Runtime` -> `Run all`

Notebook 也保留了本地备用路径：

```text
data/dating_app_behavior_dataset.csv
```

这个备用路径主要用于在本仓库中检查或运行 Notebook。

## 可选 auto-sklearn 尝试

auto-sklearn 部分默认关闭：

```python
RUN_AUTOSKLEARN = False
```

如果要尝试运行 auto-sklearn，可以改成：

```python
RUN_AUTOSKLEARN = True
```

如果 auto-sklearn 在 Colab 中安装失败，可以保留失败输出，并在报告中说明环境限制。Notebook 中引用了 auto-sklearn 官方安装说明：

<https://automl.github.io/auto-sklearn/master/installation.html>

## 运行后会得到什么

运行完整 Notebook 后，会生成：

- 数据集检查摘要
- 目标变量分布
- EDA 可视化图表
- 调参后的模型比较表
- 模型比较图
- 最佳模型混淆矩阵
- 特征重要性图
- 敏感特征消融实验结果
- 英文报告笔记
- 幻灯片大纲

## 提交前检查清单

提交前请确认：

- Notebook 中的小组成员占位符已替换成真实姓名。
- 已在全新的 Colab runtime 中从头到尾运行 Notebook。
- 数据集读取结果是 50,000 行、19 个原始字段。
- `dropoff_risk` 已正确创建。
- `match_outcome` 没有被用作模型训练特征。
- 5 个作业要求模型都成功输出评估指标。
- 已导出完成后的 `.ipynb`。
- 已导出或打印 Notebook 为 `.pdf`。
- 已使用 Notebook 中的 report notes 和 slide outline 准备最终报告与展示幻灯片。

