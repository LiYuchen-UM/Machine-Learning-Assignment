# WIA1006/WID3006 机器学习
2025/2026 学年第二学期
小组作业

## 提示：你只能使用以下模型
| Learning Paradigm        | Task Type                       | Model Name                                                                                                                |
| -------------------------- | --------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| Supervised Learning      | Regression                      | Linear Regression, Polynomial Regression, Locally Weighted Linear Regression                                              |
| Supervised Learning      | Classification                  | Logistic Regression                                                                                                       |
| Supervised Learning      | Classification & Regression     | Decision Tree, Random Forest, Artificial Neural Network (ANN) / Multilayer Perceptron (MLP), Support Vector Machine (SVM) |
| Unsupervised Learning    | Clustering                      | K-Means, Hierarchical Clustering (Agglomerative & Divisive), Spectral Clustering, Gaussian Mixture Model (GMM)            |
| Unsupervised Learning    | Dimensionality Reduction        | Principal Component Analysis (PCA)                                                                                        |
| Probabilistic Models     | Sequence / State Prediction     | Markov Models, Hidden Markov Models (HMM)                                                                                 |
| Other / Advanced Methods | Optimization & Dynamic Learning | Expectation-Maximization Algorithm (EM), Reinforcement LearningGenetic Algorithm                                          |

# 💘数据牵起红线：爱情、生活与点赞💘

本项目面向修读 2025/2026 学年第二学期 WIA1006/WID3006《机器学习》课程的马来亚大学（UM）FCSIT 本科计算机科学学生。

本作业将于第 5 周，即 2026 年 4 月 6 日（星期一）中午 12:00 开始，最终提交时间为第 13 周，即 2026 年 6 月 8 日（星期一）中午 12:00。

# 1. 案例背景

在持续互联的时代，人际关系越来越受到数字化互动的影响。如今，人们通过即时通讯进行交流，通过表情符号传递情绪，而“沉默”有时比语言本身更具含义。

与传统关系不同，现代关系往往通过回复时间、在线状态以及社交媒体互动等模式逐渐发展，并由此产生了诸如“Ghosting（突然断联）”和“Situationship（暧昧关系）”等现象。

本项目《数据牵起红线：爱情、生活与点赞》邀请学生探索如何将这些数字行为转化为数据，并利用机器学习进行分析。

通过研究互动模式与行为信号，学生可以尝试预测关系结果，例如：

- 配对结果（match outcomes）
- 关系持续时间（relationship duration）
- 被突然断联的可能性（likelihood of ghosting）
- 关系类型（relationship type）
- 等等

通过这个有趣的案例研究，学生不仅需要应用技术技能，还应更深入地理解：数据如何反映，甚至有时误导，人类在数字时代中的复杂关系。

# 2. 数据集（非常重要）

你必须使用指定的数据集。

该数据集是一个虚构交友应用中用户行为的合成数据表示，共包含 50,000 条记录以及 19 个特征，用于描述：

- 人口统计信息（demographic details）
- App 使用模式（app usage patterns）
- 滑动行为倾向（swipe tendencies）
- 配对结果（match outcomes）

数据通过程序生成，以模拟真实用户互动，因此非常适合用于：

- 探索性数据分析（EDA）
- 机器学习建模（例如预测配对结果）
- 研究在线交友平台中的用户行为趋势

关键特征包括：

- 性别（gender）
- 性取向（sexual orientation）
- 地区类型（location type）
- 收入等级（income bracket）
- 教育水平（education level）
- 用户兴趣（user interests）
- App 使用时间（app usage time）
- 滑动比例（swipe ratios）
- 收到的点赞数（likes received）
- 双向匹配数（mutual matches）
- 配对结果（例如 “Mutual Match”、“Ghosted”、“Catfished” 等）

该数据集具有良好的多样性与平衡性，包含：

- 类别型变量（categorical variables）
- 数值型变量（numerical variables）
- 标签变量（labeled variables）

适用于多种分析任务。

数据集链接：

https://www.kaggle.com/datasets/keyushnisar/dating-app-behavior-dataset

# 3. 机器学习流程

在本项目中，你们将以小组形式使用机器学习技术探索并分析一个特定主题。

项目目标是：

- 训练（train）
- 优化（optimize）
- 比较（compare）

多个机器学习模型（至少 5 个模型），并比较它们在解决现实问题中的表现。

本项目采用标准机器学习流程如下：

## 1）项目设计（Design your project）

小组首先需要进行头脑风暴，提出一个你们感兴趣的具体问题，例如：

- 配对结果
- 关系持续时间
- 被断联概率
- 关系类型
- 等等

鼓励你们参考各种资源与代码仓库获取灵感，但最终所要解决的机器学习问题必须由你们自己提出。

请为你们的项目命名。

## 2）数据收集 / 获取 / 挖掘（Data Collection / Acquisition / Mining）

确定主题与研究问题后，你们可能需要收集或寻找与项目相关的数据。

欢迎探索更多数据来源。

## 3）数据预处理（Data Pre-processing）

收集数据后，需要对数据进行预处理，以适用于机器学习任务。

这可能包括：

- 数据清洗（cleaning）
- 数据归一化（normalization）
- 数据转换（transformation）

## 4）特征选择 / 特征提取（Feature Selection / Feature Extraction）

在模型训练之前，你们可以使用 PCA 等技术进行特征选择或特征提取。

## 5）模型选择（Model Selection）

数据准备完成后，需要为项目选择合适的机器学习算法。

你们可以选择：

- 监督学习（supervised learning）
- 无监督学习（unsupervised learning）

并根据你们要解决的问题选择合适模型。

## 6）模型训练与超参数调优（Model Training and Hyperparameter Tuning）

选定模型后，需要使用预处理后的数据进行训练，并利用适当指标评估模型表现。

你们可能需要对模型进行微调，以提升性能。

## 7）模型评估（Model Evaluation）

根据你们所选择的机器学习任务：

- 最适合的评估方法是什么？
- 如何比较不同模型？
- 你的模型与 auto-sklearn 相比表现如何？

参考：

https://automl.github.io/auto-sklearn/master/

## 8）可选项：将项目开发为简单应用（Optional）

例如：

- Dashboard（仪表板）
- 交互式工具（interactive tool）

你们展示的功能与投入越多，分数可能越高。

## 9）最终成果与提交（Final Deliverable and Submission）

你们需要清晰、简洁地展示研究发现与结果。

请通过 SPECTRUM 提交以下内容：

### i. 项目展示幻灯片（Presentation Slides）

请确保包含：

- 5 分钟项目展示视频链接
- Google Colab / Kaggle Notebook 链接

如果使用本地 Notebook，请提交以下两个文件：

- `.ipynb` 文件
- `.pdf` 文件

请勿提交视频文件本身。

### ii. 小组项目报告（Group Project Report）

**请在所有提交文件中注明所有组员姓名。**

# 4. 评分标准

本次作业总分占课程总成绩的 14%。

请务必准时提交，逾期提交将不被接受，所有组员将被记为 0%。

## 最终提交时间

第 13 周：2026 年 6 月 8 日（星期一）中午 12:00

# 1）技术能力（6%）

### a）问题的相关性与重要性（1%）

### b）数据收集与预处理（2%）

### c）模型选择与模型表现（3%）

# 2）软技能评估（4%）

### a）展示质量（2%）

### b）创意与创新（1%）

### c）团队合作（1%）

### d）本作业将评估以下软技能：

#### • 沟通能力（CS1、CS2、CS3）

包括：

- 讨论流程
- 演示中的语言表达
- 流畅度
- 思路连贯性
- 团队合作表现
- 问答能力

#### • 批判性思维与问题解决能力（CT1、CT2、CT3）

包括：

- 创造性与批判性思维
- 创新能力
- 知识整合与连接
- 将想法转化为新形式 / 新方案

#### • 道德与职业伦理（EM1、EM2）

包括：

- 在线 / 现场展示结果时的职业伦理
- 道德规范

# 3）报告提交（4%）

## a）报告结构（2%）

报告应包含：

- 问题与目标（Problem and Objective）
- 方法与模型说明（Methodology and Model Explanation）
- 结果与可视化（Results and Visualization）
- 洞察与结果解释（Insights and Interpretation）
- 结论（Conclusion）

## b）写作质量与学术诚信（2%）

# 总分：14%

小组分组链接：[GROUP FORMATION.xlsx](https://365umedumy.sharepoint.com/:x:/s/WIA1006WID3006MachineLearning/IQCQ4ZbtuWOmRbV1CrFSY4iCAblh_CeA-8CHj96ZFMjiuO8?e=L53ofp)

请确保填写的所有信息准确无误，因为该表格将作为分配小组作业成绩的官方参考。