# Predicting Communication Drop-off Risk in Dating Apps Using Machine Learning

**Course:** WIA1006/WID3006 Machine Learning  
**Group members:** Replace this line with all group member names before submission.  
**Notebook:** `RESULT_dating_dropoff_risk.ipynb`  
**Dataset:** `data/dating_app_behavior_dataset.csv`

## Abstract

This project investigates whether user behavior data from a synthetic dating app dataset can be used to predict communication drop-off risk. The original dataset contains 50,000 records and 19 features describing user demographics, app usage patterns, swipe behavior, messaging behavior, and match outcomes. A binary target variable, `dropoff_risk`, was created from the original `match_outcome` column. Interactions labeled as `Ghosted` or `Chat Ignored` were treated as positive drop-off cases, while all other outcomes were treated as negative cases.

Five assignment-approved machine learning models were trained, optimized, and compared: Logistic Regression, Decision Tree, Random Forest, Support Vector Machine, and Multilayer Perceptron. A Dummy Classifier was used as a baseline, and auto-sklearn was also executed separately in a local WSL/Linux environment for comparison. The best manually selected model was Support Vector Machine with an F1 score of 0.2900. auto-sklearn achieved a higher F1 score of 0.3313, mainly because it produced very high recall. However, most models had balanced accuracy and ROC-AUC close to 0.50, showing that the available features had weak discriminative power for this binary task.

## 1. Problem and Objective

Digital dating platforms generate many behavioral signals, such as app usage time, swipe ratio, likes received, mutual matches, profile completeness, message count, and emoji usage. These signals may reflect how users interact with others before a relationship outcome is observed. This project focuses on the question:

**Can dating-app behavior data predict whether an interaction will end in communication drop-off?**

The objective is to build and compare machine learning models that predict whether a record belongs to the drop-off risk class. In this project, communication drop-off is defined as an interaction where the final `match_outcome` is either `Ghosted` or `Chat Ignored`.

The project aims to:

- Create a binary classification problem from the provided dataset.
- Perform exploratory data analysis to understand data patterns.
- Preprocess numerical, categorical, and multi-label interest tag features.
- Train and tune at least five machine learning models.
- Compare the models using appropriate classification metrics.
- Compare the manual models with auto-sklearn.
- Interpret the results and discuss limitations and ethical concerns.

## 2. Dataset and Target Definition

The dataset is a synthetic representation of user behavior on a fictional dating app. It contains:

- 50,000 records
- 19 original columns
- No missing values
- Numerical, categorical, and text-like multi-label features
- A labeled outcome column: `match_outcome`

The original target column, `match_outcome`, contains 10 outcome categories. For this project, it was converted into a binary target:

| Original `match_outcome` value | New target |
|---|---|
| `Ghosted` | `dropoff_risk = 1` |
| `Chat Ignored` | `dropoff_risk = 1` |
| All other outcomes | `dropoff_risk = 0` |

The final target distribution was:

| Class | Count | Rate |
|---|---:|---:|
| No drop-off | 40,022 | 80.04% |
| Drop-off | 9,978 | 19.96% |

This distribution shows that the dataset is imbalanced. Because only about one fifth of records are positive drop-off cases, accuracy alone is not a suitable evaluation metric.

## 3. Exploratory Data Analysis

The exploratory data analysis showed that the binary target was imbalanced but not extremely rare. The positive class rate was approximately 20%.

Numerical feature comparisons between drop-off and non-drop-off records showed only very small differences. For example:

| Feature | No drop-off mean | Drop-off mean | Difference |
|---|---:|---:|---:|
| `likes_received` | 99.4172 | 99.9625 | 0.5453 |
| `app_usage_time_min` | 150.0067 | 149.5342 | -0.4725 |
| `bio_length` | 250.1117 | 250.4257 | 0.3140 |
| `mutual_matches` | 13.8416 | 13.9854 | 0.1438 |
| `message_sent_count` | 50.0984 | 49.9656 | -0.1328 |

Categorical feature analysis also showed weak separation. Most category-level drop-off rates stayed close to the overall 20% baseline. For example, `app_usage_time_label` categories had drop-off rates from about 19.54% to 20.64%, while `location_type` categories had drop-off rates from about 19.57% to 20.21%.

These EDA findings suggest that the selected binary target is difficult to predict from the available features. The dataset appears to contain limited signal for separating drop-off and non-drop-off cases.

Recommended figures to include from the notebook:

- Figure 1: Binary target distribution
- Figure 2: Original match outcome distribution
- Figure 3: Numerical feature distributions by `dropoff_risk`
- Figure 4: Correlation heatmap
- Figure 5: Category-level drop-off rate tables

## 4. Methodology and Model Explanation

### 4.1 Data Preprocessing

The original `match_outcome` column was used only to create the binary target. After creating `dropoff_risk`, `match_outcome` was removed from the model feature matrix to prevent target leakage.

The preprocessing pipeline handled three feature groups:

1. **Numerical features**
   - Median imputation
   - Standard scaling

2. **Categorical features**
   - Most-frequent imputation
   - One-hot encoding

3. **Interest tags**
   - The `interest_tags` column contains comma-separated interests.
   - It was split into tags and encoded using `CountVectorizer`.

The data was split into training and testing sets using an 80:20 stratified split:

- Training set: 40,000 records
- Test set: 10,000 records
- Train positive rate: 19.96%
- Test positive rate: 19.96%

Stratification was used to preserve the same class distribution in both sets.

### 4.2 Models Used

The following five assignment-approved models were trained and tuned:

1. **Logistic Regression**
   - A linear classification model used as a simple and interpretable baseline.
   - `class_weight="balanced"` was used to account for class imbalance.

2. **Decision Tree**
   - A non-linear tree-based model that can capture interactions between features.
   - Hyperparameters such as tree depth and minimum leaf size were tuned.

3. **Random Forest**
   - An ensemble of decision trees designed to reduce overfitting and improve generalization.
   - The number of trees, maximum depth, and minimum leaf size were tuned.

4. **Support Vector Machine**
   - Implemented using `LinearSVC` for efficient training.
   - `class_weight="balanced"` was used.
   - The regularization parameter `C` was tuned.

5. **Multilayer Perceptron / Artificial Neural Network**
   - Implemented using `MLPClassifier`.
   - Hidden layer size and regularization strength were tuned.

A **Dummy Classifier** was also included as a baseline, but it was not counted as one of the five required models.

### 4.3 Hyperparameter Tuning

Hyperparameter tuning was performed using `GridSearchCV` with 3-fold stratified cross-validation. The scoring metric for tuning was positive-class F1 score, because the main objective was to identify drop-off risk cases.

The best hyperparameters found were:

| Model | Best hyperparameters |
|---|---|
| Logistic Regression | `C = 1.0` |
| Decision Tree | `max_depth = None`, `min_samples_leaf = 50` |
| Random Forest | `n_estimators = 100`, `max_depth = 10`, `min_samples_leaf = 10` |
| Support Vector Machine | `C = 0.1` |
| MLP / ANN | `hidden_layer_sizes = (64,)`, `alpha = 0.0001` |

## 5. Results and Visualization

### 5.1 Manual Model Comparison

The best manually selected model was Support Vector Machine. Its test F1 score was 0.2900.

| Model | Accuracy | Balanced Accuracy | Precision | Recall | F1 | ROC-AUC | Average Precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| Support Vector Machine | 0.5142 | 0.5077 | 0.2047 | 0.4970 | 0.2900 | 0.5065 | 0.2002 |
| Logistic Regression | 0.5141 | 0.5077 | 0.2047 | 0.4970 | 0.2899 | 0.5065 | 0.2002 |
| Decision Tree | 0.5240 | 0.4952 | 0.1963 | 0.4474 | 0.2728 | 0.4890 | 0.1939 |
| Random Forest | 0.6992 | 0.4966 | 0.1930 | 0.1593 | 0.1745 | 0.4879 | 0.1950 |
| MLP / ANN | 0.8004 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.5128 | 0.2040 |
| Dummy Baseline | 0.8004 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |

Although the Dummy Classifier achieved 80.04% accuracy, it had an F1 score of 0.0000 because it predicted only the majority class. This confirms that accuracy is misleading for this imbalanced task.

The SVM and Logistic Regression models produced almost identical results. Both detected about half of the true drop-off cases, but their precision was only around 20%. This means that many predicted drop-off cases were false positives.

The Random Forest achieved higher accuracy than SVM and Logistic Regression, but it had much lower recall for the positive class. This made it less suitable for the project objective. The MLP model predicted no positive cases at the default threshold, resulting in an F1 score of 0.0000.

Recommended figures to include:

- Figure 6: Model comparison bar chart
- Figure 7: Confusion matrix for the best manual model

### 5.2 Best Manual Model: Support Vector Machine

The classification report for the best manual model was:

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| No drop-off | 0.81 | 0.52 | 0.63 | 8,004 |
| Drop-off | 0.20 | 0.50 | 0.29 | 1,996 |

The model achieved:

- Accuracy: 0.51
- Macro average F1: 0.46
- Weighted average F1: 0.56

The most important interpretation is that the SVM model was able to capture about half of the true drop-off cases, but it also produced many false positives. Therefore, it should be interpreted as a weak screening model rather than a reliable prediction system.

### 5.3 auto-sklearn Comparison

auto-sklearn was run locally in a WSL/Linux Python 3.9 environment because the default Google Colab Python runtime could not install auto-sklearn successfully.

The auto-sklearn run completed with the following results:

| Metric | Value |
|---|---:|
| Accuracy | 0.2209 |
| Balanced accuracy | 0.5009 |
| Precision | 0.1999 |
| Recall | 0.9669 |
| F1 | 0.3313 |
| ROC-AUC | 0.4997 |
| Average precision | 0.2008 |

auto-sklearn achieved the highest F1 score among all methods. However, this improvement came mainly from very high recall. The recall of 0.9669 means that auto-sklearn detected almost all drop-off cases, but the precision of 0.1999 means that most records predicted as drop-off were false positives.

The ROC-AUC and balanced accuracy were both close to 0.50. This indicates that auto-sklearn did not learn a strong discriminative pattern, even though it achieved a higher F1 score by predicting the positive class very frequently.

The auto-sklearn sprint statistics were:

- Number of target algorithm runs: 282
- Successful runs: 156
- Crashed runs: 87
- Time-limit failures: 9
- Memory-limit failures: 30
- Best validation score: 0.333484

An OpenBLAS memory allocation warning occurred during the auto-sklearn run, and several candidate models exceeded memory limits. However, the final auto-sklearn run still completed and produced valid metrics.

### 5.4 Full Model Comparison Including auto-sklearn

| Model | F1 | Precision | Recall | Balanced Accuracy | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| auto-sklearn | 0.3313 | 0.1999 | 0.9669 | 0.5009 | 0.4997 |
| Support Vector Machine | 0.2900 | 0.2047 | 0.4970 | 0.5077 | 0.5065 |
| Logistic Regression | 0.2899 | 0.2047 | 0.4970 | 0.5077 | 0.5065 |
| Decision Tree | 0.2728 | 0.1963 | 0.4474 | 0.4952 | 0.4890 |
| Random Forest | 0.1745 | 0.1930 | 0.1593 | 0.4966 | 0.4879 |
| MLP / ANN | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.5128 |

If F1 score is used as the main metric, auto-sklearn is the best-performing method. If balanced accuracy or ROC-AUC is considered, none of the models performed much better than random classification.

## 6. Insights and Interpretation

### 6.1 Main Insight

The main finding is that the dataset features have limited predictive power for the selected binary target. Both manual models and auto-sklearn struggled to separate drop-off cases from non-drop-off cases.

This conclusion is supported by:

- Very small feature differences in EDA.
- Category-level drop-off rates close to the overall baseline.
- ROC-AUC values close to 0.50.
- Balanced accuracy values close to 0.50.
- Average precision values close to the positive class rate.

### 6.2 Interpretation of Best Models

The best manual model, SVM, produced a moderate recall of 0.4970 but low precision of 0.2047. This means it detected about half of the true drop-off cases, but most of its positive predictions were incorrect.

auto-sklearn produced a higher F1 score of 0.3313 by predicting the positive class much more aggressively. Its recall was very high at 0.9669, but its precision remained low at 0.1999. This suggests that auto-sklearn favored a strategy of over-detecting drop-off cases.

In practical terms, neither model is reliable enough for real decision-making. The models can be used for academic comparison and discussion, but their predictions should not be treated as strong evidence about individual users.

### 6.3 Feature Importance

For the best manual model, the top coefficient-based features included interest tags such as `MMA`, `History`, `Binge-Watching`, `Clubbing`, and app usage labels such as `Extreme User` and `Very Low`.

However, these importance values should be interpreted cautiously. Since the overall model performance is weak, the top features may reflect weak statistical patterns or noise rather than stable predictive relationships. They should not be interpreted as causal explanations.

## 7. Ethical Considerations and Limitations

### 7.1 Synthetic Data Limitation

The dataset is synthetic and generated programmatically. Therefore, the findings should not be generalized to real dating-app users. The analysis shows how machine learning can be applied to behavioral data, but it does not prove real-world social patterns.

### 7.2 Sensitive Attributes

The dataset includes sensitive or personal attributes such as gender, sexual orientation, income bracket, and education level. These features can introduce bias or unfair predictions if used irresponsibly.

An ablation test compared Logistic Regression performance with and without selected sensitive columns:

| Model variant | F1 | Precision | Recall | Balanced Accuracy | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| With sensitive columns | 0.2899 | 0.2047 | 0.4970 | 0.5077 | 0.5065 |
| Without selected sensitive columns | 0.2875 | 0.2026 | 0.4950 | 0.5046 | 0.5023 |

Removing sensitive columns produced only a very small decrease in F1 score. This suggests that the selected sensitive attributes were not essential for the overall predictive performance.

However, group-level checks showed that predicted drop-off rates varied noticeably across gender groups, even though actual drop-off rates were similar. For example, the actual drop-off rates across gender groups were around 18.8% to 20.7%, while predicted drop-off rates ranged from about 41.8% to 60.0%. This shows that even weak models can produce uneven predictions across groups.

### 7.3 Practical Limitations

The project has several limitations:

- The binary target may be too broad because `Ghosted` and `Chat Ignored` are related but not identical behaviors.
- The synthetic data may not contain strong relationships between features and the selected target.
- The models have low discriminative power.
- auto-sklearn experienced memory warnings and many failed candidate runs.
- The model should not be used for real user evaluation, ranking, moderation, or relationship prediction.

## 8. Conclusion

This project built a binary classification pipeline to predict communication drop-off risk in a synthetic dating-app behavior dataset. The target class was defined as records where `match_outcome` was either `Ghosted` or `Chat Ignored`.

Five assignment-approved models were trained and compared. The best manual model was Support Vector Machine, with an F1 score of 0.2900. auto-sklearn achieved a higher F1 score of 0.3313, but this was mainly caused by very high recall and low precision. Both auto-sklearn and the manual models had balanced accuracy and ROC-AUC close to 0.50.

The final conclusion is that the available features provide weak predictive signal for communication drop-off risk. The project still demonstrates a complete machine learning workflow, including target engineering, preprocessing, model selection, hyperparameter tuning, model comparison, auto-sklearn benchmarking, interpretation, and ethical evaluation. However, the results should be interpreted cautiously, and the models should not be used for real-world decision-making.

## References

1. Assignment guideline: `guideline.md`
2. Dataset source: https://www.kaggle.com/datasets/keyushnisar/dating-app-behavior-dataset
3. auto-sklearn documentation: https://automl.github.io/auto-sklearn/master/
4. scikit-learn documentation: https://scikit-learn.org/stable/

