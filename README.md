# Dating App Drop-off Risk Prediction

This project is a Google Colab-based machine learning assignment for predicting communication drop-off risk in a synthetic dating-app behavior dataset.

The notebook builds a binary classification task from the original `match_outcome` column:

- `dropoff_risk = 1`: `match_outcome` is `Ghosted` or `Chat Ignored`
- `dropoff_risk = 0`: all other outcomes

Main notebook:

- `dating_dropoff_risk.ipynb`

Dataset:

- `data/dating_app_behavior_dataset.csv`

## Models Used

The notebook trains and compares five assignment-approved machine learning models:

1. **Logistic Regression**
2. **Decision Tree**
3. **Random Forest**
4. **Support Vector Machine**
   - Implemented with `LinearSVC` for faster training in Google Colab.
5. **Artificial Neural Network / Multilayer Perceptron**
   - Implemented with `MLPClassifier`.

The notebook also includes:

- **DummyClassifier baseline**
  - Used only as a baseline.
  - It is not counted as one of the five required models.
- **Optional auto-sklearn section**
  - Disabled by default with `RUN_AUTOSKLEARN = False`.
  - The notebook documents installation/environment limitations if auto-sklearn cannot run.

## Evaluation Metrics

The main model-selection metric is positive-class **F1 score**, because the project focuses on detecting drop-off cases.

The notebook also reports:

- Accuracy
- Balanced accuracy
- Precision
- Recall
- ROC-AUC
- Average precision / PR-AUC
- Classification report
- Confusion matrix

## Preprocessing

The notebook uses sklearn pipelines:

- Numeric features:
  - Median imputation
  - Standard scaling
- Categorical features:
  - Most-frequent imputation
  - One-hot encoding
- `interest_tags`:
  - Split by comma
  - Encoded as multi-label tag indicators using `CountVectorizer`

The original `match_outcome` column is removed from the training features after creating `dropoff_risk`.

## How to Run in Google Colab

1. Open Google Colab: <https://colab.research.google.com/>
2. Upload `dating_dropoff_risk.ipynb`.
3. Upload the dataset file:
   - `dating_app_behavior_dataset.csv`
4. Make sure the uploaded CSV is available at:
   - `/content/dating_app_behavior_dataset.csv`
5. In Colab, click:
   - `Runtime` -> `Run all`

The notebook also has a local fallback path:

```text
data/dating_app_behavior_dataset.csv
```

This fallback is only for running or checking the notebook from this repository.

## Optional auto-sklearn Attempt

The auto-sklearn section is optional and disabled by default:

```python
RUN_AUTOSKLEARN = False
```

To attempt it, change the value to:

```python
RUN_AUTOSKLEARN = True
```

If auto-sklearn installation fails in Colab, keep the failure output and explain it in the report. The notebook cites the official auto-sklearn installation documentation:

<https://automl.github.io/auto-sklearn/master/installation.html>

## Expected Outputs

After running the notebook, it should produce:

- Dataset validation summary
- Target distribution
- Exploratory data analysis charts
- Tuned model comparison table
- Model comparison chart
- Best model confusion matrix
- Feature importance chart
- Sensitive-feature ablation results
- Report notes
- Slide outline

## Submission Checklist

Before submission:

- Replace the group member placeholder in the notebook with real names.
- Run the notebook from top to bottom in a fresh Colab runtime.
- Confirm the dataset loads as 50,000 rows and 19 original columns.
- Confirm `dropoff_risk` is created correctly.
- Confirm `match_outcome` is not used as a model feature.
- Confirm all five required models produce evaluation metrics.
- Export the completed notebook as `.ipynb`.
- Export or print the notebook as `.pdf`.
- Use the notebook's report notes and slide outline to prepare the final report and presentation.

