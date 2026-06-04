# SwipeSense 6-Speaker Presentation Script

**Deck:** `SwipeSense_6_Speaker_Presentation.pptx`  
**Suggested total time:** about 5-6 minutes  
**Group members:** replace Speaker 1-6 with real names before presenting.

---

## Speaker 1: Opening and Problem Framing

**Slides:** 1-2  
**Target time:** 50-60 seconds

Good morning everyone. We are Group 4, and our project is called **SwipeSense**. Our topic is based on the assignment theme, *Tying the Data Knot: Love, Life and Likes*. We study whether dating app behavior can be used to predict user engagement and whether higher engagement is actually connected to better relationship outcomes.

---

Our main prediction target is `app_usage_time_label`, which separates users into seven engagement segments, from Barely to Extreme User. At the same time, we analyze `match_outcome`, including outcomes such as Mutual Match, Ghosted, Chat Ignored, Date Happened, and Relationship Formed.

The key idea of our project is simple: digital behavior can describe activity, but it should not automatically be treated as romantic success. So our two research questions are: first, can machine learning predict a user's engagement segment? Second, does higher engagement meaningfully relate to better match outcomes?

---

## Speaker 2: Dataset and Preprocessing

**Slides:** 3-4  
**Target time:** 50-60 seconds

For our dataset, we used the Dating App Behavior Dataset from Kaggle. The dataset is clean and large, but the target is heavily imbalanced.

The features include demographics, interests, app usage time, swipe behavior, likes received, mutual matches, profile details, messaging behavior, emoji usage, active hours, and match outcomes. Our target variable, `app_usage_time_label`, has seven classes. The largest class is Extreme User, which makes up 40.28% of the data. This imbalance is important because a model can get high accuracy by mostly predicting the largest class.

---

For preprocessing, we converted `interest_tags` into multi-hot encoded features. Categorical variables were one-hot encoded, and numeric variables were scaled using `StandardScaler`. We also used an 80/20 stratified train-test split so that the target class distribution stays similar in training and testing.

We tested two setups. The strict no-leakage setup removes `app_usage_time_min`, while the high-accuracy setup includes it. `match_outcome` is not used as a prediction feature because we reserve it for the relationship outcome analysis.

---

## Speaker 3: Models and Strict Results

**Slides:** 5-6  
**Target time:** 55-65 seconds

We trained five supervised classification models: Logistic Regression, Decision Tree, Random Forest, Support Vector Machine, and MLP or ANN. We used accuracy, macro F1, weighted F1, classification reports, and confusion matrices to evaluate them.

Macro F1 is especially important in our project because the target is imbalanced. Accuracy alone can be misleading. For example, the majority baseline reaches 40.28% accuracy by relying on the largest class, but its macro F1 is only 0.0820.

---

In the strict no-leakage setup, we removed `app_usage_time_min`. This means the models had to predict engagement using indirect signals such as demographics, swipes, likes, profile details, messages, emoji usage, and interests.

The best strict result was the tuned Random Forest, with 0.2202 accuracy and 0.1421 macro F1. This performance is modest. The important point is that without the direct usage-time field, the remaining features do not strongly predict engagement segment.

---

## Speaker 4: High-Accuracy Setup and Auto-sklearn

**Slides:** 7-8  
**Target time:** 55-65 seconds

Next, we tested the high-accuracy setup, where `app_usage_time_min` is included as a feature. The result changed dramatically. The Decision Tree achieved 1.0000 accuracy and 1.0000 macro F1. Random Forest also performed almost perfectly, with 0.9992 accuracy and 0.9978 macro F1.

This happens because `app_usage_time_label` is a binned version of `app_usage_time_min`. A tree model can learn the threshold boundaries directly. So this result is very strong technically, but it needs careful interpretation. It shows that usage time is the dominant feature defining engagement labels.

---

We also compared our models with auto-sklearn. In the strict setup, auto-sklearn achieved 0.2413 accuracy and 0.1408 macro F1, which is very close to our tuned Random Forest. This means AutoML could not significantly improve the strict result.

In the high-accuracy setup, auto-sklearn also reached 1.0000 accuracy and 1.0000 macro F1. This confirms that the performance gap is caused by the feature signal, not simply by our choice of model.

---

## Speaker 5: Match Outcome Analysis and Ethics

**Slides:** 9-10  
**Target time:** 55-65 seconds

After predicting engagement, we analyzed whether engagement level is associated with match outcomes. We used a stacked outcome composition view, a normalized heatmap, a chi-square test, and Cramer's V.

The result was very clear. Cramer's V was only 0.0102, and the p-value was 0.995044. This means the relationship between engagement segment and match outcome is extremely weak in practical terms. In other words, users who spend more time on the app do not necessarily get better romantic outcomes.

---

This is one of the most important insights in our project. Machine learning can predict app activity very well when direct usage time is available, but activity does not automatically mean dating success.

There is also an ethical point. If dating apps use models like this in real systems, they should avoid making unfair assumptions based on gender, orientation, income, education, or location. They should also avoid overclaiming what digital activity can reveal about personal relationships.

---

## Speaker 6: Final Conclusion and Q&A

**Slides:** 11-12  
**Target time:** 50-60 seconds

To conclude, our project has three main findings. First, engagement can be predicted almost perfectly when `app_usage_time_min` is included. The Decision Tree reached 1.0000 accuracy and macro F1.

Second, when we remove the direct usage-time field in the strict no-leakage setup, model performance becomes much weaker. The best manually tuned model reached 0.1421 macro F1, and auto-sklearn reached 0.1408 macro F1. This shows that indirect behavior and demographic features alone are weak predictors of engagement labels.

Third, higher engagement does not meaningfully translate into better match outcomes. The Cramer's V value was only 0.0102, so app activity intensity should not be interpreted as romantic success.

Overall, SwipeSense shows both the power and limitation of machine learning. ML can predict labels well when direct signals are available, but human relationship outcomes are more complex.

Thank you for listening.
