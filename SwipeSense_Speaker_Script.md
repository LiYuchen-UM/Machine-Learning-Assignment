# SwipeSense 6-Speaker Presentation Script

**Deck:** `SwipeSense_6_Speaker_Presentation.pptx`  
**Group members:** replace Speaker 1-6 with real names before presenting.

---

## Speaker 1: Opening and Problem Framing

**Slides:** 1-2  
**Target time:** 50-60 seconds

Good morning everyone. We are Group 4, and our project is called **SwipeSense**. Our topic is based on the assignment theme, *Tying the Data Knot: Love, Life and Likes*. We study whether dating app behavior can be used to predict user engagement and whether higher engagement is actually connected to better relationship outcomes.

---

Our prediction target is `app_usage_time_label`, which groups users into seven engagement levels, from Barely to Extreme User. We also analyze `match_outcome`, including Ghosted, Date Happened, and Relationship Formed.

Our key idea is simple: app activity is not the same as romantic success. So we ask two questions. First, can machine learning predict user engagement? Second, does higher engagement meaningfully relate to better match outcomes?

---

## Speaker 2: Dataset and Preprocessing

**Slides:** 3-4  
**Target time:** 50-60 seconds

We used the Dating App Behavior Dataset from Kaggle. It is large and clean, but the target is imbalanced.

The dataset includes demographics, interests, usage time, swipes, likes, mutual matches, profile details, messages, emoji usage, active hours, and match outcomes. Our target, `app_usage_time_label`, has seven classes. The largest class is Extreme User, at 40.28% of the data. This matters because a model can appear accurate by mostly predicting the majority class.

---

For preprocessing, we multi-hot encoded `interest_tags`, one-hot encoded categorical variables, and scaled numeric variables with `StandardScaler`. We used an 80/20 stratified train-test split.

We tested two setups. The strict no-leakage setup removes `app_usage_time_min`. The high-accuracy setup includes it. `match_outcome` is never used as a prediction feature because it is reserved for relationship analysis.

---

## Speaker 3: Models and Strict Results

**Slides:** 5-6  
**Target time:** 55-65 seconds

We trained five supervised classification models: Logistic Regression, Decision Tree, Random Forest, Support Vector Machine, and MLP or ANN. We evaluated them using accuracy, macro F1, weighted F1, classification reports, and confusion matrices.

Macro F1 is important because the target is imbalanced. Accuracy alone can be misleading. For example, the majority baseline reaches 40.28% accuracy, but its macro F1 is only 0.0820.

---

In the strict setup, we removed `app_usage_time_min`. The models had to use indirect signals such as demographics, swipes, likes, profiles, messages, emojis, and interests.

The best strict result was the tuned Random Forest, with 0.2202 accuracy and 0.1421 macro F1. This is modest, but meaningful. It shows that without direct usage time, the remaining features do not strongly predict engagement segment.

---

## Speaker 4: High-Accuracy Setup and Auto-sklearn

**Slides:** 7-8  
**Target time:** 55-65 seconds

Next, we tested the high-accuracy setup, where `app_usage_time_min` is included. The result changed dramatically. Decision Tree achieved 1.0000 accuracy and 1.0000 macro F1. Random Forest was also almost perfect, with 0.9992 accuracy and 0.9978 macro F1.

This happens because `app_usage_time_label` is created from `app_usage_time_min`. Tree models can learn those thresholds directly. So the result is technically strong, but it mainly shows target leakage.

---

We also compared our work with auto-sklearn. In the strict setup, it achieved 0.2413 accuracy and 0.1408 macro F1, close to our tuned Random Forest. AutoML did not significantly improve the strict result.

In the high-accuracy setup, auto-sklearn also reached 1.0000 accuracy and macro F1. This confirms that the performance gap comes from the feature signal, not just the model choice.

---

## Speaker 5: Match Outcome Analysis and Ethics

**Slides:** 9-10  
**Target time:** 55-65 seconds

After predicting engagement, we analyzed whether engagement level is associated with match outcomes. We used a stacked bar chart, normalized heatmap, chi-square test, and Cramer's V.

The result was clear. Cramer's V was only 0.0102, and the p-value was 0.995044. This means the relationship between engagement segment and match outcome is extremely weak. Users who spend more time on the app do not necessarily get better romantic outcomes.

---

This is one of our key insights. Machine learning can predict app activity very well when direct usage time is available, but activity does not automatically mean dating success.

There is also an ethical concern. Dating apps should avoid unfair assumptions based on gender, orientation, income, education, or location. They should also avoid overclaiming what digital behavior can reveal about personal relationships.

---

## Speaker 6: Final Conclusion and Q&A

**Slides:** 11-12  
**Target time:** 50-60 seconds

To conclude, our project has three findings. First, engagement can be predicted almost perfectly when `app_usage_time_min` is included. The Decision Tree reached 1.0000 accuracy and macro F1.

Second, when that direct field is removed, performance becomes much weaker. The best tuned model reached 0.1421 macro F1, and auto-sklearn reached 0.1408 macro F1.

Third, higher engagement does not meaningfully translate into better match outcomes. Cramer's V was only 0.0102.

Overall, SwipeSense shows both the power and limitation of machine learning. ML can predict labels when direct signals are available, but relationship outcomes are more complex.

Thank you for listening.
