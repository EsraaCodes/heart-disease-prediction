# 🫀 Heart Disease Prediction – Machine Learning Project

## 📌 Project Overview

This project focuses on building a **machine learning classification model** to predict the presence of heart disease based on patient medical attributes. The goal is to create a reliable, well-evaluated model that can generalize to unseen data and demonstrate a full end-to-end ML workflow.

---

## 🎯 Objective

* Predict whether a patient has heart disease (binary classification)
* Evaluate model performance using **robust and reliable metrics**
* Avoid overfitting and ensure good generalization
* Save and reload the trained model for real-world usage

---

## 🧠 Machine Learning Workflow

1. Problem definition
2. Data loading and exploration
3. Data preprocessing
4. Train–test split
5. Model selection (Random Forest)
6. Model training
7. Model evaluation using multiple metrics
8. Cross-validation
9. ROC curve and AUC analysis
10. Model persistence (saving & loading)

---

## 📊 Model Used

**RandomForestClassifier**

Why Random Forest?

* Handles non-linear relationships
* Reduces overfitting by averaging multiple decision trees
* Performs well on structured/tabular medical data

---

## 📈 Evaluation Metrics

Multiple evaluation metrics were used to ensure a fair and complete assessment:

* **Accuracy**: Overall correctness of predictions
* **Precision**: How many predicted positive cases are actually positive
* **Recall (Sensitivity)**: Ability to detect actual heart disease cases (very important in medical problems)
* **F1-score**: Balance between precision and recall
* **ROC–AUC**: Measures how well the model separates classes independent of threshold

---

## 🔁 Cross-Validation

To obtain a reliable estimate of model performance, **5-fold cross-validation** was applied.

Cross-validation benefits:

* Reduces dependency on a single train/test split
* Provides a more realistic estimate of generalization performance
* Helps detect overfitting

The final reported scores are the **mean values across all folds**.

---

## 📉 ROC Curve & AUC

The ROC curve visualizes the trade-off between:

* True Positive Rate (Recall)
* False Positive Rate

The **Area Under the Curve (AUC)** summarizes this performance into a single value.
A higher AUC indicates better class separation.

---

## 💾 Model Persistence (Saving & Loading)

The trained model was saved using `joblib` and later reloaded.

Both the original model and the loaded model achieved the **same accuracy score**, confirming:

* Correct model serialization
* No loss of learned parameters
* Reproducibility of results

This step enables deployment in real-world applications such as web services or APIs.

---

## ✅ Key Results

* Model achieved stable performance across cross-validation folds
* Test accuracy of approximately **83%**
* Consistent results before and after loading the saved model

---

## 🧾 Conclusion

This project demonstrates a complete machine learning pipeline, from data preparation to model deployment. By using cross-validation, multiple evaluation metrics, and model persistence, the final model is both reliable and production-ready.

---

## 🚀 Future Improvements

* Hyperparameter tuning using GridSearchCV or RandomizedSearchCV
* Feature importance analysis
* Trying additional models (Logistic Regression, XGBoost)
* Deployment as a web or API service

---

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Joblib

---

## 📎 Notes

This project emphasizes **understanding, evaluation, and reliability** over simply achieving high accuracy.
