# 📧 Spam Detection using Machine Learning

A Machine Learning project that classifies text messages/emails as **Spam** or **Ham (Not Spam)** using Natural Language Processing (NLP) and supervised Machine Learning techniques.

The project compares multiple classification algorithms, performs hyperparameter tuning, evaluates the models using classification metrics, selects the best-performing model, and saves the trained model and TF-IDF vectorizer for later predictions.

---

## 📌 Project Overview

Spam messages are unwanted messages that may contain promotional content, suspicious links, misleading offers, or other unwanted information.

This project uses text classification techniques to automatically determine whether a given message is:

* 🚨 **Spam**
* ✅ **Ham**

The main text-processing technique used in the project is **TF-IDF (Term Frequency–Inverse Document Frequency)**, which converts text into numerical features that Machine Learning models can process.

---

## 🎯 Project Objectives

The main objectives of this project are:

* 📥 Load and analyze the email/message dataset
* 🧹 Prepare the text data for Machine Learning
* 🔤 Convert text into numerical features using TF-IDF
* ✂️ Split the dataset into training and testing data
* 🤖 Train multiple classification models
* ⚙️ Perform hyperparameter tuning using `RandomizedSearchCV`
* 📊 Compare model performance
* 🏆 Select the best-performing model
* 🔮 Predict whether new messages are Spam or Ham
* 💾 Save the trained model and TF-IDF vectorizer

---

## 🧰 Technologies Used

### Programming Language

* 🐍 Python

### Data Processing

* Pandas
* NumPy

### Natural Language Processing

* TF-IDF Vectorization

### Machine Learning

* Scikit-learn

### Models Evaluated

* Logistic Regression
* Linear SVM
* Naive Bayes
* K-Nearest Neighbors (KNN)

### Model Optimization

* RandomizedSearchCV

### Development Environment

* Jupyter Notebook
* VS Code

### Model Serialization

* Pickle

### Version Control

* Git
* GitHub

---

## 📂 Project Structure

```text
spam-detection-ml/
│
├── 📓 spam_detection(1).ipynb
├── 📊 email.csv
├── 🤖 spam_classifier.pkl
├── 🔤 tfidf_vectorizer.pkl
├── 📄 .gitignore
└── 📄 README.md
```

> The file names above correspond to the project files used for the Spam Detection project.

---

## 📊 Dataset

The project uses:

```text
email.csv
```
[The Data Set Source Link ](https://www.kaggle.com/datasets/ashfakyeafi/spam-email-classification)

The dataset contains text messages/emails with labels used for Spam/Ham classification.

The project uses the dataset available in the repository rather than making an unsupported claim about its original source.

### Dataset Source

The exact external Kaggle dataset URL is **not stated here because it has not been verified from the project files**.

This avoids incorrectly linking a different Kaggle dataset to this project.

---

## 🔄 Machine Learning Workflow

The project follows this workflow:

```text
📊 Dataset
    ↓
📥 Load Data
    ↓
🔍 Data Understanding
    ↓
🧹 Text Data Preparation
    ↓
✂️ Train/Test Split
    ↓
🔤 TF-IDF Vectorization
    ↓
🤖 Train Multiple Models
    ↓
⚙️ RandomizedSearchCV
    ↓
📊 Model Evaluation
    ↓
🏆 Best Model Selection
    ↓
🔮 Spam/Ham Prediction
    ↓
💾 Save Model + Vectorizer
```

---

# 🔍 Data Preparation

The dataset is loaded using Pandas and prepared for Machine Learning.

The text data is transformed into numerical representations before being supplied to the classification algorithms.

The target task is binary classification:

```text
Spam → Spam message
Ham  → Non-spam message
```

---

# 🔤 TF-IDF Vectorization

The project uses **TF-IDF (Term Frequency–Inverse Document Frequency)** to convert text into numerical feature vectors.

TF-IDF assigns importance to words based on their occurrence within documents and across the dataset.

The vectorizer is fitted on the training text and then used to transform the test text.

Conceptually:

```text
Text Message
     ↓
TF-IDF Vectorizer
     ↓
Numerical Feature Vector
     ↓
Machine Learning Model
```

The fitted vectorizer is saved as:

```text
tfidf_vectorizer.pkl
```

---

# ✂️ Train/Test Split

The dataset is divided into training and testing portions.

The training data is used to train the Machine Learning models, while the test data is used to evaluate their performance on unseen messages.

The transformed test data is represented as:

```python
X_test_tfidf
```

---

# 🤖 Machine Learning Models

Multiple classification algorithms were evaluated in the project.

## 1️⃣ Logistic Regression

Logistic Regression was used as one of the classification algorithms for Spam/Ham prediction.

The model was tuned using `RandomizedSearchCV`.

The recorded F1-score from the project evaluation was approximately:

```text
0.9180
```

---

## 2️⃣ Linear SVM

A Linear Support Vector Machine was also evaluated for text classification.

Linear SVM is suitable for high-dimensional text feature spaces such as TF-IDF representations.

The recorded F1-score was approximately:

```text
0.9174
```

---

## 3️⃣ Naive Bayes

Naive Bayes was evaluated as another classification approach.

Naive Bayes is commonly used for text classification because it can work effectively with sparse text-based feature representations.

The recorded F1-score was approximately:

```text
0.9053
```

---

## 4️⃣ K-Nearest Neighbors

K-Nearest Neighbors (KNN) was also included in the model comparison.

The recorded F1-score was approximately:

```text
0.5287
```

The result was substantially lower than the other evaluated models in this project.

---

# ⚙️ Hyperparameter Tuning

`RandomizedSearchCV` was used to search through model hyperparameters.

The purpose of hyperparameter tuning was to identify better-performing parameter combinations rather than relying only on default model settings.

General workflow:

```text
Model
  ↓
Parameter Search
  ↓
RandomizedSearchCV
  ↓
Best Estimator
  ↓
Test Evaluation
```

---

# 📊 Model Evaluation

The project compared the trained models using evaluation results.

The recorded F1-scores were:

| Model                  | F1-Score |
| ---------------------- | -------: |
| 🥇 Logistic Regression |  ~0.9180 |
| 🥈 Linear SVM          |  ~0.9174 |
| 🥉 Naive Bayes         |  ~0.9053 |
| KNN                    |  ~0.5287 |

Based on the recorded F1-score comparison, **Logistic Regression** was the best-performing model among the evaluated models.

> These values are the recorded results from the project run. Model results can change if the notebook is rerun with different data splits, preprocessing, library versions, or tuning results.

---

# 🏆 Best Model Selection

The project stores the tuned models in a dictionary:

```python
best_models
```

The best model name is obtained from the model comparison results:

```python
best_model_name = results_df.iloc[0]["Model"]
```

The corresponding tuned model is then selected:

```python
best_model = best_models[best_model_name]
```

This ensures that the model selected for final prediction is actually the model identified as the best-performing model in the evaluation results.

For the recorded results, the selected model was:

```text
Logistic Regression
```

---

# 🔮 Spam/Ham Prediction

After selecting the best model, new messages can be transformed using the fitted TF-IDF vectorizer and passed to the trained classifier.

Example messages tested in the project include:

```text
"Congratulations! You won a free iPhone. Click here now."
```

and:

```text
"Hey, are we still meeting tomorrow at 10 AM"
```

The classifier predicts the corresponding class:

```text
Spam
```

or:

```text
Ham
```

---

# 🐛 Model Selection Issue Fixed

During development, the variable `best_model` was being overwritten inside the model-training loop.

The project initially contained logic similar to:

```python
best_model = search.best_estimator_
best_models[name] = best_model
```

Because this happened repeatedly for every model, `best_model` could contain the estimator from the last iteration instead of the model that actually achieved the best evaluation result.

The issue was corrected by explicitly selecting the best model from `best_models`:

```python
best_model_name = results_df.iloc[0]["Model"]
best_model = best_models[best_model_name]
```

This made the final prediction model consistent with the model-ranking results.

---

# 💾 Saving the Trained Model

The selected classifier is saved using Python's `pickle` module.

```python
import pickle

with open("spam_classifier.pkl", "wb") as file:
    pickle.dump(best_model, file)
```

The saved classifier is:

```text
spam_classifier.pkl
```

---

# 💾 Saving the TF-IDF Vectorizer

The fitted TF-IDF vectorizer is also saved:

```python
with open("tfidf_vectorizer.pkl", "wb") as file:
    pickle.dump(tfidf, file)
```

The two saved files work together:

```text
spam_classifier.pkl
        +
tfidf_vectorizer.pkl
        ↓
New Message
        ↓
TF-IDF Transformation
        ↓
Spam Classifier
        ↓
Spam / Ham
```

---

# 📁 Project Files

## `spam_detection(1).ipynb`

The main Jupyter Notebook containing the data processing, TF-IDF transformation, model training, hyperparameter tuning, evaluation, model selection, and prediction workflow.

## `email.csv`

The dataset used for the Spam/Ham classification project.

## `spam_classifier.pkl`

The serialized trained classification model selected for final prediction.

## `tfidf_vectorizer.pkl`

The serialized TF-IDF vectorizer used to transform text into numerical features.

## `.gitignore`

Used to prevent unnecessary files such as Python cache files, virtual environments, and local configuration files from being committed.

---

# 🧠 Key Concepts Demonstrated

This project demonstrates practical knowledge of:

### 🐍 Python

* Python programming
* Pandas
* NumPy
* File handling

### 📝 Natural Language Processing

* Text classification
* TF-IDF
* Sparse text representations

### 🤖 Machine Learning

* Supervised learning
* Binary classification
* Logistic Regression
* Linear SVM
* Naive Bayes
* KNN
* Hyperparameter tuning

### ⚙️ Model Optimization

* RandomizedSearchCV
* Model comparison
* Best estimator selection

### 📊 Model Evaluation

* F1-score
* Comparing classification models
* Testing predictions on new text

### 💾 Model Deployment Preparation

* Model serialization
* Pickle
* Saving the TF-IDF vectorizer

---

# 🚀 How to Run the Project

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/Rajeshkumar0707/spam-detection-ml.git
```

## 2️⃣ Open the Project

```bash
cd spam-detection-ml
```

## 3️⃣ Install Required Libraries

```bash
pip install pandas numpy scikit-learn jupyter
```

## 4️⃣ Start Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
spam_detection(1).ipynb
```

## 5️⃣ Run the Notebook

Run the notebook cells in order to:

```text
Load Dataset
    ↓
Prepare Text Data
    ↓
Create TF-IDF Features
    ↓
Train Models
    ↓
Tune Hyperparameters
    ↓
Evaluate Models
    ↓
Select Best Model
    ↓
Test Predictions
    ↓
Save Model
```

---

# ⚠️ Important Notes

* The project is a Machine Learning classification experiment based on the supplied dataset.
* The reported F1-scores are from the recorded evaluation run.
* Model performance may vary when the notebook is rerun.
* The project should not be interpreted as a production-grade email security system.
* A Machine Learning prediction is not a guarantee that a message is actually malicious or safe.
* The exact original external dataset source is intentionally not claimed without verification.

---

# 🔮 Future Improvements

Possible improvements include:

* 🔤 More advanced NLP preprocessing
* 🧠 Testing additional text-classification algorithms
* ⚙️ More systematic hyperparameter optimization
* 📊 Add confusion matrix visualization
* 📈 Evaluate Precision, Recall, and ROC-AUC
* 🔄 Use reproducible train/test splitting
* 🧪 Perform cross-validation
* 💻 Build a simple web interface
* 🚀 Deploy the model as an API or web application
* 📦 Create a complete inference pipeline combining the vectorizer and classifier

---

# 📌 Project Highlights

* 📧 Built a Spam/Ham text classification system
* 🔤 Used TF-IDF for text feature extraction
* 🤖 Compared four Machine Learning algorithms
* ⚙️ Used `RandomizedSearchCV` for hyperparameter tuning
* 📊 Compared models using F1-score
* 🏆 Logistic Regression achieved the highest recorded F1-score
* 🔮 Tested the trained model on new messages
* 🐛 Fixed the final-model selection logic
* 💾 Saved the trained classifier and TF-IDF vectorizer
* 🐍 Developed using Python and Jupyter Notebook

---

# 🔗 GitHub Repository

[Spam Detection ML – GitHub Repository](https://github.com/Rajeshkumar0707/spam-detection-ml?utm_source=chatgpt.com)

---

# 👨‍💻 Author

## Rajesh Kumar

Computer Science Engineering Graduate | Aspiring Data Scientist | Machine Learning Enthusiast

### GitHub

[Rajesh Kumar – GitHub](https://github.com/Rajeshkumar0707?utm_source=chatgpt.com)

---

## ⭐ Project Summary

This project demonstrates an end-to-end Machine Learning workflow for classifying text messages as **Spam** or **Ham**.

The project covers text feature extraction with **TF-IDF**, training and comparison of multiple classification algorithms, hyperparameter tuning using **RandomizedSearchCV**, model evaluation using **F1-score**, best-model selection, prediction of new messages, and serialization of the trained model and vectorizer.

```text
📧 Text Message
      ↓
🔤 TF-IDF
      ↓
🤖 Machine Learning Model
      ↓
📊 Model Evaluation
      ↓
🏆 Best Model
      ↓
🔮 Spam / Ham Prediction
      ↓
💾 Saved Model
```

**Built with Python 🐍 | Machine Learning 🤖 | NLP 🔤 | Scikit-learn 📊**
