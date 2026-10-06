# Spam Mail Detector

A machine learning-based spam message detection system built using **Python, Natural Language Processing (NLP), TF-IDF, and Multinomial Naive Bayes**.

The project classifies SMS messages as either **Spam** or **Ham (Not Spam)**.

---

## 1. Project Overview

Spam messages are unwanted messages that may contain advertisements, scams, fraudulent offers, or other unwanted content.

This project uses machine learning and natural language processing techniques to automatically classify messages as:

* **Ham** – Normal or legitimate messages
* **Spam** – Unwanted or potentially fraudulent messages

The model was trained using the **SMS Spam Collection dataset**.

---

## 2. Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Natural Language Processing (NLP)
* TF-IDF
* Multinomial Naive Bayes
* Joblib

---

## 3. Dataset

The project uses the **SMS Spam Collection Dataset**, which contains labeled SMS messages classified as either `ham` or `spam`.

Each message contains:

* Message label
* Message text

Example:

```text
ham    Go until jurong point, crazy.. Available only in bugis n great world la e buffet...
spam   Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005.
```

The dataset was divided into training and testing data.

The test dataset contains **1,115 messages**:

* Ham: 966
* Spam: 149

---

## 4. Project Workflow

The overall machine learning pipeline is:

```text
SMS Dataset
     ↓
Text Preprocessing
     ↓
TF-IDF Feature Extraction
     ↓
Train/Test Split
     ↓
Multinomial Naive Bayes
     ↓
Model Evaluation
     ↓
Custom Message Prediction
```

---

## 5. Text Preprocessing

Before training the machine learning model, the messages were cleaned.

The preprocessing steps include:

1. Converting text to lowercase
2. Removing URLs
3. Removing email addresses
4. Removing unnecessary special characters
5. Removing extra spaces

### Example

#### Original Message

```text
Go until jurong point, crazy.. Available only in bugis n great world la e buffet...
```

#### Cleaned Message

```text
go until jurong point crazy available only in bugis n great world la e buffet
```

---

## 6. TF-IDF Feature Extraction

Machine learning models cannot directly understand raw text.

Therefore, **TF-IDF (Term Frequency-Inverse Document Frequency)** was used to convert the text messages into numerical feature vectors.

The vectorizer was configured with a maximum of **5,000 features**.

```python
TfidfVectorizer(
    stop_words="english",
    max_features=5000
)
```

---

## 7. Machine Learning Model

A **Multinomial Naive Bayes** classifier was used for spam detection.

Naive Bayes is a commonly used machine learning algorithm for text classification because it works well with high-dimensional text features such as TF-IDF vectors.

---

## 8. Model Evaluation

The model was evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

### Final Results

| Metric    |       Score |
| --------- | ----------: |
| Accuracy  |  **96.86%** |
| Precision | **100.00%** |
| Recall    |  **76.51%** |
| F1 Score  |  **86.69%** |

### Classification Report

```text
              precision    recall  f1-score   support

         Ham       0.97      1.00      0.98       966
        Spam       1.00      0.77      0.87       149

    accuracy                           0.97      1115
   macro avg       0.98      0.88      0.92      1115
weighted avg       0.97      0.97      0.97      1115
```

---

## 9. Confusion Matrix

The confusion matrix obtained from the test dataset was:

```text
[[966   0]
 [ 35 114]]
```

This means:

* 966 Ham messages were correctly classified as Ham.
* 114 Spam messages were correctly classified as Spam.
* 35 Spam messages were incorrectly classified as Ham.
* 0 Ham messages were incorrectly classified as Spam.

The confusion matrix visualization is available in:

```text
results/confusion_matrix.png
```

---

## 10. Custom Message Testing

The trained model was also tested with new messages that were not part of the training dataset.

### Example 1

```text
Congratulations! You have won a free cash prize. Claim now!
```

Prediction:

```text
SPAM
Confidence: 97.99%
```

### Example 2

```text
Hey, are you free today? Let's meet in the evening.
```

Prediction:

```text
HAM
Confidence: 99.19%
```

### Example 3

```text
URGENT! You have won a lottery. Send your details to claim.
```

Prediction:

```text
SPAM
Confidence: 79.97%
```

### Example 4

```text
Can you please send me the assignment?
```

Prediction:

```text
HAM
Confidence: 89.10%
```

---

## 11. Project Structure

```text
Spam-Mail-Detector/
│
├── data/
│   └── SMSSpamCollection
│
├── models/
│   ├── spam_classifier.pkl
│   └── tfidf_vectorizer.pkl
│
├── results/
│   ├── class_distribution.png
│   └── confusion_matrix.png
│
├── src/
│   └── spam_detector.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

> The `venv/` virtual environment is used locally and is excluded from Git using `.gitignore`.

---

## 12. How to Run the Project

### Step 1: Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

### Step 2: Open the project

```bash
cd Spam-Mail-Detector
```

### Step 3: Create a virtual environment

```bash
python -m venv venv
```

### Step 4: Activate the virtual environment

#### Windows PowerShell

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 5: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Run the program

```bash
python src\spam_detector.py
```

---

## 13. Output

The project generates:

* Spam vs Ham class distribution graph
* Confusion matrix
* Model performance metrics
* Trained machine learning model
* TF-IDF vectorizer
* Predictions for custom messages

Generated files include:

```text
results/class_distribution.png
results/confusion_matrix.png
models/spam_classifier.pkl
models/tfidf_vectorizer.pkl
```

---

## 14. Limitations

The model has some limitations:

* It is trained on SMS messages and may not perform equally well on all types of emails.
* New spam patterns may not always be detected correctly.
* Some spam messages may be classified as ham.
* The dataset is relatively small compared with large-scale real-world email datasets.
* The model may perform differently on messages with vocabulary or patterns that are very different from the training data.

---

## 15. Future Improvements

Possible improvements include:

* Using larger and more diverse datasets
* Comparing multiple machine learning algorithms
* Adding advanced NLP techniques
* Using word and character n-grams
* Handling multilingual spam messages
* Developing a web interface for real-time spam detection
* Periodically retraining the model with new spam examples

---

## 16. Conclusion

This project demonstrates a basic Natural Language Processing and machine learning pipeline for spam message detection.

The combination of text preprocessing, TF-IDF feature extraction, and Multinomial Naive Bayes achieved an accuracy of **96.86%** on the test dataset.

The project provided practical experience in:

* Text preprocessing
* Feature extraction
* Machine learning classification
* Model evaluation
* Confusion matrix analysis
* Working with real-world text data

Overall, the project demonstrates how traditional machine learning techniques can be used to build a simple and effective text classification system for spam detection.

---

## 17. Requirements

The main Python libraries used in this project are:

```text
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
```

All dependencies can be installed using:

```bash
pip install -r requirements.txt
```

---

## 18. Git Ignore

The virtual environment and temporary Python files should not be uploaded to GitHub.

The `.gitignore` file contains:

```text
venv/
__pycache__/
*.pyc
.vscode/
.idea/
```

---

## 19. Author

**Enayat Ullah**

Computer Science Graduate
B.Tech in Computer Science and Engineering

GitHub: [ENAYATULLA](https://github.com/ENAYATULLA)
#   S p a m - M a i l - D e t e c t o r  
 