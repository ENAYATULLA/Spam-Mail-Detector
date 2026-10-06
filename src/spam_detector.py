import os
import re
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. LOAD UCI SMS SPAM DATASET
# ============================================================

DATA_PATH = "data/SMSSpamCollection"

df = pd.read_csv(
    DATA_PATH,
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8"
)

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)

print("\nFirst five messages:")
print(df.head())


# ============================================================
# 2. CHECK DATASET
# ============================================================

print("\nDataset information:")
print(df.info())

print("\nClass distribution:")
print(df["label"].value_counts())


# ============================================================
# 3. REMOVE MISSING VALUES
# ============================================================

df = df.dropna()

print("\nDataset after removing missing values:")
print(df.shape)


# ============================================================
# 4. VISUALIZE SPAM VS HAM
# ============================================================

os.makedirs("results", exist_ok=True)

plt.figure(figsize=(6, 4))

sns.countplot(
    data=df,
    x="label"
)

plt.title("Spam vs Ham Messages")
plt.xlabel("Message Type")
plt.ylabel("Number of Messages")

plt.tight_layout()

plt.savefig(
    "results/class_distribution.png",
    dpi=300
)

plt.show()


# ============================================================
# 5. TEXT PREPROCESSING
# ============================================================

def clean_text(text):
    """
    Clean SMS text before feature extraction.
    """

    # Convert text to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+|https\S+",
        "",
        text
    )

    # Remove email addresses
    text = re.sub(
        r"\S+@\S+",
        "",
        text
    )

    # Keep only letters and spaces
    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


df["clean_message"] = df["message"].apply(clean_text)


print("\nText preprocessing example:")

print("\nOriginal message:")
print(df["message"].iloc[0])

print("\nCleaned message:")
print(df["clean_message"].iloc[0])


# ============================================================
# 6. CONVERT LABELS INTO NUMBERS
# ============================================================

df["label_numeric"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

df = df.dropna(
    subset=["label_numeric"]
)

df["label_numeric"] = df[
    "label_numeric"
].astype(int)


# ============================================================
# 7. TRAIN / TEST SPLIT
# ============================================================

X = df["clean_message"]
y = df["label_numeric"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 8. TF-IDF FEATURE EXTRACTION
# ============================================================

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X_train_tfidf = vectorizer.fit_transform(
    X_train
)

X_test_tfidf = vectorizer.transform(
    X_test
)

print("\nTF-IDF feature extraction completed!")

print(
    "Training feature matrix:",
    X_train_tfidf.shape
)

print(
    "Testing feature matrix:",
    X_test_tfidf.shape
)


# ============================================================
# 9. TRAIN NAIVE BAYES MODEL
# ============================================================

model = MultinomialNB()

model.fit(
    X_train_tfidf,
    y_train
)

print("\nNaive Bayes model training completed!")


# ============================================================
# 10. PREDICTIONS
# ============================================================

y_pred = model.predict(
    X_test_tfidf
)


# ============================================================
# 11. MODEL EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)


print("\n===================================")
print("       MODEL PERFORMANCE")
print("===================================")

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)


# ============================================================
# 12. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Ham", "Spam"]
    )
)


# ============================================================
# 13. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Ham", "Spam"],
    yticklabels=["Ham", "Spam"]
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.tight_layout()

plt.savefig(
    "results/confusion_matrix.png",
    dpi=300
)

plt.show()


# ============================================================
# 14. SAVE MODEL AND VECTORIZER
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    model,
    "models/spam_classifier.pkl"
)

joblib.dump(
    vectorizer,
    "models/tfidf_vectorizer.pkl"
)

print("\nModel saved:")
print("models/spam_classifier.pkl")

print("\nTF-IDF vectorizer saved:")
print("models/tfidf_vectorizer.pkl")


# ============================================================
# 15. CUSTOM MESSAGE PREDICTION
# ============================================================

def predict_spam(message):

    cleaned_message = clean_text(
        message
    )

    message_vector = vectorizer.transform(
        [cleaned_message]
    )

    prediction = model.predict(
        message_vector
    )[0]

    probabilities = model.predict_proba(
        message_vector
    )[0]

    if prediction == 1:

        result = "SPAM"
        confidence = probabilities[1]

    else:

        result = "HAM"
        confidence = probabilities[0]

    return result, confidence


# ============================================================
# 16. TEST CUSTOM MESSAGES
# ============================================================

test_messages = [

    "Congratulations! You have won a free cash prize. Claim now!",

    "Hey, are you free today? Let's meet in the evening.",

    "URGENT! You have won a lottery. Send your details to claim.",

    "Can you please send me the assignment?"

]


print("\n===================================")
print("       CUSTOM MESSAGE TESTING")
print("===================================")


for message in test_messages:

    result, confidence = predict_spam(
        message
    )

    print("\nMessage:")
    print(message)

    print(
        "Prediction:",
        result
    )

    print(
        f"Confidence: {confidence * 100:.2f}%"
    )


print("\n===================================")
print("       PROJECT COMPLETED")
print("===================================")