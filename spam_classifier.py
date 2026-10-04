import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# 1. Load dataset
data = pd.read_csv(
    "SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)

print("Dataset loaded successfully!")
print("Total messages:", len(data))


# 2. Convert labels
data["label"] = data["label"].map({"ham": 0, "spam": 1})


# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(
    data["message"],
    data["label"],
    test_size=0.2,
    random_state=42,
    stratify=data["label"]
)


# 4. Convert text into numbers using TF-IDF
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# 5. Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)


# 6. Test model
y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Ham", "Spam"]
))


# 7. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)


# 8. Spam vs Ham graph
counts = data["label"].value_counts()

plt.bar(["Ham", "Spam"], [counts.get(0, 0), counts.get(1, 0)])
plt.title("Spam vs Ham Messages")
plt.xlabel("Message Type")
plt.ylabel("Number of Messages")
plt.show()


# 9. Predict a new message
print("\n--- Email Spam Classifier ---")

new_message = input("Enter an email/message: ")

new_message_tfidf = vectorizer.transform([new_message])
prediction = model.predict(new_message_tfidf)[0]

if prediction == 1:
    print("Result: SPAM 🚨")
else:
    print("Result: HAM (Not Spam) ✅")