# Email Spam Classifier

## Project Overview

Email Spam Classifier is a machine learning project that classifies text messages as **Spam** or **Ham (Not Spam)**.

The project uses Natural Language Processing (NLP) techniques and a Naive Bayes machine learning algorithm to identify unwanted messages.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Matplotlib

## Dataset

The project uses the **UCI SMS Spam Collection** dataset.

- Total Messages: 5572
- Ham Messages: 4825
- Spam Messages: 747

## Machine Learning Algorithm

The project uses:

- TF-IDF Vectorization for text feature extraction
- Multinomial Naive Bayes for classification

## Model Accuracy

The trained model achieved approximately:

**97.04% Accuracy**

## How It Works

1. Load the SMS dataset.
2. Separate messages into Spam and Ham.
3. Split the dataset into training and testing data.
4. Convert text into numerical features using TF-IDF.
5. Train the Multinomial Naive Bayes model.
6. Evaluate the model using accuracy and classification metrics.
7. Enter a new message and predict whether it is Spam or Ham.

## Example

Input:

> Congratulations! You have won a free prize. Click now to claim.

Output:

**SPAM**

Input:

> Hello, are you coming to college tomorrow?

Output:

**HAM (Not Spam)**

## Project Structure

```text
Email_Spam_Classifier/
│
├── spam_classifier.py
├── SMSSpamCollection
├── requirements.txt
└── README.md