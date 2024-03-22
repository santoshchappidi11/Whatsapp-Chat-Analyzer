import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# Load your dataset
data = pd.read_csv('modified_train_df.csv')

X = data['message']
Y = data['sentiment']

# Drop rows with missing values
data.dropna(subset=['message'], inplace=True)

# Text Preprocessing
# def preprocess_text(text):
#     tokens = word_tokenize(text)  # Tokenize text
#     tokens = [word for word in tokens if word not in string.punctuation]  # Remove punctuation
#     tokens = [word.lower() for word in tokens]  # Lowercase
#     stop_words = set(stopwords.words('english'))
#     tokens = [word for word in tokens if word not in stop_words] # Remove stopwords
#     porter = PorterStemmer()
#     tokens = [porter.stem(word) for word in tokens]  # Stemming
#     preprocessed_text = ' '.join(tokens)  # Join tokens back into text
#     return preprocessed_text

# Apply text preprocessing to the 'message' column
# data['message'] = data['message'].apply(preprocess_text)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Apply text preprocessing to X_train and X_test
# X_train_preprocessed = X_train.apply(preprocess_text)
# X_test_preprocessed = X_test.apply(preprocess_text)
X_train_preprocessed = X_train
X_test_preprocessed = X_test


# Convert text data into numerical feature vectors using TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train_preprocessed)
X_test_tfidf = vectorizer.transform(X_test_preprocessed)

# Train the SVM model
model = SVC(kernel='linear')
model.fit(X_train_tfidf, y_train)

# Now the model is trained, you can save it using pickle
import pickle

# Save the trained model to pickle
with open('svm_model.pkl', 'wb') as f:
    pickle.dump((model, vectorizer), f)