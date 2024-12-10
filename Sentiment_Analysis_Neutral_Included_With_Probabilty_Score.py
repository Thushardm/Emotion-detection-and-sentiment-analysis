import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def load_text_files_with_unsup(base_directory):
    """
    Load text files from the 'neg' and 'pos' folders in the specified directory.
    - 'neg': Contains negative sentiment files, labeled as 0.
    - 'pos': Contains positive sentiment files, labeled as 1.
    Returns:
        texts: List of text content from files.
        labels: Corresponding labels for supervised learning.
    """
    texts = []  # List to store text content
    labels = []  # List to store labels for each text

    # Iterate through subfolders ('neg' and 'pos')
    for subfolder in os.listdir(base_directory):
        subfolder_path = os.path.join(base_directory, subfolder)
        if os.path.isdir(subfolder_path):  # Ensure it's a directory
            for filename in os.listdir(subfolder_path):  # Iterate through files
                if filename.endswith(".txt"):  # Process only .txt files
                    with open(os.path.join(subfolder_path, filename), 'r', encoding='utf-8') as f:
                        text = f.read()  # Read the file content

                        # Assign labels based on folder name
                        if subfolder.lower() == "neg":
                            texts.append(text)
                            labels.append(0)  # Negative sentiment
                        elif subfolder.lower() == "pos":
                            texts.append(text)
                            labels.append(1)  # Positive sentiment
    return texts, labels

# Load text data and labels
base_directory = r"E:\Python\Sentiment-Analysis-NN\Sentiment\train"
texts, labels = load_text_files_with_unsup(base_directory)

# Initialize TF-IDF Vectorizer for feature extraction
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
# Transform texts into a matrix of TF-IDF features
X = tfidf_vectorizer.fit_transform(texts)

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Train a Logistic Regression classifier
classifier = LogisticRegression(max_iter=1000)  # Setting max iterations to avoid convergence issues
classifier.fit(X_train, y_train)  # Fit the model on the training data

# Calculate predicted probabilities for the test set
y_proba = classifier.predict_proba(X_test)  # Get probabilities for both classes (negative and positive)

# Extract the probability of the positive sentiment (class 1)
scores = y_proba[:, 1]  # Positive sentiment probabilities

# Save the model and vectorizer for deployment
joblib.dump(classifier, "model.pkl")
joblib.dump(tfidf_vectorizer, "vectorizer.pkl")

# Interpret scores and assign sentiment categories
for idx, score in enumerate(scores):
    if score < 0.4:  # Below 0.4 is considered negative
        sentiment = "Negative"
    elif 0.4 <= score <= 0.6:  # Between 0.4 and 0.6 is neutral
        sentiment = "Neutral"
    else:  # Above 0.6 is positive
        sentiment = "Positive"
    print(f"Text {idx + 1}: Score = {score:.2f}, Sentiment = {sentiment}")

# Evaluate the classifier on the test set
y_pred = classifier.predict(X_test)  # Predicted labels
print("Classification Report:")  # Detailed evaluation metrics
print(classification_report(y_test, y_pred))

# Function to predict sentiment for user-provided text
def predict_sentiment(user_input):
    """
    Predict the sentiment of a user-provided text.
    Steps:
    1. Transform input text into TF-IDF features using the trained vectorizer.
    2. Calculate probabilities using the trained classifier.
    3. Map probabilities to a sentiment category (Negative, Neutral, Positive).
    """
    # Transform the input text using TF-IDF vectorizer
    input_features = tfidf_vectorizer.transform([user_input])
    
    # Predict probabilities for the input text
    y_proba = classifier.predict_proba(input_features)
    
    # Extract the probability for positive sentiment (class 1)
    positive_score = y_proba[0][1]
    
    # Interpret the score into sentiment
    if positive_score < 0.4:
        sentiment = "Negative"
    elif 0.4 <= positive_score <= 0.6:
        sentiment = "Neutral"
    else:
        sentiment = "Positive"
    
    # Print the analysis
    print(f"\nText: {user_input}")
    print(f"Score: {positive_score:.2f}")
    print(f"Sentiment: {sentiment}")

# Take user input and predict its sentiment
user_input = input("\nEnter a text to analyze sentiment: ")
predict_sentiment(user_input)