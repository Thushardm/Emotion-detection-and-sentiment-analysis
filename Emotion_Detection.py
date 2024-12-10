import string
import pickle
from collections import Counter
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
def train_emotion_model(input_text_path, emotions_path):
    # Read and preprocess the input text
    text = open(input_text_path, encoding='utf-8').read()
    lower_case = text.lower()
    cleaned_text = lower_case.translate(str.maketrans('', '', string.punctuation))

    # Tokenize and remove stop words
    tokenized_words = word_tokenize(cleaned_text, "english")
    final_words = [word for word in tokenized_words if word not in stopwords.words('english')]

    # Lemmatize words
    lemma_words = [WordNetLemmatizer().lemmatize(word) for word in final_words]

    # Match words with emotions dataset
    emotion_list = []
    with open(emotions_path, 'r') as file:
        for line in file:
            clear_line = line.replace("\n", '').replace(",", '').replace("'", '').strip()
            word, emotion = clear_line.split(':')
            if word.strip() in lemma_words:
                emotion_list.append(emotion.strip())

    # Count emotions
    emotion_counter = Counter(emotion_list)

    # Save the emotion model
    model_path = "emotion_model.pkl"
    with open(model_path, 'wb') as model_file:
        pickle.dump(emotion_counter, model_file)

    print(f"Model saved to {model_path}")
    return emotion_counter
