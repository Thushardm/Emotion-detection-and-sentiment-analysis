import string
from collections import Counter
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Step 1: Reading the input text file
text = open('InputText.txt', encoding='utf-8').read()   # Read the content of the text file
lower_case = text.lower()                               # Convert the entire text to lowercase for uniformity
cleaned_text = lower_case.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation marks

# Step 2: Tokenizing the text into words
# Using word_tokenize (faster and more robust than split()) to break text into individual words
tokenized_words = word_tokenize(cleaned_text, "english")

# Step 3: Removing stopwords
# Stopwords are common words (e.g., "the", "is") that do not contribute to meaningful analysis
final_words = []
for word in tokenized_words:
    if word not in stopwords.words('english'):  # Exclude stopwords from the tokenized list
        final_words.append(word)

# Step 4: Lemmatization
# Convert words to their base form (e.g., "running" -> "run", "better" -> "good")
lemma_words = []
lemmatizer = WordNetLemmatizer()
for word in final_words:
    word = lemmatizer.lemmatize(word)       # Apply lemmatization to each word
    lemma_words.append(word)

# Step 5: Associating emotions with words
# Read a custom emotion mapping file and match words to emotions
emotion_list = []                           # List to store detected emotions
with open('emotions.txt', 'r') as file:     # Open the emotion mapping file
    for line in file:
        clear_line = line.replace("\n", '').replace(",", '').replace("'", '').strip()  # Clean the line
        word, emotion = clear_line.split(':')  # Split the line into word and emotion

        if word in lemma_words:             # If the word is in our processed text
            emotion_list.append(emotion)    # Append the associated emotion

print(emotion_list)         # Print the list of emotions detected
w = Counter(emotion_list)   # Count the frequency of each emotion
print(w)                    # Print the frequency count of emotions

# Step 6: Sentiment Analysis
# Analyze the sentiment of the cleaned text using NLTK's VADER SentimentIntensityAnalyzer
def sentiment_analyse(sentiment_text):
    """
    Perform sentiment analysis on the input text.
    - Calculates sentiment scores for Positive, Negative, and Neutral.
    - Determines overall sentiment based on the highest score.
    """
    score = SentimentIntensityAnalyzer().polarity_scores(sentiment_text)  # Get sentiment scores
    if score['neg'] > score['pos']:     # If the negative score is higher
        print("Negative Sentiment")
    elif score['neg'] < score['pos']:   # If the positive score is higher
        print("Positive Sentiment")
    else:                               # If positive and negative scores are equal
        print("Neutral Sentiment")

# Perform sentiment analysis on the cleaned text
sentiment_analyse(cleaned_text)

# Step 7: Plotting Emotion Distribution
# Visualize the emotion frequency as a bar chart
fig, ax1 = plt.subplots()       # Create a subplot for the bar chart
ax1.bar(w.keys(), w.values())   # Create a bar chart with emotion labels and their frequencies
fig.autofmt_xdate()             # Format the x-axis labels for better readability
plt.savefig('graph.png')        # Save the graph as an image file
plt.show()                      # Display the bar chart
