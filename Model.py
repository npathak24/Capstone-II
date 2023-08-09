from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pandas as pd
import sqlite3 as sql
import nltk

# Make sure to download the required NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')

def preprocess_text(text):
    # Tokenize the text
    words = nltk.word_tokenize(text)
    
    # Remove punctuation and convert to lowercase
    words = [word.lower() for word in words if word.isalnum()]
    
    # Lemmatize the words to their base forms
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    
    # Join the processed words back into a single string
    return ' '.join(words)

def load_data():
    # create connection with the database file
    database = "music_recommendation.db"
    connection = sql.connect(database)

    query = "SELECT * FROM tracks"

    df = pd.read_sql_query(query, connection)

    # Preprocess the text in the 'tags' column
    df['tags'] = df['artist'].astype(str) + df['danceability'].astype(str) + df['energy'].astype(str) + df['popularity'].astype(str)
    df['tags'] = df['tags'].apply(preprocess_text)

    # Filter out empty or very short strings
    new_df = df[df['tags'].str.len() > 2].copy()

    # Calculate the TF-IDF matrix
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(new_df['tags'])
    tfidf_array = tfidf_matrix.toarray()

    return new_df, tfidf_array

new_df, vector = load_data()
similarity = cosine_similarity(vector)


new_df['album'] == 'Lover'

sorted(list(enumerate(similarity[0])), reverse=True, key=lambda x: x[1])

sorted(list(enumerate(similarity[0])), reverse=True, key=lambda x: x[1])[1:6]

def recommend(album):
    # Check if the provided album exists in the DataFrame
    if album not in new_df['album'].values:
        print("Album not found in the database.")
        return

    # Find the index of the provided album
    index = new_df[new_df['album'] == album].index[0]

    # Calculate cosine similarity between the provided album and all other albums
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])

    # Print the top 5 similar albums
    print(f"Albums similar to '{album}':")
    for i in distances[1:6]:
        print(new_df.iloc[i[0]].album)

recommend('reputation')

# Rest of the code remains the same
