from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
import pandas as pd
import sqlite3 as sql
import nltk

app = Flask(__name__, static_url_path='/static', static_folder='static')

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

    # Calculate cosine similarity
    similarity = cosine_similarity(tfidf_array)

    return new_df, similarity

new_df, similarity = load_data()

def recommend(album):
    index = new_df[new_df['album'] == album].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommendations = [new_df.iloc[i[0]]['album'] for i in distances[1:6]]
    return recommendations

@app.route('/', methods=['GET', 'POST'])
def index():
    album = None
    recommendations = []

    if request.method == 'POST':
        album = request.form['album']
        recommendations = recommend(album)

    return render_template('index.html', album=album, recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
