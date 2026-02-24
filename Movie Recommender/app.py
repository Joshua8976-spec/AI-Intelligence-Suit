from flask import Flask, render_template, request
import pandas as pd
import ast # This helps us read the Genre JSON data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)

# Helper function to convert JSON genres to a clean string
def convert_genres(genre_str):
    try:
        genres = ast.literal_eval(genre_str)
        return ", ".join([g['name'] for g in genres])
    except:
        return ""

# --- LOAD DATA ---
df = pd.read_csv('tmdb_movies.csv', encoding='latin1')
df['overview'] = df['overview'].fillna('')
# Clean up genres and ratings
df['genres_list'] = df['genres'].apply(convert_genres)
df['vote_average'] = df['vote_average'].fillna(0)

# --- THE MATH ---
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['overview'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    movie_name = request.form.get('movie_name')
    try:
        idx = df[df['title'].str.lower() == movie_name.lower()].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
        
        # We now store a list of DICTIONARIES for each movie
        recommended_data = []
        for i in sim_scores:
            movie_row = df.iloc[i[0]]
            recommended_data.append({
                'title': movie_row['title'],
                'genres': movie_row['genres_list'],
                'rating': movie_row['vote_average']
            })
            
        return render_template('index.html', results=recommended_data, original_title=movie_name)
    except:
        return render_template('index.html', error="Movie not found!")

if __name__ == '__main__':
    app.run(debug=True)