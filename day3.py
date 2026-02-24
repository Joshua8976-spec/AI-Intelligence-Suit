import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression

# 1. Load Data
print("Loading Data...")
df = pd.read_csv('IMDB Dataset.csv').head(5000)
df['sentiment_num'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

# 2. Setup the "Map" (Recommender)
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = vectorizer.fit_transform(df['review'])
similarity = cosine_similarity(tfidf_matrix)

# 3. Setup the "Judge" (Sentiment - using the same brain from Day 1)
model = LogisticRegression()
model.fit(tfidf_matrix, df['sentiment_num'])

def get_smart_recommendation(movie_idx):
    # Find similar movies
    scores = list(enumerate(similarity[movie_idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:4]
    
    print(f"\nSince you liked Review #{movie_idx}:")
    for idx, dist in scores:
        # Analyze the sentiment of the recommended movie's review
        pred = model.predict(tfidf_matrix[idx])
        vibe = "🔥 HIGHLY RATED" if pred[0] == 1 else "⚠️ MIXED REVIEWS"
        
        print(f"-> Recommend Review #{idx} (Similarity: {dist:.2f})")
        print(f"   AI Vibe Check: {vibe}")

# Run it!
# 5. SEARCH BY KEYWORD
print("\n" + "="*30)
search_term = input("What kind of movie are you looking for? (e.g. 'space', 'scary', 'action'): ")

# Transform your search into the same "Math Map"
search_vec = vectorizer.transform([search_term])
search_sim = cosine_similarity(search_vec, tfidf_matrix)

# Find the top 3 matches for that word
top_matches = search_sim.argsort()[0][-3:][::-1]

print(f"\nTop matches for '{search_term}':")
for idx in top_matches:
    pred = model.predict(tfidf_matrix[idx])
    vibe = "🔥 HIGHLY RATED" if pred[0] == 1 else "⚠️ MIXED REVIEWS"
    print(f"-> Review #{idx} | Vibe: {vibe}")
    print(f"   Excerpt: {df['review'].iloc[idx][:300]}...\n")