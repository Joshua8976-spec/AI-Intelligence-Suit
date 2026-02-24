import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# --- STEP 1: FIND AND LOAD FILE ---
target_file = 'tmdb_movies.csv'
if not os.path.exists(target_file):
    files = [f for f in os.listdir('.') if 'tmdb' in f.lower()]
    target_file = files[0] if files else None

if target_file:
    print(f"✅ Step 1: Found {target_file}. Loading now...")
    try:
        df = pd.read_csv(target_file, encoding='utf-8')
    except:
        df = pd.read_csv(target_file, encoding='latin1')
else:
    print("❌ Error: No file found. Make sure tmdb_movies.csv is in this folder!")
    exit()

# --- STEP 2: PREPARE DATA ---
print("⏳ Step 2: Cleaning plot summaries...")
df['overview'] = df['overview'].fillna('')

# --- STEP 3: THE MATH (VECTORIZATION) ---
print("⏳ Step 3: Converting words to math (This might take 30 seconds)...")
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['overview'])

# --- STEP 4: CALCULATE SIMILARITY ---
print("⏳ Step 4: Calculating similarity matrix...")
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# --- STEP 5: RECOMMENDATION FUNCTION ---
def get_recommendations(title):
    try:
        idx = df[df['title'].str.lower() == title.lower()].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
        
        print(f"\n🌟 Results for '{title}':")
        for i in sim_scores:
            print(f"-> {df['title'].iloc[i[0]]}")
    except:
        print("\n❌ Movie not found! Try 'Avatar' or 'Spectre'.")

# --- STEP 6: THE SEARCH BOX ---
print("\n🚀 SETUP COMPLETE!")
print("="*30)
user_movie = input("Enter a movie title you like: ")
get_recommendations(user_movie)