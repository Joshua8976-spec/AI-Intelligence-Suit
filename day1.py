import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 1. THE DATA: Small examples to teach the AI
reviews = [
    "I loved this movie, it was fantastic!", 
    "A complete waste of time, terrible acting.",
    "The best film I have seen this year!",
    "I hated every minute of it, so boring."
]
# positive = 1, negative = 0
labels = [1, 0, 1, 0] 

# 2. THE TRANSLATOR: Turn words into numbers (Math)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(reviews)

# 3. THE BRAIN: Create and train the model
model = LogisticRegression()
model.fit(X, labels)

# 4. THE TEST: Give it a brand new review it has never seen
new_review = ["The ending was brilliant and the acting was top notch!"]
new_review_transformed = vectorizer.transform(new_review)
prediction = model.predict(new_review_transformed)

# 5. THE RESULT
sentiment = "POSITIVE" if prediction[0] == 1 else "NEGATIVE"
print(f"\nReview: {new_review[0]}")
print(f"AI Sentiment Analysis: {sentiment}")