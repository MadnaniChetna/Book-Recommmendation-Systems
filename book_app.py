import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import streamlit as st

# Load the data
data = pd.read_csv("C:/Users/chetn/Website/books_data.csv")

# Handle missing values
data['Book-Title'] = data['Book-Title'].fillna('')
data['authors'] = data['authors'].fillna('Unknown Author')

# TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(data['Book-Title'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Indexing
indices = pd.Series(data.index, index=data['Book-Title']).drop_duplicates()

# Recommendation function
def get_recommendations(title, cosine_sim=cosine_sim):
    # Ensure we only get the first matching index
    if title not in indices:
        return pd.DataFrame()
    
    idx = indices[title]
    if isinstance(idx, pd.Series):
        idx = idx.iloc[0]
    
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    book_indices = [i[0] for i in sim_scores]
    return data.iloc[book_indices][['Book-Title', 'authors']]

# Streamlit UI
st.title("📚 Book Recommendation System")

# Optional author dropdown (for information or highlighting)
author_list = sorted(data['authors'].unique())
selected_author = st.selectbox("Author", ['None'] + author_list)

# Book dropdown (always shows all books)
all_books = data['Book-Title'].unique()
selected_book = st.selectbox("Choose a Book Title:", all_books)

# Show recommendations
if st.button("Get Recommendations"):
    recommendations = get_recommendations(selected_book)
    if not recommendations.empty:
        st.write("### Recommended Books:")
        for _, row in recommendations.iterrows():
            st.write(f"- **{row['Book-Title']}** by *{row['authors']}*")
    else:
        st.warning("No recommendations found.")
