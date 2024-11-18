import streamlit as st
import gensim.downloader as api
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Load the Word2Vec model
model = api.load('word2vec-google-news-300')

# Function to get similar words
def get_similar_words(word, top_n=10):
    try:
        similar_words = model.most_similar(word, topn=top_n)
        return similar_words
    except KeyError:
        return []

# Function to visualize embeddings with PCA
def visualize_embeddings_pca(similar_words):
    words = [word for word, _ in similar_words]
    vectors = [model[word] for word in words]
    
    # PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(vectors)
    
    df_pca = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
    df_pca['word'] = words
    
    # Plot using Plotly
    fig = px.scatter(df_pca, x='PC1', y='PC2', text='word', title='PCA Visualization of Word Embeddings')
    fig.update_traces(textposition='top center')
    st.plotly_chart(fig)

# Function to visualize embeddings with t-SNE
def visualize_embeddings_tsne(similar_words):
    words = [word for word, _ in similar_words]
    vectors = [model[word] for word in words]
    
    # t-SNE
    tsne = TSNE(n_components=2, perplexity=15, n_iter=3000, random_state=42)
    tsne_result = tsne.fit_transform(vectors)
    
    df_tsne = pd.DataFrame(tsne_result, columns=['x', 'y'])
    df_tsne['word'] = words
    
    # Plot using Plotly
    fig = px.scatter(df_tsne, x='x', y='y', text='word', title='t-SNE Visualization of Word Embeddings')
    fig.update_traces(textposition='top center')
    st.plotly_chart(fig)

# Streamlit app UI
st.title("Word Embeddings Exploration with Word2Vec")
st.write("This app allows you to explore word embeddings using Word2Vec, visualize their relationships, and understand concepts like PCA and t-SNE.")

# Input from user
word_input = st.text_input("Enter a word to find similar words:", "king")
top_n = st.slider("Number of similar words to display", 1, 20, 10)

# Show similar words
similar_words = get_similar_words(word_input, top_n)
if similar_words:
    st.write(f"Top {top_n} similar words to '{word_input}':")
    st.write(similar_words)
    
    # Visualizations
    st.subheader("PCA Visualization of Embeddings")
    visualize_embeddings_pca(similar_words)
    
    st.subheader("t-SNE Visualization of Embeddings")
    visualize_embeddings_tsne(similar_words)
else:
    st.write(f"Sorry, the word '{word_input}' is not in the vocabulary.")
