

# 🧠 Day 2: Exploring Word Embeddings - A Friendly Guide

## **1. What Are Word Embeddings?**
Imagine trying to understand the meaning of words just from their position in a dictionary — it’s tough! Word embeddings make this easier by representing words as **vectors** in a continuous space, where similar words (like "king" and "queen") have similar positions.

### **Why Do We Need Word Embeddings?**
- Traditional methods like **one-hot encoding** treat every word as unique, ignoring the relationships between words.
- **Word embeddings** solve this by capturing the **semantic meaning** — words that appear in similar contexts have similar vectors. For example, "happy" and "joyful" will have similar vector representations because they are often used in similar situations.

## **2. Two Popular Models for Word Embeddings**
Let’s explore two common ways to create these embeddings:

### **a. Word2Vec**
- **Developed by Google**, Word2Vec comes in two flavors:
  - **Skip-gram**: Predicts the surrounding words given a target word.
  - **CBOW** (Continuous Bag of Words): Predicts a word given its surrounding context.
- **Example**: In vector math, you can show relationships like:
  \[
  \text{"king"} - \text{"man"} + \text{"woman"} \approx \text{"queen"}
  \]
  This means that if you take the vector for "king," subtract "man," and add "woman," you get a vector close to "queen"!

### **b. GloVe (Global Vectors)**
- **Developed by Stanford**, GloVe focuses on understanding how often words co-occur in a large text corpus.
- **Difference**: Unlike Word2Vec, which focuses on local context (surrounding words), GloVe uses global word co-occurrence statistics, capturing both local and global word relationships.

## **3. Making Sense of High-Dimensional Data with Visualization**
Embeddings usually have **300 dimensions** (like 300 features). To visualize these in 2D, we need to **reduce dimensions**. Here’s how:

### **a. PCA (Principal Component Analysis)**
- **What It Does**: PCA identifies the directions (principal components) that capture the most variance in the data and reduces the dimensions accordingly.
- **Pros**: Fast and efficient.
- **Cons**: Can lose some detailed (local) relationships between words.

### **b. t-SNE (t-Distributed Stochastic Neighbor Embedding)**
- **What It Does**: t-SNE focuses on keeping similar points (words) close together in the reduced space, which is great for visualizing clusters.
- **Pros**: Reveals hidden structures and clusters.
- **Cons**: Computationally heavy and sensitive to settings.

## **4. Project Overview: What We’re Building**
We’re going to:
1. **Load Pre-trained Word Embeddings** (like Word2Vec).
2. **Find Similar Words**: Given a word, we’ll find 10 similar words.
3. **Visualize Embeddings** using PCA and t-SNE to see the relationships.
4. **Create an Interactive App** using Streamlit where you can input a word and explore its similar words visually.

## **5. Quick Code Preview**
Here’s a sneak peek at some of the code you’ll be using:

### **Finding Similar Words:**
```python
similar_words = model.most_similar("king", topn=10)
print(similar_words)
```
This code finds the top 10 words most similar to "king" using the Word2Vec model.

### **Visualizing with t-SNE:**
```python
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2)
reduced_embeddings = tsne.fit_transform(vectors)
```
This reduces your high-dimensional vectors to 2D, making them easier to plot.

## **6. Key Takeaways**
- **Word embeddings** capture the semantic relationships between words, making them powerful for NLP tasks.
- **Dimensionality reduction** techniques like PCA and t-SNE help us visualize these embeddings effectively.
- **Visual Exploration** can reveal insights into how words relate to one another, showing clusters of similar words.

### 🚀 **Explore and Play**: Check out the interactive app to see this in action!
