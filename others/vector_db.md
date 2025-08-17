# Vector Databases: A Practical Guide




## Table of Contents
1. [What is a Vector Database?](#what-is-a-vector-database)
2. [When to Use a Vector Database](#when-to-use-a-vector-database)
3. [How to Create a Vector Database with Python](#how-to-create-a-vector-database-with-python)
    - [Using PDFs, CSVs, Docs, Audio, Video](#using-pdfs-csvs-docs-audio-video)
    - [Code Examples](#code-examples)
4. [Embeddings and Indexes in Vector DB](#embeddings-and-indexes-in-vector-db)
    - [Embeddings](#embeddings)
    - [Vector Indexing](#vector-indexing)
    - [Popular Vector Indexing Algorithms](#popular-vector-indexing-algorithms)
    - [Code Example: Building and Querying with HNSW Index](#code-example-building-and-querying-with-hnsw-index)
5. [Business Use Cases](#business-use-cases)
6. [Real-world Examples](#real-world-examples)
7. [Vector DB, Semantic Search, and AI Applications](#vector-db-semantic-search-and-ai-applications)
8. [Building a Chatbot with Vector DB](#building-a-chatbot-with-vector-db)
    - [OpenAI Embeddings](#openai-embeddings)
    - [Chunking and Text Splitting](#chunking-and-text-splitting)
    - [Vector Database Operations and Features](#vector-database-operations-and-features)
    - [Use Cases](#use-cases)
    - [Examples of Vector Databases and Libraries](#examples-of-vector-databases-and-libraries)
    - [Vector Indexing and Search Algorithms](#vector-indexing-and-search-algorithms)
    - [Example: Chatbot with ChromaDB and OpenAI Embeddings](#example-chatbot-with-chromadb-and-openai-embeddings)
9. [How to Interact with a Vector Database Using Python](#how-to-interact-with-a-vector-database-using-python)
    - [Adding Documents and Vectors](#1-adding-documents-and-vectors)
    - [Querying for Similarity](#2-querying-for-similarity)
    - [Updating Documents or Vectors](#3-updating-documents-or-vectors)
    - [Deleting Documents](#4-deleting-documents)
    - [Filtering with Metadata](#5-filtering-with-metadata)
10. [Building a Plain Vector Database (No LLM)](#building-a-plain-vector-database-no-llm)
    - [Example: Plain Vector DB with ChromaDB](#example-plain-vector-db-with-chromadb)
11. [CLIP, GloVe, and Word2Vec Embedding Models](#clip-glove-and-word2vec-embedding-models)
    - [CLIP (Image/Text Embeddings)](#clip-imagetext-embeddings)
    - [GloVe (Text Embeddings)](#glove-text-embeddings)
    - [Word2Vec (Text Embeddings)](#word2vec-text-embeddings)
    - [Audio to Vector](#audio-to-vector)
12. [ChromaDB, OpenAI, and Ollama](#chromadb-openai-and-ollama)
    - [What is ChromaDB?](#what-is-chromadb)
    - [Why Do We Need OpenAI?](#why-do-we-need-openai)
    - [How to Build a Vector DB with Ollama (Local LLM)](#how-to-build-a-vector-db-with-ollama-local-llm)
    - [Example: Using Ollama for Local Embeddings with ChromaDB](#example-using-ollama-for-local-embeddings-with-chromadb)
    - [When to Use OpenAI vs. Ollama](#when-to-use-openai-vs-ollama)
    - [Summary](#summary)
13. [References](#references)

---

## What is a Vector Database?
A **vector database** is a specialized database designed to store, index, and search high-dimensional vectors (numerical arrays). These vectors typically represent data such as text, images, audio, or video after being transformed by machine learning models (embeddings). Vector databases enable efficient similarity search, which is crucial for AI, recommendation systems, and semantic search.

---

## When to Use a Vector Database
- **Semantic Search:** Find similar documents, images, or audio based on meaning, not keywords.
- **Recommendation Systems:** Suggest products, content, or users based on vector similarity.
- **AI Applications:** Power chatbots, question-answering, and retrieval-augmented generation.
- **Multimedia Search:** Search across images, audio, and video using embeddings.
- **Large-scale Data:** Handle millions of items with fast similarity queries.

---

## How to Create a Vector Database with Python

### Using PDFs, CSVs, Docs, Audio, Video
You can use libraries like [ChromaDB](https://docs.trychroma.com/), [FAISS](https://github.com/facebookresearch/faiss), or [Milvus](https://milvus.io/) for vector databases. Below is an example using ChromaDB and OpenAI embeddings.

#### Install Required Libraries
```bash
pip install chromadb openai pypdf pandas
```

#### Code Examples
```python
import chromadb
from chromadb.utils import embedding_functions
import openai
import pandas as pd
from PyPDF2 import PdfReader

# Set your OpenAI API key
openai.api_key = "YOUR_OPENAI_API_KEY"

# Function to get embeddings from OpenAI
# Converts text to vector using OpenAI
# We need this to represent text semantically for similarity search
def get_embedding(text):
    response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
    return response['data'][0]['embedding']

# Initialize Chroma DB
client = chromadb.Client()
collection = client.create_collection(name="documents")

# Index PDF
pdf = PdfReader("sample.pdf")
for page in pdf.pages:
    text = page.extract_text()
    embedding = get_embedding(text)
    collection.add(documents=[text], embeddings=[embedding])

# Index CSV
# We need this to search structured data semantically
# Replace 'text_column' with your actual column name
df = pd.read_csv("sample.csv")
for row in df['text_column']:
    embedding = get_embedding(row)
    collection.add(documents=[row], embeddings=[embedding])

# Index plain text docs
with open("sample.txt") as f:
    text = f.read()
    embedding = get_embedding(text)
    collection.add(documents=[text], embeddings=[embedding])

# Note: For audio/video, use a model to transcribe or extract features, then embed.
```

Using ollama

```python

import chromadb
import pandas as pd
from PyPDF2 import PdfReader
import requests
import json

# Function to get embeddings from Ollama
# Converts text to vector using a local Ollama model
def get_embedding(text):
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={
            "model": "llama2",  # or another model you have installed in Ollama
            "prompt": text
        }
    )
    if response.status_code == 200:
        return response.json()["embedding"]
    else:
        raise Exception(f"Error getting embedding: {response.text}")

# Initialize Chroma DB
#
# By default, when you create a ChromaDB client without specifying parameters, it uses an in-memory database. This means:
# The database exists only in RAM
# All data will be lost when your program terminates
# No files are created on disk
#
#
# If you want to persist the data to disk, you would need to use the PersistentClient instead:
# from chromadb.config import Settings
# persistent_client = chromadb.PersistentClient(path="/path/to/save/data", settings=Settings(allow_reset=True))

client = chromadb.Client()
collection = client.create_collection(name="documents")

# Index PDF
pdf = PdfReader("sample.pdf")
for page in pdf.pages:
    text = page.extract_text()
    embedding = get_embedding(text)
    collection.add(documents=[text], embeddings=[embedding])

# Index CSV
# Replace 'text_column' with your actual column name
df = pd.read_csv("sample.csv")
for row in df['text_column']:
    embedding = get_embedding(row)
    collection.add(documents=[row], embeddings=[embedding])

# Index plain text docs
with open("sample.txt") as f:
    text = f.read()
    embedding = get_embedding(text)
    collection.add(documents=[text], embeddings=[embedding])

# Note: For audio/video, use a model to transcribe or extract features, then embed.

```

---
## Embeddings and Indexes in Vector DB

### Embeddings
- **Definition:** Numeric representations (vectors) of data (text, images, audio, etc.) generated by ML models
- **Purpose:** Capture semantic meaning and relationships in a mathematical space
- **Examples:** OpenAI's text-embedding-ada-002 converts text to 1536-dimensional vectors

### Vector Indexing
- **Challenge:** Naive similarity search requires comparing a query vector against every vector in the database (O(n) complexity)
- **Solution:** Vector indexes organize vectors to enable sublinear (faster than O(n)) search
- **Why needed:** Critical for performance when searching through millions or billions of vectors

#### Popular Vector Indexing Algorithms
1. **HNSW (Hierarchical Navigable Small World)**
    - Creates a multi-layered graph structure for efficient navigation
    - Excellent balance between speed and recall
    - Used by ChromaDB, Milvus, and others

2. **IVF (Inverted File Index)**
    - Divides vector space into clusters (Voronoi cells)
    - Only searches relevant clusters instead of entire space
    - Good for very large datasets

3. **Flat Index**
    - No indexing structure, performs exhaustive search
    - 100% accurate but slow for large datasets
    - Used as baseline or for small collections

### Code Example: Building and Querying with HNSW Index

```python
import numpy as np
import faiss
import time

# Generate sample data (10,000 vectors with 128 dimensions)
dimension = 128
num_vectors = 10000
vectors = np.random.random((num_vectors, dimension)).astype(np.float32)
query = np.random.random((1, dimension)).astype(np.float32)

# Approach 1: Flat index (no optimization)
flat_index = faiss.IndexFlatL2(dimension)
flat_index.add(vectors)

# Approach 2: HNSW index
# M = connections per layer, efConstruction = build quality parameter
hnsw_index = faiss.IndexHNSWFlat(dimension, 32)  
hnsw_index.hnsw.efConstruction = 40
hnsw_index.hnsw.efSearch = 16
hnsw_index.add(vectors)

# Compare search speeds
start = time.time()
_, flat_results = flat_index.search(query, k=10)  # Find 10 nearest neighbors
flat_time = time.time() - start

start = time.time()
_, hnsw_results = hnsw_index.search(query, k=10)
hnsw_time = time.time() - start

print(f"Flat index search time: {flat_time:.6f} seconds")
print(f"HNSW index search time: {hnsw_time:.6f} seconds")
print(f"Speed improvement: {flat_time/hnsw_time:.2f}x faster")

# Check overlap in results (accuracy)
overlap = len(set(flat_results[0]) & set(hnsw_results[0]))
print(f"Result accuracy: {overlap/10*100:.1f}% overlap")
```

The indexing structures create a trade-off between:
- **Search speed:** How quickly results are returned
- **Recall/accuracy:** How closely the results match the true nearest neighbors
- **Memory usage:** How much additional memory the index requires
- **Build time:** How long it takes to construct the index

For production systems, properly configured indexes can provide 100-1000x speed improvements with minimal accuracy loss.

---

## Business Use Cases
- **Customer Support:** Power semantic search for FAQs and chatbots.
- **Product Recommendations:** Suggest similar products based on user behavior.
- **Content Moderation:** Detect similar harmful content.
- **Fraud Detection:** Find patterns in transaction data.

---

## Real-world Examples
- **Spotify:** Music recommendation using audio embeddings.
- **Google Photos:** Image search by content, not filename.
- **E-commerce:** Personalized product suggestions.
- **Legal/Medical:** Semantic search in large document repositories.

---

## Vector DB, Semantic Search, and AI Applications
Vector DBs enable **semantic search**—finding items by meaning, not keywords. They are foundational for AI applications like chatbots, document retrieval, and recommendation engines.

---

## Building a Chatbot with Vector DB

### OpenAI Embeddings
OpenAI provides high-quality embeddings for text, images, and more. These are used to represent user queries and documents as vectors for semantic search.

### Chunking and Text Splitting
Chunking helps break large documents into smaller pieces for better retrieval accuracy.

### Vector Database Operations and Features
- **Add:** Insert new vectors.
- **Query:** Find similar vectors.
- **Update/Delete:** Modify or remove vectors.
- **Metadata:** Store extra info with vectors.

### Use Cases
- Semantic search
- Recommendation engines
- Multimedia retrieval
- Fraud detection
- Chatbots

### Examples of Vector Databases and Libraries
- [ChromaDB](https://docs.trychroma.com/): Easy Python API, local or cloud.
- [FAISS](https://github.com/facebookresearch/faiss): Facebook’s library for fast similarity search.
- [Milvus](https://milvus.io/): Scalable, cloud-native vector DB.
- [Weaviate](https://weaviate.io/): RESTful API, supports multiple data types.

### Vector Indexing and Search Algorithms
- **HNSW (Hierarchical Navigable Small World):** Fast, scalable search.
- **IVF (Inverted File Index):** Efficient for large datasets.
- **Flat Index:** Brute-force, accurate but slow for big data.

#### Example: Chatbot with ChromaDB and OpenAI Embeddings
```python
import chromadb
from chromadb.utils import embedding_functions
import openai

# Set your OpenAI API key
openai.api_key = "YOUR_OPENAI_API_KEY"

# What is get_embedding?
# This function sends text to OpenAI's embedding API, which uses a model (text-embedding-ada-002)
# to convert the text into a high-dimensional vector (embedding). This vector captures the semantic meaning of the text.
# Why do we send text as input?
# We need to represent both our documents and user queries as vectors so we can compare them for similarity.
# What is 'text-embedding-ada-002'?
# This is an OpenAI model specifically trained to generate embeddings for text. It is widely used for semantic search and retrieval tasks.
def get_embedding(text):
    response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
    # Returns a list of floats (the embedding vector) representing the input text
    return response['data'][0]['embedding']

# Chunking: Split large documents into smaller pieces for better retrieval accuracy
def chunk_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# Initialize ChromaDB (the vector database)
client = chromadb.Client()
collection = client.create_collection(name="chatbot_docs")

# Index documents
# For each chunk, we generate an embedding and store it in the vector DB
# This allows us to later search for similar chunks based on user queries
with open("knowledge_base.txt") as f:
    text = f.read()
    chunks = chunk_text(text)
    for chunk in chunks:
        embedding = get_embedding(chunk)
        collection.add(documents=[chunk], embeddings=[embedding])

# Chatbot query
# When a user asks a question, we:
# 1. Generate an embedding for the query (using get_embedding)
# 2. Search the vector DB for the most similar document chunks
# 3. Return those chunks as the chatbot's answer
# What does collection.query do?
# It finds the stored document chunks whose embeddings are closest to the query embedding (semantic similarity)
# Where does the LLM kick in?
# In this example, the LLM (OpenAI's embedding model) is used to generate embeddings, not to generate answers.
# For a full chatbot, you could pass the retrieved chunks to an LLM (like GPT) to generate a natural language response.
def chatbot_query(query):
    query_embedding = get_embedding(query)  # Convert user query to embedding
    results = collection.query(query_embeddings=[query_embedding], n_results=3)  # Find top 3 similar chunks
    return results['documents']  # Return the most relevant chunks

# Example usage
user_input = "How do I reset my password?"
answers = chatbot_query(user_input)
print("Chatbot answers:", answers)
```

---



## How to Interact with a Vector Database Using Python

Once your vector database is set up (with ChromaDB or similar), you can perform various operations programmatically. Here are common interactions:

### 1. Adding Documents and Vectors
```python
collection.add(documents=["New Doc"], embeddings=[[0.1, 0.2, 0.3, ...]])
# Adds a new document and its vector to the collection
```

### 2. Querying for Similarity
```python
query_vector = [0.1, 0.2, 0.3, ...]  # Your query vector
results = collection.query(query_embeddings=[query_vector], n_results=3)
print("Top matches:", results['documents'])
# Returns the top 3 most similar documents
```

### 3. Updating Documents or Vectors
```python
# ChromaDB supports upserts (update or insert)
collection.upsert(documents=["Updated Doc"], embeddings=[[0.4, 0.5, 0.6, ...]])
# Updates the vector for the given document
```

### 4. Deleting Documents
```python
collection.delete(documents=["Doc to delete"])
# Removes the document and its vector from the collection
```

### 5. Filtering with Metadata
```python
collection.add(documents=["Doc with meta"], embeddings=[[0.7, 0.8, 0.9, ...]], metadatas=[{"type": "example"}])
results = collection.query(query_embeddings=[[0.7, 0.8, 0.9, ...]], where={"type": "example"})
print("Filtered results:", results['documents'])
# Adds metadata and queries with filters
```

**Comments:**
- All operations are performed via the collection object in Python.
- You can automate ingestion, search, and management of your vector data.
- For more advanced features, see the [ChromaDB documentation](https://docs.trychroma.com/).

---

Sometimes you may want to build a vector database without using any large language model (LLM) or external embedding service. This is useful for prototyping, working with numeric data, or when you already have vector representations (e.g., from image features, sensor data, or manual encoding).

### Example: Plain Vector DB with ChromaDB

```python
import chromadb
import numpy as np

# Create some sample vectors manually or from numeric features
# Here, we use random vectors for demonstration
documents = ["Item A", "Item B", "Item C"]
vectors = [np.random.rand(10).tolist() for _ in documents]  # 10-dimensional vectors

# Initialize ChromaDB
client = chromadb.Client()
collection = client.create_collection(name="plain_vectors")

# Add documents and their vectors to the DB
for doc, vec in zip(documents, vectors):
    collection.add(documents=[doc], embeddings=[vec])

# Query: Find the most similar item to a new vector
query_vector = np.random.rand(10).tolist()
results = collection.query(query_embeddings=[query_vector], n_results=1)
print("Most similar item:", results['documents'])
```

**Comments:**
- No LLM or external API is used; vectors are created manually or from numeric features.
- This approach is useful for similarity search in numeric datasets, image features, sensor data, etc.
- You can use any method to generate vectors, as long as they are lists of numbers.

---


## CLIP, GloVe, and Word2Vec Embedding Models

### CLIP (Image/Text Embeddings)
CLIP (Contrastive Language–Image Pre-training) is a model by OpenAI that can embed both images and text into the same vector space, enabling cross-modal search and retrieval.

#### Example: CLIP Usage in Python
```python
# Install: pip install torch torchvision clip-by-openai
import clip
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Prepare image and text
image = preprocess(Image.open("cat.jpg")).unsqueeze(0).to(device)
text = clip.tokenize(["a photo of a cat", "a photo of a dog"]).to(device)

# Get embeddings
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

# Compare similarity
similarity = (image_features @ text_features.T).softmax(dim=-1)
print("Similarity scores:", similarity)
```

### GloVe (Text Embeddings)
GloVe (Global Vectors for Word Representation) is a popular unsupervised learning algorithm for obtaining vector representations for words.

#### Example: GloVe Usage in Python
```python
# Download GloVe vectors from https://nlp.stanford.edu/projects/glove/
import numpy as np

# Load GloVe vectors (example for glove.6B.50d.txt)
glove_path = "glove.6B.50d.txt"
glove = {}
with open(glove_path, "r", encoding="utf8") as f:
    for line in f:
        parts = line.split()
        word = parts[0]
        vector = np.array(parts[1:], dtype=np.float32)
        glove[word] = vector

# Get embedding for a word
embedding = glove["cat"]
print("GloVe embedding for 'cat':", embedding)
```

### Word2Vec (Text Embeddings)
Word2Vec is a neural network-based technique to learn word associations from a large corpus of text. It produces word embeddings that capture semantic relationships.

#### Example: Word2Vec Usage in Python
```python
# Install: pip install gensim
from gensim.models import Word2Vec

# Train Word2Vec on sample sentences
sentences = [["cat", "sat", "on", "the", "mat"], ["dog", "barked"]]
model = Word2Vec(sentences, vector_size=50, window=5, min_count=1, workers=2)

# Get embedding for a word
embedding = model.wv["cat"]
print("Word2Vec embedding for 'cat':", embedding)
```

### Audio to Vector
To convert audio to vectors, you typically extract features (MFCCs, spectrograms, etc.) using libraries like librosa, then use those features as embeddings.

#### Example: Audio Feature Extraction in Python
```python
# Install: pip install librosa
import librosa

# Load audio file
audio_path = "audio_sample.wav"
y, sr = librosa.load(audio_path)

# Extract MFCC features
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
mfcc_mean = mfcc.mean(axis=1)
print("Audio vector (MFCC mean):", mfcc_mean)
```

**Comments:**
- CLIP is used for cross-modal (image/text) search and retrieval.
- GloVe and Word2Vec are used for text embeddings and NLP tasks.
- Audio features (MFCCs) can be used for similarity search in audio databases.
- All these embeddings can be stored and queried in a vector database like ChromaDB.

---

## ChromaDB, OpenAI, and Ollama

### What is ChromaDB?
[ChromaDB](https://docs.trychroma.com/) is an open-source vector database designed for AI and semantic search applications. It’s easy to use, supports fast similarity search, and can run locally or in the cloud. ChromaDB is popular for prototyping and production use in Python projects.

**Key Features:**
- Simple Python API
- Fast similarity search
- Supports metadata and filtering
- Integrates with popular embedding models (OpenAI, local LLMs, etc.)
- Can be used for chatbots, semantic search, recommendation, and more

### Why Do We Need OpenAI?
[OpenAI](https://platform.openai.com/docs/guides/embeddings) provides state-of-the-art embedding models (like `text-embedding-ada-002`) that convert text, images, or other data into high-dimensional vectors. These embeddings capture semantic meaning, making it possible to search by meaning rather than keywords.

**Reasons to use OpenAI embeddings:**
- High-quality, general-purpose embeddings
- Easy integration with Python and ChromaDB
- Excellent performance for semantic search and retrieval tasks

**Note:** Using OpenAI requires an API key and internet access. For privacy, cost, or local deployment, you may want to use a local LLM.

### How to Build a Vector DB with Ollama (Local LLM)
[Ollama](https://ollama.com/) is a tool for running large language models (LLMs) locally on your machine. You can use Ollama to generate embeddings without sending data to external servers.

#### Example: Using Ollama for Local Embeddings with ChromaDB

1. **Install Ollama and ChromaDB**
```bash
# Install Ollama (see https://ollama.com for details)
brew install ollama

# Start Ollama server
ollama serve

# Pull a model (e.g., llama2)
ollama pull llama2

# Install ChromaDB
pip install chromadb
```

2. **Generate Embeddings Locally with Ollama**
```python
import chromadb
import requests

# Function to get embeddings from Ollama (local LLM)
# Keeps all data local for privacy and cost control
def get_local_embedding(text):
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": "llama2", "prompt": text}
    )
    return response.json()["embedding"]

# Initialize ChromaDB
client = chromadb.Client()
collection = client.create_collection(name="local_docs")

# Index documents using local embeddings
documents = ["First document text", "Second document text"]
for doc in documents:
    embedding = get_local_embedding(doc)
    collection.add(documents=[doc], embeddings=[embedding])

# Querying
query = "Find similar docs"
query_embedding = get_local_embedding(query)
results = collection.query(query_embeddings=[query_embedding], n_results=2)
print("Results:", results['documents'])
```

**Comments:**
- This approach keeps all data local—no external API calls.
- You can use any LLM supported by Ollama for embeddings.
- Useful for privacy, cost control, and offline scenarios.

### When to Use OpenAI vs. Ollama
- **OpenAI:** Best for high-quality, general-purpose embeddings, cloud scalability, and rapid prototyping.
- **Ollama (Local LLM):** Best for privacy, cost savings, and when you need to run everything locally.

**Summary:**  
ChromaDB is a flexible vector database that works with both cloud (OpenAI) and local (Ollama) embedding models. Choose the embedding provider based on your business needs, privacy requirements, and infrastructure.

---

## References
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Milvus](https://milvus.io/)
- [Weaviate](https://weaviate.io/)
- [Ollama](https://ollama.com/)

---
