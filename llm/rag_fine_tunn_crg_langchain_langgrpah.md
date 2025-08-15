# Advanced LLM Concepts: RAG, CAG, Fine-Tuning, Prompt Engineering, LangGraph, and LangChain

## Table of Contents
1. [What is RAG (Retrieval-Augmented Generation)?](#what-is-rag-retrieval-augmented-generation)
    - [Python Example: RAG with Ollama and OpenAI](#python-example-rag-with-ollama-and-openai)
2. [What is CAG (Context-Augmented Generation)?](#what-is-cag-context-augmented-generation)
    - [Python Example: CAG with Ollama](#python-example-cag-with-ollama)
3. [Fine-Tuning LLMs](#fine-tuning-llms)
    - [Python Example: Fine-Tuning with OpenAI](#python-example-fine-tuning-with-openai)
4. [Prompt Engineering](#prompt-engineering)
    - [Python Example: Prompt Engineering with OpenAI](#python-example-prompt-engineering-with-openai)
5. [LangGraph](#langgraph)
    - [Python Example: LangGraph Workflow](#python-example-langgraph-workflow)
6. [LangChain](#langchain)
    - [Python Example: LangChain Pipeline](#python-example-langchain-pipeline)
7. [References](#references)

---

## What is RAG (Retrieval-Augmented Generation)?
RAG combines retrieval of external knowledge (from databases, documents, etc.) with generative models (LLMs) to answer queries with up-to-date, context-rich information. It is widely used for chatbots, search, and Q&A systems.

### Python Example: RAG with Ollama and OpenAI
docs = ["Python is a programming language.", "Ollama runs LLMs locally."]
```python
import requests
import openai
import numpy as np

openai.api_key = "YOUR_OPENAI_API_KEY"

# Step 1: Prepare documents and query
docs = ["Python is a programming language.", "Ollama runs LLMs locally."]
query = "What is Ollama?"



# Step 2: Get embeddings for query and documents
# We use OpenAI's embedding model to convert the user's query into a high-dimensional vector.
# For the documents, embeddings should be created once (when the documents are added to your system) and cached/stored for reuse.
# This avoids recomputing document embeddings for every query, making retrieval fast and efficient.
query_emb = openai.Embedding.create(input=query, model="text-embedding-ada-002")['data'][0]['embedding']
# In a real system, you would load precomputed doc_embeddings from storage:
# doc_embeddings = load_cached_doc_embeddings()
# For demonstration, we compute them here, but this is NOT recommended for production:
doc_embeddings = [openai.Embedding.create(input=doc, model="text-embedding-ada-002")['data'][0]['embedding'] for doc in docs]

# Step 3: Compare query embedding with document embeddings using cosine similarity
# Why? Cosine similarity measures how close two vectors are in direction, which reflects how similar their meanings are.
# We want to find which document is most semantically similar to the user's query.
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Calculate similarity scores for each document
similarities = [cosine_similarity(query_emb, doc_emb) for doc_emb in doc_embeddings]
# Find the index of the document with the highest similarity score
best_idx = np.argmax(similarities)
# Select the most relevant document as context for the LLM
context = docs[best_idx]  # The most relevant doc based on similarity
# This process ensures the context is the most semantically similar to the query, improving the LLM's answer quality.

# Step 4: Generate answer using Ollama (local LLM)
ollama_response = requests.post(
    "http://localhost:11434/api/generate",
    json={"model": "llama2", "prompt": f"Context: {context}\nQuestion: {query}"}
)
print("RAG Answer:", ollama_response.json()["response"])
```

---

## What is CAG (Context-Augmented Generation)?
CAG is similar to RAG but focuses on enriching the prompt with additional context (metadata, user history, etc.) before passing it to the LLM. Useful for personalized assistants and adaptive chatbots.

### Python Example: CAG with Ollama
```python
import requests

user_history = "User previously asked about Python."
query = "Tell me more about Python."
context = f"History: {user_history}"

ollama_response = requests.post(
    "http://localhost:11434/api/generate",
    json={"model": "llama2", "prompt": f"{context}\nCurrent question: {query}"}
)
print("CAG Answer:", ollama_response.json()["response"])
```

---

## Fine-Tuning LLMs
Fine-tuning adapts a pre-trained LLM to specific tasks or domains by training it further on custom data. This improves accuracy and relevance for specialized use cases.

### Python Example: Fine-Tuning with OpenAI
```python
import openai

openai.api_key = "YOUR_OPENAI_API_KEY"

# Upload training data (JSONL file)
openai.File.create(file=open("train.jsonl", "rb"), purpose="fine-tune")

# Start fine-tuning job
openai.FineTuningJob.create(training_file="file-xxxxxx", model="gpt-3.5-turbo")

# Use the fine-tuned model
response = openai.ChatCompletion.create(
    model="ft:gpt-3.5-turbo:your-org:custom-task",
    messages=[{"role": "user", "content": "Your custom prompt"}]
)
print(response.choices[0].message["content"])
```

---

## Prompt Engineering
Prompt engineering is the art of designing effective prompts to guide LLMs toward desired outputs. It is crucial for getting accurate, relevant, and safe responses.

### Python Example: Prompt Engineering with OpenAI
```python
import openai

openai.api_key = "YOUR_OPENAI_API_KEY"

prompt = "You are a helpful assistant. Summarize the following text: Python is a popular language."
response = openai.Completion.create(
    model="text-davinci-003",
    prompt=prompt,
    max_tokens=50
)
print(response.choices[0].text.strip())
```

---

## LangGraph
LangGraph is a framework for building multi-step, graph-based workflows with LLMs. It allows chaining and branching of tasks, useful for complex reasoning and automation.

### Python Example: LangGraph Workflow
```python
# Pseudocode for LangGraph (actual API may differ)
from langgraph import Graph, Node

# Define nodes (steps)
node1 = Node(task="Extract entities from text")
node2 = Node(task="Generate summary")

# Build graph
graph = Graph()
graph.add_node(node1)
graph.add_node(node2)
graph.connect(node1, node2)

# Run workflow
result = graph.run(input_text="Python is great for AI.")
print(result)
```

---

## LangChain
LangChain is a popular Python library for building LLM-powered applications. It supports chaining prompts, retrieval, agents, and more.

### Python Example: LangChain Pipeline
```python
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

llm = Ollama(model="llama2")
prompt = PromptTemplate.from_template("Explain: {topic}")
chain = LLMChain(llm=llm, prompt=prompt)

result = chain.run({"topic": "Python"})
print(result)
```

---

## References
- [OpenAI API Docs](https://platform.openai.com/docs)
- [Ollama](https://ollama.com/)
- [LangChain](https://python.langchain.com/)
- [LangGraph](https://langgraph.com/)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)

---

Feel free to copy, adapt, and extend this document for your needs!
