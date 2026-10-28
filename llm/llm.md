
# Complete LLM Learning Roadmap: Zero to Building Your Own LLM

## Goal

Learn everything required to build, train, fine-tune, and deploy Large Language Models (LLMs) using Python, PyTorch, and Hugging Face.

**Level:** Beginner to Advanced
**Duration:** 6–9 months
**Prerequisites:** None
**Learning approach:** Theory + Mathematics + Coding + Projects

---

# Table of Contents

1. Introduction to AI, ML, Deep Learning, and LLMs
2. Python Programming
3. Mathematics for Deep Learning
4. NumPy and Data Processing
5. Machine Learning Fundamentals
6. Neural Networks from Scratch
7. PyTorch
8. TensorFlow and Keras
9. Deep Learning Architectures
10. Natural Language Processing
11. Tokenization and Embeddings
12. Transformer Architecture
13. Hugging Face Ecosystem
14. Building a GPT Model from Scratch
15. LLM Training and Fine-Tuning
16. LoRA and QLoRA
17. Retrieval-Augmented Generation (RAG)
18. LLM Evaluation and Optimization
19. LLM Deployment
20. GPU and Cloud Computing
21. Practical Projects
22. Learning Resources
23. Final Checklist

---

# 1. Introduction to AI, ML, Deep Learning, and LLMs

## 1.1 Artificial Intelligence

Artificial Intelligence (AI) is the field of creating systems that perform tasks associated with intelligence.

## 1.2 Machine Learning

Machine Learning (ML) enables computers to learn patterns from data.

Examples:

- Classification
- Regression
- Clustering
- Prediction

## 1.3 Deep Learning

Deep Learning uses neural networks with multiple layers to learn complex patterns.

## 1.4 Natural Language Processing

NLP focuses on processing, understanding, and generating human language.

## 1.5 Large Language Models

LLMs are neural networks trained on large text datasets to predict and generate sequences of tokens.

Examples of model families include GPT, Llama, Qwen, and Mistral.

## 1.6 Understand the relationship

Artificial Intelligence
- Machine Learning
  - Deep Learning
    - Natural Language Processing
      - Transformers
        - Large Language Models

## 1.7 Understand how an LLM works

Training text
-> Tokenization
-> Token IDs
-> Embeddings
-> Transformer layers
-> Next-token prediction
-> Loss calculation
-> Backpropagation
-> Weight updates
-> Trained model

## Topics to understand

- Parameters and weights
- Training versus inference
- Training datasets
- Loss functions
- Gradient descent
- Backpropagation
- Pretraining
- Fine-tuning
- Context windows
- Prompting and generation

**Goal:** Explain how an LLM learns to predict the next token.

---

# 2. Python Programming

Python is the primary programming language used throughout this roadmap.

## 2.1 Fundamentals

Learn:

- Variables and data types
- Strings and numbers
- Operators
- Conditional statements
- Loops
- Functions
- Lists
- Tuples
- Dictionaries
- Sets
- List comprehensions

## 2.2 Intermediate Python

Learn:

- Modules and packages
- File handling
- Exception handling
- Classes and objects
- Inheritance
- Iterators and generators
- Decorators
- Type hints
- Virtual environments
- Package management using pip

## 2.3 Development tools

Learn:

- Jupyter Notebook
- Google Colab
- Visual Studio Code
- Git and GitHub
- Linux terminal basics
- Virtual environments

## 2.4 Practice exercises

1. Calculate the mean of a list of numbers.
2. Calculate variance and standard deviation.
3. Read a text file and count word frequencies.
4. Build a text-cleaning function.
5. Create a class representing a neural network layer.
6. Load and process a CSV dataset.

## Project: Text Analyzer

Build a Python program that calculates:

- Total characters
- Total words
- Unique words
- Most frequent words
- Average word length

**Goal:** Write Python programs independently.

---

# 3. Mathematics for Deep Learning

Learn mathematics alongside programming instead of waiting until you have mastered every mathematical topic.

## 3.1 Linear Algebra

Learn:

- Scalars
- Vectors
- Matrices
- Tensors
- Matrix multiplication
- Dot products
- Matrix transpose
- Matrix dimensions and shapes
- Vector norms
- Eigenvalues and eigenvectors

Example:

```python
import numpy as np

x = np.array([1, 2, 3])
w = np.array([0.2, 0.5, 0.3])

result = np.dot(x, w)

print(result)
````

## 3.2 Calculus

Learn:

- Functions
- Derivatives
- Partial derivatives
- Chain rule
- Gradients
- Gradient descent

## 3.3 Probability and Statistics

Learn:

- Mean and median
- Variance and standard deviation
- Probability distributions
- Conditional probability
- Bayes' theorem
- Expected value
- Sampling
- Entropy
- Cross-entropy
- Softmax

## 3.4 Optimization

Learn:

- Loss functions
- Gradient descent
- Stochastic gradient descent
- Mini-batch gradient descent
- Learning rates
- Momentum
- Adam
- AdamW
- Learning-rate schedules

Goal: Understand the mathematics behind neural network training.

# 4. NumPy and Data Processing

## 4.1 NumPy

Learn:

- Array creation
- Indexing and slicing
- Reshaping
- Broadcasting
- Vectorization
- Matrix multiplication
- Random number generation

## 4.2 Pandas

Learn:

- DataFrames
- Reading CSV and JSON
- Missing values
- Filtering and sorting
- Data cleaning
- Grouping and aggregation

## 4.3 Data visualization

Learn Matplotlib and Seaborn.

Practice visualizing:

- Training loss
- Validation loss
- Data distributions
- Model predictions
- Learning curves

## Project

Create a dataset analysis pipeline that loads, cleans, analyzes, and visualizes data.

# 5. Machine Learning Fundamentals

## 5.1 Core concepts

Learn:

- Supervised learning
- Unsupervised learning
- Self-supervised learning
- Classification
- Regression
- Clustering
- Feature engineering
- Training, validation, and test sets
- Overfitting and underfitting
- Regularization
- Data leakage

## 5.2 Algorithms

Understand the basics of:

- Linear regression
- Logistic regression
- Decision trees
- Random forests
- K-means clustering
- Support vector machines
- Gradient boosting

## 5.3 Evaluation metrics

Classification:

- Accuracy
- Precision
- Recall
- F1 score

Regression:

- Mean squared error
- Mean absolute error

Language models:

- Training loss
- Validation loss
- Perplexity

## Project

Build a classification model using scikit-learn.

# 6. Neural Networks from Scratch

This is one of the most important sections.

Before relying on deep learning frameworks, understand how a neural network learns.

## 6.1 What is a neuron?

A simplified neuron calculates:

z = w1x1 + w2x2 + ... + wn\*xn + b

a = activation\_function(z)

Where:

- x represents inputs
- w represents weights
- b represents bias
- a represents the output

## 6.2 Activation functions

Learn:

- Sigmoid
- Tanh
- ReLU
- Leaky ReLU
- GELU
- Softmax

## 6.3 Forward propagation

Forward propagation passes input data through the network to calculate predictions.

## 6.4 Loss functions

A loss function measures the difference between a model's predictions and the target values.

Learn:

- Mean squared error
- Binary cross-entropy
- Categorical cross-entropy
- Next-token cross-entropy

## 6.5 Backpropagation

Backpropagation calculates gradients of the loss with respect to the model parameters.

A gradient-based weight update can be written as:

w\_new = w\_old - learning\_rate \* gradient

## 6.6 Implement a neural network using NumPy

Build these components:

1. Single neuron
2. Perceptron
3. Two-layer neural network
4. Forward propagation
5. Loss calculation
6. Backpropagation
7. Gradient descent
8. Training loop

## Project: XOR Neural Network

Build a neural network that learns the XOR function.

Goal: Understand the internal mechanics of neural networks.

# 7. PyTorch

PyTorch should be your primary deep learning framework for this roadmap.

## 7.1 What is PyTorch?

PyTorch is a deep learning framework that provides tensors, automatic differentiation, neural network layers, optimizers, and GPU support.

## 7.2 Tensors

Learn:

- Tensor creation
- Tensor dimensions
- Tensor shapes
- Data types
- Indexing
- Broadcasting
- Matrix multiplication
- CPU and GPU devices

Example:

python

```
import torch

x = torch.tensor([1.0, 2.0, 3.0])

print(x)
print(x.shape)
print(x.dtype)
```

## 7.3 Automatic differentiation

python

```
import torch

x = torch.tensor(3.0, requires_grad=True)

y = x ** 2

y.backward()

print(x.grad)
```

Understand how PyTorch calculates gradients automatically.

## 7.4 Neural networks using torch.nn

Learn:

- nn.Module
- nn.Linear
- Activation functions
- Loss functions
- Embedding layers
- Dropout
- Layer normalization
- Weight initialization

Example:

python

```
import torch
from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.layers(x)

model = NeuralNetwork()

x = torch.randn(4, 10)
output = model(x)

print(output.shape)
```

## 7.5 Training loop

Understand the purpose of:

- Forward pass
- Loss calculation
- optimizer.zero\_grad()
- loss.backward()
- optimizer.step()
- Epochs
- Mini-batches
- Validation
- Checkpoint saving

## 7.6 Dataset and DataLoader

Learn:

- torch.utils.data.Dataset
- DataLoader
- Batching
- Shuffling
- Custom datasets
- Padding and collation

## 7.7 GPU training

Learn:

- CUDA basics
- Moving tensors to a GPU
- GPU memory
- Mixed-precision training
- Gradient accumulation
- Gradient clipping

## 7.8 Advanced PyTorch

Learn:

- Learning-rate schedulers
- Custom loss functions
- Checkpoint management
- Reproducibility
- Profiling
- Distributed training basics
- torch.compile

## Projects

1. Image classifier
2. Text classifier
3. Neural network trained on your own dataset
4. Tiny language model

# 8. TensorFlow and Keras

TensorFlow is another major deep learning framework. Keras provides a high-level interface for creating neural networks.

## 8.1 Topics

Learn:

- TensorFlow tensors
- Automatic differentiation
- GradientTape
- Keras models
- Sequential API
- Functional API
- Layers
- Optimizers
- Loss functions
- Model training
- Validation
- Saving and loading models
- GPU support

Example:

python

```
import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Input(shape=(10,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(2)
])

model.compile(
    optimizer="adam",
    loss=keras.losses.SparseCategoricalCrossentropy(
        from_logits=True
    ),
    metrics=["accuracy"]
)

model.summary()
```

## 8.2 PyTorch vs. TensorFlow

| Feature | PyTorch | TensorFlow/Keras |
| --- | --- | --- |
| Model construction | torch.nn | Keras layers |
| Automatic gradients | Autograd | GradientTape |
| Training | Explicit loops or utilities | fit() or custom loops |
| Hugging Face integration | Strong | Supported for selected workflows |
| Priority | Learn first | Learn afterward |

Recommendation: Master PyTorch first. Learn TensorFlow when your projects or workplace require it.

# 9. Deep Learning Architectures

## 9.1 Feedforward Neural Networks

Learn how fully connected layers transform input features.

## 9.2 Convolutional Neural Networks

Learn:

- Convolutions
- Filters
- Kernels
- Pooling
- Feature maps

## 9.3 Recurrent Neural Networks

Learn:

- RNNs
- Vanishing gradients
- LSTMs
- GRUs

## 9.4 Attention

Learn:

- Query
- Key
- Value
- Scaled dot-product attention
- Self-attention
- Multi-head attention
- Attention masks

## 9.5 Transformers

Learn:

- Encoder architecture
- Decoder architecture
- Encoder-decoder architecture
- Positional embeddings
- Residual connections
- Layer normalization
- Feedforward layers
- Causal attention masks

Goal: Understand why Transformers are widely used for language modeling.

# 10. Natural Language Processing

## 10.1 Text preprocessing

Learn:

- Text normalization
- Unicode
- Punctuation
- Sentence segmentation
- Dataset cleaning
- Duplicate removal
- Train-validation-test splits

Avoid removing punctuation or stop words indiscriminately from generative training data.

## 10.2 Text representations

Learn:

- One-hot encoding
- Bag of words
- TF-IDF
- Word embeddings
- Contextual embeddings

## 10.3 Language modeling

Understand:

- Probability of a sequence
- Autoregressive prediction
- Next-token prediction
- Conditional probability
- Teacher forcing
- Sampling during generation

A language model estimates:

P(x1, x2, ..., xn) = product of P(xt | x1, ..., x(t-1))

Goal: Understand how a model predicts the next token using preceding context.

# 11. Tokenization and Embeddings

## 11.1 Tokenization

Tokenization converts text into smaller units called tokens.

Example:

Text: "I am learning language models."

Tokens: \["I", "am", "learning", "language", "models", "."\]

These tokens are then converted into integer IDs using a tokenizer's vocabulary.

## 11.2 Tokenization algorithms

Learn:

- Character-level tokenization
- Word-level tokenization
- Byte Pair Encoding
- WordPiece
- Unigram tokenization
- Special tokens
- Padding
- Truncation
- Attention masks

## 11.3 Embeddings

Learn:

- Embedding matrices
- Token embeddings
- Positional embeddings
- Embedding dimensions
- Contextual representations

## Project

Train a tokenizer on a small text corpus and inspect its output.

# 12. Transformer Architecture

## 12.1 Decoder-only Transformer

A simplified architecture:

Input token IDs -\> Token embeddings -\> Positional information -\> Causal self-attention -\> Feedforward network -\> Residual connections and normalization -\> Transformer blocks -\> Vocabulary projection -\> Next-token logits

## 12.2 Attention mathematics

Scaled dot-product attention is:

Attention(Q, K, V) = softmax(QK^T / sqrt(d\_k))V

Where:

- Q = Queries
- K = Keys
- V = Values
- d\_k = Key dimension

Understand how attention combines information from different positions in a sequence.

## 12.3 Important concepts

Learn:

- Causal masking
- Multi-head attention
- Feedforward blocks
- Residual connections
- Layer normalization
- RMSNorm
- Positional encodings
- Rotary positional embeddings
- Vocabulary logits
- Cross-entropy loss
- KV caching
- Context length
- Autoregressive generation

## 12.4 Implementation exercises

1. Build token embeddings.
2. Implement positional representations.
3. Implement causal self-attention.
4. Implement multi-head attention.
5. Build a feedforward network.
6. Implement a Transformer block.
7. Add a language-model output layer.
8. Train on next-token prediction.
9. Generate text.

Goal: Implement a small decoder-only Transformer.

# 13. Hugging Face Ecosystem

Hugging Face provides tools for pretrained models, datasets, tokenizers, training, and model sharing.

## 13.1 Important libraries

### Transformers

Load pretrained models, tokenizers, configurations, and training utilities.

### Datasets

Load and process datasets.

### Tokenizers

Train and use efficient tokenizers.

### Accelerate

Simplify training across supported hardware configurations.

### PEFT

Perform parameter-efficient fine-tuning using methods such as LoRA.

### TRL

Support language-model training and post-training workflows.

### Hugging Face Hub

Discover and share models, datasets, and other artifacts.

## 13.2 Create an account

Visit:

[https://huggingface.co/](<https://huggingface.co/?utm_source=chatgpt.com>)

Then:

1. Create an account.
2. Explore the Models section.
3. Read model cards.
4. Review model licenses.
5. Explore datasets.
6. Learn how repositories work.

## 13.3 Install libraries

bash

```
pip install torch transformers datasets tokenizers accelerate
```

For fine-tuning:

bash

```
pip install peft trl
```

Install optional hardware-related dependencies according to your environment.

## 13.4 Load a pretrained model

python

```
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen2.5-0.5B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

prompt = "Artificial intelligence is"

inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

This uses a pretrained model; it does not train a new model.

Check the model card, license, hardware requirements, and current library compatibility before use.

Project: Build a command-line text generator.

# 14. Build a GPT-Style Language Model from Scratch

## 14.1 Define the objective

Build a small decoder-only Transformer that predicts the next token.

## 14.2 Prepare the dataset

Tasks:

- Collect or download a legally usable corpus.
- Clean corrupted records.
- Remove unwanted duplicates.
- Split the data.
- Train or select a tokenizer.
- Convert text into token IDs.
- Create fixed-length training sequences.

## 14.3 Create training examples

Example:

Input: \[A, B, C, D\]

Target: \[B, C, D, E\]

The model predicts the next token using the preceding context.

## 14.4 Implement the model

Build:

- Token embeddings
- Positional representations
- Transformer blocks
- Causal self-attention
- Feedforward layers
- Normalization
- Vocabulary projection
- Cross-entropy loss

## 14.5 Train the model

Implement:

- Batching
- Forward passes
- Loss calculation
- Backpropagation
- Optimizer updates
- Validation
- Checkpoint saving
- Learning-rate scheduling

## 14.6 Generate text

Implement:

- Greedy decoding
- Temperature
- Top-k sampling
- Top-p sampling
- End-of-sequence handling

## 14.7 Evaluate the model

Measure:

- Validation loss
- Perplexity
- Text coherence
- Repetition
- Memorization risks

Project: Train a tiny GPT-style model and generate text from it.

# 15. LLM Training and Fine-Tuning

## 15.1 Pretraining from scratch

Pretraining initializes a model and trains it on a large text corpus.

You need:

- Model architecture
- Tokenizer
- Training dataset
- Compute resources
- Optimizer
- Learning-rate scheduler
- Checkpointing
- Evaluation pipeline

Training a capable general-purpose LLM requires substantial data and compute. Start with small educational models.

## 15.2 Fine-tuning

Fine-tuning adapts an existing pretrained model to a task or domain.

Examples:

- Customer support
- Document question answering
- Code assistance
- Domain-specific writing
- Information extraction

## 15.3 Supervised fine-tuning

Prepare examples containing prompts and target responses.

Example:

json

```
{
  "messages": [
    {
      "role": "user",
      "content": "What is a neural network?"
    },
    {
      "role": "assistant",
      "content": "A neural network is a model that learns patterns from data."
    }
  ]
}
```

Learn:

- Dataset formatting
- Chat templates
- Training masks
- Padding
- Truncation
- Validation splits
- Loss calculation
- Checkpointing
- Evaluation

## 15.4 Fine-tune using Hugging Face

Learn:

- Model loading
- Tokenizer loading
- Dataset preparation
- Trainer API
- Training arguments
- Batch sizes
- Gradient accumulation
- Learning rates
- Evaluation
- Saving and loading models

Project: Fine-tune a small pretrained model on a domain-specific dataset.

# 16. LoRA and QLoRA

## 16.1 LoRA

Low-Rank Adaptation adds trainable low-rank updates to selected model weights.

Learn:

- Low-rank matrices
- Target modules
- Rank
- Alpha
- Dropout
- Adapter saving
- Adapter loading
- Adapter merging

## 16.2 QLoRA

QLoRA combines a quantized base model with trainable LoRA adapters.

Learn:

- Quantization
- Four-bit loading
- NF4
- Compute data types
- GPU memory
- Hardware compatibility

## Project

Fine-tune a small instruction model using LoRA or QLoRA.

Compare the original and adapted models on the same test prompts.

# 17. Retrieval-Augmented Generation (RAG)

RAG combines an LLM with a retrieval system that supplies relevant external information.

It is often a good choice for applications that need to answer questions about private documents or frequently changing information.

## 17.1 RAG architecture

Documents -\> Text extraction -\> Text cleaning -\> Chunking -\> Embedding generation -\> Vector index -\> User question -\> Retrieval -\> Context construction -\> LLM generation -\> Answer with sources

## 17.2 Topics to learn

- Document ingestion
- Chunking strategies
- Embedding models
- Vector similarity
- Vector databases
- Metadata filtering
- Retrieval evaluation
- Reranking
- Context construction
- Citation generation
- Hallucination evaluation

## 17.3 Tools

- Sentence Transformers
- FAISS
- Chroma
- Other vector databases
- Hugging Face embedding models

## Project: PDF Question-Answering System

Build an application that:

1. Reads PDF documents.
2. Extracts text.
3. Splits text into chunks.
4. Generates embeddings.
5. Retrieves relevant chunks.
6. Supplies the context to an LLM.
7. Generates an answer.
8. Displays the sources.

# 18. LLM Evaluation and Optimization

## 18.1 Evaluation

Learn:

- Training loss
- Validation loss
- Perplexity
- Held-out datasets
- Benchmark limitations
- Data contamination

## 18.2 Instruction-following evaluation

Measure:

- Correctness
- Relevance
- Completeness
- Instruction adherence
- Safety
- Format compliance
- Consistency

## 18.3 Optimization

Learn:

- Quantization
- Mixed precision
- KV caching
- Batching
- Efficient attention
- Memory management
- Model compilation
- Latency measurement
- Throughput measurement

## 18.4 Experiment tracking

Record:

- Model version
- Dataset version
- Hyperparameters
- Training steps
- Evaluation results
- Hardware
- Checkpoint identifiers

Project: Compare a pretrained model, a fine-tuned model, and a RAG application.

# 19. LLM Deployment

## 19.1 Deployment options

Learn:

- Local inference
- Hugging Face Spaces
- Hugging Face Inference Providers
- FastAPI
- Docker
- Cloud GPU instances
- Compatible inference servers

## 19.2 Build an API

Learn:

- HTTP
- REST APIs
- Request and response formats
- Input validation
- Error handling
- Authentication
- Rate limiting
- Logging
- Streaming responses

## 19.3 Build a user interface

Explore Gradio or another suitable interface framework.

Include:

- Text input
- Generation settings
- Output display
- Error messages
- Optional streaming

## 19.4 Production considerations

- GPU memory
- Concurrent requests
- Cold starts
- Latency
- Token limits
- Cost monitoring
- Privacy
- Security
- Model licensing
- Monitoring

Project: Deploy a fine-tuned model with a web interface.

# 20. GPU and Cloud Computing

## 20.1 Learn the basics

- CPU versus GPU
- GPU memory
- CUDA
- Tensor operations
- Batch size
- Mixed precision
- Quantization
- Memory bottlenecks
- Training versus inference

## 20.2 Learning environments

- Google Colab
- Kaggle Notebooks
- Local CPU
- Local NVIDIA GPU
- Cloud GPU providers

Check current hardware availability, pricing, and session limits before starting a training run.

## 20.3 Model memory

A simple approximation for FP32 model weights is:

Weight memory = Number of parameters × 4 bytes

For example, one billion FP32 parameters require approximately 4 GB for weights alone.

Training requires additional memory for gradients, activations, optimizer states, and temporary tensors.

# 21. Practical Projects

## Beginner

- [ ]Python text analyzer
- [ ]Dataset cleaner
- [ ]Word-frequency analyzer
- [ ]Data visualization project

## Intermediate

- [ ]Linear regression from scratch
- [ ]XOR neural network
- [ ]Image classifier
- [ ]Text classifier
- [ ]PyTorch training pipeline

## Advanced

- [ ]Custom tokenizer
- [ ]Self-attention implementation
- [ ]Transformer block
- [ ]Tiny GPT-style model
- [ ]Pretrained Hugging Face text generator
- [ ]Fine-tuned domain-specific model
- [ ]LoRA fine-tuning project
- [ ]PDF RAG chatbot
- [ ]LLM evaluation pipeline
- [ ]Deployed LLM application

# 22. Learning Resources

## Python

- Python tutorial: [https://docs.python.org/3/tutorial/](<https://docs.python.org/3/tutorial/?utm_source=chatgpt.com>)
- NumPy: [https://numpy.org/doc/stable/](<https://numpy.org/doc/stable/?utm_source=chatgpt.com>)
- Pandas: [https://pandas.pydata.org/docs/](<https://pandas.pydata.org/docs/?utm_source=chatgpt.com>)

## Mathematics and Machine Learning

- scikit-learn: [https://scikit-learn.org/stable/](<https://scikit-learn.org/stable/?utm_source=chatgpt.com>)
- 3Blue1Brown Linear Algebra: [https://www.3blue1brown.com/topics/linear-algebra](<https://www.3blue1brown.com/topics/linear-algebra?utm_source=chatgpt.com>)
- 3Blue1Brown Neural Networks: [https://www.3blue1brown.com/topics/neural-networks](<https://www.3blue1brown.com/topics/neural-networks?utm_source=chatgpt.com>)

## PyTorch

- Official tutorials: [https://pytorch.org/tutorials/](<https://pytorch.org/tutorials/?utm_source=chatgpt.com>)
- PyTorch basics: [https://pytorch.org/tutorials/beginner/basics/intro.html](<https://pytorch.org/tutorials/beginner/basics/intro.html?utm_source=chatgpt.com>)

## TensorFlow

- Tutorials: [https://www.tensorflow.org/tutorials](<https://www.tensorflow.org/tutorials?utm_source=chatgpt.com>)
- Keras guides: [https://keras.io/guides/](<https://keras.io/guides/?utm_source=chatgpt.com>)

## Hugging Face

- LLM course: [https://huggingface.co/learn/llm-course/](<https://huggingface.co/learn/llm-course/?utm_source=chatgpt.com>)
- Transformers: [https://huggingface.co/docs/transformers/](<https://huggingface.co/docs/transformers/?utm_source=chatgpt.com>)
- Datasets: [https://huggingface.co/docs/datasets/](<https://huggingface.co/docs/datasets/?utm_source=chatgpt.com>)
- Tokenizers: [https://huggingface.co/docs/tokenizers/](<https://huggingface.co/docs/tokenizers/?utm_source=chatgpt.com>)
- PEFT: [https://huggingface.co/docs/peft/](<https://huggingface.co/docs/peft/?utm_source=chatgpt.com>)
- TRL: [https://huggingface.co/docs/trl/](<https://huggingface.co/docs/trl/?utm_source=chatgpt.com>)
- Accelerate: [https://huggingface.co/docs/accelerate/](<https://huggingface.co/docs/accelerate/?utm_source=chatgpt.com>)

## Transformers and GPT

- Illustrated Transformer: [https://jalammar.github.io/illustrated-transformer/](<https://jalammar.github.io/illustrated-transformer/?utm_source=chatgpt.com>)
- Attention Is All You Need: [https://arxiv.org/abs/1706.03762](<https://arxiv.org/abs/1706.03762?utm_source=chatgpt.com>)
- nanoGPT: [https://github.com/karpathy/nanoGPT](<https://github.com/karpathy/nanoGPT?utm_source=chatgpt.com>)

# 23. Suggested 24-Week Learning Plan

| Weeks | Topic | Deliverable |
| --- | --- | --- |
| 1–2 | Python | Text analyzer |
| 3–4 | Mathematics and NumPy | Matrix exercises |
| 5–6 | Machine learning | Classification model |
| 7–8 | Neural networks | XOR network |
| 9–11 | PyTorch | Trained neural network |
| 12 | TensorFlow and Keras | Basic Keras model |
| 13–14 | NLP and tokenization | Text preprocessing pipeline |
| 15–16 | Transformers | Tiny Transformer |
| 17–18 | Hugging Face | Pretrained model application |
| 19–20 | Language-model training | Tiny GPT-style model |
| 21–22 | Fine-tuning | LoRA-adapted model |
| 23 | RAG and evaluation | Document Q&A application |
| 24 | Deployment | Working demo |

This is a suggested schedule. Take longer where necessary.

# 24. Final Checklist

## Programming and Mathematics

- [ ]Write Python programs independently.
- [ ]Work with NumPy arrays and matrices.
- [ ]Understand gradients and the chain rule.
- [ ]Understand loss functions and gradient descent.

## Neural Networks and PyTorch

- [ ]Explain forward propagation.
- [ ]Explain backpropagation.
- [ ]Build a neural network with nn.Module.
- [ ]Write a training loop.
- [ ]Use datasets and batches.
- [ ]Save and load model checkpoints.

## Transformers

- [ ]Explain tokenization.
- [ ]Explain embeddings.
- [ ]Explain queries, keys, and values.
- [ ]Implement causal self-attention.
- [ ]Explain next-token prediction.
- [ ]Train a tiny language model.

## Hugging Face and LLMs

- [ ]Load a pretrained model.
- [ ]Generate text.
- [ ]Prepare a training dataset.
- [ ]Fine-tune a small model.
- [ ]Evaluate model quality.
- [ ]Build a RAG application.
- [ ]Deploy an LLM application.
- [ ]Document the project.

# Final Recommendation

Follow this learning sequence:

Python -\> Mathematics -\> NumPy -\> Machine Learning -\> Neural Networks from Scratch -\> PyTorch -\> NLP -\> Tokenization -\> Transformers -\> Hugging Face -\> Tiny GPT from Scratch -\> Fine-Tuning -\> LoRA and QLoRA -\> RAG -\> Evaluation -\> Deployment

Learn PyTorch deeply before spending significant time on TensorFlow.

Distinguish between three different goals:

1. Using an LLM: Run a pretrained model.
2. Customizing an LLM: Fine-tune a pretrained model or connect it to your documents.
3. Training an LLM from scratch: Initialize a model and train it on a text corpus.

Start with small models that you can understand completely. Increase the model size and complexity as your programming, mathematical, and hardware skills improve.

First milestone: Complete Python fundamentals, implement a neural network in NumPy, and then implement the same network in PyTorch. Next, learn tokenization and build a tiny GPT-style model.
