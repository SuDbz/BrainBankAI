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
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolExecutor, tools_to_execute_tools
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
import operator

# Define custom tools
@tool
def search_database(query: str) -> str:
    """Search the company database for information"""
    # In a real application, this would query a database
    return f"Found results for: {query}"

@tool
def summarize_text(text: str) -> str:
    """Summarize the provided text"""
    # This could use a dedicated summarization model
    return f"Summary of: {text}"

# Initialize LLM
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Define the state schema for our workflow
class State:
    messages: list
    next_step: str = START

# Define node functions
def router(state):
    """Determines which step to execute next"""
    messages = state["messages"]
    response = llm.invoke(
        [
            HumanMessage(content="Based on the conversation, what should I do next? Options: [search, summarize, finish]"),
            *messages
        ]
    )
    
    # Extract decision from response
    decision = response.content.strip().lower()
    
    if "search" in decision:
        return "search"
    elif "summarize" in decision:
        return "summarize"
    else:
        return "finish"

def search_step(state):
    """Performs database search based on the conversation"""
    messages = state["messages"]
    response = llm.invoke(
        [
            HumanMessage(content="Extract the search query from the conversation"),
            *messages
        ]
    )
    
    # Execute search tool
    search_query = response.content.strip()
    tools = [search_database]
    tool_executor = ToolExecutor(tools)
    tool_response = tool_executor.invoke({"name": "search_database", "arguments": {"query": search_query}})
    
    # Add results to messages
    return {
        "messages": messages + [AIMessage(content=f"Search results: {tool_response}")]
    }

def summarize_step(state):
    """Summarizes information gathered so far"""
    messages = state["messages"]
    # Collect all message content to summarize
    all_content = " ".join([m.content for m in messages])
    
    # Execute summarize tool
    tools = [summarize_text]
    tool_executor = ToolExecutor(tools)
    summary = tool_executor.invoke({"name": "summarize_text", "arguments": {"text": all_content}})
    
    # Add summary to messages
    return {
        "messages": messages + [AIMessage(content=f"Summary: {summary}")]
    }

def final_step(state):
    """Generate final response to user"""
    messages = state["messages"]
    response = llm.invoke(
        [
            HumanMessage(content="Synthesize all information and provide a final helpful response"),
            *messages
        ]
    )
    
    return {
        "messages": messages + [response]
    }

# Build the graph
workflow = StateGraph(State)

# Add nodes
workflow.add_node("router", router)
workflow.add_node("search", search_step)
workflow.add_node("summarize", summarize_step)
workflow.add_node("finish", final_step)

# Add edges
workflow.add_edge(START, "router")
workflow.add_conditional_edges(
    "router",
    {
        "search": (lambda state: state["next_step"] == "search", "search"),
        "summarize": (lambda state: state["next_step"] == "summarize", "summarize"),
        "finish": (lambda state: state["next_step"] == "finish", "finish")
    }
)
workflow.add_edge("search", "router")
workflow.add_edge("summarize", "router")
workflow.add_edge("finish", END)

# Compile the graph
app = workflow.compile()

# Run the workflow
inputs = {
    "messages": [HumanMessage(content="I need information about our customer database structure")]
}
result = app.invoke(inputs)

# Print the final messages
for message in result["messages"]:
    if isinstance(message, AIMessage):
        print(f"AI: {message.content}")
    elif isinstance(message, HumanMessage):
        print(f"Human: {message.content}")
```

---

## LangChain
LangChain is a popular Python library for building LLM-powered applications. It provides a framework for developing applications that combine LLMs with other components for more complex, context-aware, and interactive AI systems.

### Python Example: LangChain Pipeline

```python
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.tools import DuckDuckGoSearchRun
from langchain.memory import ConversationBufferMemory

# Initialize LLM
llm = Ollama(model="llama2")

# Create tools for the agent
search_tool = DuckDuckGoSearchRun()
tools = [
    Tool(
        name="Search",
        func=search_tool.run,
        description="Useful for searching information on the internet"
    )
]

# Create prompt templates for different stages
research_prompt = PromptTemplate(
    template="Research the following topic thoroughly: {topic}",
    input_variables=["topic"]
)

analysis_prompt = PromptTemplate(
    template="Analyze the following research data and identify key points:\n{research_data}",
    input_variables=["research_data"]
)

summary_prompt = PromptTemplate(
    template="Create a concise summary based on these key points:\n{key_points}",
    input_variables=["key_points"]
)

# Create individual chains
research_chain = LLMChain(llm=llm, prompt=research_prompt, output_key="research_data")
analysis_chain = LLMChain(llm=llm, prompt=analysis_prompt, output_key="key_points")
summary_chain = LLMChain(llm=llm, prompt=summary_prompt, output_key="summary")

# Chain the chains together for sequential processing
sequential_chain = SimpleSequentialChain(
    chains=[research_chain, analysis_chain, summary_chain],
    verbose=True
)

# Create an agent with memory for interactive tasks
memory = ConversationBufferMemory(memory_key="chat_history")
agent_prompt = PromptTemplate.from_template(
    """You are a helpful AI assistant. Use the following tools to help answer questions:
    {tools}
    
    Use the following format:
    Question: the input question
    Thought: your reasoning about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    
    Begin!
    
    Question: {

---

## References
- [OpenAI API Docs](https://platform.openai.com/docs)
- [Ollama](https://ollama.com/)
- [LangChain](https://python.langchain.com/)
- [LangGraph](https://langgraph.com/)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)

---
