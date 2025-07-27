# Large Language Models (LLMs) Overview

## Table of Contents
- [Introduction](#introduction)
- [Popular LLMs](#popular-llms)
  - [GPT (OpenAI)](#gpt-openai)
  - [Gemini (Google)](#gemini-google)
  - [Claude (Anthropic)](#claude-anthropic)
  - [Other Notable LLMs](#other-notable-llms)
- [LLM Comparison Table](#llm-comparison-table)
- [Training Data Sources](#training-data-sources)
- [Use Cases](#use-cases)
- [How to Use LLMs](#how-to-use-llms)
- [Prompting Basics](#prompting-basics)
  - [Prompt Engineering Tips](#prompt-engineering-tips)
  - [Prompt Examples](#prompt-examples)
- [References](#references)

---

## Introduction
Large Language Models (LLMs) are advanced AI systems trained on massive text datasets to understand and generate human-like language. They power chatbots, code assistants, search engines, and more.

---

## Popular LLMs

### GPT (OpenAI)
- **Models:** GPT-3, GPT-3.5, GPT-4
- **Training Data:** Web pages, books, Wikipedia, code, forums (up to Apr 2023 for GPT-4)
- **Use Cases:** Chatbots, code generation, summarization, translation, creative writing
- **How to Use:**
  - OpenAI API: [https://platform.openai.com](https://platform.openai.com)
  - Python example:
    ```python
    import openai
    openai.api_key = "YOUR_API_KEY"
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Tell me a joke."}]
    )
    print(response.choices[0].message.content)
    ```

### Gemini (Google)
- **Models:** Gemini Pro, Gemini Ultra
- **Training Data:** Web pages, books, code, images, videos (up to early 2024)
- **Use Cases:** Multimodal tasks (text, image, video), search, code, chat
- **How to Use:**
  - Google AI Studio: [https://aistudio.google.com](https://aistudio.google.com)
  - API (coming soon)
  - Example prompt:
    > "Summarize the main points of this article: [URL]"

### Claude (Anthropic)
- **Models:** Claude 2, Claude 3 (Haiku, Sonnet, Opus)
- **Training Data:** Web pages, books, code, Q&A, Wikipedia (up to early 2024)
- **Use Cases:** Safe chatbots, summarization, code, research, enterprise AI
- **How to Use:**
  - Claude API: [https://console.anthropic.com](https://console.anthropic.com)
  - Python example:
    ```python
    import anthropic
    client = anthropic.Client("YOUR_API_KEY")
    response = client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=100,
        messages=[{"role": "user", "content": "Explain quantum computing."}]
    )
    print(response.content[0].text)
    ```

### Other Notable LLMs
- **Llama (Meta):** Open-source, trained on web and books, used for research and private deployments
- **Mistral:** Open-source, focused on efficiency and performance
- **PaLM (Google):** Predecessor to Gemini, strong in reasoning and code
- **Cohere:** Enterprise-focused, strong in retrieval and search

---

## LLM Comparison Table

| Model      | Provider    | Latest Version | Training Data Cutoff | Multimodal | API Available | Typical Use Cases           |
|------------|-------------|----------------|----------------------|------------|---------------|----------------------------|
| GPT-4      | OpenAI      | GPT-4o         | Apr 2023             | Yes        | Yes           | Chat, code, writing        |
| Gemini     | Google      | Ultra          | Early 2024           | Yes        | Soon          | Text, image, video, code   |
| Claude 3   | Anthropic   | Opus           | Early 2024           | Yes        | Yes           | Safe chat, code, research  |
| Llama 3    | Meta        | Llama 3        | Mar 2024             | No         | No            | Research, private deploys  |
| Mistral    | Mistral AI  | Mistral Large  | 2023                 | No         | Yes           | Open-source, efficiency    |
| PaLM 2     | Google      | PaLM 2         | 2023                 | Yes        | No            | Reasoning, code            |
| Cohere     | Cohere      | Command R      | 2023                 | No         | Yes           | Enterprise, retrieval      |

---

## Training Data Sources
- **Web pages:** Common Crawl, Wikipedia, news, blogs
- **Books:** Public domain and licensed books
- **Code:** GitHub, StackOverflow, open-source repositories
- **Q&A:** Forums, help sites
- **Images/Videos:** (for multimodal models) YouTube, image datasets

---

## Use Cases
- **Chatbots & Virtual Assistants**
- **Code Generation & Review**
- **Text Summarization**
- **Translation**
- **Search & Retrieval**
- **Content Creation**
- **Enterprise Automation**
- **Multimodal Tasks (text, image, video)**

---

## How to Use LLMs
- **Cloud APIs:** Most providers offer APIs (OpenAI, Anthropic, Cohere)
- **Open-source Models:** Download and run locally (Meta Llama, Mistral)
- **Web Interfaces:** ChatGPT, Claude Console, Google AI Studio
- **Integrations:** VS Code extensions, Slack bots, custom apps

---

## Prompting Basics

Prompting is the art of instructing LLMs to get the desired output. Good prompts are clear, specific, and provide context.

### Prompt Engineering Tips
- **Be explicit:** State exactly what you want
- **Provide context:** Give background or examples
- **Use step-by-step instructions:** For complex tasks
- **Specify format:** Ask for lists, tables, or code
- **Iterate:** Refine prompts based on output

### Prompt Examples
- **Simple Q&A:**
    > "What are the benefits of solar energy?"
- **Summarization:**
    > "Summarize the following text in 3 bullet points: ..."
- **Code Generation:**
    > "Write a Python function to reverse a string."
- **Table Output:**
    > "List the top 5 programming languages in a markdown table."
- **Role Play:**
    > "Act as a helpful tutor. Explain calculus to a beginner."

---

## References
- [OpenAI API Docs](https://platform.openai.com/docs)
- [Anthropic Claude API](https://docs.anthropic.com/claude/reference)
- [Google Gemini](https://aistudio.google.com)
- [Meta Llama](https://ai.meta.com/llama/)
- [Mistral AI](https://mistral.ai/)
- [Cohere API](https://docs.cohere.com/docs)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)

---

*This page provides a practical overview of the most popular LLMs, their capabilities, and how to use them effectively for a wide range of tasks.*
