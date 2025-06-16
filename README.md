# 🤖 LangChain Agent with Custom Tool & In-Memory Cache – Coursera Project

> **Built with the help of my personal AI agent, _Jules_, as part of the [LangChain for LLM Application Development (Coursera)](https://www.coursera.org/projects/langchain-for-llm-application-development) project.**

---

## 🧠 Project Overview

This repository demonstrates a simple yet functional **LangChain-based agent** implementation. It includes:

- A **custom tool** for string length calculation
- A **ReAct-style reasoning agent**
- And an **in-memory caching mechanism** to optimize repeated LLM calls.

Everything was designed and developed in collaboration with **Jules**, my personalized AI coding assistant.

---

## 🚀 Key Features

### 🧠 ReAct-based LangChain Agent
- Uses a **ReAct prompt** to reason step-by-step before using tools.
- Can autonomously decide which tool to use based on the query.

### 🛠️ Custom Tool: `StringLengthTool`
- Dynamically computes the **length of a given string**.
- Accepts **natural language queries**, even with varying phrasing or casing.

### ⚡ In-Memory Caching
- Implements `InMemoryCache` from LangChain to **store previous LLM responses**.
- Speeds up repeated queries and **minimizes redundant API calls**.

### 🧪 Built-in Testing
- Includes **3 basic tests** to validate:
  1. Tool functionality
  2. Natural language interpretation
  3. Caching effectiveness (performance & response reuse)

---

## 📁 Project Structure

- `string_length_tool_func`: the main function behind the custom tool.
- `Tool`: wraps the function with a name and description usable by the agent.
- `AgentExecutor`: runs the agent using the provided tools and model.
- `run_tests()`: runs predefined unit tests to check core functionality.

---

## ⚙️ Requirements

- Python 3.9 or higher
- OpenAI API Key set as an environment variable:

```bash
export OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx
