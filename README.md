# Graph-Based Automated Technical Blog Writer Agent

An autonomous, multi-agent pipeline built with **LangGraph**, **ChatOllama (Qwen 2.5:3b)**, and **Tavily Search**, designed to dynamically research, plan, write, and illustrate professional technical blog posts locally.

## Architectural Overview

The system uses an **Orchestrator-Worker-Reducer (Fan-out/Fan-in)** graph pattern to scale content generation and maintain high structural consistency:


## Key Technical Features

* **Advanced Agentic Routing:** Dynamically evaluates topics via a router node to select between `closed_book`, `hybrid`, or `open_book` research modes.
* **Resilient Web Search Integration:** Leverages **Tavily** to fetch high-signal evidence for up-to-date technical articles.
* **Parallel Worker Fan-Out:** Uses LangGraph's `Send` API to parallelize section generation, ensuring unique objectives and word counts per section.
* **Automated Asset Generation:** Integrates free **Pollinations.ai API** pipelines to plan, prompt, fetch, and embed custom vector/concept diagrams directly into Markdown outputs.
* **100% Local LLM Execution:** Runs lightweight, robust open-source models (`qwen2.5:3b`) locally via Ollama and LangChain.