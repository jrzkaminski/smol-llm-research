# ToolSandbox: A Stateful, Conversational, Interactive Evaluation Benchmark for LLM Tool Use Capabilities

---

> [!NOTE]
> https://arxiv.org/pdf/2408.04682

> [!TIP]
> https://github.com/apple/ToolSandbox

> [!IMPORTANT]
> TL;DR: ToolSandbox is a benchmark for evaluating LLMs' tool-use capabilities in stateful, conversational, and interactive contexts. It is distinct because it integrates state dependencies, dynamic milestones, and minefields, providing deeper insights into intermediate and final model performance. The benchmark identifies key challenges, like reasoning over tool dependencies and handling insufficient information, where even leading models show limitations.


## Purpose and Design
TOOLSANDBOX is an evaluation benchmark designed to test the tool-use capabilities of large language models (LLMs) in stateful, conversational, and interactive environments. Unlike previous benchmarks, it integrates state dependencies, conversational dynamics, and interactive tasks to better reflect real-world challenges. It uses Python-based tools and an execution environment to simulate complex tool interactions.

## Key Features
1. **Stateful Tools**:
   - Evaluates how LLMs interact with tools that require understanding and manipulating implicit world states.
   - Includes dependencies between tools, where successful tool usage may depend on changes to the world state (e.g., enabling WiFi to access the internet).
   
2. **Conversational and On-Policy Evaluation**:
   - Utilizes a simulated user for realistic interactions and testing the LLM's ability to manage dialog history and state tracking.
   - Tests multi-turn dialog capabilities rather than static or pre-defined trajectories.

3. **Interactive Evaluation**:
   - Benchmarks include dynamic milestones (critical task completion events) and minefields (prohibited actions or outcomes) to assess both intermediate and final model performance.

4. **Tool Design and Usage**:
   - A diverse set of Python-based and wrapped RapidAPI tools to represent realistic, complex scenarios.
   - Augmented tools with distractions, scrambled names, and reduced descriptions to challenge LLM reasoning.

5. **Evaluation Metrics**:
   - Combines trajectory-based analysis with flexible similarity measures for intermediate and final outcomes.
   - Evaluates on dimensions like state dependencies, canonicalization, and insufficient information handling.

## Benchmark Characteristics
- **Scenarios**:
   - Includes 1,032 test cases designed to probe various tool-use challenges, such as single/multiple tool calls, user turns, and state-dependent scenarios.
   - Incorporates tasks requiring canonicalization of input data and reasoning over missing information.

- **Tool Properties**:
   - Each tool is documented with a schema for clear usage but may be deliberately obfuscated to test LLM adaptability.
   - Stateful scenarios mimic real-world environments with dependencies like network connectivity and service availability.

- **Performance Insights**:
   - Demonstrates significant performance gaps between open-source and proprietary models.
   - Highlights challenges in areas like state dependency and canonicalization, where even state-of-the-art (SOTA) models struggle.

## **Key Findings**
- Proprietary models like GPT-4 outperform open-source models but still encounter difficulties with nested state dependencies and tool selection efficiency.
- Models often hallucinate tools or arguments in insufficient information scenarios, showcasing gaps in reasoning capabilities.
