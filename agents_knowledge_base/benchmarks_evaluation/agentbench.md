# AgentBench: Evaluating LLMs as Agents

---

> [!NOTE]
> https://openreview.net/pdf?id=zAdUB0aCTQ

> [!TIP]
> https://github.com/THUDM/AgentBench

> [!IMPORTANT]
> TL;DR: AGENTBENCH is a pioneering benchmark for evaluating Large Language Models (LLMs) as agents across 8 diverse environments representing code-grounded, game-grounded, and web-grounded tasks. Unlike previous benchmarks, AGENTBENCH uniquely integrates real-world challenges like database management, digital card games, and web browsing into a unified framework to test reasoning, decision-making, and instruction-following abilities. It evaluates 29 LLMs (both commercial and open-source) and reveals significant gaps between top-performing models like GPT-4 and open-source alternatives. Distinct features include multi-turn interaction, toolkits for modular evaluation, and an emphasis on real-world usability of LLMs.

The paper introduces **AGENTBENCH**, a benchmark designed to evaluate the reasoning and decision-making capabilities of Large Language Models (LLMs) as agents in interactive environments. The benchmark spans eight distinct environments across three categories: **code-grounded**, **game-grounded**, and **web-grounded** tasks. It aims to address the lack of standardized evaluation methods for LLMs functioning as autonomous agents.

## Key Contributions:

1. **Diverse Environments for Evaluation:**
   - Code-grounded tasks like operating system interaction, database management, and knowledge graph queries.
   - Game-grounded challenges including digital card games, lateral thinking puzzles, and household simulations.
   - Web-grounded scenarios like web shopping and browsing.

2. **Comprehensive Evaluation of LLMs:**
   - AGENTBENCH evaluates 29 LLMs, including both API-based commercial models (e.g., GPT-4, Claude) and open-source models (e.g., LLaMA, Vicuna).
   - Performance metrics such as success rate (SR), reward, and F1 score highlight significant gaps between commercial and open-source LLMs.

3. **Insights and Challenges:**
   - Commercial models like GPT-4 exhibit strong performance in most environments, but open-source models lag significantly, emphasizing the need for further development.
   - Failures often stem from poor instruction following, limited long-term reasoning, and decision-making capabilities.
   - Training on high-quality alignment data and careful code-specific tuning can improve agent performance, but overemphasis on code can sometimes harm general reasoning.

4. **Toolkit for Evaluation:**
   - A plug-and-play evaluation framework facilitates modular testing across various environments using HTTP APIs, ensuring scalability and reproducibility.

5. **Major Findings:**
   - A substantial performance gap exists between top-tier commercial models and their open-source counterparts, with GPT-4 leading in most tasks.
   - Open-source models show uneven performance, often excelling in specific scenarios but failing in others due to resource and training limitations.
   - Training strategies, such as code tuning and alignment data usage, significantly impact performance.
